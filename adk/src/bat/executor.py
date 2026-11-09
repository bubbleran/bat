from contextlib import aclosing
from typing import (
    Any,
    Collection,
    ContextManager,
    Dict,
    List,
    Optional,
    Tuple,
    override,
)

from a2a.helpers import (
    get_data_parts,
    get_message_text,
    new_task_from_user_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Task
from a2a.utils.errors import InvalidParamsError, UnsupportedOperationError
from google.protobuf.json_format import MessageToDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, ValuesStreamPart
from opentelemetry.trace import Span
from pydantic import JsonValue

from .a2a_stream import _parse_resume, _serialize_text_and_data, _to_part
from .logging import create_logger
from .telemetry import SpanKind, extract_context, get_tracer
from .telemetry import attributes as attrs

logger = create_logger(__name__, "debug")
tracer = get_tracer(__name__)


def _agent_span(
    agent_name: str,
    task: Task,
    message: Message,
) -> ContextManager[Span]:
    """The `invoke_agent` span of one request, under the caller's trace."""
    metadata = MessageToDict(message.metadata)
    carrier = {k: v for k, v in metadata.items() if isinstance(v, str)}
    return tracer.start_as_current_span(
        f"{attrs.OP_INVOKE_AGENT} {agent_name}",
        context=extract_context(carrier),
        kind=SpanKind.INTERNAL,
        attributes={
            attrs.GEN_AI_OPERATION_NAME: attrs.OP_INVOKE_AGENT,
            attrs.GEN_AI_AGENT_NAME: agent_name,
            attrs.GEN_AI_CONVERSATION_ID: task.context_id,
            attrs.SESSION_ID: task.context_id,
            attrs.BAT_TASK_ID: task.id,
        },
    )


class BATAgentExecutor(AgentExecutor):
    """Run a compiled LangGraph graph for each A2A request.

    The graph's state needs a `messages` field with stable message IDs (use
    `BATAgentState` or `add_messages`). The user's text arrives as a
    `HumanMessage`, and the final
    `AIMessage` is the answer if its ID is new in this request. The
    request's first data part arrives in `data_in` if declared; otherwise
    all data parts are appended to the user's text as JSON. `data_out`
    goes back as a data part next to the answer. Interrupt replies update
    the same fields, excluding the resume control envelope. `interrupt()`
    returns the inner `resume` value, or the text for text-only replies.
    With default data channels, put successive interrupts in separate
    sequential nodes: LastValue allows only one update per graph step.
    """

    def __init__(self, graph: CompiledStateGraph, agent_name: str) -> None:
        if "messages" not in graph.channels:
            raise ValueError(
                "The graph's state has no 'messages' field: add messages: "
                "Annotated[list[AnyMessage], add_messages]."
            )
        self.graph = graph
        self.agent_name = agent_name

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task = await _task(context, event_queue)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        with _agent_span(self.agent_name, task, context.message):
            try:
                await self._run(context.message, task, updater)
            except InvalidParamsError:
                raise
            except Exception as e:
                await _fail_update(e, task, updater)

    @override
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise UnsupportedOperationError(
            message="AgentExecutor does not support task cancel operation yet."
        )

    async def _run(
        self,
        message: Message,
        task: Task,
        updater: TaskUpdater,
    ) -> None:
        config: RunnableConfig = {
            "configurable": {"thread_id": task.context_id}
        }
        text = get_message_text(message)
        data = get_data_parts(message.parts)
        pending = await self._pending(config)
        answer, data = _parse_resume(text, data, waiting=bool(pending))
        values = _state_from_parts(text, data, self.graph.channels)
        input: Command | Dict[str, object]
        if pending:
            input = Command(
                update=values,
                resume={pending[0].id: answer},
            )
        else:
            input = values
        last, initial_message_ids = await self._stream(input, config, updater)
        await _publish_update(
            last,
            await self._pending(config),
            task,
            updater,
            initial_message_ids,
        )

    async def _pending(self, config: RunnableConfig) -> Tuple[Interrupt, ...]:
        """The `interrupt()`s the thread waits on; the first one was asked."""
        pending: Tuple[Interrupt, ...] = ()
        if self.graph.checkpointer:
            # retrieve the last state of the graph for that thread
            snapshot = await self.graph.aget_state(config)
            pending = snapshot.interrupts
        return pending

    async def _stream(
        self,
        input: Command | Dict[str, object],
        config: RunnableConfig,
        updater: TaskUpdater,
    ) -> Tuple[Optional[ValuesStreamPart], set[str]]:
        """Stream progress and return final values plus this request's
        initial message IDs."""
        last: Optional[ValuesStreamPart] = None
        initial_message_ids: Optional[set[str]] = None
        stream = self.graph.astream(
            input,
            config,
            stream_mode=["values", "custom"],
            subgraphs=True,
            version="v2",
        )
        async with aclosing(stream):
            async for part in stream:
                if part["type"] == "custom":
                    progress = updater.new_agent_message(
                        [_to_part(part["data"])]
                    )
                    await updater.start_work(progress)
                if part["type"] == "values" and not part["ns"]:
                    if initial_message_ids is None:
                        messages = _field(part["data"], "messages") or []
                        initial_message_ids = set(
                            _message_id(message) for message in messages
                        )
                    last = part
        return last, initial_message_ids or set()


async def _task(context: RequestContext, event_queue: EventQueue) -> Task:
    """The request's task; a new one is announced first, as A2A requires."""
    task = context.current_task
    if not task:
        task = new_task_from_user_message(context.message)
        await event_queue.enqueue_event(task)
    return task


async def _publish_update(
    last: Optional[ValuesStreamPart],
    pending: Tuple[Interrupt, ...],
    task: Task,
    updater: TaskUpdater,
    initial_message_ids: set[str],
) -> None:
    """Ask the first pending question, or send the answer, or fail without
    one."""
    if pending:
        question = [_to_part(pending[0].value)]
        metadata = None
        if pending[0].response_schema is not None:
            metadata = {
                "interrupt": {"response_schema": pending[0].response_schema}
            }
        await updater.requires_input(
            updater.new_agent_message(question, metadata=metadata)
        )
    else:
        parts = _final_answer(last["data"] if last else {}, initial_message_ids)
        if not parts:
            message = "Agent finished without a final result."
            logger.error(f"[{task.context_id}]: {message}")
            await updater.failed(updater.new_agent_message([_to_part(message)]))
        else:
            await updater.add_artifact(parts)
            await updater.complete()


async def _fail_update(
    error: Exception, task: Task, updater: TaskUpdater
) -> None:
    """Report a failed run by exception type only: its text may hold secrets."""
    logger.error(f"[{task.context_id}]: Agent execution failed", exc_info=error)
    message = f"Agent execution failed ({type(error).__name__})."
    await updater.failed(updater.new_agent_message([_to_part(message)]))


def _state_from_parts(
    text: str, data: List[JsonValue], channels: Collection[str]
) -> Dict[str, object]:
    """Map an incoming message to declared fields, falling back to JSON
    in the text when the graph has no `data_in`. Clear any answer data."""
    input: Dict[str, object] = {}
    if "data_in" in channels:
        input["data_in"] = data[0] if data else None
    elif data:
        label = "Attached data (JSON):"
        text = _serialize_text_and_data(
            f"{text}\n\n{label}" if text else label, data
        )
    if "data_out" in channels:
        input["data_out"] = None
    if text:
        input["messages"] = [HumanMessage(text)]
    return input


def _final_answer(state: object, initial_message_ids: set[str]) -> List[Part]:
    """The final AIMessage with a new ID as text, and `data_out` as data."""
    combined_parts: List[Part] = []
    messages: Optional[List[BaseMessage]] = _field(state, "messages")
    if messages:
        last = messages[-1]
        if (
            isinstance(last, AIMessage)
            and last.text
            and _message_id(last) not in initial_message_ids
        ):
            combined_parts.append(new_text_part(last.text))
    data = _field(state, "data_out")
    if data is not None:
        combined_parts.append(_to_part(data))
    return combined_parts


def _message_id(message: BaseMessage) -> str:
    """Require stable message identity instead of guessing from position."""
    if message.id is None:
        raise ValueError(
            "Messages must have IDs: use add_messages on the graph's "
            "messages field."
        )
    return message.id


def _field(state: object, name: str) -> Any:
    # v2 streams the state in its schema type: a dict or a Pydantic model
    if isinstance(state, dict):
        value = state.get(name)
    else:
        value = getattr(state, name, None)
    return value
