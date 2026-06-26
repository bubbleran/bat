from typing import Dict

from a2a.helpers import (
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Task, TaskState
from a2a.utils.errors import (
    InternalError,
    InvalidRequestError,
    UnsupportedOperationError,
)
from google.protobuf.json_format import MessageToDict
from typing_extensions import Any, override

from ..logging import create_logger
from ..telemetry import SpanKind, extract_context, get_tracer
from ..telemetry import attributes as attrs
from .graph import AgentGraph
from .state import AgentTaskResult

logger = create_logger(__name__, "debug")
tracer = get_tracer(__name__)


def _carrier_from_message(message: Any) -> Dict[str, str]:
    """Extract a propagation carrier from an A2A message.

    Returns the message metadata as a flat ``{str: str}`` dict (which may
    contain a W3C ``traceparent``). Never raises: telemetry must not affect
    request handling.
    """
    try:
        metadata = message.metadata
        as_dict = MessageToDict(metadata)
    except Exception:
        return {}
    return {k: v for k, v in as_dict.items() if isinstance(v, str)}


class MinimalAgentExecutor(AgentExecutor):
    """Minimal Agent Executor.

    Minimal implementation of the AgentExecutor interface used by the
    `AgentApplication` class to execute agent tasks.
    """

    def __init__(
        self,
        agent_graph: AgentGraph,
    ):
        self.agent_graph = agent_graph

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if not self._request_ok(context):
            raise InvalidRequestError(
                message="AgentExecutor could not validate the request."
            )

        query = context.get_user_input()
        task = context.current_task
        if not task:
            logger.debug("task not found, creating new task from user message")
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        # Continue the caller's distributed trace: the called agent's spans
        # share the caller's trace_id (propagated via the W3C traceparent), so
        # multi-agent token usage can be recomposed by trace_id downstream.
        parent_ctx = extract_context(_carrier_from_message(context.message))

        # The agent name is derived from the graph class name (the "Graph"
        # suffix dropped), so it travels with the graph instead of being
        # propagated as a separate attribute.
        agent_name = type(self.agent_graph).__name__.removesuffix("Graph")
        try:
            with tracer.start_as_current_span(
                f"{attrs.OP_INVOKE_AGENT} {agent_name}",
                context=parent_ctx,
                kind=SpanKind.INTERNAL,
            ) as span:
                span.set_attribute(
                    attrs.GEN_AI_OPERATION_NAME, attrs.OP_INVOKE_AGENT
                )
                span.set_attribute(attrs.GEN_AI_AGENT_NAME, agent_name)
                if task.context_id:
                    span.set_attribute(
                        attrs.GEN_AI_CONVERSATION_ID, task.context_id
                    )
                    # Group all turns/resumes of this conversation under one
                    # Phoenix Session (each turn is its own trace).
                    span.set_attribute(attrs.SESSION_ID, task.context_id)
                # Per-turn correlation key. Usage/tool-call metadata is no
                # longer sent in the A2A message (clean output); consumers
                # read it from the spans instead, keyed by conversation.id +
                # this task id.
                if task.id:
                    span.set_attribute(attrs.BAT_TASK_ID, task.id)

                config = {"configurable": {"thread_id": task.context_id}}
                keep_streaming = True
                prev_item = None
                async for item in self.agent_graph.astream(query, config):
                    if item != prev_item:
                        if keep_streaming:
                            keep_streaming = await self._process_task_result(
                                task=task,
                                task_result=item,
                                updater=updater,
                            )
                        else:
                            logger.warning(
                                "Artifact has been updated: ignoring item."
                            )
                    prev_item = item
        except Exception as e:
            logger.error(f"An error occurred while streaming the response: {e}")
            raise InternalError(
                message="Error encountered while streaming the response."
            ) from e

    def _request_ok(self, context: RequestContext) -> bool:
        return True

    async def _process_task_result(
        self,
        task: Task,
        task_result: AgentTaskResult,
        updater: TaskUpdater,
    ) -> bool:
        match task_result.task_status:
            case (
                TaskState.TASK_STATE_WORKING
                | TaskState.TASK_STATE_INPUT_REQUIRED
            ) as state:
                message = new_text_message(
                    text=task_result.content,
                    context_id=task.context_id,
                    task_id=task.id,
                )
                await updater.update_status(
                    state=state,
                    message=message,
                )
            case TaskState.TASK_STATE_COMPLETED:
                await updater.add_artifact(
                    [new_text_part(task_result.content)],
                )
                await updater.update_status(
                    state=TaskState.TASK_STATE_COMPLETED,
                )
            case TaskState.TASK_STATE_FAILED:
                raise InternalError(message=task_result.content)
            case _:
                logger.error(f"Unknown task status: {task_result.task_status}")
        return task_result.task_status == TaskState.TASK_STATE_WORKING

    @override
    async def cancel(
        self,
        request: RequestContext,
        event_queue: EventQueue,
    ) -> Task | None:
        raise UnsupportedOperationError(
            message="AgentExecutor does not support task cancel operation yet."
        )
