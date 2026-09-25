import asyncio
import contextlib
from typing import Any, AsyncIterable, Callable, Dict, Literal, Optional, Type

from a2a.client import ClientConfig, create_client
from a2a.types import (
    AgentCard,
    AgentInterface,
    Message,
    SendMessageRequest,
    StreamResponse,
    TaskState,
)
from httpx import AsyncClient
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START
from typing_extensions import override

from ..agent.config import AgentConfig
from ..agent.state import AgentState, AgentTaskResult, AgentTaskStatus
from ..logging import create_logger
from ..telemetry import SpanKind, inject_context
from ..telemetry import attributes as attrs
from ..telemetry import get_tracer as _get_tracer
from .prebuilt_workflow import PrebuiltWorkflow

logger = create_logger(__name__, level="debug")
tracer = _get_tracer(__name__)


def _inject_traceparent(message: Message, span: Any = None) -> None:
    """Inject the trace context of ``span`` into an outgoing A2A message.

    Adds the W3C ``traceparent`` (and friends) to the message metadata so the
    called agent can continue the same distributed trace. The span is passed
    explicitly rather than read from the current context, because the caller is
    an async generator that yields across tasks. Never raises.
    """
    carrier: Dict[str, str] = {}
    inject_context(carrier, span=span)
    if not carrier:
        return
    try:
        message.metadata.update(carrier)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"Could not inject trace context into message: {e}")


class CallAgentNode(PrebuiltWorkflow):
    """CallAgentNode implements agent-to-agent communication using the A2A
    protocol.

    This workflow abstracts streaming communication between agents as an
    internal mini-graph.
    It handles the complete lifecycle of calling another agent, consuming its
    streamed responses, and managing state updates.

    The internal mini-graph flow is:
        START → call_agent → consume_stream → router
        → (consume_stream | cleanup) → END

    The loop continues consuming streamed responses from the target agent until
    either:
    - The target agent completes its task
    - The target agent requests user input
    - An error occurs

    Example
    -------
    ```python
    from a2a.types import Message
    from bat.agent import AgentGraph, AgentState
    from bat.prebuilt import CallAgentNode
    from langchain_core.runnables import RunnableConfig
    from typing import List, Optional
    from langchain_core.messages import BaseMessage

    class MyAgentState(AgentState):
        agent_input_text: str
        agent_output_text: str
        agent_response_status: str
        agent_input_required: bool = False
        agent_response_content: Optional[str] = None
        ...

    def build_agent_message(config: RunnableConfig, text: str) -> Message:
        \"\"\"Build a message to send to the SMO Agent.\"\"\"
        return Message(
            role="user",
            parts=[{"type": "text", "text": text}],
        )

    class MyAgentGraph(AgentGraph):
        def setup(self, config: AgentConfig) -> None:
            self.call_agent_node = CallAgentNode(
                config=config,
                StateType=MyAgentState,
                loop_name="domain_agent_loop",
                agent_name="SMO Agent",
                input_key="agent_input_text",
                output_key="agent_output_text",
                status_key="agent_status",
                agent_input_required_key="agent_input_required",
                agent_response_status_key="agent_response_status",
                agent_response_content_key="agent_response_content",
                build_message=build_agent_message,
            )
            ...
            self.graph_builder.add_node(
                "call_agent_node",
                self.call_agent_node.as_runnable()
            )
    ```
    """

    def __init__(
        self,
        config: AgentConfig,
        StateType: Type[AgentState],
        loop_name: str,
        agent_name: str,
        build_message: Callable[[RunnableConfig, str], Message],
        *,
        input_key: str = "input",
        output_key: str = "output",
        status_key: str = "status",
        agent_input_required_key: str = "agent_input",
        agent_response_status_key: str = "agent_response_status",
        agent_response_content_key: str = "agent_response_content",
        recursion_limit: int = 50,
    ) -> None:
        """Initialize the CallAgentNode workflow with the given configuration
        and parameters.

        Args:
            config (AgentConfig): Configuration for the agent, including
                checkpointing options.
            StateType (Type[AgentState]): The AgentState schema used in the
                loop.
            loop_name (str): The name of this workflow loop
                (e.g., "domain_agent_loop").
            agent_name (str): The name of the target agent to call
                (e.g., "SMO Agent").
                The agent card will be retrieved from the configuration using
                this name.
            build_message (Callable[[RunnableConfig, str], Message]): Callback
                function to build the request message from the input text.
                Should accept a RunnableConfig and a string, and return an A2A
                Message object.
            input_key (str, optional): A key pointing to a string in the state.
                Defaults to "input".
                The value at this key is used as input to send to the target
                agent.
            output_key (str, optional): A key pointing to a string in the
                state.
                Defaults to "output".
                The value at this key is updated with responses from the target
                agent.
            status_key (str, optional): A key pointing to a string in the
                state.
                Defaults to "status".
                The value at this key is updated with the overall status of
                the communication.
                Useful to display the current operation to the user.
            agent_input_required_key (str, optional): A key pointing to a bool
                in the state.
                Defaults to "agent_input".
                The value at this key is set to True when the target agent
                requests user input.
            agent_response_status_key (str, optional): A key pointing to a
                string in the state.
                Defaults to "agent_response_status".
                The value at this key is updated with the status from the
                target agent.
            agent_response_content_key (str, optional): A key pointing to a
                string in the state.
                Defaults to "agent_response_content".
                The value at this key is updated with the content from the
                target agent.
            recursion_limit (int, optional): Maximum recursion depth for nested
                calls.
                Defaults to 50.
                This prevents infinite loops in agent-to-agent communication.
        """
        keys = [
            input_key,
            output_key,
            status_key,
            agent_input_required_key,
            agent_response_content_key,
            agent_response_status_key,
        ]
        for key in keys:
            if key not in StateType.model_fields:
                logger.error(
                    f"key '{key}' not available in the provided AgentState "
                    f"type '{StateType.__name__}'"
                )
                raise KeyError(
                    f"key '{key}' not available in the provided AgentState "
                    f"type '{StateType.__name__}'"
                )

        # Initialize PrebuiltWorkflow
        super().__init__(
            config=config,
            StateType=StateType,
            loop_name=loop_name,
            agent_name=agent_name,
            build_message=build_message,
            input=input_key,
            output=output_key,
            global_status=status_key,
            agent_input_required=agent_input_required_key,
            agent_response_status=agent_response_status_key,
            agent_response_content=agent_response_content_key,
            recursion_limit=recursion_limit,
        )

    def _router(
        self, state: Type[AgentState]
    ) -> Literal["consume_stream", "cleanup"]:
        """Route between consuming more stream data or cleaning up.

        This method determines the next step in the workflow based on the
        current state. It evaluates two conditions:
        - Whether the stream has completed (stream_done flag)
        - Whether the target agent has requested user input
            (agent_input_required field)

        Args:
            state (Type[AgentState]): The current state of the workflow.

        Returns:
            Literal["consume_stream", "cleanup"]: Returns "cleanup" if the
                stream is done or input is required, otherwise returns
                "consume_stream" to continue processing.
        """
        stream_done_val = self.stream_done
        needs_input_val = bool(getattr(state, self.agent_input_required))

        return (
            "cleanup"
            if stream_done_val or needs_input_val
            else "consume_stream"
        )

    @override
    def _setup(
        self,
        loop_name: str,
        agent_name: str,
        build_message: Callable[[RunnableConfig, str], Message],
        *,
        input: str = "question",
        output: str = "answer",
        global_status: Optional[str] = None,
        agent_input_required: str = "agent_input",
        agent_response_status: str = "agent_response_status",
        agent_response_content: str = "agent_response_content",
        recursion_limit: int = 50,
    ) -> None:
        """Set up the internal mini-graph with nodes and edges.

        This method initializes the internal state and constructs the workflow
        graph with the following nodes:
        - call_agent: Prepares and initiates the call to the target agent
        - consume_stream: Consumes one item from the streaming response queue
        - cleanup: Final cleanup after the workflow completes

        The graph flow is:
            START → call_agent → consume_stream → router
            → (consume_stream | cleanup) → END

        Args:
            loop_name (str): The name of this workflow loop.
            agent_name (str): The name of the target agent to call.
            build_message (Callable[[RunnableConfig, str], Message]): Callback
                to build the request message.
            input (str): State field name for input text.
            output (str): State field name for output text.
            global_status (str): State field name for global status.
            agent_input_required (str): State field name indicating if agent
                needs input.
            agent_response_status (str): State field name for agent-specific
                status.
            agent_response_content (str): State field name for agent-specific
                content.
            recursion_limit (int): Maximum recursion depth for nested calls.
        """
        self._agent_name = agent_name
        self.input = input
        self.output = output
        self.global_status = global_status
        self.agent_input_required = agent_input_required
        self.agent_response_status = agent_response_status
        self.agent_response_content = agent_response_content
        self._build_message = build_message
        self.recursion_limit = recursion_limit
        self.loop_name = loop_name
        self._agent_card = None
        self.stream_done: bool = False
        self._queue: Optional[asyncio.Queue[Optional[AgentTaskResult]]] = None
        self._stream_task: Optional[asyncio.Task[None]] = None

        self.graph_builder.add_node("call_agent", self._call_agent)
        self.graph_builder.add_node("consume_stream", self._consume_stream)
        self.graph_builder.add_node("cleanup", self._cleanup)

        self.graph_builder.add_edge(START, "call_agent")
        self.graph_builder.add_edge("call_agent", "consume_stream")
        self.graph_builder.add_conditional_edges("consume_stream", self._router)
        self.graph_builder.add_edge("cleanup", END)

    @override
    async def _astream(
        self,
        state: Type[AgentState],
        config: RunnableConfig,
    ) -> AsyncIterable[Type[AgentState]]:
        """Stream execution of the internal graph.

        This method orchestrates the streaming execution of the CallAgentNode
        workflow.
        It ensures the recursion limit is set appropriately (minimum 200 or
        the configured limit) to handle potentially deep agent-to-agent
        communication chains.

        Args:
            state (Type[AgentState]): The initial state for the workflow.
            config (RunnableConfig): The runnable configuration, which may
                include checkpointing settings and recursion limits.

        Yields:
            Type[AgentState]: The updated state after each step in the workflow,
                validated against the StateType schema.
        """
        cfg: Dict[str, Any] = dict(config or {})

        if "recursion_limit" not in cfg or (
            isinstance(cfg["recursion_limit"], int)
            and cfg["recursion_limit"] < 200
        ):
            cfg["recursion_limit"] = self.recursion_limit

        stream = self.graph.astream(state, cfg)

        async for item in stream:
            state_item = self.StateType.model_validate(item)
            yield state_item

    async def _call_agent(
        self,
        state: Type[AgentState],
        config: RunnableConfig,
    ) -> AsyncIterable[Type[AgentState]]:
        """Initial node: prepare and start the agent stream.

        This method performs the following operations:
        1. Retrieves the agent card for the target agent (if not already cached)
        2. Resets dynamic state fields (agent_status, agent_content,
            agent_input_required)
        3. Extracts the input text from the state (using input or output key)
        4. Builds the request message using the provided build_message callback
        5. Updates the global status to indicate work is in progress
        6. Starts the background streaming worker to consume responses from the
            target agent

        Args:
            state (Type[AgentState]): The current state of the workflow.
            config (RunnableConfig): The runnable configuration.

        Yields:
            Type[AgentState]: The updated state after initiating the agent call.
        """
        url = self.agent_config.get_a2a_agent_connection(self._agent_name).url

        if (
            self._agent_card is None
            or self._agent_card.name != self._agent_name
        ):
            cards = await self.agent_config.list_agent_cards([self._agent_name])
            self._agent_card = cards[self._agent_name]
            if self._agent_card.supported_interfaces:
                self._agent_card.supported_interfaces[0].url = url
            else:
                self._agent_card.supported_interfaces.append(
                    AgentInterface(
                        url=url,
                        protocol_binding="JSONRPC",
                        protocol_version="2.0",
                    )
                )
            logger.debug(
                f"Node `{self.loop_name}.call_agent`: "
                f"Set agent card URL to {url} for agent {self._agent_card.name}"
            )

        # Reset dynamic fields
        self.stream_done = False
        setattr(state, self.agent_response_status, None)
        setattr(state, self.agent_response_content, None)
        setattr(state, self.agent_input_required, False)

        # Get input text
        text = getattr(state, self.input, "") or getattr(
            state, self.output, None
        )
        if not isinstance(text, str):
            text = str(text)

        request = self._build_message(config, text)

        # Update global state
        if self.global_status:
            setattr(
                state,
                self.global_status,
                AgentTaskStatus.AGENT_TASK_STATUS_WORKING,
            )
        setattr(state, self.output, f"Forwarding request to {self.loop_name}…")

        # Start streaming worker
        await self._start_stream(request)

        yield state

    async def _consume_stream(
        self,
        state: Type[AgentState],
        config: RunnableConfig,
    ) -> AsyncIterable[Type[AgentState]]:
        """Consume one item from the stream queue and update state.

        This method retrieves one item from the background worker's queue and
        updates the state accordingly. It handles three types of queue items:
        1. None (sentinel): Indicates the end of the stream
        2. (status, content) tuple: Regular update from the target agent
        3. Special case: status="input-required" triggers early termination

        When an item is consumed, the method updates both agent-specific fields
        (agent_status, agent_content) and global fields (global_status, output).

        Args:
            state (Type[AgentState]): The current state of the workflow.
            config (RunnableConfig): The runnable configuration.

        Yields:
            Type[AgentState]: The updated state after consuming one stream item.
        """
        q = self._queue
        if q is None:
            # Nothing to consume: consider stream finished
            self.stream_done = True
            yield state
            return

        atr = await q.get()

        # Sentinel: end of stream
        if atr is None:
            self.stream_done = True
            await self._stop_stream()
            yield state
            return

        # Update agent-specific fields
        setattr(state, self.agent_response_status, atr.task_status)
        setattr(state, self.agent_response_content, atr.content)

        # Update global fields
        if self.global_status:
            setattr(state, self.global_status, atr.task_status)
        if atr.content:
            setattr(state, self.output, atr.content)

        setattr(state, self.agent_input_required, atr.requires_input())

        if atr.requires_input():
            # If user input is required, stop the stream
            self.stream_done = True
            await self._stop_stream()

        yield state

    def _cleanup(self, state: Type[AgentState]) -> AgentState:
        """Final cleanup node (only an endpoint)."""
        return state

    # -------------------------------------------------------------------------
    # STREAM HELPERS
    # -------------------------------------------------------------------------
    async def _stop_stream(self) -> None:
        """Stop the streaming worker task and clear the queue.

        1. Cancels the background worker task if it's running
        2. Awaits the task cancellation to ensure clean shutdown
        3. Clears references to the task and queue

        This method is safe to call multiple times and handles the case where
        no streaming task is currently running.
        """
        if self._stream_task:
            self._stream_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._stream_task
        self._stream_task = None
        self._queue = None

    async def _start_stream(self, request: Message) -> None:
        """Start a background worker to consume the agent stream and populate
        the queue.

        1. Stops any existing stream to ensure clean state
        2. Creates a new asyncio Queue for inter-task communication
        3. Spawns a background worker task that:
           - Consumes the agent stream using consume_agent_stream()
           - Maps each stream item to (status, content) tuples
           - Pushes items to the queue for the main workflow to consume
           - Handles errors and ensures a sentinel (None) is sent at the end

        The background worker runs independently, allowing the main workflow
        to consume stream items at its own pace.

        Args:
            request (Message): The A2A message to send to the target agent.
        """
        await self._stop_stream()

        q: asyncio.Queue[Optional[tuple[str, str]]] = asyncio.Queue()
        self._queue = q

        async def _worker():
            """Background worker that consumes agent stream and pushes items
            to queue.
            """
            try:
                async for chunk in self.consume_agent_stream(
                    agent_card=self._agent_card,
                    message=request,
                ):
                    atr = AgentTaskResult.from_send_message_stream(chunk)
                    await q.put(atr)
                    logger.debug(
                        f"Worker: put {(atr.task_status, atr.content)}"
                    )
                    if atr.task_status in [
                        TaskState.TASK_STATE_COMPLETED,
                        TaskState.TASK_STATE_INPUT_REQUIRED,
                        TaskState.TASK_STATE_FAILED,
                    ]:
                        break
            except Exception as e:
                atr = AgentTaskResult(
                    task_status=TaskState.TASK_STATE_FAILED,
                    content=f"CallAgentNode stream error: {e}",
                )
                await q.put(atr)
            finally:
                # Sentinel: end of stream
                await q.put(None)

        self._stream_task = asyncio.create_task(_worker())

    async def consume_agent_stream(
        self,
        agent_card: AgentCard,
        message: Message,
    ) -> AsyncIterable[StreamResponse]:
        """Consume the agent stream from another A2A agent.
        The following operations are performed:
        1. Creates an A2A client configured for streaming (120s timeout)
        2. Opens a CLIENT span and propagates its trace context to the called
            agent via the message metadata (W3C ``traceparent``)
        3. Sends the request message to the target agent
        4. Yields each stream item (events or messages)
        5. Handles errors and ensures the span is closed

        Token usage and tool calls of the called agent are captured through its
        own OpenTelemetry spans (correlated to this call by the shared
        ``trace_id``), not extracted from the A2A messages here.

        Args:
            agent_card (AgentCard): The agent card of the target agent,
                containing connection details and capabilities.
            message (Message): The A2A message to send to the agent.

        Yields:
            StreamResponse: Stream items from the agent, including status
                updates, artifacts, and final messages.

        Raises:
            Exception: If the streaming connection fails or encounters an
                error.
        """
        TIMEOUT = 120.0  # seconds
        span = tracer.start_span(
            f"{attrs.OP_INVOKE_AGENT} {agent_card.name}",
            kind=SpanKind.CLIENT,
        )
        span.set_attribute(attrs.GEN_AI_OPERATION_NAME, attrs.OP_INVOKE_AGENT)
        span.set_attribute(attrs.GEN_AI_AGENT_NAME, agent_card.name)
        # Propagate the trace to the called agent through message metadata.
        _inject_traceparent(message, span)

        client = await create_client(
            agent=agent_card,
            client_config=ClientConfig(
                httpx_client=AsyncClient(timeout=TIMEOUT),
                streaming=True,
            ),
        )
        stream = client.send_message(SendMessageRequest(message=message))
        try:
            async for chunk in stream:
                yield chunk

        except Exception as e:
            logger.error(f"consume_agent_stream: Streaming failed: {e}")
            span.record_exception(e)
            raise
        finally:
            span.end()
