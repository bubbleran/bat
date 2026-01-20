import asyncio
import bisect
import time
from ..agent.config import AgentConfig
from ..agent.state import AgentState
from ..chat_model_client import UsageMetadata
from ..logging import create_logger
from .prebuilt_workflow import PrebuiltWorkflow
from a2a.client import ClientConfig, ClientEvent, ClientFactory
from a2a.types import (
    AgentCard,
    Message,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TaskState,
    Part,
)
from a2a.utils.parts import get_text_parts, get_data_parts, get_file_parts
from functools import reduce
from httpx import AsyncClient
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from typing import Any, AsyncIterable, Callable, Dict, List, Literal, Optional, Sequence, Type
from typing_extensions import override

logger = create_logger(__name__, level="debug")
USAGE_METADATA_KEY = "usage"

class CallAgentNode(PrebuiltWorkflow):
    """CallAgentNode implements agent-to-agent communication using the A2A protocol.

    This workflow abstracts streaming communication between agents as an internal mini-graph.
    It handles the complete lifecycle of calling another agent, consuming its streamed responses,
    and managing state updates.

    The internal mini-graph flow is:
        START → call_agent → consume_stream → router → (consume_stream | cleanup) → END

    The loop continues consuming streamed responses from the target agent until either:
    - The target agent completes its task
    - The target agent requests user input
    - An error occurs

    Args
    -------
        config (AgentConfig): Configuration for the agent, including checkpointing options.
        StateType (Type[AgentState]): The AgentState schema used in the loop.
        loop_name (str): The name of this workflow loop (e.g., "domain_agent_loop").
        agent_name (str): The name of the target agent to call (e.g., "SMO Agent").
        build_message (Callable[[RunnableConfig, str], Message]): Callback function to build
            the request message from the input text.
        input (str, optional): A key pointing to a string in the state. Defaults to "question".
            The value at this key is used as input to send to the target agent.
        output (str, optional): A key pointing to a string in the state. Defaults to "answer".
            The value at this key is updated with responses from the target agent.
        global_status (str, optional): A key pointing to a string in the state. Defaults to "status".
            The value at this key is updated with the overall status of the communication.
        agent_input_required (str, optional): A key pointing to a bool in the state. Defaults to "agent_input".
            The value at this key is set to True when the target agent requests user input.
        agent_status (str, optional): A key pointing to a string in the state. Defaults to "agent_status".
            The value at this key is updated with the status from the target agent.
        agent_content (str, optional): A key pointing to a string in the state. Defaults to "agent_content".
            The value at this key is updated with the content from the target agent.
        input_required (str, optional): A key pointing to a bool in the state. Defaults to "input_required".
            The value at this key is set to True when user input is required.
        recursion_limit (int, optional): Maximum recursion depth for nested calls. Defaults to 50.

    Example
    -------
    ```python
    from a2a.types import Message
    from bat.agent import AgentGraph, AgentState
    from bat.prebuilt import CallAgentNode
    from langchain_core.runnables import RunnableConfig
    from typing import List, Optional
    from langchain_core.messages import BaseMessage

    class OrchestratorAgentState(AgentState):
        agent_input_text: str
        agent_output_text: str
        agent_status: str
        agent_input_required: bool = False
        agent_content: Optional[str] = None
        input_required: bool = False
        ...

    def build_agent_message(config: RunnableConfig, text: str) -> Message:
        \"\"\"Build a message to send to the SMO Agent.\"\"\"
        return Message(
            role="user",
            parts=[{"type": "text", "text": text}],
        )

    class OrchestratorGraph(AgentGraph):
        def setup(self, config: AgentConfig) -> None:
            self.call_agent_node = CallAgentNode(
                config=config,
                StateType=OrchestratorAgentState,
                loop_name="domain_agent_loop",
                agent_name="SMO Agent",
                input="agent_input_text",
                output="agent_output_text",
                global_status="agent_status",
                agent_input_required="agent_input_required",
                agent_status="agent_status",
                agent_content="agent_content",
                input_required="input_required",
                build_message=build_agent_message,
            )
            ...
            self.graph_builder.add_node("call_agent_node", self.call_agent_node.as_runnable())
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
        input: str = "question",
        output: str = "answer",
        global_status: str = "status",
        agent_input_required: str = "agent_input",
        agent_status: str = "agent_status",
        agent_content: str = "agent_content",
        input_required: str = "input_required",
        recursion_limit: int = 50,
    ) -> None:
        """Initialize the CallAgentNode workflow with the given configuration and parameters.

        Args:
            config (AgentConfig): Configuration for the agent, including checkpointing options.
            StateType (Type[AgentState]): The AgentState schema used in the loop.
            loop_name (str): The name of this workflow loop (e.g., "domain_agent_loop").
            agent_name (str): The name of the target agent to call (e.g., "SMO Agent").
                The agent card will be retrieved from the configuration using this name.
            build_message (Callable[[RunnableConfig, str], Message]): Callback function to build
                the request message from the input text. Should accept a RunnableConfig and a
                string, and return an A2A Message object.
            input (str, optional): A key pointing to a string in the state. Defaults to "question".
                The value at this key is used as input to send to the target agent.
            output (str, optional): A key pointing to a string in the state. Defaults to "answer".
                The value at this key is updated with responses from the target agent.
            global_status (str, optional): A key pointing to a string in the state. Defaults to "status".
                The value at this key is updated with the overall status of the communication.
                Useful to display the current operation to the user.
            agent_input_required (str, optional): A key pointing to a bool in the state. Defaults to "agent_input".
                The value at this key is set to True when the target agent requests user input.
            agent_status (str, optional): A key pointing to a string in the state. Defaults to "agent_status".
                The value at this key is updated with the status from the target agent.
            agent_content (str, optional): A key pointing to a string in the state. Defaults to "agent_content".
                The value at this key is updated with the content from the target agent.
            input_required (str, optional): A key pointing to a bool in the state. Defaults to "input_required".
                The value at this key is set to True when user input is required.
            recursion_limit (int, optional): Maximum recursion depth for nested calls. Defaults to 50.
                This prevents infinite loops in agent-to-agent communication.
        """
        # Initialize PrebuiltWorkflow 
        super().__init__(
            config=config,
            StateType=StateType,
            loop_name=loop_name,
            agent_name=agent_name,
            build_message=build_message,
            input=input,
            output=output,
            global_status=global_status,
            agent_input_required=agent_input_required,
            agent_status=agent_status,
            agent_content=agent_content,
            input_required=input_required,
            recursion_limit=recursion_limit,
        )
        
        
    def _router(self, state: Type[AgentState]) -> Literal["consume_stream", "cleanup"]:
        """Route between consuming more stream data or cleaning up.
        
        This method determines the next step in the workflow based on the current state.
        It evaluates two conditions:
        - Whether the stream has completed (stream_done flag)
        - Whether the target agent has requested user input (agent_input_required field)
        
        Args:
            state (Type[AgentState]): The current state of the workflow.
            
        Returns:
            Literal["consume_stream", "cleanup"]: Returns "cleanup" if the stream is done
                or input is required, otherwise returns "consume_stream" to continue processing.
        """
        stream_done_val = self.stream_done 
        needs_input_val = bool(getattr(state, self.agent_input_required))

        return "cleanup" if stream_done_val or needs_input_val else "consume_stream"
    
    @override
    def _setup(
        self,
        loop_name: str,
        agent_name: str,
        build_message: Callable[[RunnableConfig, str], Message],
        *,
        input: str = "question",
        output: str = "answer",
        global_status: str = "status",
        agent_input_required: str = "agent_input",
        agent_status: str = "agent_status",
        agent_content: str = "agent_content",
        input_required: str = "input_required",
        recursion_limit: int = 50,
    ) -> None:
        """Set up the internal mini-graph with nodes and edges.
        
        This method initializes the internal state and constructs the workflow graph
        with the following nodes:
        - call_agent: Prepares and initiates the call to the target agent
        - consume_stream: Consumes one item from the streaming response queue
        - cleanup: Final cleanup after the workflow completes
        
        The graph flow is:
            START → call_agent → consume_stream → router → (consume_stream | cleanup) → END
        
        Args:
            loop_name (str): The name of this workflow loop.
            agent_name (str): The name of the target agent to call.
            build_message (Callable[[RunnableConfig, str], Message]): Callback to build the request message.
            input (str): State field name for input text.
            output (str): State field name for output text.
            global_status (str): State field name for global status.
            agent_input_required (str): State field name indicating if agent needs input.
            agent_status (str): State field name for agent-specific status.
            agent_content (str): State field name for agent-specific content.
            input_required (str): State field name indicating if user input is required.
            recursion_limit (int): Maximum recursion depth for nested calls.
        """
        self._agent_name = agent_name
        self.input = input
        self.output = output
        self.global_status = global_status
        self.agent_input_required = agent_input_required
        self.agent_status = agent_status
        self.agent_content = agent_content
        self.input_required = input_required
        self._build_message = build_message
        self.recursion_limit = recursion_limit
        self.loop_name = loop_name
        
        self._agent_card = None  
        self.stream_done: bool = False
        self._queue: Optional[asyncio.Queue[Optional[tuple[str, str]]]] = None
        self._stream_task: Optional[asyncio.Task[None]] = None
        self._usage_metadatas: List[tuple[float,UsageMetadata]] = []

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
        
        This method orchestrates the streaming execution of the CallAgentNode workflow.
        It ensures the recursion limit is set appropriately (minimum 200 or the configured
        limit) to handle potentially deep agent-to-agent communication chains.
        
        Args:
            state (Type[AgentState]): The initial state for the workflow.
            config (RunnableConfig): The runnable configuration, which may include
                checkpointing settings and recursion limits.
        
        Yields:
            Type[AgentState]: The updated state after each step in the workflow,
                validated against the StateType schema.
        """
        cfg: Dict[str, Any] = dict(config or {})

        if "recursion_limit" not in cfg or (isinstance(cfg["recursion_limit"], int) and cfg["recursion_limit"] < 200):
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
        2. Resets dynamic state fields (agent_status, agent_content, agent_input_required)
        3. Extracts the input text from the state (using input or output key)
        4. Builds the request message using the provided build_message callback
        5. Updates the global status to indicate work is in progress
        6. Starts the background streaming worker to consume responses from the target agent
        
        Args:
            state (Type[AgentState]): The current state of the workflow.
            config (RunnableConfig): The runnable configuration.
            
        Yields:
            Type[AgentState]: The updated state after initiating the agent call.
        """
        
        if self._agent_card is None or self._agent_card.name != self._agent_name:
            cards = await self.agent_config.list_agent_cards([self._agent_name])
            self._agent_card = cards[self._agent_name]
            
        # Reset dynamic fields
        self.stream_done = False
        setattr(state, self.agent_status, None)
        setattr(state, self.agent_content, None)
        setattr(state, self.agent_input_required, False)

        # Get input text
        text = getattr(state, self.input, "") or getattr(state, self.output, None) 
        if not isinstance(text, str):
            text = str(text)

        request = self._build_message(config, text)

        # Update global state
        setattr(state, self.global_status, "working")
        setattr(state, self.output, f"Forwarding request to {self.loop_name}…")
        setattr(state, self.input_required, False)

        # Start streaming worker
        await self._start_stream(request)

        yield state

    async def _consume_stream(
        self,
        state: Type[AgentState],
        config: RunnableConfig,
    ) -> AsyncIterable[Type[AgentState]]:
        """Consume one item from the stream queue and update state.
        
        This method retrieves one item from the background worker's queue and updates
        the state accordingly. It handles three types of queue items:
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

        item = await q.get()

        # Sentinel: end of stream
        if item is None:
            self.stream_done = True
            await self._stop_stream()
            yield state
            return

        status, content = item

        # Update agent-specific fields
        setattr(state, self.agent_status, status)
        setattr(state, self.agent_content, content)

        # Update global fields
        setattr(state, self.global_status, status)
        if content:
            setattr(state, self.output, content)

        needs_input = (status == "input-required")
        setattr(state, self.agent_input_required, needs_input)
        setattr(state, self.input_required, needs_input)

        if needs_input:
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
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
        self._stream_task = None
        self._queue = None

    async def _start_stream(self, request: Message) -> None:
        """Start a background worker to consume the agent stream and populate the queue.
        
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
            """Background worker that consumes agent stream and pushes items to queue."""
            try:
                async for item in self.consume_agent_stream(
                    agent_card=self._agent_card,
                    message=request,
                ):
                    status, content = self._map_stream_item(item)
                    status = status.value
                    await q.put((status, content))
                    logger.debug(f"Worker: put {(status, content)}")
                    if status in ("completed", "error", "input-required"):
                        break

            except Exception as e:
                await q.put(("error", f"AgentLoop stream error: {e}"))

            finally:
                # Sentinel: end of stream
                await q.put(None)

        self._stream_task = asyncio.create_task(_worker())



    @staticmethod
    def _parts_to_text(parts: Sequence[Part] | None) -> str:
        """Convert A2A message parts to a text string.
        
        This utility method extracts text content from A2A message parts, which can
        contain different types of content (text, data, files). It prioritizes:
        1. Text parts: Joined with double newlines
        2. Data parts: Converted to string representation
        3. File parts: Summarized as "Length: N files"
        
        Args:
            parts (Sequence[Part] | None): The message parts to convert.
            
        Returns:
            str: The extracted text content, or an empty string if no parts are provided.
        """
        if not parts:
            return ""
        texts = get_text_parts(list(parts))
        if texts:
            return "\n\n".join(texts)
        datas = get_data_parts(list(parts))
        if datas:
            return str(datas)
        files = get_file_parts(list(parts))
        if files:
            return f"Length: {len(files)} files"
        return ""


    def _map_stream_item(self, item: ClientEvent | Message) -> tuple[TaskState, str]:
        """Map a stream item to (status, content) tuple.
        Handles different types of stream items:
        1. Message: Direct message response → (completed, message_text)
        2. TaskArtifactUpdateEvent: Artifact update → (completed, artifact_text)
        3. TaskStatusUpdateEvent: Status update → (state, status_message)
        4. Other events: Default → (working, empty_string)
        
        The extracted status and content are used to update the workflow state
        and provide feedback to the user.
        
        Args:
            item (ClientEvent | Message): Stream item from the agent, either a task
                update event or a direct message.
            
        Returns:
            tuple[TaskState, str]: A tuple containing the task state and the extracted
                text content.
        """
        if (isinstance(item, Message)):
            return TaskState.completed, CallAgentNode._parts_to_text(item.parts)
        
        _, update = item

        if isinstance(update, TaskArtifactUpdateEvent):
            artifact = update.artifact
            msg = CallAgentNode._parts_to_text(artifact.parts)
            return TaskState.completed, msg

        if isinstance(update, TaskStatusUpdateEvent):
            status_obj = update.status
            message = status_obj.message
            msg = CallAgentNode._parts_to_text(message.parts)
            st = getattr(status_obj, "state", None)

            return st, msg if st else (TaskState.working, msg)

        return TaskState.working, ""


    async def consume_agent_stream(
        self,
        agent_card: AgentCard,
        message: Message,
    ) -> AsyncIterable[ClientEvent | Message]:
        """Consume the agent stream from another A2A agent.
        The following operations are performed:
        1. Creates an A2A client configured for streaming (120s timeout)
        2. Sends the request message to the target agent
        3. Yields each stream item (events or messages)
        4. Tracks usage metadata from the stream for metrics collection
        5. Handles errors and ensures proper cleanup
        
        The method automatically extracts and stores usage metadata from each
        stream item, which can later be retrieved using get_usage_metadata().
        
        Args:
            agent_card (AgentCard): The agent card of the target agent, containing
                connection details and capabilities.
            message (Message): The A2A message to send to the agent.
        
        Yields:
            ClientEvent | Message: Stream items from the agent, including status
                updates, artifacts, and final messages.
                
        Raises:
            Exception: If the streaming connection fails or encounters an error.
        """
        TIMEOUT = 120.0  # seconds
        client_factory = ClientFactory(
            ClientConfig(
                httpx_client=AsyncClient(timeout=TIMEOUT),
                streaming=True,
            )
        )
        client = client_factory.create(card=agent_card)
        stream = client.send_message(request=message)
        try:
            async for item in stream:
                # Track usage metadata
                t=time.time()
                if isinstance(item, Message):
                    metadata = item.metadata
                else:
                    event = item[1]
                    metadata = event.metadata if event else None

                if metadata and USAGE_METADATA_KEY in metadata:
                    usage = metadata[USAGE_METADATA_KEY]
                    self._usage_metadatas.append((t, UsageMetadata.model_validate(usage)))
                yield item

        except Exception as e:
            logger.error(f"consume_agent_stream: Streaming failed: {e}")
            raise
    
    def get_usage_metadata(
        self,
        from_timestamp: Optional[float] = None,
    ) -> UsageMetadata:
        """
        Get the aggregated usage metadata collected from the agent communication stream.

        Args:
            from_timestamp (Optional[float]): If provided, only usage metadata after this timestamp will be considered.
                If None, all usage metadata will be considered.
        
        Returns:
            UsageMetadata: The aggregated usage metadata from all stream events.
        """
        # lower bound binary search to find the first usage metadata after the timestamp
        i = bisect.bisect_left(
            self._usage_metadatas,
            0 if from_timestamp is None else from_timestamp,
            key=lambda x: x[0]
        )
        # call reduce to aggregate usage metadata
        return reduce(
            lambda acc, metadata: acc + metadata[1],
            self._usage_metadatas[i:],
            UsageMetadata(),
        )
