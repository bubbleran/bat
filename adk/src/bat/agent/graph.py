from abc import ABC, abstractmethod
from typing import AsyncIterable, Dict, List, Optional, Type

from a2a.helpers import new_text_message
from a2a.types import Message, Role
from langchain_core.messages import ToolCall
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from ..chat_model_client import ChatModelClient, TraceMetadata, UsageMetadata
from ..logging import create_logger
from ..prebuilt import CallAgentNode
from .config import AgentConfig
from .state import AgentState, AgentTaskResult, AgentTaskStatus

logger = create_logger(__name__, level="debug")

USAGE_METADATA_KEY = "usage"


class AgentGraph(ABC):
    """Abstract base class for agent graphs.

    Extend this class to implement the specific behavior of an agent.

    Example
    -------
    ```python
    from bat.agent import AgentGraph, AgentState
    from langgraph.runnables import RunnableConfig
    from langgraph.graph import StateGraph

    class MyAgentState(BaseModel):
        # Your state here
        # ...
        pass

    class MyAgentGraph(AgentGraph):
        def __init__(self):
            # Define the agent graph using langgraph.graph.StateGraph class
            graph_builder = StateGraph(MyAgentState)
            # Add nodes and edges to the graph as needed ...
            super().__init__(
                graph_builder=graph_builder,
                use_checkpoint=True,
                logger_name="my_agent"
            )
            self._log("Graph initialized", "info")

        # Your nodes logic here
        # ...
    ```
    """

    StateType: Type[AgentState]
    _chat_model_clients: Dict[str, ChatModelClient] = {}
    _graph_builder: StateGraph
    _graph: CompiledStateGraph

    def __init__(
        self,
        config: AgentConfig,
        StateType: Type[AgentState],
    ):
        """Initialize the AgentGraph with a state graph and optional checkpointing and logger.
        Compile the state graph and set up the logger if the logger_name is provided.

        Args:
            graph_builder (StateGraph): The state graph builder.
            use_checkpoint (bool): Whether to use checkpointing. Defaults to False.
            logger_name (Optional[str]): The name of the logger to use. Defaults to None.
        """
        self.StateType = StateType
        self._graph_builder = StateGraph(StateType)
        self.setup(config)
        self._common_setup(config)

    @property
    def graph_builder(self) -> StateGraph:
        """Get the state graph builder.

        Returns:
            StateGraph: The state graph builder.
        """
        return self._graph_builder

    @property
    def compiled_graph(self) -> CompiledStateGraph:
        """Get the compiled state graph.

        Returns:
            CompiledStateGraph: The compiled state graph of the agent.
        """
        return self._graph

    @abstractmethod
    def setup(
        self,
        config: AgentConfig,
    ) -> None:
        """Set up the agent graph with the provided configuration.
        Subclasses must implement this method.

        Args:
            config (AgentConfig): The agent configuration.
        """
        pass

    def _common_setup(
        self,
        config: AgentConfig,
    ) -> None:
        """Common setup logic for the agent graph.

        Args:
            config (AgentConfig): The agent configuration.
        """
        self._memory = MemorySaver() if config.checkpoints else None
        self._graph = self._graph_builder.compile(checkpointer=self._memory)

        self._usage_buffer = UsageMetadata()

    async def astream(
        self,
        query: str,
        config: RunnableConfig,
    ) -> AsyncIterable[AgentTaskResult]:
        """Asynchronously stream results from the agent graph based on the query and configuration.
        This method performes the following steps:
        1. Looks for a checkpoint associated with the provided configuration.
        2. If no checkpoint is found, creates a new agent state from the query,
            using the `from_query` method of the `StateType`.
        3. If a checkpoint is found, restores the state from the checkpoint and updates it with the query
            using the `update_after_checkpoint_restore` method.
        4. Prepares the input for the graph execution, wrapping the state in a `Command` if the
            `is_waiting_for_human_input` method of the state returns `True`.
        5. Executes the graph with the `astream` method, passing the input and configuration.
        6. For each item in the stream:
            - If it is an interrupt, yields an `AgentTaskResult` with the status
            `input-required`. This enables human-in-the-loop interactions.
            - Otherwise, validates the item as an `StateType` and converts it to an
            `AgentTaskResult` using the `to_task_result` method of the state. Then it yields the result.

        This method prints debug logs in the format `[<thread_id>]: <message>`.

        Args:
            query (str): The query to process.
            config (RunnableConfig): Configuration for the runnable.
        Returns:
            AsyncIterable[AgentTaskResult]: An asynchronous iterable of agent task results.
        """
        thread_id = config.get("configurable", {}).get("thread_id")

        checkpoint = self._memory.get(config) if self._memory else None
        if checkpoint is None:
            logger.debug(f"[{thread_id}]: No checkpoint")
            state = self.StateType.from_query(query)
            logger.debug(f"[{thread_id}]: State initialized")
        else:
            logger.debug(f"[{thread_id}]: Checkpoint found")
            channel_values = checkpoint.get("channel_values", {})
            state = self.StateType.model_validate(channel_values)
            logger.debug(f"[{thread_id}]: State restored")
            state.update_after_checkpoint_restore(query)
            logger.debug(f"[{thread_id}]: State updated")

        input = (
            Command(resume=state)
            if state.is_waiting_for_human_input()
            else state
        )

        stream = self._graph.astream(
            input=input,
            config=config,
            stream_mode="values",
            subgraphs=True,
        )
        logger.debug(
            f"[{thread_id}]: Graph execution started {'with Command' if state.is_waiting_for_human_input() else ''}"
        )

        try:
            async for item in stream:
                try:
                    state_item = self.StateType.model_validate(item[1])
                    task_result_item = state_item.to_task_result()
                    logger.debug(
                        f"[{thread_id}]: Yielding AgentTaskResult: [{task_result_item.task_status}] {task_result_item.content}"
                    )
                    yield task_result_item
                except Exception as ve:
                    logger.error(f"[{thread_id}]: Validation error: {ve}")
                    yield AgentTaskResult(
                        task_status=AgentTaskStatus.AGENT_TASK_STATUS_FAILED,
                        content="Invalid state format",
                    )
        except Exception as e:
            logger.error(f"[{thread_id}]: Error during stream processing: {e}")
            yield AgentTaskResult(
                task_status=AgentTaskStatus.AGENT_TASK_STATUS_FAILED,
                content=f"Stream error: {str(e)}",
            )

        # Checkpoints are possible only if memory is enabled
        if self._memory:
            current_state = self._graph.get_state(config=config)
            intr = (
                current_state.tasks[0].interrupts[0]
                if current_state.tasks
                else None
            )
            if intr:
                logger.debug(f"[{thread_id}]: Yielding Interrupt: {intr.value}")
                yield AgentTaskResult(
                    task_status=AgentTaskStatus.AGENT_TASK_STATUS_INPUT_REQUIRED,
                    content=intr.value,
                )
        logger.debug(f"[{thread_id}]: Graph execution completed")

    def draw_mermaid(
        self,
        file_path: Optional[str] = None,
    ) -> None:
        """Draw the agent graph in Mermaid format. If a file path is provided, save the diagram to
        the file, otherwise print it to the console.

        Args:
            file_path (Optional[str]): The path to the file where the Mermaid diagram should be saved.
        """
        mermaid_str = self._graph.get_graph().draw_mermaid()
        if file_path:
            with open(file_path, "w") as f:
                f.write(mermaid_str)
        else:
            print(mermaid_str)

    def _pop_usage_metadata_from_buf(
        self,
    ) -> UsageMetadata:
        """Pop the usage metadata from the buffer and return it."""
        usage = self._usage_buffer
        self._usage_buffer = UsageMetadata()
        return usage

    def _get_usage_metadata(
        self,
        from_timestamp: Optional[float] = None,
    ) -> UsageMetadata:
        """Get the total usage metadata for the graph including all ChatModelClient instances
        and other agents called using the `consume_agent_stream` method.

        Args:
            from_timestamp (Optional[float]): If provided, only usage after this timestamp is considered.
                If None, all usage metadata is returned.
        Returns:
            UsageMetadata: The total usage metadata for the graph.
        """
        total = self._pop_usage_metadata_from_buf()
        for _, value in self.__dict__.items():
            if isinstance(value, (ChatModelClient, CallAgentNode)):
                total += value.get_usage_metadata(
                    from_timestamp=from_timestamp,
                )
        return total

    def _get_trace_metadata(
        self,
        from_timestamp: Optional[float] = None,
    ) -> TraceMetadata:
        """Get aggregated trace metadata for the graph from components that expose it.

        Components opt in by implementing a callable method:
            get_trace_metadata(from_timestamp: Optional[float]) -> TraceMetadata

        Args:
            from_timestamp (Optional[float]): If provided, only entries after this
                timestamp are returned.

        Returns:
            TraceMetadata: Aggregated trace metadata. The ``tool_calls`` list is
                empty when no component has trace data to report.
        """
        tool_calls: List[ToolCall] = []

        for _, value in self.__dict__.items():
            get_trace_metadata = getattr(value, "get_trace_metadata", None)
            if not callable(get_trace_metadata):
                continue

            trace_item = get_trace_metadata(from_timestamp=from_timestamp)
            if not isinstance(trace_item, TraceMetadata):
                continue

            tool_calls.extend(trace_item.tool_calls)

        return TraceMetadata(tool_calls=tool_calls)

    @staticmethod
    def build_message(
        config: RunnableConfig,
        text: str,
    ) -> Message:
        cfg = config["configurable"] or {}
        thread_id = cfg.get("thread_id", "default")
        task_id = cfg.get("task_id", None)

        return new_text_message(
            text=text,
            context_id=thread_id,
            task_id=task_id,
            role=Role.ROLE_USER,
        )
