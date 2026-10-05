from abc import ABC, abstractmethod
from contextlib import aclosing, suppress
from typing import Any, AsyncIterable, Dict, Optional, Tuple, Type

from a2a.helpers import new_text_message
from a2a.types import Message, Role
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import ValidationError

from ..logging import create_logger
from .config import AgentConfig
from .state import AgentState, AgentTaskResult, AgentTaskStatus

logger = create_logger(__name__, level="debug")


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
    _graph_builder: StateGraph
    _graph: CompiledStateGraph

    def __init__(
        self,
        config: AgentConfig,
        StateType: Type[AgentState],
    ):
        """Initialize the AgentGraph with a state graph and optional
        checkpointing and logger.
        Compile the state graph and set up the logger if the logger_name
        is provided.

        Args:
            graph_builder (StateGraph): The state graph builder.
            use_checkpoint (bool): Whether to use checkpointing.
                Defaults to False.
            logger_name (Optional[str]): The name of the logger to use.
                Defaults to None.
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

    async def astream(
        self,
        query: str,
        config: RunnableConfig,
    ) -> AsyncIterable[AgentTaskResult]:
        """Asynchronously stream results from the agent graph based on the
        query and configuration.

        While the graph runs, WORKING results are yielded as progress, without
        consecutive duplicates. When it stops, exactly one final result is
        yielded:
        - INPUT_REQUIRED if the graph is paused on an interrupt;
        - otherwise `to_task_result()` of the graph's final state;
        - FAILED if the graph raised or finished without a final result.

        Other results of intermediate states are ignored: the answer always
        comes from the final state. The executor publishes every result
        yielded here, so an override must follow these rules.

        This method performs the following steps:
        1. Looks for a checkpoint associated with the provided configuration.
        2. If no checkpoint is found, creates a new agent state from the query,
            using the `from_query` method of the `StateType`.
        3. If a checkpoint is found, restores the state from the checkpoint and
            updates it with the query using the
            `update_after_checkpoint_restore` method. A checkpoint that is not
            a valid `StateType` is dropped: the state starts over from the
            query.
        4. Prepares the input for the graph execution, wrapping the state in a
            `Command` if the `is_waiting_for_human_input` method of the state
            returns `True`.
        5. Executes the graph with the `astream` method, including nested
            graphs, so that prebuilt workflows can report progress.
        6. Converts each streamed state with `to_task_result`, and yields it
            if it is new WORKING progress. A state of a nested graph that is
            not a `StateType` is skipped.
        7. Once the graph stops, yields the final result.

        This method prints debug logs in the format `[<thread_id>]: <message>`.

        Args:
            query (str): The query to process.
            config (RunnableConfig): Configuration for the runnable.
        Returns:
            AsyncIterable[AgentTaskResult]: An asynchronous iterable of agent
                task results.
        """
        thread_id = config.get("configurable", {}).get("thread_id")

        checkpoint = None
        if self._memory:
            snapshot = await self._graph.aget_state(config)
            if snapshot.created_at:
                checkpoint = snapshot.values
        if checkpoint is None:
            logger.debug(f"[{thread_id}]: No checkpoint")
            state = self.StateType.from_query(query)
            logger.debug(f"[{thread_id}]: State initialized")
        else:
            logger.debug(f"[{thread_id}]: Checkpoint found")
            try:
                state = self.StateType.model_validate(checkpoint)
            except ValidationError:
                logger.warning(
                    f"[{thread_id}]: Checkpoint is not a valid state, "
                    "starting over",
                    exc_info=True,
                )
                # Passing every field marks them all as set: LangGraph skips a
                # field that is None and unset, so its old value would stay.
                state = self.StateType.model_construct(
                    **dict(self.StateType.from_query(query))
                )
            else:
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
            f"[{thread_id}]: Graph execution started "
            f"{'with Command' if state.is_waiting_for_human_input() else ''}"
        )

        working = AgentTaskStatus.AGENT_TASK_STATUS_WORKING
        final_result: Optional[AgentTaskResult] = None
        progress: Optional[AgentTaskResult] = None
        try:
            async with aclosing(stream):
                async for namespace, value in stream:
                    result = self._to_task_result(namespace, value)
                    if not namespace:
                        final_result = result
                    if (
                        result is not None
                        and result.task_status == working
                        and result != progress
                    ):
                        progress = result
                        yield result
            if self._memory:
                interrupted = await self._interrupt_result(config)
                if interrupted is not None:
                    final_result = interrupted
        except Exception as e:
            logger.exception(f"[{thread_id}]: Agent execution failed")
            final_result = AgentTaskResult(
                task_status=AgentTaskStatus.AGENT_TASK_STATUS_FAILED,
                content=f"Agent execution failed ({type(e).__name__}).",
            )
        if final_result is None or final_result.task_status == working:
            final_result = AgentTaskResult(
                task_status=AgentTaskStatus.AGENT_TASK_STATUS_FAILED,
                content="Agent finished without a final result.",
            )
        yield final_result
        logger.debug(f"[{thread_id}]: Graph execution completed")

    def _to_task_result(
        self,
        namespace: Tuple[str, ...],
        value: Dict[str, Any],
    ) -> Optional[AgentTaskResult]:
        """Convert a streamed state to an `AgentTaskResult`.

        An empty `namespace` means the state comes from this graph. Otherwise
        it comes from a nested graph, which may use its own schema: such a
        state gives None.
        """
        result = None
        if namespace:
            with suppress(ValidationError):
                result = self.StateType.model_validate(value).to_task_result()
        else:
            result = self.StateType.model_validate(value).to_task_result()
        return result

    async def _interrupt_result(
        self,
        config: RunnableConfig,
    ) -> Optional[AgentTaskResult]:
        """Return an INPUT_REQUIRED result for the interrupt the graph is
        paused on, or None if there is none.
        """
        interrupts = []
        snapshot = await self._graph.aget_state(config)
        for task in snapshot.tasks:
            interrupts.extend(task.interrupts)
        return AgentTaskResult(
            task_status=AgentTaskStatus.AGENT_TASK_STATUS_INPUT_REQUIRED,
            content=interrupts[0].value,
        ) if interrupts else None

    def draw_mermaid(
        self,
        file_path: Optional[str] = None,
    ) -> None:
        """Draw the agent graph in Mermaid format. If a file path is provided,
        save the diagram to the file, otherwise print it to the console.

        Args:
            file_path (Optional[str]): The path to the file where the Mermaid
                diagram should be saved.
        """
        mermaid_str = self._graph.get_graph().draw_mermaid()
        if file_path:
            with open(file_path, "w") as f:
                f.write(mermaid_str)
        else:
            print(mermaid_str)

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
