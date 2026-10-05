from abc import ABC, abstractmethod
from typing import AsyncGenerator, AsyncIterable, Type

from langchain_core.runnables import RunnableConfig, RunnableGenerator
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..agent.config import AgentConfig
from ..agent.state import AgentState
from ..logging import create_logger

logger = create_logger(__name__, level="debug")


class PrebuiltWorkflow(ABC):
    """Abstract base class for prebuilt workflows.
    Extend this class to implement custom workflows.
    """

    _StateType: Type[AgentState]
    _graph_builder: StateGraph
    _graph: CompiledStateGraph
    _agent_config: AgentConfig

    def __init__(
        self,
        config: AgentConfig,
        StateType: Type[AgentState],
        *args,
        **kwargs,
    ):
        """Initialize the workflow: set up the graph with `_setup` and compile
        it.

        The graph is compiled without a checkpointer: run as a node of an
        `AgentGraph`, it uses the checkpointer of that graph.

        Args:
            config (AgentConfig): The agent configuration.
            StateType (Type[AgentState]): The AgentState schema of the
                workflow.
            *args, **kwargs: Forwarded to `_setup`.
        """
        self._StateType = StateType
        self._graph_builder = StateGraph(StateType)
        # TODO: poor design, prebuilts should not need config: drop it.
        self._agent_config = config

        self._setup(*args, **kwargs)

        self._graph = self._graph_builder.compile()

    @property
    def StateType(self) -> Type[AgentState]:
        """Get the AgentState type used in the workflow.

        Returns:
            Type[AgentState]: The AgentState type.
        """
        return self._StateType

    @property
    def graph_builder(self) -> StateGraph:
        """Get the state graph builder.

        Returns:
            StateGraph: The state graph builder.
        """
        return self._graph_builder

    @property
    def graph(self) -> CompiledStateGraph:
        """Get the compiled state graph.

        Returns:
            CompiledStateGraph: The compiled state graph.
        """
        return self._graph

    @property
    def agent_config(self) -> AgentConfig:
        """Get the AgentConfig used in the workflow.

        Returns:
            AgentConfig: The AgentConfig.
        """
        return self._agent_config

    @abstractmethod
    def _setup(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Set up the workflow by adding nodes and edges to the graph builder.
        This method should be implemented by subclasses to define the specific
        workflow.
        """
        pass

    @abstractmethod
    async def _astream(
        self,
        state: Type[AgentState],
        config: RunnableConfig,
    ) -> AsyncIterable[Type[AgentState]]:
        """Streams the PrebuiltWorkflow for a single AgentState instance.

        Args:
            state (Type[AgentState]): The AgentState instance to process.
            config (RunnableConfig): The runnable configuration.

        Yields:
            Type[AgentState]: The updated AgentState after each step in the
                loop.
        """
        pass

    async def _astream_wrap(
        self,
        generator: AsyncGenerator[Type[AgentState], None],
        config: RunnableConfig,
    ) -> AsyncIterable[Type[AgentState]]:
        """Streams the PrebuiltWorkflow for each AgentState instance from the
        input generator.
        It expects the input generator to yield exactly one AgentState
        instance; if multiple instances are found, only the first instance is
        processed but the stream is fully consumed.

        Args:
            generator (AsyncGenerator): An async generator yielding AgentState
                instances.
            config (RunnableConfig): The runnable configuration.

        Yields:
            Type[AgentState]: The updated AgentState after each step in the
                loop.
        """
        instance_found = False
        async for item in generator:
            if instance_found:
                logger.warning(
                    "Warning: Multiple instances found in input generator."
                    "Only the first will be processed."
                )
            else:
                instance_found = True
                async for _sub_item in self._astream(item, config):
                    # sub-items are automatically available to the outer
                    # generator thanks to `subgraphs=True` in the astream
                    # method of AgentGraph
                    continue
                yield _sub_item

    def as_runnable(self) -> RunnableGenerator:
        """Returns a RunnableGenerator for the ReActLoop, wrapping the
        `astream` method.
        Allows the ReActLoop to be used as a node (subgraph) in a larger graph.
        """
        return RunnableGenerator(
            self._astream_wrap,
            name=self.__class__.__name__,
        )
