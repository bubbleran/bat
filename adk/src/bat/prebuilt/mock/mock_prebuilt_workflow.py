from abc import ABC, abstractmethod
from typing import Type

from langchain_core.runnables import Runnable, RunnableLambda

from ...agent.state import AgentState


class MockPrebuiltWorkflow(ABC):
    """Base class for mock nodes. Subclasses must override modify_state."""

    def __init__(self, mock_output: str) -> None:
        self.mock_output = mock_output

    @abstractmethod
    def modify_state(
        self,
        state: Type[AgentState],
    ) -> Type[AgentState]:
        """Modify the state with mock output. Must be overridden by subclasses.

        Args:
            state: The current agent state.

        Returns:
            The modified state.
        """
        pass

    def as_runnable(self) -> Runnable:
        """Return this mock node as a LangChain Runnable."""
        return RunnableLambda(self.modify_state)
