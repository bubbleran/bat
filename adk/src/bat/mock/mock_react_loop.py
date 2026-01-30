from .mock_node import MockNode
from ..agent.config import AgentConfig
from ..agent.state import AgentState
from ..chat_model_client import ChatModelClient
from typing import Optional, Type
from typing_extensions import override


class MockReActLoop(MockNode):
    """Mock version of ReActLoop that modifies state without actual LLM calls.
    
    Args:
        mock_output: The simulated output string that will be returned instead of
            making actual LLM calls. This value will be used to populate the
            output_key field in the state.
        for all other parameters, see ReActLoop.
    
    Example:
        mock_react_loop = MockReActLoop(
            config=config,
            StateType=MyAgentState,
            loop_name="monitoring_react_loop",
            chat_model_client=chat_client,
            input_key="question",
            output_key="answer",
            messages_key="history",
            status_key="status",
            mock_output="All systems operational.",
        )
    """

    def __init__(
        self,
        config: AgentConfig,
        StateType: Type[AgentState],
        loop_name: str,
        chat_model_client: ChatModelClient,
        mock_output: str,
        input_key: str = "input",
        output_key: str = "output",
        messages_key: Optional[str] = None,
        status_key: Optional[str] = None,
    ) -> None:
        super().__init__(mock_output)
        self.config = config
        self.StateType = StateType
        self.loop_name = loop_name
        self.chat_model_client = chat_model_client
        self.input_key = input_key
        self.output_key = output_key
        self.messages_key = messages_key
        self.status_key = status_key

    @override
    def modify_state(
        self,
        state: Type[AgentState],
    ) -> Type[AgentState]:
        """Modify state like ReActLoop would, but with mock output instead of real LLM calls."""
        updates = {self.output_key: self.mock_output}
        if self.status_key:
            updates[self.status_key] = "completed"
        return state.model_copy(update=updates)
