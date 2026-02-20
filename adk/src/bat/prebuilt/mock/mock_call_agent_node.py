from .mock_prebuilt_workflow import MockPrebuiltWorkflow
from ...agent.config import AgentConfig
from ...agent.state import AgentState
from a2a.types import Message
from langchain_core.runnables import RunnableConfig
from typing import Callable, Type
from typing_extensions import override


class MockCallAgentNode(MockPrebuiltWorkflow):
    """Mock version of CallAgentNode that modifies state without actual agent calls.
    
    Args:
        mock_output: The simulated output string that will be returned instead of
            making actual agent calls. This value will be used to populate the
            output field and agent_response_content field in the state.
            
        for all other parameters, see CallAgentNode.
    
    Example:
        mock_call_agent = MockCallAgentNode(
            config=config,
            StateType=MyAgentState,
            loop_name="call_external_agent",
            agent_name="ExternalAgent",
            input="agent_question",
            output="answer",
            global_status="status",
            agent_input_required="agent_input",
            agent_response_status="agent_response_status",
            agent_response_content="agent_response_content",
            build_message=build_agent_message,
            mock_output="Operation completed successfully.",
        )
    """
    
    def __init__(
        self,
        config: AgentConfig,
        StateType: Type[AgentState],
        loop_name: str,
        agent_name: str,
        build_message: Callable[[RunnableConfig, str], Message],
        mock_output: str,
        *,
        input: str = "question",
        output: str = "answer",
        global_status: str = "status",
        agent_input_required: str = "agent_input",
        agent_response_status: str = "agent_response_status",
        agent_response_content: str = "agent_response_content",
        recursion_limit: int = 50,
    ) -> None:
        super().__init__(mock_output)
        self.config = config
        self.StateType = StateType
        self.loop_name = loop_name
        self.agent_name = agent_name
        self.build_message = build_message
        self.input = input
        self.output = output
        self.global_status = global_status
        self.agent_input_required = agent_input_required
        self.agent_response_status = agent_response_status
        self.agent_response_content = agent_response_content
        self.recursion_limit = recursion_limit

    @override
    def modify_state(
        self,
        state: Type[AgentState],
    ) -> Type[AgentState]:
        """Modify state like CallAgentNode would, but with mock output instead of real calls."""
        return state.model_copy(update={
            self.output: self.mock_output,
            self.agent_response_content: self.mock_output,
            self.agent_response_status: "completed",
            self.global_status: "completed",
            self.agent_input_required: False,
        })
