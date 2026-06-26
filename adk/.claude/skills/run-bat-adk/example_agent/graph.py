"""Minimal, LLM-free agent used to drive bat-adk end to end.

The graph has a single node that echoes the incoming query back as the
response, so the whole A2A request/response path can be exercised without
any model provider or API key.
"""

from typing import Optional, Self

from bat.agent import AgentGraph, AgentState, AgentTaskResult, AgentTaskStatus
from langgraph.graph import END, START
from typing_extensions import override


class EchoAgentState(AgentState):
    query: str
    response: Optional[str] = None

    @classmethod
    @override
    def from_query(cls, query: str) -> Self:
        return cls(query=query)

    @override
    def to_task_result(self) -> AgentTaskResult:
        return AgentTaskResult(
            task_status=(
                AgentTaskStatus.AGENT_TASK_STATUS_COMPLETED
                if self.response
                else AgentTaskStatus.AGENT_TASK_STATUS_WORKING
            ),
            content=self.response or "Generating response...",
        )


class EchoAgentGraph(AgentGraph):
    @override
    def setup(self, config) -> None:
        def respond(state: EchoAgentState) -> EchoAgentState:
            return state.model_copy(
                update={"response": f"echo: {state.query}"}
            )

        self.graph_builder.add_node("respond", respond)
        self.graph_builder.add_edge(START, "respond")
        self.graph_builder.add_edge("respond", END)
