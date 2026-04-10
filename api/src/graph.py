from bat.agent import AgentGraph, AgentState, AgentTaskResult
from bat.prebuilt import ReActLoop
from langgraph.graph import START, END
from typing import Optional, Self
from typing_extensions import override

from .llm_clients.example_client import ExampleClient


class ApiAgentState(AgentState):
    query: str
    response: Optional[str] = None

    @classmethod
    @override
    def from_query(
        cls,
        query: str,
    ) -> Self:
        return cls(query=query)

    @override
    def to_task_result(
        self,
    ) -> AgentTaskResult:
        return AgentTaskResult(
            task_status="completed" if self.response else "working",
            content=self.response or "Generating response...",
        )


class ApiAgentGraph(AgentGraph):
    @override
    def setup(
        self,
        config,
    ) -> None:
        #Client setup 
        self.example_client = ExampleClient(
            tools=[],
        )

    # Graph wiring 
    self.graph_builder.add_edge(
        START,
        END,
    )

