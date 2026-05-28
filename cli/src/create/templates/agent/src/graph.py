from bat.agent import AgentGraph, AgentState, AgentTaskResult, AgentTaskStatus
from bat.prebuilt import ReActLoop
from langgraph.graph import START, END
from typing import Optional, Self
from typing_extensions import override

__CLIENT_IMPORTS__


class __AGENT_CLASS_NAME__AgentState(AgentState):
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
            task_status=(
                AgentTaskStatus.AGENT_TASK_STATUS_COMPLETED
                if self.response
                else AgentTaskStatus.AGENT_TASK_STATUS_WORKING
            ),
            content=self.response or "Generating response...",
        )


class __AGENT_CLASS_NAME__AgentGraph(AgentGraph):
    @override
    def setup(
        self,
        config,
    ) -> None:
        #Client setup
__CLIENT_SETUP__

        # Graph wiring
        self.graph_builder.add_edge(
            START,
            END,
        )

