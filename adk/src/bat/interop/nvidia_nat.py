from typing import Optional

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from bat.agent import AgentApplication, AgentState


class NATConnector:
    """LangGraph connector between NAT A2A server and BAT agent.
    Provides a LangGraph wrapper that converts NAT message format to BAT agent state.

    !!! NOT supporting multi-turn conversations at the moment.
    """

    class _WState(MessagesState):
        """WState wraps MessagesState adding a query field"""

        query: str
        bat_state: Optional[AgentState] = None

    def _extract_query(
        state: MessagesState,
    ) -> str:
        """Extract query string from the last message in state."""
        messages = state["messages"]
        if len(messages) == 0:
            raise RuntimeError(
                "Received messages list with 0 messages, expected >= 1."
            )
        last = messages[-1]
        query = last.content.strip()
        return query

    def __init__(
        self,
        agent_app: AgentApplication,
    ):
        """
        Initialize the connector and build the graph.

        Args:
            agent_app(AgentApplication): the agent application to wrap for NAT.
        """
        self.agent_app = agent_app
        self.StateType = agent_app._AgentStateType
        self.bat_agent_graph = agent_app.agent_graph.compiled_graph

    def _input_adaptor(
        self,
        state: _WState,
    ) -> _WState:
        """Extract query from messages."""
        state["query"] = NATConnector._extract_query(state)
        return state

    async def _output_adaptor(
        self,
        state: _WState,
    ) -> _WState:
        """Run BAT agent and convert result to message."""
        if "bat_state" not in state or state["bat_state"] is None:
            print("setting bat state")
            state["bat_state"] = self.StateType.from_query(state["query"])
        else:
            state["bat_state"].update_after_checkpoint_restore(state["query"])
        out = await self.bat_agent_graph.ainvoke(
            state["bat_state"].model_dump()
        )

        if hasattr(out, "response"):
            text = out.response or ""
        elif isinstance(out, dict) and "response" in out:
            text = out["response"] or ""
        else:
            text = str(out)
        state["messages"].append(AIMessage(content=text))
        return state

    def compile(self) -> CompiledStateGraph:
        """
        Builds and compiles a LangGraph compatible with the format required by NAT.

        Returns:
            Compiled LangGraph wrapping the AgentApplication Graph.
        """
        # Graph wrapper
        self.wgraph = StateGraph(NATConnector._WState)

        self.wgraph.add_node("input_adaptor", self._input_adaptor)
        self.wgraph.add_node("output_adaptor", self._output_adaptor)

        self.wgraph.add_edge(START, "input_adaptor")
        self.wgraph.add_edge("input_adaptor", "output_adaptor")
        self.wgraph.add_edge("output_adaptor", END)
        compiled = self.wgraph.compile()
        return compiled
