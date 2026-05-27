from bat.agent import AgentApplication
from src.graph import __AGENT_CLASS_NAME__AgentGraph, __AGENT_CLASS_NAME__AgentState

if __name__ == '__main__':
    agent = AgentApplication(
        AgentGraphType=__AGENT_CLASS_NAME__AgentGraph,
        AgentStateType=__AGENT_CLASS_NAME__AgentState,
    )
    agent.run()
