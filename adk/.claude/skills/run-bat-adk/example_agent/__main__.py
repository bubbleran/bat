"""Launch the Echo agent as an A2A server (Starlette + uvicorn).

Run from this directory so AGENT_CARD_PATH / CONFIG resolve, e.g.:

    URL=localhost PORT=9900 \
        .venv/bin/python -m example_agent

Required env: URL. Optional: PORT (default 9900), CONFIG, AGENT_CARD_PATH.
"""

from bat.agent import AgentApplication
from graph import EchoAgentGraph, EchoAgentState

if __name__ == "__main__":
    agent = AgentApplication(
        AgentGraphType=EchoAgentGraph,
        AgentStateType=EchoAgentState,
    )
    agent.run()
