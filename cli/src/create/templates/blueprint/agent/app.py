"""Entrypoint for the __AGENT_NAME__ agent.

`run` is what the blueprint's __main__.py imports once this agent is
selected, which is why the import of the graph lives in this module rather
than in the dispatcher.
"""

from bat.agent import AgentApplication

from .src.graph import (
    __AGENT_CLASS_NAME__AgentGraph,
    __AGENT_CLASS_NAME__AgentState,
)


def run() -> None:
    AgentApplication(
        AgentGraphType=__AGENT_CLASS_NAME__AgentGraph,
        AgentStateType=__AGENT_CLASS_NAME__AgentState,
    ).run()
