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

try:
    # The telemetry privacy floor, one for every agent of the blueprint:
    # config.yaml can raise it but never lower it. The Dockerfile writes this
    # module (at the blueprint root) from its TELEMETRY_PRIVACY_FLOOR build
    # arg right before freezing, so it exists only inside the image; run from
    # source, config.yaml alone decides.
    from telemetry_floor import TELEMETRY_PRIVACY_FLOOR
except ModuleNotFoundError:
    TELEMETRY_PRIVACY_FLOOR = "none"


def run() -> None:
    AgentApplication(
        AgentGraphType=__AGENT_CLASS_NAME__AgentGraph,
        AgentStateType=__AGENT_CLASS_NAME__AgentState,
        telemetry_privacy_floor=TELEMETRY_PRIVACY_FLOOR,
    ).run()
