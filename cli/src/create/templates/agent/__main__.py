from bat.agent import AgentApplication
from src.graph import (
    __AGENT_CLASS_NAME__AgentGraph,
    __AGENT_CLASS_NAME__AgentState,
)

try:
    # The telemetry privacy floor: config.yaml can raise it but never lower
    # it. The Dockerfile writes this module from its TELEMETRY_PRIVACY_FLOOR
    # build arg right before freezing, so it exists only inside the image;
    # run from source, config.yaml alone decides.
    from telemetry_floor import TELEMETRY_PRIVACY_FLOOR
except ModuleNotFoundError:
    TELEMETRY_PRIVACY_FLOOR = "none"

if __name__ == "__main__":
    agent = AgentApplication(
        AgentGraphType=__AGENT_CLASS_NAME__AgentGraph,
        AgentStateType=__AGENT_CLASS_NAME__AgentState,
        telemetry_privacy_floor=TELEMETRY_PRIVACY_FLOOR,
    )
    agent.run()
