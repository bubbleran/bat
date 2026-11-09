import os

# The A2A SDK reads its instrumentation switch at import: the ADK opens its
# own spans instead.
os.environ.setdefault("OTEL_INSTRUMENTATION_A2A_SDK_ENABLED", "false")

from .a2a_stream import RemoteAgentResponse, RemoteTaskContext
from .config import BATAgentConfig, load_agent_config
from .executor import BATAgentExecutor
from .mcp_tools import init_mcp_clients, list_mcp_tools
from .model import init_model
from .persistence import init_checkpointer
from .remote_agents import RemoteAgentClient, init_a2a_clients
from .server import BATAgentServer, load_agent_card
from .state import BATAgentState
from .telemetry import init_telemetry

__all__ = [
    "BATAgentConfig",
    "RemoteTaskContext",
    "BATAgentState",
    "BATAgentExecutor",
    "BATAgentServer",
    "RemoteAgentClient",
    "RemoteAgentResponse",
    "init_a2a_clients",
    "init_checkpointer",
    "init_mcp_clients",
    "init_model",
    "init_telemetry",
    "list_mcp_tools",
    "load_agent_card",
    "load_agent_config",
]
