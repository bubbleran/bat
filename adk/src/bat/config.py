import os
from typing import Any, Dict, List, Literal, Optional, Self
from urllib.parse import urlsplit

import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .logging import create_logger
from .telemetry import TelemetryPrivacy, parse_privacy

logger = create_logger(__name__, "debug")

class EndpointConfig(BaseModel):
    """Where the agent is served. A port in `url` moves to `port`."""

    url: Optional[str] = None
    port: int = 9900

    @model_validator(mode="after")
    def _split_port_from_url(self) -> Self:
        url = self.url or ""
        raw = url if "://" in url else f"http://{url}"
        parts = urlsplit(raw)
        if parts.port is not None:
            if self.port != 9900 and self.port != parts.port:
                raise ValueError(
                    f"Conflicting ports: url has ':{parts.port}' but "
                    f"endpoint.port is {self.port}. Set the port in one place."
                )
            self.port = parts.port
            self.url = raw.replace(f":{parts.port}", "", 1).rstrip("/")
        return self


class ModelConfig(BaseModel):
    """The chat model. MODEL, MODEL_PROVIDER and BASE_URL override it."""

    provider: Optional[
        Literal["anthropic", "deepseek", "nvidia", "ollama", "openai"]
    ] = None
    name: Optional[str] = None
    base_url: Optional[str] = None
    reasoning_effort: Optional[str] = None
    service_tier: Optional[str] = None


class PersistenceConfig(BaseModel):
    """Where conversations are kept between messages."""

    type: Literal["memory", "postgres"]
    url: Optional[str] = None

    @model_validator(mode="after")
    def _postgres_needs_url(self) -> Self:
        if self.type == "postgres" and not self.url:
            raise ValueError("persistence.url is required for postgres.")
        return self


class OutputConfig(BaseModel):
    """One telemetry destination: a JSONL file, an OTLP collector or stdout."""

    type: Literal["local", "remote", "console"]
    file_path: Optional[str] = None
    endpoint: Optional[str] = None


class TelemetrySettings(BaseModel):
    """OpenTelemetry settings; telemetry is on when `output` is not empty.

    `privacy` is how much of the agent's internals may leave the process,
    each level a superset of the one below: `none` (default) exports
    everything, `content` redacts prompts, messages, completions and tool
    arguments, `names` also redacts span names, `full` also tool names.
    """

    service_name: Optional[str] = None
    project_name: Optional[str] = None
    privacy: TelemetryPrivacy = TelemetryPrivacy.NONE
    output: List[OutputConfig] = Field(default_factory=list)

    @field_validator("privacy", mode="before")
    @classmethod
    def _parse_privacy(cls, value: object) -> TelemetryPrivacy:
        # A typo must fail, not fall back to `none` and export in the clear.
        return parse_privacy(value)


class MCPServerConfig(BaseModel):
    """An MCP server. An unreachable server stops the agent."""

    name: str
    url: str
    timeout: float = 60

    @model_validator(mode="before")
    @classmethod
    def _required_is_deprecated(cls, data: object) -> object:
        if isinstance(data, dict) and "required" in data:
            logger.warning(
                "`required` in mcp-servers is deprecated and ignored: "
                "every server must answer at startup."
            )
        return data


class RemoteAgentConfig(BaseModel):
    """A remote A2A agent. An unreachable agent stops the agent."""

    name: str
    url: str
    timeout: float = 60

    @model_validator(mode="before")
    @classmethod
    def _required_is_deprecated(cls, data: object) -> object:
        if isinstance(data, dict) and "required" in data:
            logger.warning(
                "`required` in remote-agents is deprecated and ignored: "
                "every agent must answer at startup."
            )
        return data


class BATAgentConfig(BaseModel):
    """What config.yaml says about the agent. Data only: the `bat.init_*`
    functions open the connections."""

    model_config = ConfigDict(populate_by_name=True)

    agent_card: Optional[str] = None
    endpoint: EndpointConfig = Field(default_factory=EndpointConfig)
    model: Optional[ModelConfig] = None
    persistence: Optional[PersistenceConfig] = None
    checkpoints: bool = False
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    mcp_servers: List[MCPServerConfig] = Field(
        default_factory=list, alias="mcp-servers"
    )
    remote_agents: List[RemoteAgentConfig] = Field(
        default_factory=list, alias="remote-agents"
    )

    @model_validator(mode="after")
    def _checkpoints_mean_memory(self) -> Self:
        if self.checkpoints and self.persistence is None:
            logger.warning(
                "`checkpoints: true` is deprecated: use "
                "`persistence: {type: memory}`."
            )
            self.persistence = PersistenceConfig(type="memory")
        return self


def load_agent_config(path: Optional[str] = None) -> BATAgentConfig:
    """Load config.yaml (`CONFIG_PATH`, default ./config.yaml) without
    touching the network. `.env` is loaded and `${VAR}` expanded first."""
    load_dotenv()
    path = path or os.getenv("CONFIG_PATH", "./config.yaml")
    data = _read_yaml(path)
    model = _model_from_env(data.get("model") or {})
    if model:
        data["model"] = model
    return BATAgentConfig.model_validate(data)


def _read_yaml(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as file:
            data = yaml.safe_load(os.path.expandvars(file.read())) or {}
    else:
        logger.warning(f"No config file at '{path}': using an empty config.")
    return data


def _model_from_env(model: Dict[str, object]) -> Dict[str, object]:
    """`model` with MODEL (`provider:name` allowed), MODEL_PROVIDER and
    BASE_URL applied: the operator sets them."""
    result = dict(model)
    name = os.getenv("MODEL")
    provider = os.getenv("MODEL_PROVIDER")
    base_url = os.getenv("BASE_URL")
    if name and not provider and ":" in name:
        provider, name = name.split(":", 1)
    if name:
        result["name"] = name
    if provider:
        result["provider"] = provider
    if base_url:
        result["base_url"] = base_url
    return result
