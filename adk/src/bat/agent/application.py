import os
from typing import Type

import uvicorn
from a2a.helpers import display_agent_card
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentInterface
from dotenv import load_dotenv
from google.protobuf.json_format import Parse
from jsonschema import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..chat_model_client import ChatModelClientConfig
from ..logging import create_logger
from ..telemetry import TelemetryConfig, setup_telemetry
from ._executor import MinimalAgentExecutor
from .config import AgentConfig, TelemetrySettings
from .graph import AgentGraph
from .state import AgentState

load_dotenv()
logger = create_logger(__name__, "debug")

A2A_APPLICATION_DEFAULT_PORT = 9900
DEFAULT_AGENT_CARD_PATH = "./agent.json"


class AgentApplication:
    """Agent Application based on `Starlette`.
    This class sets up an agent application that serves the A2A protocol.

    Configuration:
        Settings are read from ``./config.yaml`` (see :class:`AgentConfig`):
        - ``endpoint.url`` (required): base URL where the agent is hosted.
        - ``endpoint.port``: A2A application port. Defaults to 9900.
        - ``model``: provider/name/base_url for the chat model (the env vars
          ``MODEL``/``MODEL_PROVIDER``/``BASE_URL`` still override these).
        - ``telemetry``: OpenTelemetry settings (see :class:`AgentConfig`).
        - ``agent_card``: path to the agent card JSON. Optional; defaults to
          ``./agent.json`` (``AGENT_CARD_PATH`` env still works as a fallback).

        Only secrets stay in the environment (API keys, e.g.
        ``OPENAI_API_KEY``). ``CONFIG_PATH`` (default ``./config.yaml``) and
        ``AGENT_CARD_DISPLAY`` (default true) are still read from the
        environment.

    Attributes
    -------
        agent_card (AgentCard): The agent card containing metadata about the
            agent.
        agent_graph (AgentGraph): The agent graph that defines the agent's
            behavior and capabilities.

    Example
    -------
    ```python
        from bat.agent import AgentApplication

        agent = AgentApplication(
            AgentGraphType=MyAgentGraph,
            AgentStateType=MyAgentState,
        )
        agent.run()
    ```
    """

    def __init__(
        self,
        AgentGraphType: Type[AgentGraph],
        AgentStateType: Type[AgentState],
    ):
        """Initialize the AgentApplication with the given agent card path and
        agent graph.

        Args:
            AgentGraphType (Type[AgentGraph]): The class to use to instantiate
                the agent graph.
            AgentStateType (Type[AgentState]): The class to use to instantiate
                the agent state.
        """
        config_path = os.getenv("CONFIG_PATH", "./config.yaml")
        self._config = AgentConfig.load(config_path)


        if self._config.model is not None:
            ChatModelClientConfig._set_defaults(
                provider=self._config.model.provider,
                name=self._config.model.name,
                base_url=self._config.model.base_url,
            )

        endpoint = self._config.endpoint
        self.a2a_port = (
            endpoint.port
            if endpoint is not None and endpoint.port is not None
            else A2A_APPLICATION_DEFAULT_PORT
        )
        self._url = endpoint.url if endpoint is not None else None

        self._agent_card_display = (
            os.getenv("AGENT_CARD_DISPLAY", "true").strip().lower() == "true"
        )


        agent_card_path = (
            self._config.agent_card
            or os.getenv("AGENT_CARD_PATH")
            or DEFAULT_AGENT_CARD_PATH
        )
        self._agent_card = self.load_agent_card(agent_card_path)
        if self._agent_card_display:
            display_agent_card(self._agent_card)


        enabled = False
        if self._config.telemetry is not None:
            telemetry = self._config.telemetry
            enabled = bool(telemetry.output)
        else:
            telemetry = TelemetrySettings()
            logger.debug(
                "No telemetry config found in config.yaml; telemetry is "
                "disabled by default. To enable, add a `telemetry` section "
                "with a valid output to config.yaml."
            )
        telemetry_config = TelemetryConfig.from_settings(
            enabled=enabled,
            service_name=telemetry.service_name,
            project_name=telemetry.project_name,
            outputs=[o.model_dump() for o in telemetry.output],
            default_service_name=self._agent_card.name,
        )
        setup_telemetry(config=telemetry_config)

        self._AgentStateType = AgentStateType
        self._AgentGraphType = AgentGraphType
        agent_graph = AgentGraphType(
            config=self._config,
            StateType=AgentStateType,
        )
        self._agent_executor = MinimalAgentExecutor(agent_graph)
        self._request_handler = DefaultRequestHandlerV2(
            agent_executor=self._agent_executor,
            task_store=InMemoryTaskStore(),
            agent_card=self._agent_card,
            extended_agent_card=self._agent_card,
        )
        agent_card_routes = create_agent_card_routes(
            agent_card=self._agent_card,
        )
        json_rpc_routes = create_jsonrpc_routes(
            request_handler=self._request_handler,
            rpc_url="/",
        )
        ping_routes = [
            Route("/ping", self._ping, methods=["GET"]),
        ]
        self._a2a_server = Starlette(
            routes=ping_routes + agent_card_routes + json_rpc_routes
        )

    async def _ping(self, _: Request) -> JSONResponse:
        """Liveness endpoint.

        Replies `"pong"` when the agent is up and its agent card endpoint is
        exposed.
        """
        return JSONResponse("pong")

    def load_agent_card(
        self,
        agent_card_path: str,
    ) -> AgentCard:
        """Load the Agent Card from a JSON file.

        Args:
            agent_card_path (str): The path to the Agent Card JSON file.

        Returns:
            AgentCard: The loaded Agent Card.

        Raises:
            Exception: For general errors during loading.
            EnvironmentError: If ``endpoint.url`` is not set in config.yaml.
            FileNotFoundError: If the agent card file does not exist.
            ValidationError: If the agent card JSON is invalid.
        """
        logger.info(f"Loading AgentCard from '{agent_card_path}'")
        url = self._url
        if url is None:
            logger.error("Agent endpoint URL is not set.")
            raise EnvironmentError(
                "Agent endpoint URL is not set. Add 'endpoint.url' to "
                "config.yaml."
            )
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
        url = url.rstrip("/")
        port = self.a2a_port

        interfaceUrl = f"{url}:{port}"
        try:
            with open(agent_card_path, "r") as file:
                json_str = file.read()
                agent_card = Parse(json_str, AgentCard())

                if agent_card.supported_interfaces:
                    logger.error("interfaces already defined: will be ignored")
                    raise Exception(
                        "AgentCard's supportedInterfaces field is set, please "
                        "remove supportedInterfaces from your agent card."
                    )
                agent_card.supported_interfaces.append(
                    AgentInterface(
                        url=interfaceUrl,
                        protocol_binding="JSONRPC",
                        protocol_version="2.0",
                    )
                )
        except FileNotFoundError as e:
            raise FileNotFoundError("Agent card file not found.") from e
        except ValidationError as e:
            raise ValidationError("Invalid agent card format.") from e
        except Exception as e:
            raise Exception(f"Error loading agent card: {e}") from e

        return agent_card

    @property
    def agent_graph(self) -> AgentGraph:
        """Get the agent graph."""
        return self._agent_executor.agent_graph

    @property
    def agent_card(self) -> AgentCard:
        """Get the agent card."""
        return self._agent_card

    def run(self) -> None:
        """Run the agent application, serving the A2A protocol."""
        uvicorn.run(
            app=self._a2a_server,
            host="0.0.0.0",
            port=self.a2a_port,
            reload=False,
        )
