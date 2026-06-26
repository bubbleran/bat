import asyncio
import os
import uuid
from threading import Thread
from typing import Optional, Type

import httpx
import uvicorn
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers import (
    display_agent_card,
    get_stream_response_text,
    new_text_message,
)
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentInterface, Role
from dotenv import load_dotenv
from google.protobuf.json_format import Parse
from jsonschema import ValidationError
from mcp.server import FastMCP
from starlette.applications import Starlette

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
MCP_APPLICATION_DEFAULT_PORT = 9800
DEFAULT_HTTPX_CLIENT_TIMEOUT = 180
DEFAULT_AGENT_CARD_PATH = "./agent.json"


class AgentApplication:
    """Agent Application based on `Starlette`.
    This class sets up an agent application that can handle A2A and MCP
    protocols.

    Configuration:
        Settings are read from ``./config.yaml`` (see :class:`AgentConfig`):
        - ``endpoint.url`` (required): base URL where the agent is hosted.
        - ``endpoint.port``: A2A application port. Defaults to 9900.
        - ``endpoint.mcp_port``: MCP application port. Defaults to 9800.
        - ``model``: provider/name/base_url for the chat model (the env vars
          ``MODEL``/``MODEL_PROVIDER``/``BASE_URL`` still override these).
        - ``telemetry``: OpenTelemetry settings (see :class:`AgentConfig`).
        - ``agent_card``: path to the agent card JSON. Optional; defaults to
          ``./agent.json`` (``AGENT_CARD_PATH`` env still works as a fallback).

        Only secrets stay in the environment (API keys, e.g.
        ``OPENAI_API_KEY``). ``AGENT_CARD_DISPLAY`` (default true) is still read
        from the environment.

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
        # config.yaml (in the cwd) is the source of truth for settings.
        config_path = os.getenv("CONFIG_PATH", "./config.yaml")
        self._config = AgentConfig.load(config_path)

        # Model: install the config.yaml defaults so ChatModelClient picks them
        # up. The env vars MODEL/MODEL_PROVIDER/BASE_URL still override these.
        if self._config.model is not None:
            ChatModelClientConfig._set_defaults(
                provider=self._config.model.provider,
                name=self._config.model.name,
                base_url=self._config.model.base_url,
            )

        # Endpoint (ports + base URL) from config.
        endpoint = self._config.endpoint
        self.a2a_port = (
            endpoint.port
            if endpoint is not None and endpoint.port is not None
            else A2A_APPLICATION_DEFAULT_PORT
        )
        self.mcp_port = (
            endpoint.mcp_port
            if endpoint is not None and endpoint.mcp_port is not None
            else MCP_APPLICATION_DEFAULT_PORT
        )
        self._url = endpoint.url if endpoint is not None else None

        self._agent_card_display = os.getenv(
            "AGENT_CARD_DISPLAY", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Agent card path: config.yaml's `agent_card` is the source of truth;
        # AGENT_CARD_PATH env stays as a fallback, then the default location.
        agent_card_path = (
            self._config.agent_card
            or os.getenv("AGENT_CARD_PATH")
            or DEFAULT_AGENT_CARD_PATH
        )
        self._agent_card = self.load_agent_card(agent_card_path)
        if self._agent_card_display:
            display_agent_card(self._agent_card)

        # Opt-in telemetry, configured from config.yaml's `telemetry` section
        # (no-op unless an output is configured). The agent card name is the
        # default OTel service.name.
        enabled = False
        if self._config.telemetry is not None:
            telemetry = self._config.telemetry
            enabled = bool (telemetry.output)
        else:
            telemetry = TelemetrySettings() 
            logger.debug("No telemetry config found in config.yaml; telemetry is " "disabled by default. To enable, add a `telemetry` section with a valid output to config.yaml.")
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
        self._a2a_server = Starlette(routes=agent_card_routes + json_rpc_routes)

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

    def _build_mcp_application(self) -> FastMCP:
        mcp = FastMCP(
            name=self.agent_card.name,
            host="0.0.0.0",
            port=self.mcp_port,
        )

        @mcp.tool(
            name=f"get_{self.agent_card.name.lower().replace(' ', '_')}_card",
        )
        def get_agent_card() -> str:
            """
            Get the Agent Card as a JSON string, i.e. a description of the Agent
            and its capabilities.

            Returns:
                str: The Agent Card in JSON format.
            """
            return self.agent_card.model_dump_json()

        @mcp.tool(
            name=f"call_{self.agent_card.name.lower().replace(' ', '_')}",
        )
        def call_agent(
            query: str,
            context_id: Optional[str] = None,
            message_id: str = "1",
        ) -> str:
            """
            Call the Agent with a query and return the response.

            Args:
                query (str): The input query for the Agent.
                context_id (Optional[str]): The context ID for the conversation.
                    Defaults to None.
                    If None, a random context ID will be generated calling
                    `uuid.uuid4()`.
                message_id (str): The message ID in the conversation.
                    Defaults to "1".

            Returns:
                str: The Agent's response.
            """

            async def get_response_from_stream() -> str:
                client_factory = ClientFactory(
                    config=ClientConfig(
                        streaming=False,
                        httpx_client=httpx.AsyncClient(
                            timeout=DEFAULT_HTTPX_CLIENT_TIMEOUT
                        ),
                    ),
                )
                client = client_factory.create(card=self.agent_card)
                message = new_text_message(
                    text=query,
                    context_id=context_id or str(uuid.uuid4()),
                    task_id=message_id,
                    role=Role.ROLE_USER,
                )
                stream = client.send_message(message)
                chunk = await anext(stream)

                response = get_stream_response_text(chunk)
                if not response:
                    response = "No valid response received."
                    logger.warning(
                        "No valid response was obtained from the agent stream."
                    )
                return response

            try:
                result = {}

                def runner(coro):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result["value"] = loop.run_until_complete(coro)
                    except Exception as e:
                        result["error"] = e
                    finally:
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                        loop.close()

                t = Thread(
                    target=runner,
                    args=(get_response_from_stream(),),
                )
                t.start()
                t.join()

                if "error" in result:
                    raise result["error"]
                response = result["value"]
            except Exception as e:
                logger.error(f"Error while getting response from Agent: {e}")
                response = (
                    f"An error occurred while processing your request: {e}"
                )
            return response

        return mcp

    def run(
        self,
        expose_mcp: bool = False,
    ) -> None:
        """Run the agent application.

        Args:
            expose_mcp (bool, optional): Whether to expose the MCP protocol.
                Defaults to False.
                **This parameter isn't fully supported yet and may lead to
                unexpected behavior when set to True.**
        """

        if expose_mcp:
            a2a_app = self._a2a_server
            mcp_app = self._build_mcp_application()

            a2a_server_config = uvicorn.Config(
                app=a2a_app,
                host="0.0.0.0",
                port=self.a2a_port,
                reload=False,
            )
            a2a_server = uvicorn.Server(config=a2a_server_config)

            t_a2a = Thread(target=lambda: a2a_server.run())
            t_mcp = Thread(
                target=lambda: asyncio.run(mcp_app.run_streamable_http_async())
            )

            t_a2a.start()
            t_mcp.start()
            t_mcp.join()
            t_a2a.join()
        else:
            a2a_app = self._a2a_server
            uvicorn.run(
                app=a2a_app,
                host="0.0.0.0",
                port=self.a2a_port,
                reload=False,
            )
