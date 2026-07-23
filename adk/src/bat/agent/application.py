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

from ..logging import create_logger
from ._executor import MinimalAgentExecutor
from .config import AgentConfig
from .graph import AgentGraph
from .state import AgentState

load_dotenv()
logger = create_logger(__name__, "debug")

A2A_APPLICATION_DEFAULT_PORT = 9900
DEFAULT_AGENT_CARD_PATH = "./agent.json"


class AgentApplication:
    """Agent Application based on `Starlette`.
    This class sets up an agent application that serves the A2A protocol.

    Supported Environment Variables:
        - `URL` (required): The base URL where the agent will be hosted.
        - `PORT`: The port for the A2A application. Defaults to 9900.
        - `CONFIG`: Path to a configuration file for the agent.
        Defaults to "config.yaml".
        - `AGENT_CARD_PATH`: Path to the agent card. Defaults to "./agent.json"
        - `AGENT_CARD_DISPLAY`: Whether to display the AgentCard when the
        agent starts. Defaults to True.

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
        self.a2a_port = int(os.getenv("PORT", A2A_APPLICATION_DEFAULT_PORT))

        self._agent_card_display = bool(os.getenv("AGENT_CARD_DISPLAY", "1"))
        agent_card_path = os.getenv("AGENT_CARD_PATH", DEFAULT_AGENT_CARD_PATH)
        self._agent_card = self.load_agent_card(agent_card_path)
        if self._agent_card_display:
            display_agent_card(self._agent_card)

        self._config_path = os.getenv("CONFIG", "config.yaml")
        self._config = AgentConfig.load(self._config_path)

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
            EnvironmentError: If the URL environment variable is not set.
            FileNotFoundError: If the agent card file does not exist.
            ValidationError: If the agent card JSON is invalid.
        """
        logger.info(f"Loading AgentCard from '{agent_card_path}'")
        url = os.getenv("URL")
        if url is None:
            logger.error("URL environment variable is not set.")
            raise EnvironmentError("URL environment variable is not set.")
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
        url = url.rstrip("/")
        port = int(os.getenv("PORT", A2A_APPLICATION_DEFAULT_PORT))

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
