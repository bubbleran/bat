import asyncio
import httpx
import json
import os
import uuid
import uvicorn
from ..logging import create_logger
from ._executor import MinimalAgentExecutor
from .config import AgentConfig
from .graph import AgentGraph
from .state import AgentState
from a2a.client import ClientConfig, ClientFactory
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, Message, TextPart
from dotenv import load_dotenv
from jsonschema import ValidationError
from mcp.server import FastMCP
from starlette.applications import Starlette
from threading import Thread
from typing import Optional, Type

load_dotenv()
_logger = create_logger(__name__, "debug")

A2A_APPLICATION_DEFAULT_PORT = 9900
MCP_APPLICATION_DEFAULT_PORT = 9800
DEFAULT_HTTPX_CLIENT_TIMEOUT = 180

class AgentApplication:
    f"""Agent Application based on `Starlette`.
    This class sets up an agent application that can handle A2A and MCP protocols.
    Supported Environment Variables:
        - `URL` (required): The base URL where the agent will be hosted.
        - `PORT`: The port for the A2A application. Defaults to `{A2A_APPLICATION_DEFAULT_PORT}`.
        - `MCP_PORT`: The port for the MCP application. Defaults to `{MCP_APPLICATION_DEFAULT_PORT}`.
        - `CONFIG`: Path to a configuration file for the agent. Defaults to _"config.yaml"_.

    Attributes
    -------
        agent_card (AgentCard): The agent card containing metadata about the agent.
        agent_graph (AgentGraph): The agent graph that defines the agent's behavior and capabilities.
    
    Example
    -------
    ```python
        from bat.agent import AgentApplication

        agent = AgentApplication(
            agent_card_path='./agent.json',
            agent_graph=MyAgentGraph(),
        )
        agent.run()
    ```
    """

    def __init__(
        self,
        AgentGraphType: Type[AgentGraph],
        AgentStateType: Type[AgentState],
        agent_card_path: str = './agent.json',
    ):
        """
        Initialize the AgentApplication with the given agent card path and agent graph.

        Args:
            agent_graph (AgentGraph): The agent graph implementing the agent's logic.
            agent_card_path (str): The path to the agent card JSON file. Defaults to _"./agent.json"_.
        """
        self.a2a_port = int(os.getenv("PORT", A2A_APPLICATION_DEFAULT_PORT))
        self.mcp_port = int(os.getenv("MCP_PORT", MCP_APPLICATION_DEFAULT_PORT))

        self._agent_card = self.load_agent_card(agent_card_path)
        self._config_path = os.getenv("CONFIG", "config.yaml")
        self._config = AgentConfig.load(self._config_path)

        self._AgentStateType = AgentStateType
        self._AgentGraphType = AgentGraphType
        agent_graph = AgentGraphType(
            config=self._config,
            StateType=AgentStateType,
        )
        self._agent_executor = MinimalAgentExecutor(agent_graph)
        self._request_handler = DefaultRequestHandler(
            agent_executor=self._agent_executor,
            task_store=InMemoryTaskStore(),
        )
        self._a2a_server = A2AStarletteApplication(
            agent_card=self._agent_card,
            http_handler=self._request_handler
        )
    
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
        url = os.getenv("URL")
        if url is None:
            _logger.error("URL environment variable is not set.")
            raise EnvironmentError("URL environment variable is not set.")
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
        url = url.rstrip("/")
        port = int(os.getenv("PORT", A2A_APPLICATION_DEFAULT_PORT))

        try:
            with open(agent_card_path, 'r') as file:
                agent_data = json.load(file)
                agent_data.setdefault('url', f'{url}:{port}')
                agent_card = AgentCard.model_validate(agent_data)
                _logger.debug('Agent Card loaded.')
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Agent card file not found.') from e
        except ValidationError as e:
            raise ValidationError(f'Invalid agent card format.') from e
        except Exception as e:
            raise Exception(f'Error loading agent card: {e}') from e

        return agent_card

    @property
    def agent_graph(self) -> AgentGraph:
        """Get the agent graph."""
        return self._agent_executor.agent_graph
    
    @property
    def agent_card(self) -> AgentCard:
        """Get the agent card."""
        return self._agent_card
    
    def _build_a2a_application(self) -> Starlette:
        """Build the A2A Starlette application.
        
        Returns:
            Starlette: The built Starlette application.
        """
        return self._a2a_server.build()

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
            Get the Agent Card as a JSON string, i.e. a description of the Agent and its capabilities.

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
                context_id (Optional[str]): The context ID for the conversation. Defaults to None.
                    If None, a random context ID will be generated calling `uuid.uuid4()`.
                message_id (str): The message ID in the conversation. Defaults to "1".

            Returns:
                str: The Agent's response.
            """
            async def get_response_from_stream() -> str:
                client_factory = ClientFactory(
                    config=ClientConfig(
                        streaming=False,
                        httpx_client=httpx.AsyncClient(timeout=DEFAULT_HTTPX_CLIENT_TIMEOUT),
                    ),
                )
                client = client_factory.create(card=self.agent_card)
                message = Message(
                    context_id=context_id or str(uuid.uuid4()),
                    message_id=message_id,
                    role="user",
                    parts=[TextPart(text=query)]
                )
                stream = client.send_message(message)
                item = await anext(stream)

                response = None
                if isinstance(item, Message):
                    if item.parts and item.parts[0].root.kind == "text":
                        response = item.parts[0].root.text
                    else:
                        _logger.warning("Received Message with non-text part; ignoring.")
                else:
                    task = item[0]
                    if task.artifacts:
                        artifact = task.artifacts[0]
                        if artifact.parts and artifact.parts[0].root.kind == "text":
                            response = artifact.parts[0].root.text
                        else:
                            _logger.warning("Received Artifact with non-text part; ignoring.")

                if response is None:
                    response = "No valid response received."
                    _logger.warning("No valid response was obtained from the agent stream.")
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
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
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
                _logger.error(f"Error while getting response from Agent: {e}")
                response = f"An error occurred while processing your request: {e}"
            return response

        return mcp

    def run(
        self,
        expose_mcp: bool = False,
    ) -> None:
        """Run the agent application.

        Args:
            expose_mcp (bool, optional): Whether to expose the MCP protocol. Defaults to False.
                **This parameter isn't fully supported yet and may lead to unexpected behavior
                when set to True.**
        """

        if expose_mcp:
            a2a_app = self._build_a2a_application()
            mcp_app = self._build_mcp_application()

            a2a_server_config = uvicorn.Config(
                app=a2a_app,
                host="0.0.0.0",
                port=self.a2a_port,
                reload=False,
            )
            a2a_server = uvicorn.Server(config=a2a_server_config)

            t_a2a = Thread(target=lambda: a2a_server.run())
            t_mcp = Thread(target=lambda: asyncio.run(mcp_app.run_streamable_http_async()))

            t_a2a.start()
            t_mcp.start()
            t_mcp.join()        
            t_a2a.join()
        else:
            a2a_app = self._build_a2a_application()
            uvicorn.run(
                app=a2a_app,
                host="0.0.0.0",
                port=self.a2a_port,
                reload=False,
            )
