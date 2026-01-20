import asyncio
import yaml
from ..logging import create_logger
from a2a.client import A2ACardResolver
from a2a.types import AgentCard
from httpx import AsyncClient
from langchain.tools import BaseTool
from langchain_mcp_adapters.sessions import StreamableHttpConnection as MCPConnection
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, BeforeValidator, Field
from typing import Dict, List, Literal, Tuple
from typing_extensions import Annotated, Self

_logger = create_logger(__name__, "debug")

DEFAULT_TIMEOUT = 60

class MCPServerConfig(BaseModel):
    f"""Configuration for an MCP server connection.

    Attributes:
        name (str): The name of the MCP server.
        url (str): The URL of the MCP server.
        required (bool): Whether the MCP server is required to be reachable. Defaults to True.
            When set to False, connection failures will be treated as if the server is available
            but always returns empty responses (e.g., no tools).
        timeout (int): The timeout in seconds for connecting to the MCP server. Defaults to {DEFAULT_TIMEOUT}.

    Note: the `required` attribute is not well supported yet.
    """
    name: str
    url: str
    required: bool = True
    timeout: int = DEFAULT_TIMEOUT

"""Type for inter-agent communication protocol ('A2A' or 'MCP')."""
InteragentCommunicationProtocol = Annotated[
    Literal['A2A', 'MCP'],
    BeforeValidator(lambda v: v.upper())
]

class RemoteAgentConfig(BaseModel):
    f"""Configuration for remote A2A or MCP agent connections.

    Attributes:
        name (str): The name of the remote agent.
        url (str): The URL of the remote agent.
        protocol (InteragentCommunicationProtocol): The communication protocol used by the remote agent ('A2A' or 'MCP').
        required (bool): Whether the remote agent is required to be reachable. Defaults to True.
            When set to False, connection failures will be treated by ignoring the agent, i.e. no
            `AgentCard` will be returned.
        timeout (int): The timeout in seconds for connecting to the remote agent. Defaults to {DEFAULT_TIMEOUT}.

    Note: the `required` attribute is not well supported yet.
    """
    name: str
    url: str
    protocol: InteragentCommunicationProtocol
    required: bool = True
    timeout: int = DEFAULT_TIMEOUT

class A2AConnection(BaseModel):
    url: str
    timeout: int

class AgentConfig(BaseModel):
    """
    Agent Configuration, including MCP servers and remote agents.

    Attributes
    -------
        mcp_servers (List[MCPServerConfig]): List of MCP server configurations.
        remote_agents (List[RemoteAgentConfig]): List of remote agent configurations.

    Methods
    -------
        is_remote_agent_required(agent_name: str) -> bool:
            Check if a remote agent is marked as required in the configuration.
        is_mcp_server_required(server_name: str) -> bool:
            Check if an MCP server is marked as required in the configuration.
        load(path: str = 'config.yaml') -> AgentConfig:
            Load the agent configuration from a YAML file.
        get_tools(mcp_server_names: List[str]) -> List[BaseTool]:
            Retrieve tools from the specified MCP servers.
        get_agent_cards(agent_names: List[str]) -> List[AgentCard]:
            Retrieve agent cards from the specified remote agents.
    """
    checkpoints: bool = Field(default=False)
    mcp_servers: List[MCPServerConfig] = Field(default=[], alias='mcp-servers')
    remote_agents: List[RemoteAgentConfig] = Field(default=[], alias='remote-agents')

    _mcp_server_connections: dict[str, MCPConnection] = {}
    _mcp_agent_connections: dict[str, MCPConnection] = {}
    _a2a_agent_connections: dict[str, A2AConnection] = {}

    _required_mcp_servers: Dict[str, bool] = {}
    _required_remote_agents: Dict[str, bool] = {}
    _mcp_servers_aliases: Dict[str, str] = {}
    _remote_agents_aliases: Dict[str, str] = {}

    def list_mcp_servers_names(self) -> List[str]:
        """List the names of all configured MCP servers.

        Returns:
            List[str]: List of MCP server names.
        """
        return [server.name for server in self.mcp_servers]

    def list_remote_agents_names(self) -> List[str]:
        """List the names of all configured remote agents.

        Returns:
            List[str]: List of remote agent names.
        """
        return [agent.name for agent in self.remote_agents]

    def is_remote_agent_required(
        self,
        agent_name: str,
    ) -> bool:
        """Check if a remote agent is marked as required in the configuration.

        Args:
            agent_name (str): The name of the remote agent.

        Returns:
            bool: True if the agent is required, False otherwise (or if the agent is not in the configuration).
        """
        if agent_name in self._required_remote_agents:
            return self._required_remote_agents.get(agent_name, False)
        return self._required_remote_agents.get(
            self._remote_agents_aliases.get(agent_name, ""),
            False,
        )

    def is_mcp_server_required(
        self,
        server_name: str,
    ) -> bool:
        """Check if an MCP server is marked as required in the configuration.

        Args:
            server_name (str): The name of the MCP server.

        Returns:
            bool: True if the server is required, False otherwise (or if the server is not in the configuration).
        """
        if server_name in self._required_mcp_servers:
            return self._required_mcp_servers.get(server_name, False)
        return self._required_mcp_servers.get(
            self._mcp_servers_aliases.get(server_name, ""),
            False,
        )

    @classmethod
    def load(
        cls,
        path: str = 'config.yaml',
    ) -> Self:
        """Load the agent configuration from a YAML file. If the YAML file is not found,
        an empty configuration is used.

        Args:
            path (str): The path to the configuration YAML file.

        Returns:
            AgentConfig: The loaded agent configuration or an empty configuration if the file is not found.
        
        Raises:
            ValueError: If the configuration file cannot be loaded or validated.
        """
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            cfg = cls.model_validate(data)
            cfg._mcp_server_connections = _build_mcp_server_connections(
                mcp_servers=cfg.mcp_servers,
            )
            cfg._a2a_agent_connections, cfg._mcp_agent_connections = _build_remote_agent_connections(
                remote_agents=cfg.remote_agents,
            )
            for server in cfg.mcp_servers:
                cfg._required_mcp_servers[server.name] = server.required
            for agent in cfg.remote_agents:
                cfg._required_remote_agents[agent.name] = agent.required
            
            asyncio.run(cfg._build_mcp_servers_aliases_map())
            asyncio.run(cfg._build_remote_agents_aliases_map())

            _logger.info(f"Agent configuration loaded [checkpoints={cfg.checkpoints}] [#agents={len(cfg.remote_agents)}] [#mcp={len(cfg.mcp_servers)}]")
        except FileNotFoundError as e:
            _logger.warning(f"Configuration file not found. The Agent won't have access to remote agents or MCP servers.")
            cfg = cls()
        except Exception as e:
            raise ValueError(f"Failed to load and validate agent configuration: {e}") from e
        return cfg
    
    def get_mcp_server_connection(
        self,
        server_name: str,
    ) -> MCPConnection | None:
        """Get the MCP connection for a given MCP server.

        Args:
            server_name (str): The name of the MCP server.

        Returns:
            MCPConnection | None: The MCP connection object if the MCP server is found, otherwise None
        """
        if server_name in self._mcp_server_connections:
            return self._mcp_server_connections[server_name]
        server_alias = self._mcp_servers_aliases.get(server_name, "")
        if server_alias in self._mcp_server_connections:
            return self._mcp_server_connections[server_alias]
        return None

    def get_mcp_agent_connection(
        self,
        agent_name: str,
    ) -> MCPConnection | None:
        """Get the MCP connection for a given remote agent, if it uses MCP protocol.

        Args:
            agent_name (str): The name of the remote agent.

        Returns:
            MCPConnection | None: The MCP connection object if the agent is found and uses MCP protocol, otherwise None.
        """
        if agent_name in self._mcp_agent_connections:
            return self._mcp_agent_connections[agent_name]
        agent_alias = self._remote_agents_aliases.get(agent_name, "")
        if agent_alias in self._mcp_agent_connections:
            return self._mcp_agent_connections[agent_name]
        return None
    
    def get_a2a_agent_connection(
        self,
        agent_name: str,
    ) -> A2AConnection | None:
        """Get the A2A connection for a given remote agent name, if it uses A2A protocol.

        Args:
            agent_name (str): The name of the remote agent.

        Returns:
            A2AConnection | None: The A2A connection object if the agent is found and uses A2A protocol, otherwise None.
        """
        if agent_name in self._a2a_agent_connections:
            return self._a2a_agent_connections[agent_name]
        agent_alias = self._remote_agents_aliases.get(agent_name, "")
        if agent_alias in self._a2a_agent_connections:
            return self._a2a_agent_connections[agent_alias]
        return None

    async def list_tools(
        self,
        mcp_server_names: List[str],
    ) -> List[BaseTool]:
        """Retrieve tools from the specified MCP servers.

        Args:
            mcp_server_names (List[str]): List of MCP server names to retrieve tools from.

        Returns:
            List[BaseTool]: List of tools available from the specified MCP servers.
        
        Raises:
            ConnectionError: If a `required` MCP server cannot be connected to.
        """
        connections = {}
        for name in mcp_server_names:
            if conn := self.get_mcp_server_connection(name):
                connections[name] = conn

        client = MultiServerMCPClient(connections=connections)
        tools = []
        for name in mcp_server_names:
            if name in connections:
                # TODO: Parallelize this across servers
                try:
                    server_tools = await client.get_tools(server_name=name)
                    tools.extend(server_tools)
                except Exception as e:
                    if self.is_mcp_server_required(name):
                        raise ConnectionError(f"Failed to retrieve tools from MCP server '{name}': {e}") from e
            else:
                _logger.warning(f"MCP Server {name} not found in configuration.")
        return tools
    
    async def list_agent_cards(
        self,
        agent_names: List[str],
    ) -> Dict[str, AgentCard]:
        """Retrieve agent cards from the specified remote agents.
        If a non-required remote agent cannot be reached, it is not included in the result.

        Args:
            agent_names (List[str]): List of remote agent names to retrieve agent cards from.

        Returns:
            Dict[str, AgentCard]: Dictionary of Agent Cards available from the specified remote agents.

        Raises:
            ConnectionError: If a `required` remote agent cannot be connected to.
        """
        mcp_connections: dict[str, MCPConnection] = {}
        for name in agent_names:
            if conn := self.get_mcp_agent_connection(name):
                mcp_connections[name] = conn

        mcp_client = MultiServerMCPClient(connections=mcp_connections)
        client = None
        agent_cards = {}
        for name in agent_names:
            try:
                if a2a_conn := self.get_a2a_agent_connection(name):
                    if client is None or client.timeout != a2a_conn.timeout:
                        client = AsyncClient(timeout=a2a_conn.timeout)
                    card_resolver = A2ACardResolver(
                        httpx_client=client,
                        base_url=a2a_conn.url,
                    )
                    agent_card = await card_resolver.get_agent_card()
                    agent_cards[name] = agent_card
                elif name in mcp_connections:
                    async with mcp_client.session(name) as session:
                        call_tool_result = await session.call_tool('get_agent_card')
                    if call_tool_result.isError:
                        raise RuntimeError("The MCP remote agent 'get_agent_card' tool returned an error.")
                    agent_card = AgentCard.model_validate(call_tool_result.result)
                    agent_cards[name] = agent_card
                else:
                    _logger.warning(f"Remote Agent {name} not found in configuration.")
            except Exception as e:
                if self.is_remote_agent_required(name):
                    raise ConnectionError(f"Failed to retrieve agent card from remote agent '{name}': {e}") from e
        return agent_cards

    async def _build_mcp_servers_aliases_map(
        self,
    ):
        """Tries a first connection attempt to all the MCP servers, filling the MCP server aliases map.
        Makes the program crash if at least one of the `required` MCP servers is not reachable.
        
        Raises:
            Exception: if the connection to one of the `required` MCP servers fails.
        """
        mcp_client = MultiServerMCPClient(connections=self._mcp_server_connections)
        aliases: Dict[str, str] = {}

        for mcp in self.mcp_servers:
            try:
                alias = await _request_mcp_name(mcp_client, mcp.name)
                aliases[alias] = mcp.name
            except Exception as e:
                if mcp.required:
                    _logger.error(f"Failed to get name from required MCP server '{mcp.name}': {e}")
                    raise e
                else:
                    _logger.warning(f"Failed to get name from MCP server '{mcp.name}': {e}")

        _logger.debug(f"Retrieved alias for {len(aliases)}/{len(self.mcp_servers)} MCP Servers.")
        self._mcp_servers_aliases = aliases

    async def _build_remote_agents_aliases_map(
        self,
    ):
        """Tries a first connection attempt to all the agents, filling the agents aliases map.
        Makes the program crash if at least one of the `required` agents is not reachable.
        
        Raises:
            Exception: if the connection to one of the `required` agents fails.
        """
        mcp_client = MultiServerMCPClient(connections=self._mcp_server_connections)
        async_client = AsyncClient(timeout=DEFAULT_TIMEOUT)
        aliases: Dict[str, str] = {}

        for agent in self.remote_agents:
            try:
                if agent.protocol == 'A2A':
                    alias = await _request_a2a_name(async_client, agent.url)
                else:
                    alias = await _request_mcp_name(mcp_client, agent.name)
                aliases[alias] = agent.name
            except Exception as e:
                if agent.required:
                    _logger.error(f"Failed to get name from required Agent '{agent.name}': {e}")
                    raise e
                else:
                    _logger.warning(f"Failed to get name from Agent '{agent.name}': {e}")

        _logger.debug(f"Retrieved alias for {len(aliases)}/{len(self.remote_agents)} Agents.")
        self._remote_agents_aliases = aliases


async def _request_mcp_name(
    client: MultiServerMCPClient,
    server_name: str,
) -> str:
    async with client.session(server_name, auto_initialize=False) as session:
        initialized_session = await session.initialize()
        return initialized_session.serverInfo.name
    raise ValueError(f"Cannot retrieve name for {server_name} (MCP).")

async def _request_a2a_name(
    client: AsyncClient,
    server_url: str,
) -> str:
    card_resolver = A2ACardResolver(
        httpx_client=client,
        base_url=server_url,
    )
    agent_card = await card_resolver.get_agent_card()
    return agent_card.name

def _build_mcp_server_connections(
    mcp_servers: List[MCPServerConfig],
) -> Dict[str, MCPConnection]:
    """Build MCP server connections from configuration.

    Args:
        mcp_servers (List[MCPServerConfig]): List of MCP server configurations.

    Returns:
        dict[str, MCPConnection]: Dictionary of MCP server connections.
    """
    mcp_connections = {
        server.name: MCPConnection(
            url=server.url,
            timeout=server.timeout,
            transport='streamable_http',
        )
        for server in mcp_servers
    }
    return mcp_connections

def _build_remote_agent_connections(
    remote_agents: List[RemoteAgentConfig],
) -> Tuple[Dict[str, A2AConnection], Dict[str, MCPConnection]]:
    """Build MCP and A2A agent connections from configuration.

    Args:
        remote_agents (List[RemoteAgentConfig]): List of remote agent configurations.

    Returns:
        Tuple[Dict[str, A2AConnection], Dict[str, MCPConnection]]:
            Two dictionaries, one with A2A agent connections and one with MCP agent connections.
    """
    mcp_connections = {
        agent.name: MCPConnection(
            url=agent.url,
            timeout=agent.timeout,
            transport='streamable_http',
        )
        for agent in remote_agents
        if agent.protocol == 'MCP'
    }
    a2a_connections = {
        agent.name: A2AConnection(
            url=agent.url,
            timeout=agent.timeout,
        )
        for agent in remote_agents
        if agent.protocol == 'A2A'
    }
    return a2a_connections, mcp_connections
