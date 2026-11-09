from typing import List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import MCPServerConfig
from .logging import create_logger

logger = create_logger(__name__, "debug")


async def init_mcp_clients(
    servers: List[MCPServerConfig],
) -> MultiServerMCPClient:
    """One client for config.yaml's `mcp-servers`. A server that does not
    answer raises `ConnectionError`, so the agent does not start."""
    client = MultiServerMCPClient(
        {
            server.name: {
                "transport": "streamable_http",
                "url": server.url,
                "timeout": server.timeout,
            }
            for server in servers
        }
    )
    for server in servers:
        await _check_connection_required(client, server.name)
    return client


async def list_mcp_tools(
    client: MultiServerMCPClient,
    *servers: str,
) -> List[BaseTool]:
    """The tools of the named servers, or of all of them. A server that
    fails now is skipped with a warning."""
    names = servers or tuple(client.connections)
    unknown = set(names) - set(client.connections)
    if unknown:
        raise ValueError(f"Unknown MCP servers: {sorted(unknown)}")
    tools: List[BaseTool] = []
    for name in names:
        tools.extend(await _server_tools(client, name))
    return tools


async def _check_connection_required(
    client: MultiServerMCPClient,
    name: str,
) -> None:
    try:
        async with client.session(name):
            pass
    except Exception as e:
        raise ConnectionError(
            f"MCP server '{name}' does not answer: {e}"
        ) from e


async def _server_tools(
    client: MultiServerMCPClient,
    name: str,
) -> List[BaseTool]:
    tools: List[BaseTool] = []
    try:
        tools = await client.get_tools(server_name=name)
    except Exception as e:
        logger.warning(f"MCP server '{name}' skipped, it does not answer: {e}")
    return tools
