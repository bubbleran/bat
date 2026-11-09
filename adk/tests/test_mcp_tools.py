import asyncio

import pytest
from conftest import free_port, serving
from mcp.server.fastmcp import FastMCP

from bat.config import MCPServerConfig
from bat.mcp_tools import init_mcp_clients, list_mcp_tools


def _mcp_app():
    mcp = FastMCP("Networks")

    @mcp.tool()
    def list_networks() -> str:
        """List the networks."""
        return "net-a, net-b"

    return mcp.streamable_http_app()


def _down():
    url = f"http://127.0.0.1:{free_port()}/mcp"
    return MCPServerConfig(name="down", url=url, timeout=2)


async def _client_with_a_server_down(up):
    """A client of `up` and of 'down', a server that stops after startup."""
    async with serving(_mcp_app(), free_port()) as url:
        down = MCPServerConfig(name="down", url=f"{url}/mcp", timeout=2)
        return await init_mcp_clients([up, down])


def _run(main):
    port = free_port()

    async def with_server():
        async with serving(_mcp_app(), port) as url:
            up = MCPServerConfig(name="up", url=f"{url}/mcp")
            return await main(up)

    return asyncio.run(with_server())


def test_tools_of_all_servers():
    async def main(up):
        client = await init_mcp_clients([up])
        tools = await list_mcp_tools(client)
        return [tool.name for tool in tools], await tools[0].ainvoke({})

    names, result = _run(main)

    assert names == ["list_networks"]
    assert "net-a, net-b" in str(result)


def test_server_down_stops_the_agent():
    async def main(up):
        await init_mcp_clients([up, _down()])

    with pytest.raises(ConnectionError, match="'down'"):
        _run(main)


def test_server_down_after_startup_is_skipped():
    async def main(up):
        client = await _client_with_a_server_down(up)
        return [tool.name for tool in await list_mcp_tools(client)]

    assert _run(main) == ["list_networks"]


def test_tools_of_named_servers_only():
    async def main(up):
        client = await _client_with_a_server_down(up)
        return await list_mcp_tools(client, "down")

    assert _run(main) == []


def test_unknown_server_name_is_an_error():
    async def main(up):
        client = await init_mcp_clients([up])
        await list_mcp_tools(client, "typo")

    with pytest.raises(ValueError, match="typo"):
        _run(main)
