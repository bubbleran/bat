import asyncio
import socket
from contextlib import asynccontextmanager

import uvicorn
from a2a.types import AgentCard
from google.protobuf.json_format import ParseDict

CARD = {
    "name": "Echo Agent",
    "description": "Echoes.",
    "version": "1.0.0",
    "capabilities": {"streaming": True},
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
    "skills": [
        {
            "id": "echo",
            "name": "Echo",
            "description": "Echoes the message.",
            "tags": ["echo"],
        }
    ],
}


def agent_card(name: str) -> AgentCard:
    return ParseDict({**CARD, "name": name}, AgentCard())


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@asynccontextmanager
async def serving(app, port: int):
    """Serve `app` on 127.0.0.1:`port` in the running loop until the block
    ends."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


def send_message(text: str) -> dict:
    """A JSON-RPC SendMessage body."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "SendMessage",
        "params": {
            "message": {
                "messageId": text,
                "role": "ROLE_USER",
                "parts": [{"text": text}],
            }
        },
    }
