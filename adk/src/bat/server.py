import asyncio
import os
from typing import Optional

import uvicorn
from a2a.helpers import get_data_parts, get_message_text
from a2a.server.agent_execution import (
    AgentExecutor,
    RequestContext,
    SimpleRequestContextBuilder,
)
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandlerV2
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.types import AgentCard, AgentInterface, SendMessageRequest, Task
from google.protobuf.json_format import Parse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from .a2a_stream import _parse_resume
from .config import BATAgentConfig, EndpointConfig
from .executor import BATAgentExecutor
from .persistence import _task_store


def load_agent_card(path: Optional[str] = None) -> AgentCard:
    """Read the agent card (`AGENT_CARD_PATH`, default ./agent.json)."""
    path = path or os.getenv("AGENT_CARD_PATH", "./agent.json")
    with open(path) as file:
        card = Parse(file.read(), AgentCard())
    if card.supported_interfaces:
        raise ValueError(
            "Remove supportedInterfaces from the agent card: the server "
            "adds it from `endpoint` in config.yaml."
        )
    return card


class BATAgentServer:
    """Serve an agent over A2A on `endpoint.port`: its card and JSON-RPC.
    The card's interface URL comes from `endpoint`."""

    def __init__(
        self,
        agent_card: AgentCard,
        agent_config: BATAgentConfig,
        executor: AgentExecutor,
    ) -> None:
        self.port = agent_config.endpoint.port
        self.agent_card = _served_card(agent_card, agent_config.endpoint)
        handler = DefaultRequestHandlerV2(
            agent_executor=executor,
            task_store=_task_store(agent_config.persistence),
            agent_card=self.agent_card,
            extended_agent_card=self.agent_card,
            request_context_builder=(
                _ResumeContextBuilder(executor)
                if isinstance(executor, BATAgentExecutor)
                else None
            ),
        )
        self.app = Starlette(
            routes=[
                *create_agent_card_routes(agent_card=self.agent_card),
                *create_jsonrpc_routes(request_handler=handler, rpc_url="/"),
            ]
        )

    async def serve(self) -> None:
        """Serve in the running event loop, the one that opened the graph's
        resources."""
        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port)
        await uvicorn.Server(config).serve()

    def run(self) -> None:
        asyncio.run(self.serve())


class _ResumeContextBuilder(SimpleRequestContextBuilder):
    """Reject invalid resume requests before the SDK starts task execution.

    The SDK marks unhandled executor exceptions as terminal failures, so
    validating only inside the executor would prevent correcting a reply.
    This hook covers both blocking and streaming requests.
    """

    def __init__(self, executor: BATAgentExecutor) -> None:
        super().__init__()
        self.executor = executor

    async def build(
        self,
        context: ServerCallContext,
        params: Optional[SendMessageRequest] = None,
        task_id: Optional[str] = None,
        context_id: Optional[str] = None,
        task: Optional[Task] = None,
    ) -> RequestContext:
        request = await super().build(
            context, params, task_id, context_id, task
        )
        if request.message is not None:
            pending = await self.executor._pending(
                {"configurable": {"thread_id": request.context_id}}
            )
            _parse_resume(
                get_message_text(request.message),
                get_data_parts(request.message.parts),
                waiting=bool(pending),
            )
        return request


def _served_card(card: AgentCard, endpoint: EndpointConfig) -> AgentCard:
    if endpoint.url is None:
        raise ValueError("Set endpoint.url in config.yaml: the card needs it.")
    url = endpoint.url
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    url = url.rstrip("/")
    card.supported_interfaces.append(
        AgentInterface(
            url=f"{url}:{endpoint.port}",
            protocol_binding="JSONRPC",
            protocol_version="2.0",
        )
    )
    return card

async def ping_route(_: Request) -> JSONResponse:
    return JSONResponse("pong")
