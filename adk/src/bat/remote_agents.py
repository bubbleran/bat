import re
import uuid
from typing import Dict, List, Optional

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_data_parts, get_text_parts, new_message
from a2a.types import (
    AgentCard,
    AgentInterface,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.config import get_config, get_stream_writer
from langgraph.func import task
from langgraph.types import interrupt
from opentelemetry.trace import Span

from .a2a_stream import (
    _FAILED,
    _INPUT_REQUIRED,
    RemoteAgentResponse,
    RemoteTaskContext,
    _read_agent_response,
    _serialize_text_and_data,
    _to_part,
)
from .config import RemoteAgentConfig
from .logging import create_logger
from .telemetry import SpanKind, get_tracer, inject_context, mark_span_error
from .telemetry import attributes as attrs

logger = create_logger(__name__, "debug")
tracer = get_tracer(__name__)


class RemoteAgentClient:
    """A client for one agent of config.yaml's `remote-agents`."""

    def __init__(self, config: RemoteAgentConfig) -> None:
        self.name = config.name
        self._config = config
        self._card: Optional[AgentCard] = None

    async def agent_card(self) -> AgentCard:
        """The remote's card, fetched once. Its interface points at the
        configured url, whatever address the card advertises."""
        if self._card is None:
            self._card = await _fetch_card(self._config)
        return self._card

    async def send_request(
        self,
        message: object,
        session: Optional[RemoteTaskContext] = None,
    ) -> RemoteAgentResponse:
        """Send text or data to the remote and wait for its response.

        Inside a graph, forward remote questions with `interrupt()` and
        resume the same remote task when the caller answers. A checkpointer
        keeps completed exchanges from being repeated on interrupt resume.
        Progress goes to the graph's stream.

        Outside a graph, return INPUT_REQUIRED to the caller. Answer with
        `send_request({"resume": value}, response.session)`.
        """
        session = session or self._session()
        try:
            get_config()
        except RuntimeError:  # not inside a graph run
            return await self._request(message, session)

        response = await _send_once(self, message, session)
        while response.status == _INPUT_REQUIRED:
            question = _serialize_text_and_data(response.text, response.data)
            answer = interrupt(
                question,
                response_schema=response.response_schema,
            )
            response = await _send_once(
                self, {"resume": answer}, response.session
            )
        return response

    async def _request(
        self,
        message: object,
        session: RemoteTaskContext,
    ) -> RemoteAgentResponse:
        """Perform one A2A exchange, including tracing and progress."""
        request = new_message(
            [_to_part(message)],
            context_id=session.context_id,
            task_id=session.task_id,
            role=Role.ROLE_USER,
        )
        with tracer.start_as_current_span(
            f"{attrs.OP_INVOKE_AGENT} {self.name}",
            kind=SpanKind.CLIENT,
            attributes={
                attrs.GEN_AI_OPERATION_NAME: attrs.OP_INVOKE_AGENT,
                attrs.GEN_AI_AGENT_NAME: self.name,
                attrs.INPUT_VALUE: str(message),
            },
        ) as span:
            _inject_trace(request, span)
            reply = await self._call(request, session)
            _record_trace(span, reply)
        return reply

    async def as_tool(self) -> BaseTool:
        """This agent as a LangChain tool for a model. A question from the
        remote pauses the graph with `interrupt()`, combining its text and
        every data part as JSON. The answer goes back to the remote, which
        is not sent the message again on resume. The model gets the remote's
        text and each data part as JSON. A pending reply's response schema
        is forwarded to the local interrupt to describe the expected input.
        """
        card = await self.agent_card()

        async def call(message: str) -> str:
            reply = await self.send_request(message)
            return _serialize_text_and_data(reply.text, reply.data)

        return StructuredTool.from_function(
            coroutine=call,
            name=re.sub(r"[^a-zA-Z0-9_-]+", "_", self.name),
            description=_card_description(card),
        )

    async def _call(
        self,
        message: Message,
        session: RemoteTaskContext,
    ) -> RemoteAgentResponse:
        try:
            reply = await self._stream(message, session)
        except Exception as e:
            logger.error(f"Call to agent '{self.name}' failed", exc_info=e)
            text = f"Call to agent '{self.name}' failed ({type(e).__name__})."
            reply = RemoteAgentResponse(
                status=_FAILED, text=text, session=session
            )
        return reply

    async def _stream(
        self,
        message: Message,
        session: RemoteTaskContext,
    ) -> RemoteAgentResponse:
        card = await self.agent_card()
        async with httpx.AsyncClient(timeout=self._config.timeout) as http:
            config = ClientConfig(httpx_client=http, streaming=True)
            client = await create_client(agent=card, client_config=config)
            async with client:
                request = SendMessageRequest(message=message)
                stream = client.send_message(request)
                return await _read_agent_response(
                    stream, session, _write_progress
                )

    def _session(self) -> RemoteTaskContext:
        try:
            thread_id = get_config()["configurable"].get("thread_id")
        except RuntimeError:  # not inside a graph run
            thread_id = None
        if thread_id is None:
            context_id = str(uuid.uuid4())
        else:
            name = f"bat:{thread_id}:{self.name}"
            context_id = str(uuid.uuid5(uuid.NAMESPACE_URL, name))
        return RemoteTaskContext(context_id=context_id)


async def init_a2a_clients(
    agents: List[RemoteAgentConfig],
) -> Dict[str, RemoteAgentClient]:
    """A client per agent of config.yaml's `remote-agents`, by name. An
    agent whose card cannot be fetched raises `ConnectionError`, so the
    agent does not start."""
    clients = {agent.name: RemoteAgentClient(agent) for agent in agents}
    for client in clients.values():
        try:
            await client.agent_card()
        except Exception as e:
            raise ConnectionError(
                f"Agent '{client.name}' does not answer: {e}"
            ) from e
    return clients


@task
async def _send_once(
    agent: RemoteAgentClient,
    message: object,
    session: RemoteTaskContext,
) -> RemoteAgentResponse:
    # A LangGraph task: on resume its result comes from the checkpoint, so
    # the remote does not get the same message twice.
    return await agent._request(message, session.model_copy())


async def _fetch_card(config: RemoteAgentConfig) -> AgentCard:
    async with httpx.AsyncClient(timeout=config.timeout) as http:
        resolver = A2ACardResolver(httpx_client=http, base_url=config.url)
        card = await resolver.get_agent_card()
    card.ClearField("supported_interfaces")
    card.supported_interfaces.append(
        AgentInterface(
            url=config.url,
            protocol_binding="JSONRPC",
            protocol_version="2.0",
        )
    )
    return card


def _write_progress(parts: List[Part]) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:  # not inside a graph run
        pass
    else:
        text = "\n".join(get_text_parts(parts))
        if text:
            writer(text)
        for data in get_data_parts(parts):
            writer(data)


def _inject_trace(message: Message, span: Span) -> None:
    """Pass the trace to the remote through the message metadata."""
    carrier: Dict[str, str] = {}
    inject_context(carrier, span=span)
    message.metadata.update(carrier)


def _record_trace(span: Span, reply: RemoteAgentResponse) -> None:
    span.set_attribute(attrs.OUTPUT_VALUE, reply.text)
    span.set_attribute(attrs.BAT_A2A_TASK_STATE, TaskState.Name(reply.status))
    if reply.status == _FAILED:
        mark_span_error(span, reply.text)


def _card_description(card: AgentCard) -> str:
    lines = [f"Send a message to {card.name}.", card.description]
    for skill in card.skills:
        lines.append(f"Skill {skill.name}: {skill.description}")
        if skill.tags:
            lines.append("Tags: " + ", ".join(skill.tags))
        if skill.examples:
            lines.append("Examples: " + "; ".join(skill.examples[:5]))
    return "\n".join(lines)
