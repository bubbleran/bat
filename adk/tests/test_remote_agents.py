import asyncio
import uuid
from itertools import cycle

import pytest
from conftest import agent_card, free_port, serving
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind

from bat import executor as executor_module
from bat import remote_agents as remote_module
from bat.a2a_stream import (
    _COMPLETED,
    _FAILED,
    _INPUT_REQUIRED,
    RemoteTaskContext,
)
from bat.config import BATAgentConfig, EndpointConfig, RemoteAgentConfig
from bat.executor import BATAgentExecutor
from bat.remote_agents import RemoteAgentClient, init_a2a_clients
from bat.server import BATAgentServer
from bat.telemetry import setup as telemetry_setup


class RemoteState(MessagesState):
    data_out: dict
    request: str


class _FakeModel(GenericFakeChatModel):
    def bind_tools(self, *args, **kwargs):
        return self


def _remote_graph(received):
    def receive(state):
        received.append(state["messages"][-1].text)
        get_stream_writer()("working on it")
        return {"request": state["messages"][-1].text}

    def answer(state):
        text = state["request"]
        if text == "deploy":
            reply = interrupt("Confirm?")
            return {"messages": [AIMessage(f"confirmed {reply}")]}
        if text == "pick":
            reply = interrupt({"options": ["c7", "c8"]})
            return {"messages": [AIMessage(f"picked {reply['cell']}")]}
        return {
            "messages": [AIMessage(f"echo {text}")],
            "data_out": {"seen": text},
        }

    builder = StateGraph(RemoteState)
    builder.add_node(receive)
    builder.add_node(answer)
    builder.add_edge(START, "receive")
    builder.add_edge("receive", "answer")
    return builder.compile(checkpointer=InMemorySaver())


def _with_remote(main, received=None, *, graph=None):
    """Serve a BAT agent on a free port and run `main(agent)` against it."""
    port = free_port()
    if graph is None:
        graph = _remote_graph([] if received is None else received)
    card = agent_card("Remote")
    config = BATAgentConfig(
        endpoint=EndpointConfig(url="http://127.0.0.1", port=port)
    )
    server = BATAgentServer(card, config, BATAgentExecutor(graph, card.name))

    async def run():
        async with serving(server.app, port) as url:
            agent = RemoteAgentClient(RemoteAgentConfig(name="Remote", url=url))
            return await main(agent)

    return asyncio.run(run())


def test_send_returns_text_and_data():
    async def main(agent):
        return await agent.send_request("hi")

    reply = _with_remote(main)

    assert reply.status == _COMPLETED
    assert reply.text == "echo hi"
    assert reply.data == [{"seen": "hi"}]


def test_progress_reaches_the_stream_writer():
    async def main(agent):
        async def call(state):
            reply = await agent.send_request(state["messages"][-1].text)
            return {"messages": [AIMessage(reply.text)]}

        builder = StateGraph(MessagesState)
        builder.add_node(call)
        builder.add_edge(START, "call")
        caller = builder.compile()
        stream = caller.astream(
            {"messages": [HumanMessage("hi")]}, stream_mode="custom"
        )
        return [event async for event in stream]

    assert _with_remote(main) == ["working on it"]


def test_input_required_keeps_the_task_for_the_next_send():
    async def main(agent):
        session = RemoteTaskContext(context_id=str(uuid.uuid4()))
        asked = await agent.send_request("deploy", session)
        waiting_task = session.task_id
        done = await agent.send_request("yes", session)
        return asked, waiting_task, done, session

    asked, waiting_task, done, session = _with_remote(main)

    assert (asked.status, asked.text) == (_INPUT_REQUIRED, "Confirm?")
    assert waiting_task is not None
    assert (done.status, done.text) == (_COMPLETED, "confirmed yes")
    assert session.task_id is None


def test_structured_reply_requires_an_envelope_and_can_be_corrected():
    async def main(agent):
        asked = await agent.send_request("pick")
        session = asked.session
        task_id = session.task_id
        invalid = await agent.send_request({"cell": "c8"}, session)
        assert session.task_id == task_id
        done = await agent.send_request({"resume": {"cell": "c7"}}, session)
        return invalid, done

    invalid, done = _with_remote(main)

    assert invalid.status == _FAILED
    assert (done.status, done.text) == (_COMPLETED, "picked c7")


@pytest.mark.parametrize(
    "message, question, answer, result",
    [
        ("deploy", "Confirm?", "yes", "confirmed yes"),
        ("pick", '{"options": ["c7", "c8"]}', {"cell": "c7"}, "picked c7"),
    ],
)
@pytest.mark.parametrize("mode", ["send_request", "as_tool"])
def test_remote_call_asks_the_user_and_does_not_resend_on_resume(
    message, question, answer, result, mode
):
    received = []

    async def main(agent):
        if mode == "as_tool":
            tool = await agent.as_tool()
            call = {"name": tool.name, "args": {"message": message}, "id": "1"}
            model = _FakeModel(
                messages=cycle(
                    [AIMessage("", tool_calls=[call]), AIMessage("ok")]
                )
            )
            caller = create_agent(model, [tool], checkpointer=InMemorySaver())
        else:

            async def call(state):
                reply = await agent.send_request(message)
                return {"messages": [AIMessage(reply.text)]}

            builder = StateGraph(MessagesState)
            builder.add_node(call)
            builder.add_edge(START, "call")
            caller = builder.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t"}}
        await caller.ainvoke({"messages": [HumanMessage("go")]}, config)
        asked = (await caller.aget_state(config)).interrupts[0].value
        out = await caller.ainvoke(Command(resume=answer), config)
        if mode == "as_tool":
            results = [m.text for m in out["messages"] if m.type == "tool"]
        else:
            results = [out["messages"][-1].text]
        return asked, results

    asked, tool_result = _with_remote(main, received)

    assert asked == question
    assert tool_result == [result]
    assert received == [message]


def test_send_request_resumes_successive_questions_without_repeating_exchanges(
    monkeypatch,
):
    requests = []
    responses = []

    def window(state):
        answer = interrupt("Which window?", response_schema=str)
        return {"messages": [AIMessage(answer)]}

    def approve(state):
        answer = interrupt("Approve?", response_schema=bool)
        return {
            "messages": [AIMessage(f"{state['messages'][-1].text}: {answer}")]
        }

    builder = StateGraph(MessagesState)
    builder.add_node(window)
    builder.add_node(approve)
    builder.add_edge(START, "window")
    builder.add_edge("window", "approve")
    remote = builder.compile(checkpointer=InMemorySaver())

    async def main(agent):
        original = agent._request

        async def record(message, session):
            requests.append((message, session.model_copy()))
            response = await original(message, session)
            responses.append(response)
            return response

        monkeypatch.setattr(agent, "_request", record)

        async def call(state):
            response = await agent.send_request("schedule")
            assert response.status == _COMPLETED
            return {"messages": [AIMessage(response.text)]}

        builder = StateGraph(MessagesState)
        builder.add_node(call)
        builder.add_edge(START, "call")
        caller = builder.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "successive"}}

        await caller.ainvoke({"messages": [HumanMessage("go")]}, config)
        assert (await caller.aget_state(config)).interrupts[
            0
        ].value == "Which window?"
        assert len(requests) == 1

        await caller.ainvoke(Command(resume="evening"), config)
        assert (await caller.aget_state(config)).interrupts[
            0
        ].value == "Approve?"
        assert len(requests) == 2

        out = await caller.ainvoke(Command(resume=False), config)
        assert not (await caller.aget_state(config)).interrupts
        return out["messages"][-1].text

    assert _with_remote(main, graph=remote) == "evening: False"
    assert [message for message, _ in requests] == [
        "schedule",
        {"resume": "evening"},
        {"resume": False},
    ]
    assert requests[0][1].task_id is None
    task_id = requests[1][1].task_id
    assert task_id is not None
    assert requests[2][1].task_id == task_id
    assert len({session.context_id for _, session in requests}) == 1
    assert responses[0].session.task_id == task_id
    assert responses[1].session.task_id == task_id
    assert responses[2].session.task_id is None


def test_as_tool_gives_the_model_the_reply_data():
    async def main(agent):
        tool = await agent.as_tool()
        call = {"name": tool.name, "args": {"message": "hi"}, "id": "1"}
        model = _FakeModel(
            messages=cycle([AIMessage("", tool_calls=[call]), AIMessage("ok")])
        )
        caller = create_agent(model, [tool], checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "t"}}
        out = await caller.ainvoke({"messages": [HumanMessage("go")]}, config)
        return [m.text for m in out["messages"] if m.type == "tool"]

    assert _with_remote(main) == ['echo hi\n{"seen": "hi"}']


def test_trace_continues_in_the_remote(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry_setup, "_provider", provider)
    monkeypatch.setattr(remote_module, "tracer", provider.get_tracer("t"))
    monkeypatch.setattr(executor_module, "tracer", provider.get_tracer("t"))

    async def main(agent):
        return await agent.send_request("hi")

    _with_remote(main)

    spans = exporter.get_finished_spans()
    (client,) = [s for s in spans if s.kind == SpanKind.CLIENT]
    (remote,) = [s for s in spans if s.kind == SpanKind.INTERNAL]
    assert client.name == "invoke_agent Remote"
    assert remote.parent.span_id == client.context.span_id
    assert remote.context.trace_id == client.context.trace_id


def test_agent_down_stops_the_agent():
    down = RemoteAgentConfig(
        name="down", url=f"http://127.0.0.1:{free_port()}", timeout=2
    )

    with pytest.raises(ConnectionError, match="'down'"):
        asyncio.run(init_a2a_clients([down]))


def test_agent_down_after_startup_fails_the_call_not_the_agent():
    down = RemoteAgentConfig(
        name="down", url=f"http://127.0.0.1:{free_port()}", timeout=2
    )

    reply = asyncio.run(RemoteAgentClient(down).send_request("hi"))

    assert reply.status == _FAILED
    assert "down" in reply.text
