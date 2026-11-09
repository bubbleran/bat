import asyncio
import json
from datetime import datetime
from typing import Annotated, Optional, TypedDict
from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.helpers import new_data_part, new_message, new_text_part
from a2a.types import (
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
)
from google.protobuf.json_format import MessageToDict
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.types import interrupt
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import BaseModel

from bat import executor as executor_module
from bat.executor import BATAgentExecutor
from bat.state import BATAgentState


def _graph(*nodes, state=MessagesState, checkpointer=None):
    builder = StateGraph(state)
    previous = START
    for node in nodes:
        builder.add_node(node)
        builder.add_edge(previous, node.__name__)
        previous = node.__name__
    return builder.compile(checkpointer=checkpointer)


def _events(graph, text="question", data=None, *, extra_data=()):
    executor = BATAgentExecutor(graph, "Test Agent")
    context = MagicMock()
    parts = [new_text_part(text)]
    if data is not None:
        parts.append(new_data_part(data))
    parts.extend(new_data_part(value) for value in extra_data)
    context.message = new_message(parts, role=Role.ROLE_USER)
    context.current_task = Task(id="task", context_id="context")
    queue = AsyncMock()
    asyncio.run(executor.execute(context, queue))
    return [call.args[0] for call in queue.enqueue_event.call_args_list]


def _states(events):
    return [
        TaskState.Name(e.status.state)
        for e in events
        if isinstance(e, TaskStatusUpdateEvent)
    ]


def test_answer_is_a_text_artifact_then_completed():
    def answer(state):
        text = "hello " + state["messages"][-1].text
        return {"messages": [AIMessage(text)]}

    events = _events(_graph(answer))

    assert isinstance(events[0], TaskArtifactUpdateEvent)
    assert events[0].artifact.parts[0].text == "hello question"
    assert _states(events) == ["TASK_STATE_COMPLETED"]


class _FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_create_agent_graph_is_served():
    model = _FakeModel(messages=iter([AIMessage("from the model")]))

    events = _events(create_agent(model, [], system_prompt="Be brief."))

    assert events[0].artifact.parts[0].text == "from the model"
    assert _states(events) == ["TASK_STATE_COMPLETED"]


class _Actions(BaseModel):
    kind: str = "kaliorexi/actions"
    actions: list[dict]


class _MessagesWithOutput(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = []
    data_out: Optional[dict] = None


@pytest.mark.parametrize("state_schema", [MessagesState, _MessagesWithOutput])
@pytest.mark.parametrize("text", ["Inspect this network", ""])
def test_request_data_falls_back_to_one_message_without_data_in(
    state_schema, text, caplog
):
    data = {"network": "net-a", "enabled": True, "parent": None}
    extra = {"actions": []}
    seen = []

    def answer(state):
        messages = (
            state.messages
            if isinstance(state, BaseModel)
            else state["messages"]
        )
        seen.extend(messages)
        return {"messages": [AIMessage("ok")]}

    events = _events(
        _graph(answer, state=state_schema), text, data, extra_data=[extra]
    )

    assert _states(events) == ["TASK_STATE_COMPLETED"]
    assert len(seen) == 1
    assert seen[0].type == "human"
    prefix = f"{text}\n\n" if text else ""
    assert seen[0].text.startswith(prefix + "Attached data (JSON):\n")
    payloads = seen[0].text.split("Attached data (JSON):\n", 1)[1]
    assert [json.loads(line) for line in payloads.splitlines()] == [data, extra]
    assert not any("unknown channel" in r.message for r in caplog.records)


def test_data_out_is_a_data_part_next_to_the_answer():
    class WithData(MessagesState):
        data_out: _Actions

    networks = {"type": "navigate", "to": "/networks", "label": "Networks"}

    def answer(state):
        return {
            "messages": [AIMessage("Two networks are up.")],
            "data_out": _Actions(actions=[networks]),
        }

    parts = _events(_graph(answer, state=WithData))[0].artifact.parts

    assert parts[0].text == "Two networks are up."
    assert MessageToDict(parts[1].data) == {
        "kind": "kaliorexi/actions",
        "actions": [networks],
    }


def test_data_out_is_sent_only_by_the_run_that_set_it():
    class WithData(MessagesState):
        data_out: dict

    def answer(state):
        if state["messages"][-1].text == "first":
            return {"messages": [AIMessage("a")], "data_out": {"x": 1}}
        return {"messages": [AIMessage("b")]}

    graph = _graph(answer, state=WithData, checkpointer=InMemorySaver())
    _events(graph, "first")
    parts = _events(graph, "second")[0].artifact.parts

    assert [p.text for p in parts] == ["b"]


@pytest.mark.parametrize("with_result", [True, False])
def test_data_only_request_does_not_reuse_a_previous_answer(with_result):
    class WithData(_MessagesWithOutput):
        data_in: Optional[dict] = None

    networks = ["net-a", "net-b"]

    def answer(state: WithData) -> WithData:
        if state.data_in is None:
            state.messages.append(
                AIMessage("There are two networks: net-a and net-b.")
            )
            state.data_out = {"networks": list(networks)}
        elif with_result:
            state.data_out = {"networks": list(networks)}
        return state

    graph = _graph(answer, state=WithData, checkpointer=InMemorySaver())
    first = _events(graph, "List networks")
    assert first[0].artifact.parts[0].text == (
        "There are two networks: net-a and net-b."
    )

    networks.remove("net-b")
    second = _events(graph, "", {"refresh": True})

    if with_result:
        assert _states(second) == ["TASK_STATE_COMPLETED"]
        parts = second[0].artifact.parts
        assert len(parts) == 1
        assert parts[0].HasField("data")
        assert MessageToDict(parts[0].data) == {"networks": ["net-a"]}
    else:
        assert _states(second) == ["TASK_STATE_FAILED"]
        assert second[0].status.message.parts[0].text == (
            "Agent finished without a final result."
        )

    history = graph.get_state(
        {"configurable": {"thread_id": "context"}}
    ).values["messages"]
    assert [message.text for message in history] == [
        "List networks",
        "There are two networks: net-a and net-b.",
    ]


def test_postprocessing_publishes_the_final_appended_ai_message():
    def model(state: _MessagesWithOutput) -> _MessagesWithOutput:
        state.messages.append(AIMessage("Draft answer"))
        return state

    def adjust(state: _MessagesWithOutput) -> _MessagesWithOutput:
        state.messages.append(AIMessage("Adjusted final answer"))
        return state

    events = _events(_graph(model, adjust, state=_MessagesWithOutput))

    assert _states(events) == ["TASK_STATE_COMPLETED"]
    assert [p.text for p in events[0].artifact.parts] == [
        "Adjusted final answer"
    ]


@pytest.mark.parametrize(
    "message",
    [
        SystemMessage("Internal instructions"),
        ToolMessage("Internal tool result", tool_call_id="call"),
    ],
)
def test_non_ai_messages_are_not_published_as_final_answers(message):
    def internal(state):
        return {"messages": [message]}

    events = _events(_graph(internal))

    assert _states(events) == ["TASK_STATE_FAILED"]
    assert events[0].status.message.parts[0].text == (
        "Agent finished without a final result."
    )


def test_request_data_arrives_in_data_in_and_is_not_sent_back():
    class WithData(MessagesState):
        data_in: dict

    seen = {}

    def answer(state):
        seen["data_in"] = state["data_in"]
        seen["text"] = state["messages"][-1].text
        return {"messages": [AIMessage("ok")]}

    graph = _graph(answer, state=WithData)
    events = _events(graph, "Deploy", {"cell": "c7"})

    assert seen == {"data_in": {"cell": "c7"}, "text": "Deploy"}
    assert [p.text for p in events[0].artifact.parts] == ["ok"]


def test_declared_data_in_still_uses_only_the_first_part():
    class WithData(MessagesState):
        data_in: dict

    seen = {}

    def answer(state):
        seen.update(state)
        return {"messages": [AIMessage("ok")]}

    events = _events(
        _graph(answer, state=WithData),
        "Inspect this network",
        {"network": "net-a"},
        extra_data=[{"actions": []}],
    )

    assert _states(events) == ["TASK_STATE_COMPLETED"]
    assert seen["data_in"] == {"network": "net-a"}
    assert [m.text for m in seen["messages"]] == ["Inspect this network"]


def test_nested_models_and_dates_in_data_become_json():
    class WithData(MessagesState):
        data_out: dict

    def answer(state):
        return {
            "messages": [AIMessage("Done.")],
            "data_out": {
                "at": datetime(2026, 10, 26),
                "found": [_Actions(actions=[])],
            },
        }

    parts = _events(_graph(answer, state=WithData))[0].artifact.parts

    assert MessageToDict(parts[1].data) == {
        "at": "2026-10-26T00:00:00",
        "found": [{"kind": "kaliorexi/actions", "actions": []}],
    }


def test_structured_response_is_not_sent():
    class Structured(MessagesState):
        structured_response: dict

    def answer(state):
        return {
            "messages": [AIMessage("Sunny")],
            "structured_response": {"city": "Paris"},
        }

    parts = _events(_graph(answer, state=Structured))[0].artifact.parts

    assert [part.text for part in parts] == ["Sunny"]


def test_custom_events_are_working_updates():
    def work(state):
        write = get_stream_writer()
        write("step 1")
        write({"pct": 50})
        return {"messages": [AIMessage("done")]}

    events = _events(_graph(work))

    assert _states(events) == [
        "TASK_STATE_WORKING",
        "TASK_STATE_WORKING",
        "TASK_STATE_COMPLETED",
    ]
    assert events[0].status.message.parts[0].text == "step 1"
    assert MessageToDict(events[1].status.message.parts[0].data) == {"pct": 50}


@pytest.mark.parametrize("limit", [1, 2])
def test_graph_config_reaches_nodes_and_limits_concurrency(limit):
    active = 0
    peak = 0
    configs = []
    barrier = asyncio.Barrier(limit)

    async def answer(config):
        nonlocal active, peak
        configs.append(config)
        active += 1
        peak = max(peak, active)
        await asyncio.wait_for(barrier.wait(), timeout=5)
        active -= 1
        return {"messages": [AIMessage("Done")]}

    async def left(state, config: RunnableConfig):
        return await answer(config)

    async def right(state, config: RunnableConfig):
        return await answer(config)

    builder = StateGraph(BATAgentState)
    builder.add_node(left)
    builder.add_node(right)
    builder.add_edge(START, "left")
    builder.add_edge(START, "right")
    graph = builder.compile(checkpointer=InMemorySaver()).with_config(
        {
            "max_concurrency": limit,
            "recursion_limit": 14,
            "tags": ["my-agent"],
            "metadata": {"deployment": "test"},
            "configurable": {
                "thread_id": "agent-default",
                "custom_option": "kept",
            },
        }
    )

    events = _events(graph)

    assert _states(events) == ["TASK_STATE_COMPLETED"]
    assert peak == limit
    assert len(configs) == 2
    for config in configs:
        assert config["max_concurrency"] == limit
        assert config["recursion_limit"] == 14
        assert "my-agent" in config["tags"]
        assert config["metadata"]["deployment"] == "test"
        assert config["configurable"]["custom_option"] == "kept"
        assert config["configurable"]["thread_id"] == "context"
    assert graph.config["configurable"]["thread_id"] == "agent-default"
    snapshot = graph.get_state({"configurable": {"thread_id": "context"}})
    assert len(snapshot.values["messages"]) == 3


class _Confirm(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = []
    confirmed: Optional[str] = None


def _confirm_graph(question):
    def new_turn(state: _Confirm) -> _Confirm:
        state.confirmed = None
        return state

    def confirm(state: _Confirm) -> _Confirm:
        state.confirmed = interrupt(question)
        return state

    def answer(state: _Confirm) -> _Confirm:
        state.messages.append(AIMessage(f"confirmed: {state.confirmed}"))
        return state

    return _graph(
        new_turn,
        confirm,
        answer,
        state=_Confirm,
        checkpointer=InMemorySaver(),
    )


def test_interrupt_asks_for_input_and_the_reply_resumes():
    graph = _confirm_graph("Deploy cell 7?")

    first = _events(graph, "deploy")
    second = _events(graph, "yes")
    third = _events(graph, "deploy again")

    assert _states(first) == ["TASK_STATE_INPUT_REQUIRED"]
    assert first[0].status.message.parts[0].text == "Deploy cell 7?"
    assert second[0].artifact.parts[0].text == "confirmed: yes"
    assert _states(second) == ["TASK_STATE_COMPLETED"]
    assert _states(third) == ["TASK_STATE_INPUT_REQUIRED"]


@pytest.mark.parametrize(
    ("text", "data", "expected_resume"),
    [
        ("yes", {"cell": "c7"}, {"cell": "c7"}),
        ("", {"cell": "c7"}, {"cell": "c7"}),
        ("yes", None, "yes"),
    ],
)
def test_interrupt_reply_updates_state_and_supplies_resume(
    text, data, expected_resume
):
    class WithData(MessagesState):
        data_in: Optional[dict]
        data_out: Optional[dict]

    seen = {}

    def prepare(state):
        return {"data_out": {"stale": True}}

    def confirm(state):
        answer = interrupt("Deploy?")
        seen.update(
            text=state["messages"][-1].text,
            data_in=state["data_in"],
            data_out=state["data_out"],
            resume=answer,
        )
        return {"messages": [AIMessage("done")]}

    graph = _graph(
        prepare, confirm, state=WithData, checkpointer=InMemorySaver()
    )
    first = _events(graph, "deploy", {"cell": "old"})
    second = _events(
        graph, text, {"resume": data} if data is not None else None
    )

    assert _states(first) == ["TASK_STATE_INPUT_REQUIRED"]
    assert _states(second) == ["TASK_STATE_COMPLETED"]
    assert seen == {
        "text": text or "deploy",
        "data_in": None,
        "data_out": None,
        "resume": expected_resume,
    }
    assert [p.text for p in second[0].artifact.parts] == ["done"]


@pytest.mark.parametrize("text", ["yes", ""])
def test_interrupt_reply_without_data_in_keeps_json_and_structured_resume(text):
    seen = {}
    data = {"answer": "yes"}
    extra = {"network": "net-a"}

    def confirm(state):
        reply = interrupt("Deploy?")
        seen["reply"] = reply
        seen["messages"] = list(state["messages"])
        return {"messages": [AIMessage("done")]}

    graph = _graph(confirm, checkpointer=InMemorySaver())
    first = _events(graph, "deploy")
    second = _events(graph, text, {"resume": data}, extra_data=[extra])

    assert _states(first) == ["TASK_STATE_INPUT_REQUIRED"]
    assert _states(second) == ["TASK_STATE_COMPLETED"]
    assert seen["reply"] == data
    assert len(seen["messages"]) == 2
    assert seen["messages"][0].text == "deploy"
    latest = seen["messages"][-1].text
    prefix = f"{text}\n\n" if text else ""
    assert latest.startswith(prefix + "Attached data (JSON):\n")
    payloads = latest.split("Attached data (JSON):\n", 1)[1]
    assert [json.loads(line) for line in payloads.splitlines()] == [extra]


@pytest.mark.parametrize("value", [False, 0, "", None, [], {}])
@pytest.mark.parametrize("state_schema", [MessagesState, BATAgentState])
def test_falsy_resume_values_survive_a2a_parts_without_entering_history(
    value, state_schema
):
    seen = {}

    def confirm(state):
        seen["answer"] = interrupt("Answer?")
        messages = (
            state.messages
            if isinstance(state, BATAgentState)
            else state["messages"]
        )
        seen["history"] = [message.text for message in messages]
        if isinstance(state, BATAgentState):
            assert state.data_in is None
        return {"messages": [AIMessage("Done")]}

    graph = _graph(confirm, state=state_schema, checkpointer=InMemorySaver())
    _events(graph, "deploy")
    events = _events(graph, "", {"resume": value})

    assert _states(events) == ["TASK_STATE_COMPLETED"]
    assert seen == {"answer": value, "history": ["deploy"]}


def test_parallel_questions_are_asked_one_per_turn():
    def window(state):
        return {"messages": [AIMessage(f"window {interrupt('Which window?')}")]}

    def apply(state):
        return {"messages": [AIMessage(f"apply {interrupt('Apply?')}")]}

    builder = StateGraph(MessagesState)
    builder.add_node(window)
    builder.add_node(apply)
    builder.add_edge(START, "window")
    builder.add_edge(START, "apply")
    graph = builder.compile(checkpointer=InMemorySaver())

    first = _events(graph, "inspect and update")
    second = _events(graph, "15m")
    third = _events(graph, "yes")

    assert first[0].status.message.parts[0].text == "Which window?"
    assert second[0].status.message.parts[0].text == "Apply?"
    assert _states(third) == ["TASK_STATE_COMPLETED"]
    config = {"configurable": {"thread_id": "context"}}
    messages = graph.get_state(config).values["messages"]
    assert {m.text for m in messages[1:]} == {
        "15m",
        "window 15m",
        "yes",
        "apply yes",
    }


def test_sequential_interrupt_nodes_keep_the_latest_data_in_state():
    class WithData(MessagesState):
        data_in: Optional[dict]
        data_out: Optional[dict]
        window: dict

    seen = {}

    def choose_window(state):
        return {"window": interrupt("Which window?")}

    def confirm(state):
        apply = interrupt("Apply?")
        seen.update(
            window=state["window"],
            apply=apply,
            data_in=state["data_in"],
            messages=[m.text for m in state["messages"]],
        )
        return {"messages": [AIMessage("done")]}

    graph = _graph(
        choose_window, confirm, state=WithData, checkpointer=InMemorySaver()
    )
    _events(graph, "deploy")
    second = _events(
        graph,
        "15m",
        {"resume": {"window": "15m"}},
        extra_data=[{"network": "net-a"}],
    )
    third = _events(
        graph,
        "yes",
        {"resume": {"apply": True}},
        extra_data=[{"network": "net-b"}],
    )

    assert _states(second) == ["TASK_STATE_INPUT_REQUIRED"]
    assert _states(third) == ["TASK_STATE_COMPLETED"]
    assert seen == {
        "window": {"window": "15m"},
        "apply": {"apply": True},
        "data_in": {"network": "net-b"},
        "messages": ["deploy", "15m", "yes"],
    }


def test_interrupt_with_data_is_a_data_part():
    question = {"question": "Which cell?", "options": [7, 8]}

    events = _events(_confirm_graph(question), "deploy")

    assert MessageToDict(events[0].status.message.parts[0].data) == question


def test_agent_state_full_returns_across_sequential_interrupt_nodes():
    class WithWindow(BATAgentState):
        window: dict | None = None

    seen = {}
    configs = []

    def prepare(state: WithWindow) -> WithWindow:
        state.data_out = {"stale": True}
        return state

    def choose_window(state: WithWindow, config: RunnableConfig) -> WithWindow:
        configs.append(config)
        state.window = interrupt("Which window?")
        return state

    def confirm(state: WithWindow, config: RunnableConfig) -> WithWindow:
        configs.append(config)
        apply = interrupt("Apply?")
        seen.update(
            window=state.window,
            apply=apply,
            data_in=state.data_in,
            data_out=state.data_out,
        )
        state.messages.append(AIMessage("Done"))
        state.data_out = {"applied": True}
        return state

    graph = _graph(
        prepare,
        choose_window,
        confirm,
        state=WithWindow,
        checkpointer=InMemorySaver(),
    ).with_config({"max_concurrency": 1, "metadata": {"agent": "network"}})
    first = _events(graph, "deploy", {"network": "net-a"})
    second = _events(graph, "", {"resume": {"window": "15m"}})
    third = _events(
        graph,
        "",
        {"resume": {"apply": True}},
        extra_data=[{"operator": "alice"}],
    )

    assert _states(first) == ["TASK_STATE_INPUT_REQUIRED"]
    assert _states(second) == ["TASK_STATE_INPUT_REQUIRED"]
    assert _states(third) == ["TASK_STATE_COMPLETED"]
    assert seen == {
        "window": {"window": "15m"},
        "apply": {"apply": True},
        "data_in": {"operator": "alice"},
        "data_out": None,
    }
    assert [MessageToDict(part) for part in third[0].artifact.parts] == [
        {"text": "Done"},
        {"data": {"applied": True}},
    ]
    assert len(configs) == 4
    for config in configs:
        assert config["max_concurrency"] == 1
        assert config["metadata"]["agent"] == "network"
        assert config["configurable"]["thread_id"] == "context"


@pytest.mark.parametrize("parallel", [False, True])
def test_default_data_channels_reject_repeated_updates_in_one_step(parallel):
    def first(state):
        interrupt("First?")
        if not parallel:
            interrupt("Second?")
        return {"messages": [AIMessage("Done")]}

    def second(state):
        interrupt("Second?")
        return {"messages": [AIMessage("Done")]}

    builder = StateGraph(BATAgentState)
    builder.add_node(first)
    builder.add_edge(START, "first")
    if parallel:
        builder.add_node(second)
        builder.add_edge(START, "second")
    graph = builder.compile(checkpointer=InMemorySaver())

    assert _states(_events(graph, "start")) == ["TASK_STATE_INPUT_REQUIRED"]
    assert _states(_events(graph, "one")) == ["TASK_STATE_INPUT_REQUIRED"]
    events = _events(graph, "two")

    assert _states(events) == ["TASK_STATE_FAILED"]
    assert events[0].status.message.parts[0].text == (
        "Agent execution failed (InvalidUpdateError)."
    )


def test_graph_error_shows_only_the_exception_type():
    def broken(state):
        raise ValueError("secret detail")

    events = _events(_graph(broken))

    assert _states(events) == ["TASK_STATE_FAILED"]
    text = events[0].status.message.parts[0].text
    assert text == "Agent execution failed (ValueError)."


def test_no_answer_fails():
    def silent(state):
        return {}

    events = _events(_graph(silent))

    assert _states(events) == ["TASK_STATE_FAILED"]
    text = events[0].status.message.parts[0].text
    assert text == "Agent finished without a final result."


def test_run_is_traced_as_invoke_agent(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(executor_module, "tracer", provider.get_tracer("t"))

    def answer(state):
        return {"messages": [AIMessage("ok")]}

    _events(_graph(answer))

    (span,) = exporter.get_finished_spans()
    assert span.name == "invoke_agent Test Agent"
    assert span.attributes["gen_ai.conversation.id"] == "context"


def test_graph_without_messages_is_rejected():
    class NoMessages(TypedDict):
        x: str

    def node(state):
        return state

    with pytest.raises(ValueError, match="no 'messages' field"):
        BATAgentExecutor(_graph(node, state=NoMessages), "x")


def test_new_answer_survives_history_trimming():
    def answer(state):
        if state["messages"][-1].text == "seed":
            return {"messages": [AIMessage("Old answer", id="old")]}
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                AIMessage("Fresh answer after trimming", id="fresh"),
            ]
        }

    graph = _graph(answer, checkpointer=InMemorySaver())
    _events(graph, "seed")
    events = _events(graph, "trim")
    assert _states(events) == ["TASK_STATE_COMPLETED"]
    assert events[0].artifact.parts[0].text == "Fresh answer after trimming"


def test_resurfacing_old_message_does_not_republish_it():
    class WithData(MessagesState):
        data_in: dict | None
        data_out: dict | None

    def answer(state):
        if state["data_in"] is None:
            return {
                "messages": [
                    AIMessage("Older answer", id="older"),
                    AIMessage("Recent answer", id="recent"),
                ]
            }
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                AIMessage("Older answer", id="older"),
            ],
            "data_out": {"fresh": True},
        }

    graph = _graph(answer, state=WithData, checkpointer=InMemorySaver())
    _events(graph, "seed")
    events = _events(graph, "", {"refresh": True})
    assert _states(events) == ["TASK_STATE_COMPLETED"]
    parts = [MessageToDict(part) for part in events[0].artifact.parts]
    assert parts == [{"data": {"fresh": True}}]


def test_messages_without_ids_fail_explicitly():
    class WithoutReducer(TypedDict):
        messages: list[BaseMessage]

    def answer(state):
        return {"messages": [AIMessage("Answer", id="fresh")]}

    events = _events(_graph(answer, state=WithoutReducer))
    assert _states(events) == ["TASK_STATE_FAILED"]
    assert events[0].status.message.parts[0].text == (
        "Agent execution failed (ValueError)."
    )


def test_final_ai_message_without_id_fails_explicitly(caplog):
    class WithoutReducer(TypedDict):
        messages: list[BaseMessage]

    def answer(state):
        # No text input: the initial snapshot is empty, so the missing ID is
        # detected in the final answer rather than in the initial history.
        return {"messages": [AIMessage("Answer")]}

    events = _events(_graph(answer, state=WithoutReducer), text="")

    assert _states(events) == ["TASK_STATE_FAILED"]
    assert events[0].status.message.parts[0].text == (
        "Agent execution failed (ValueError)."
    )
    assert "Messages must have IDs: use add_messages" in caplog.text


def test_replacing_a_previous_ai_message_is_not_a_new_answer():
    class WithData(MessagesState):
        data_in: dict | None
        data_out: dict | None

    def answer(state):
        if state["data_in"] is None:
            return {"messages": [AIMessage("Old answer", id="old")]}
        return {
            "messages": [AIMessage("Edited history", id="old")],
            "data_out": {"fresh": True},
        }

    graph = _graph(answer, state=WithData, checkpointer=InMemorySaver())
    _events(graph, "seed")
    events = _events(graph, "", {"refresh": True})

    assert _states(events) == ["TASK_STATE_COMPLETED"]
    assert [MessageToDict(part) for part in events[0].artifact.parts] == [
        {"data": {"fresh": True}}
    ]


def test_postprocessing_can_replace_an_ai_message_created_in_this_request():
    def model(state):
        return {"messages": [AIMessage("Draft", id="this-run")]}

    def adjust(state):
        return {"messages": [AIMessage("Adjusted", id="this-run")]}

    events = _events(_graph(model, adjust))
    assert _states(events) == ["TASK_STATE_COMPLETED"]
    assert events[0].artifact.parts[0].text == "Adjusted"


def test_concurrent_contexts_have_independent_message_id_snapshots():
    asyncio.run(_concurrent_contexts())


async def _concurrent_contexts():
    barrier = asyncio.Barrier(2)

    async def answer(state, config: RunnableConfig):
        # Priming state is done directly, outside the executor under review.
        if isinstance(state["messages"][-1], AIMessage):
            return {}
        context_id = config["configurable"]["thread_id"]
        await barrier.wait()
        # Each response deliberately reuses an ID from the OTHER context's
        # initial history. Only a request-local snapshot permits both replies.
        other = "B" if context_id == "A" else "A"
        return {
            "messages": [AIMessage(f"Answer {context_id}", id=f"old-{other}")]
        }

    graph = _graph(answer, checkpointer=InMemorySaver())
    # The bound concurrency limit applies separately to each graph execution.
    graph = graph.with_config({"max_concurrency": 1})
    executor = BATAgentExecutor(graph, "Concurrent review")
    for context_id in ("A", "B"):
        await graph.ainvoke(
            {
                "messages": [
                    HumanMessage("seed", id=f"human-{context_id}"),
                    AIMessage(f"History {context_id}", id=f"old-{context_id}"),
                ]
            },
            {"configurable": {"thread_id": context_id}},
        )

    async def send(context_id):
        context = MagicMock()
        context.message = new_message(
            [new_text_part("reply")], role=Role.ROLE_USER
        )
        context.current_task = Task(
            id=f"task-{context_id}", context_id=context_id
        )
        queue = AsyncMock()
        await executor.execute(context, queue)
        events = [call.args[0] for call in queue.enqueue_event.call_args_list]
        assert _states(events) == ["TASK_STATE_COMPLETED"]
        artifact = next(
            event
            for event in events
            if isinstance(event, TaskArtifactUpdateEvent)
        )
        assert artifact.artifact.parts[0].text == f"Answer {context_id}"

    await asyncio.wait_for(asyncio.gather(send("A"), send("B")), timeout=10)
    for context_id in ("A", "B"):
        snapshot = await graph.aget_state(
            {"configurable": {"thread_id": context_id}}
        )
        messages = snapshot.values["messages"]
        assert messages[1].text == f"History {context_id}"
        assert messages[-1].text == f"Answer {context_id}"
    assert not hasattr(executor, "initial_message_ids")
