import asyncio
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, model_validator
from typing_extensions import Self, override

from bat.agent import (
    AgentConfig,
    AgentGraph,
    AgentState,
    AgentTaskResult,
    AgentTaskStatus,
)
from bat.prebuilt import ReActLoop

WORKING = AgentTaskStatus.AGENT_TASK_STATUS_WORKING
COMPLETED = AgentTaskStatus.AGENT_TASK_STATUS_COMPLETED
FAILED = AgentTaskStatus.AGENT_TASK_STATUS_FAILED
INPUT_REQUIRED = AgentTaskStatus.AGENT_TASK_STATUS_INPUT_REQUIRED


class _Answer(AgentState):
    query: str
    answer: Optional[str] = None

    @classmethod
    @override
    def from_query(cls, query: str) -> Self:
        return cls(query=query)

    @override
    def to_task_result(self) -> AgentTaskResult:
        if self.answer:
            return AgentTaskResult(task_status=COMPLETED, content=self.answer)
        return AgentTaskResult(task_status=WORKING, content="working")


def _collect(
    graph: AgentGraph, query: str = "question"
) -> List[AgentTaskResult]:
    async def drive():
        return [
            item
            async for item in graph.astream(
                query=query,
                config={"configurable": {"thread_id": "t"}},
            )
        ]

    return asyncio.run(drive())


class _Graph(AgentGraph):
    @override
    def setup(self, config) -> None:
        pass


def _failing_stream(exc: Exception):
    async def stream(**kwargs):
        raise exc
        yield  # pragma: no cover - makes this an async generator

    return stream


def _graph_with(tasks: tuple, stream_error: Exception) -> _Graph:
    """An AgentGraph whose compiled graph fails mid-stream and then reports
    `tasks`, so the post-stream interrupt lookup runs on a failed task."""
    graph = object.__new__(_Graph)
    graph.StateType = _Answer
    graph._memory = MagicMock()
    graph._graph = MagicMock()
    graph._graph.astream = _failing_stream(stream_error)
    graph._graph.get_state = MagicMock(
        return_value=MagicMock(tasks=tasks, created_at=None)
    )
    return graph


def test_node_error_is_reported_not_masked_by_an_empty_interrupt():
    task = MagicMock(interrupts=())
    graph = _graph_with((task,), RuntimeError("node blew up"))

    results = _collect(graph)

    assert results == [
        AgentTaskResult(
            task_status=FAILED, content="Agent execution failed (RuntimeError)."
        )
    ]


def test_graph_error_wins_over_a_pending_interrupt():
    task = MagicMock(interrupts=(MagicMock(value="need input"),))
    graph = _graph_with((task,), RuntimeError("interrupted"))

    results = _collect(graph)

    assert [r.task_status for r in results] == [
        AgentTaskStatus.AGENT_TASK_STATUS_FAILED
    ]


def test_no_tasks_yields_no_interrupt():
    graph = _graph_with((), RuntimeError("boom"))

    results = _collect(graph)

    assert len(results) == 1
    assert results[0].task_status == AgentTaskStatus.AGENT_TASK_STATUS_FAILED


class _ReActState(AgentState):
    input: str = ""
    output: Optional[str] = None
    final: Optional[str] = None
    status: Optional[str] = None

    @classmethod
    @override
    def from_query(cls, query: str) -> Self:
        return cls(input=query)

    @override
    def to_task_result(self) -> AgentTaskResult:
        # Deliberately naive: COMPLETED as soon as the ReAct loop answered,
        # although `polish` still changes the answer afterwards.
        if self.output:
            return AgentTaskResult(
                task_status=COMPLETED, content=self.final or self.output
            )
        return AgentTaskResult(
            task_status=WORKING, content=self.status or "working"
        )


@tool
def lookup() -> str:
    """Look something up."""
    return "found"


class _Client:
    """A chat model client that calls `lookup` once, then answers."""

    tools = [lookup]

    def __init__(self):
        self.calls = 0

    async def ainvoke(self, input, history=None):
        self.calls += 1
        if self.calls == 1:
            response = AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {}, "id": "call-1"}],
            )
        else:
            response = AIMessage(content="draft")
        if history is not None:
            history.append(response)
        return response


class _ReActThenPolish(AgentGraph):
    @override
    def setup(self, config: AgentConfig) -> None:
        loop = ReActLoop(
            config=config,
            StateType=_ReActState,
            loop_name="react",
            chat_model_client=_Client(),
            status_key="status",
        )

        def polish(state: _ReActState) -> _ReActState:
            state.final = f"POLISHED: {state.output}"
            return state

        self.graph_builder.add_node("react", loop.as_runnable())
        self.graph_builder.add_node("polish", polish)
        self.graph_builder.add_edge(START, "react")
        self.graph_builder.add_edge("react", "polish")
        self.graph_builder.add_edge("polish", END)


@pytest.mark.parametrize("checkpoints", [False, True])
def test_answer_is_the_final_state_not_the_prebuilt_draft(checkpoints):
    graph = _ReActThenPolish(AgentConfig(checkpoints=checkpoints), _ReActState)

    results = _collect(graph)

    assert results[-1] == AgentTaskResult(
        task_status=COMPLETED, content="POLISHED: draft"
    )
    assert all(r.task_status == WORKING for r in results[:-1])
    progress = [r.content for r in results[:-1]]
    assert "Calling tools: lookup" in progress
    assert all(a != b for a, b in zip(progress, progress[1:], strict=False))


class _Upper(BaseModel):
    text: str


class _NestedGraphWithOwnSchema(AgentGraph):
    @override
    def setup(self, config: AgentConfig) -> None:
        inner = StateGraph(_Upper)
        inner.add_node("upper", lambda s: {"text": s.text.upper()})
        inner.add_edge(START, "upper")
        inner.add_edge("upper", END)
        compiled_inner = inner.compile()

        async def node(state: _Answer) -> _Answer:
            out = await compiled_inner.ainvoke({"text": state.query})
            state.answer = out["text"]
            return state

        self.graph_builder.add_node("node", node)
        self.graph_builder.add_edge(START, "node")
        self.graph_builder.add_edge("node", END)


def test_nested_graph_with_its_own_schema_does_not_fail_the_task():
    results = _collect(_NestedGraphWithOwnSchema(AgentConfig(), _Answer), "hi")

    assert results == [
        AgentTaskResult(task_status=WORKING, content="working"),
        AgentTaskResult(task_status=COMPLETED, content="HI"),
    ]


class _Strict(_Answer):
    @model_validator(mode="after")
    def _reject_bad_answer(self) -> Self:
        if self.answer == "bad":
            raise ValueError("bad answer")
        return self


class _WritesInvalidState(AgentGraph):
    @override
    def setup(self, config: AgentConfig) -> None:
        self.graph_builder.add_node("node", lambda s: {"answer": "bad"})
        self.graph_builder.add_edge(START, "node")
        self.graph_builder.add_edge("node", END)


def test_invalid_state_of_the_agent_graph_fails_the_task():
    results = _collect(_WritesInvalidState(AgentConfig(), _Strict))

    assert results[-1].task_status == FAILED
    assert results[-1].content == "Agent execution failed (ValidationError)."
    assert all(r.task_status == WORKING for r in results[:-1])


class _EndsWithoutAnswer(AgentGraph):
    @override
    def setup(self, config: AgentConfig) -> None:
        self.graph_builder.add_edge(START, END)


def test_graph_that_ends_while_working_fails():
    results = _collect(_EndsWithoutAnswer(AgentConfig(), _Answer))

    assert results[-1].task_status == FAILED
    assert results[-1].content == "Agent finished without a final result."


class _AsksForInput(AgentGraph):
    @override
    def setup(self, config: AgentConfig) -> None:
        def ask(state: _Answer) -> _Answer:
            state.answer = interrupt("Which cell?")
            return state

        self.graph_builder.add_node("ask", ask)
        self.graph_builder.add_edge(START, "ask")
        self.graph_builder.add_edge("ask", END)


def test_interrupt_requires_input():
    results = _collect(_AsksForInput(AgentConfig(checkpoints=True), _Answer))

    assert results[-1] == AgentTaskResult(
        task_status=INPUT_REQUIRED, content="Which cell?"
    )
    assert all(r.task_status == WORKING for r in results[:-1])
