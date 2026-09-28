import asyncio
from typing import List, Optional
from unittest.mock import MagicMock

from typing_extensions import Self, override

from bat.agent import (
    AgentGraph,
    AgentState,
    AgentTaskResult,
    AgentTaskStatus,
)


class _State(AgentState):
    question: str = ""
    answer: Optional[str] = None

    @classmethod
    @override
    def from_query(cls, query: str) -> Self:
        return cls(question=query)

    @override
    def to_task_result(self) -> AgentTaskResult:
        return AgentTaskResult(
            task_status=AgentTaskStatus.AGENT_TASK_STATUS_COMPLETED,
            content=self.answer or "",
        )


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
    graph.StateType = _State
    graph._memory = MagicMock()
    graph._memory.get = MagicMock(return_value=None)
    graph._graph = MagicMock()
    graph._graph.astream = _failing_stream(stream_error)
    graph._graph.get_state = MagicMock(return_value=MagicMock(tasks=tasks))
    return graph


def _collect(graph: _Graph) -> List[AgentTaskResult]:
    async def drive():
        return [
            item
            async for item in graph.astream(
                query="question",
                config={"configurable": {"thread_id": "t"}},
            )
        ]

    return asyncio.run(drive())


def test_node_error_is_reported_not_masked_by_an_empty_interrupt():
    task = MagicMock(interrupts=())
    graph = _graph_with((task,), RuntimeError("node blew up"))

    results = _collect(graph)

    assert [r.task_status for r in results] == [
        AgentTaskStatus.AGENT_TASK_STATUS_FAILED
    ]
    assert "node blew up" in results[0].content


def test_pending_interrupt_is_still_yielded():
    task = MagicMock(interrupts=(MagicMock(value="need input"),))
    graph = _graph_with((task,), RuntimeError("interrupted"))

    results = _collect(graph)

    assert results[-1].task_status == (
        AgentTaskStatus.AGENT_TASK_STATUS_INPUT_REQUIRED
    )
    assert results[-1].content == "need input"


def test_no_tasks_yields_no_interrupt():
    graph = _graph_with((), RuntimeError("boom"))

    results = _collect(graph)

    assert len(results) == 1
    assert results[0].task_status == AgentTaskStatus.AGENT_TASK_STATUS_FAILED
