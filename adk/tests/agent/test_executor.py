import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.types import Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent
from a2a.utils.errors import InternalError

from bat.agent import AgentTaskResult, AgentTaskStatus
from bat.agent._executor import MinimalAgentExecutor


class _Graph:
    """Stands in for an AgentGraph: yields `results` and records whether its
    cleanup (`finally`) ran."""

    def __init__(self, results):
        self.results = results
        self.closed = False

    async def astream(self, query, config):
        try:
            for result in self.results:
                yield result
        finally:
            self.closed = True


def _context() -> MagicMock:
    context = MagicMock()
    context.get_user_input.return_value = "question"
    context.current_task = Task(id="task", context_id="context")
    return context


def _execute(graph: _Graph, event_queue: AsyncMock) -> None:
    executor = MinimalAgentExecutor(graph)
    asyncio.run(executor.execute(_context(), event_queue))


def test_completed_result_publishes_artifact_and_status():
    graph = _Graph(
        [
            AgentTaskResult(
                task_status=AgentTaskStatus.AGENT_TASK_STATUS_WORKING,
                content="working",
            ),
            AgentTaskResult(
                task_status=AgentTaskStatus.AGENT_TASK_STATUS_COMPLETED,
                content="answer",
            ),
        ]
    )
    event_queue = AsyncMock()

    _execute(graph, event_queue)

    events = [call.args[0] for call in event_queue.enqueue_event.call_args_list]
    assert [type(e) for e in events] == [
        TaskStatusUpdateEvent,
        TaskArtifactUpdateEvent,
        TaskStatusUpdateEvent,
    ]
    assert events[1].artifact.parts[0].text == "answer"
    assert graph.closed


def test_graph_stream_is_closed_when_publishing_fails():
    graph = _Graph(
        [
            AgentTaskResult(
                task_status=AgentTaskStatus.AGENT_TASK_STATUS_WORKING,
                content="working",
            ),
            AgentTaskResult(
                task_status=AgentTaskStatus.AGENT_TASK_STATUS_COMPLETED,
                content="answer",
            ),
        ]
    )
    event_queue = AsyncMock()
    event_queue.enqueue_event.side_effect = RuntimeError("queue closed")

    async def drive() -> bool:
        executor = MinimalAgentExecutor(graph)
        with pytest.raises(InternalError):
            await executor.execute(_context(), event_queue)
        # Checked inside the loop: `asyncio.run` closes leftover generators
        # on shutdown, which would hide a missing close.
        return graph.closed

    assert asyncio.run(drive())
