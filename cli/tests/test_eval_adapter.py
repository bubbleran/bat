"""The episode's final output is the agent's answer, read off the A2A stream.

Only the network client is faked: the chunks are the ones a scaffolded agent
actually streams, and the adapter folds them exactly as in a live run.
"""

from __future__ import annotations

import asyncio

from a2a.types import (
    Artifact,
    Part,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)

from eval.engine import adapter as adapter_module
from eval.engine.adapter import BatA2AAdapter
from eval.engine.contracts import TaskSpec


def _turn_as_the_adk_streams_it(answer: str) -> list[StreamResponse]:
    """Captured from a scaffolded agent: the answer arrives as an artifact,
    and the stream then closes on a completed status carrying no message."""
    return [
        StreamResponse(
            task=Task(
                id="t1",
                context_id="c1",
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
            )
        ),
        StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="t1",
                context_id="c1",
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            )
        ),
        StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="c1",
                artifact=Artifact(artifact_id="a1", parts=[Part(text=answer)]),
            )
        ),
        StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="t1",
                context_id="c1",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            )
        ),
    ]


def _run(monkeypatch, turns: list[list[StreamResponse]], texts: list[str]):
    remaining = list(turns)

    class _Resolver:
        def __init__(self, httpx_client, base_url) -> None:
            pass

        async def get_agent_card(self):
            return object()

    class _Client:
        async def send_message(self, request):
            for chunk in remaining.pop(0):
                yield chunk

    class _Factory:
        def __init__(self, config) -> None:
            pass

        def create(self, card):
            return _Client()

    monkeypatch.setattr(adapter_module, "A2ACardResolver", _Resolver)
    monkeypatch.setattr(adapter_module, "ClientFactory", _Factory)
    task = TaskSpec(id="task", turns=texts)
    return asyncio.run(
        BatA2AAdapter("http://agent").run_task(task, thread_id="c1")
    )


def test_final_output_is_the_answer_not_the_closing_status(
    monkeypatch,
) -> None:
    result = _run(
        monkeypatch, [_turn_as_the_adk_streams_it("echo: hi")], ["hi"]
    )

    assert result.final_status == "completed"
    assert result.final_output == "echo: hi"


def test_final_output_is_the_last_turns_answer(monkeypatch) -> None:
    result = _run(
        monkeypatch,
        [
            _turn_as_the_adk_streams_it("first answer"),
            _turn_as_the_adk_streams_it("second answer"),
        ],
        ["one", "two"],
    )

    assert result.final_output == "second answer"
