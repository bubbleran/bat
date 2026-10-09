"""The episode's final output is the agent's answer, read off the A2A stream.

Only the network client is faked: the chunks are the ones a scaffolded agent
actually streams, and the adapter folds them exactly as in a live run.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

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
from eval.engine.adapter import run_task
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


def _run(
    monkeypatch,
    turns: list[list[StreamResponse]],
    texts: list[str],
    *,
    thread_id: str = "c1",
    span_paths: list[str] | None = None,
):
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
        run_task("http://agent", task, thread_id, span_paths or [])
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


FIXTURE = Path(__file__).parent / "fixtures" / "spans" / "probe_two_turns"
PROBE_TURNS = [
    "Create a 5G network named lab-1 with two cells on band n78.",
    "Actually make it three cells.",
]


def _probe(monkeypatch, *span_paths: str):
    return _run(
        monkeypatch,
        [_turn_as_the_adk_streams_it("a"), _turn_as_the_adk_streams_it("b")],
        PROBE_TURNS,
        thread_id="probe-conv-1",
        span_paths=list(span_paths),
    )


def test_each_episode_carries_its_trajectory(monkeypatch) -> None:
    result = _probe(monkeypatch, str(FIXTURE))

    trajectory = result.trace.trajectory
    assert trajectory.found is True
    assert [turn.user for turn in trajectory.turns] == PROBE_TURNS
    assert trajectory.totals.agent_calls == 2


def test_tool_calls_include_the_called_agents_with_their_errors(
    monkeypatch,
) -> None:
    result = _probe(monkeypatch, str(FIXTURE))

    assert [call["name"] for call in result.trace.tool_calls] == [
        "list_networks",
        "check_operator",
        "list_networks",
        "check_operator",
    ]
    assert result.trace.tool_calls[1]["error"] == (
        "TimeoutError: operator did not answer in 30s"
    )


def test_a_called_agent_can_write_its_spans_elsewhere(
    monkeypatch, tmp_path
) -> None:
    """The eval's spans directory is new for every run, so a called agent
    keeps its own file, and the eval is told where it is."""
    run_dir = tmp_path / "spans-0"
    run_dir.mkdir()
    shutil.copy(FIXTURE / "supervisor.jsonl", run_dir)
    elsewhere = tmp_path / "netops.jsonl"
    shutil.copy(FIXTURE / "netops.jsonl", elsewhere)

    alone = _probe(monkeypatch, str(run_dir))
    joined = _probe(monkeypatch, str(run_dir), str(elsewhere))

    assert alone.trace.trajectory.turns[0].steps[3].steps == []
    assert [
        s.kind for s in joined.trace.trajectory.turns[0].steps[3].steps
    ] == [
        "model",
        "tool",
    ]


def test_without_a_spans_directory_the_trajectory_is_not_found(
    monkeypatch,
) -> None:
    result = _probe(monkeypatch)

    assert result.trace.trajectory.found is False
