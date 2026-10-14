"""The deterministic verdict: status, output, and checks read off the trace.

The trajectory comes from the real two-turn fixture (see
test_eval_trajectory.py): 8 model calls, 1629 + 2997 tokens, 2 calls to
``netops``, 4 failed steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.engine.adapter import read_spans
from eval.engine.contracts import (
    EpisodeResult,
    EpisodeTrace,
    ModelStep,
    TaskExpected,
    Trajectory,
    TrajectoryTotals,
    TrajectoryTurn,
)
from eval.engine.evaluator import verdict
from eval.engine.trajectory import build_trajectory, tool_calls_from

FIXTURE = Path(__file__).parent / "fixtures" / "spans" / "probe_two_turns"


@pytest.fixture(scope="module")
def failing_run() -> Trajectory:
    return build_trajectory(
        read_spans([str(FIXTURE)]),
        "probe-conv-1",
        ["Create lab-1", "Make it three cells"],
    )


NO_SPANS = Trajectory(found=False, turns=[TrajectoryTurn(user="hi")])


def _verdict(expected: dict, trajectory: Trajectory, status="completed"):
    episode = EpisodeResult(
        task_id="t",
        final_status=status,
        final_output="done",
        trace=EpisodeTrace(
            trajectory=trajectory, tool_calls=tool_calls_from(trajectory)
        ),
    )
    return verdict(episode, TaskExpected.model_validate(expected))


def test_no_errors_fails_on_a_run_with_failed_steps(failing_run) -> None:
    verdict = _verdict({"no_errors": True}, failing_run)

    assert verdict.passed is False
    assert "4 failed steps" in verdict.reason
    assert "check_operator: TimeoutError" in verdict.reason


def test_no_errors_passes_a_clean_run() -> None:
    clean_run = Trajectory(
        found=True,
        turns=[TrajectoryTurn(user="hi", steps=[ModelStep(said="hello")])],
        totals=TrajectoryTotals(model_calls=1, tokens_in=10, tokens_out=5),
    )

    assert _verdict({"no_errors": True}, clean_run).passed is True


def test_model_calls_over_the_limit_fail(failing_run) -> None:
    verdict = _verdict({"max_model_calls": 6}, failing_run)

    assert verdict.passed is False
    assert "model calls: 8, expected at most 6" in verdict.reason


def test_model_calls_at_the_limit_pass(failing_run) -> None:
    assert _verdict({"max_model_calls": 8}, failing_run).passed is True


def test_tokens_over_the_limit_fail(failing_run) -> None:
    verdict = _verdict({"max_tokens": 4000}, failing_run)

    assert verdict.passed is False
    assert "tokens: 4626, expected at most 4000" in verdict.reason


def test_expected_agent_calls_are_counted(failing_run) -> None:
    passed = _verdict(
        {"agent_calls": [{"agent": "netops", "times": 2}]}, failing_run
    )
    too_few = _verdict(
        {"agent_calls": [{"agent": "netops", "times": 3}]}, failing_run
    )
    never = _verdict({"agent_calls": [{"agent": "billing"}]}, failing_run)

    assert passed.passed is True
    assert too_few.passed is False
    assert "agent_call:netops: called 2×, expected ≥3×" in too_few.reason
    assert never.passed is False


def test_span_checks_without_spans_fail_once_and_say_why() -> None:
    verdict = _verdict(
        {"no_errors": True, "tool_calls": [{"name": "list_networks"}]},
        NO_SPANS,
    )

    assert verdict.passed is False
    assert "no spans" in verdict.reason
    # Not the misleading per-check failures on an empty trace.
    assert "called 0×" not in verdict.reason


def test_status_and_output_checks_do_not_need_spans() -> None:
    verdict = _verdict(
        {"status": "completed", "output_must_contain": ["done"]}, NO_SPANS
    )

    assert verdict.passed is True


def test_a_wrong_status_still_fails(failing_run) -> None:
    verdict = _verdict({"status": "completed"}, failing_run, status="error")

    assert verdict.passed is False
    assert "status: got 'error', expected 'completed'" in verdict.reason


def test_expected_tool_calls_see_tools_of_called_agents(failing_run) -> None:
    verdict = _verdict(
        {"tool_calls": [{"name": "check_operator", "times": 2}]}, failing_run
    )

    assert verdict.passed is True
