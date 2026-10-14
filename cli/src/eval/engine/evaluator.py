from __future__ import annotations

from typing import Any

from .contracts import AgentStep, EpisodeResult, EpisodeVerdict, TaskExpected
from .trajectory import all_steps


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _is_subset(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        remaining = list(actual)
        for item in expected:
            matches = [
                i for i, a in enumerate(remaining) if _is_subset(item, a)
            ]
            if not matches:
                return False
            del remaining[matches[0]]
        return True
    return expected == actual


def _called(label: str, count: int, times: int) -> tuple[bool, str]:
    ok = count >= times
    return ok, f"{label}: called {count}×" + (
        "" if ok else f", expected ≥{times}×"
    )


def _at_most(label: str, value: int, limit: int) -> tuple[bool, str]:
    ok = value <= limit
    return ok, f"{label}: {value}" + (
        "" if ok else f", expected at most {limit}"
    )


def _name(step: Any) -> str:
    for field in ("name", "agent", "model", "span"):
        if getattr(step, field, None):
            return f"{step.kind} {getattr(step, field)}"
    return f"{step.kind} {step.kind}"


def verdict(episode: EpisodeResult, expected: TaskExpected) -> EpisodeVerdict:
    """Every expectation of the task, checked; passed only if all hold."""
    checks: list[tuple[bool, str]] = []
    status = episode.final_status
    if expected.status is not None:
        ok = status == expected.status
        checks.append(
            (
                ok,
                f"status: '{status}'"
                if ok
                else f"status: got '{status}', expected '{expected.status}'",
            )
        )

    phrases = expected.output_must_contain
    for index, phrase in enumerate(phrases):
        label = f"output[{index}]" if len(phrases) > 1 else "output"
        ok = phrase in episode.final_output
        checks.append(
            (ok, f"{label}: {'contains' if ok else 'missing'} '{phrase}'")
        )

    trajectory = episode.trace.trajectory
    if expected.needs_spans and not trajectory.found:
        checks.append(
            (
                False,
                "trace: no spans found for this conversation -- is "
                "telemetry on, and written where the eval reads it?",
            )
        )
    elif expected.needs_spans:
        steps = all_steps(trajectory)
        for call in expected.tool_calls:
            count = sum(
                1
                for made in episode.trace.tool_calls
                if made["name"] == call.name
                and _is_subset(call.args_subset, made["args"])
            )
            checks.append(_called(f"tool_call:{call.name}", count, call.times))
        for call in expected.agent_calls:
            count = sum(
                1
                for step in steps
                if isinstance(step, AgentStep) and step.agent == call.agent
            )
            checks.append(
                _called(f"agent_call:{call.agent}", count, call.times)
            )
        totals = trajectory.totals
        if expected.max_model_calls is not None:
            checks.append(
                _at_most(
                    "model calls", totals.model_calls, expected.max_model_calls
                )
            )
        if expected.max_tokens is not None:
            used = totals.tokens_in + totals.tokens_out
            checks.append(_at_most("tokens", used, expected.max_tokens))
        if expected.no_errors:
            failed = [step for step in steps if step.error]
            shown = "; ".join(f"{_name(s)}: {s.error}" for s in failed[:3])
            more = f" (+{len(failed) - 3} more)" if len(failed) > 3 else ""
            checks.append(
                (
                    not failed,
                    f"errors: {len(failed)} failed steps -- {shown}{more}"
                    if failed
                    else "errors: none",
                )
            )

    return EpisodeVerdict(
        passed=all(ok for ok, _ in checks),
        reason="; ".join(reason for _, reason in checks),
    )
