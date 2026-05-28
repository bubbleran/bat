from __future__ import annotations

from typing import Any

from .contracts import EpisodeVerdict, ExpectedToolCall, TaskExpected


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, value in expected.items():
            if key not in actual or not _is_subset(value, actual[key]):
                return False
        return True

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) > len(actual):
            return False
        used = [False] * len(actual)
        for expected_item in expected:
            matched = False
            for idx, actual_item in enumerate(actual):
                if used[idx]:
                    continue
                if _is_subset(expected_item, actual_item):
                    used[idx] = True
                    matched = True
                    break
            if not matched:
                return False
        return True

    return expected == actual


def _count_matches(expected: ExpectedToolCall, observed: list[dict[str, Any]]) -> int:
    total = 0
    for call in observed:
        if call.get("name") != expected.name:
            continue
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if _is_subset(expected.args_subset, args):
            total += 1
    return total


class EpisodeEvaluator:
    def evaluate(
        self,
        status: str,
        output_text: str,
        tool_calls: list[dict[str, Any]],
        expected: TaskExpected,
    ) -> EpisodeVerdict:
        checks: list[tuple[bool, str]] = []

        if expected.status is not None:
            ok = status == expected.status
            reason = f"status: '{status}'" if ok else f"status: got '{status}', expected '{expected.status}'"
            checks.append((ok, reason))

        phrases = expected.output_must_contain or []
        n = len(phrases)
        for i, phrase in enumerate(phrases):
            label = f"output[{i}]" if n > 1 else "output"
            ok = phrase in output_text
            reason = f"{label}: contains '{phrase}'" if ok else f"{label}: missing '{phrase}'"
            checks.append((ok, reason))

        for exp_call in expected.tool_calls:
            count = _count_matches(exp_call, tool_calls)
            ok = count >= exp_call.times
            label = f"tool_call:{exp_call.name}"
            reason = (
                f"{label}: called {count}×"
                if ok
                else f"{label}: called {count}×, expected ≥{exp_call.times}×"
            )
            checks.append((ok, reason))

        overall = all(ok for ok, _ in checks) if checks else True
        reason = "; ".join(r for _, r in checks)
        return EpisodeVerdict(passed=overall, reason=reason)
