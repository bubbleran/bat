from __future__ import annotations

import json
from typing import Any, Dict, List

from .contracts import CheckResult, EpisodeVerdict, ExpectedToolCall, TaskExpected


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


def _count_matches(expected: ExpectedToolCall, observed: List[Dict[str, Any]]) -> int:
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
        tool_calls: List[Dict[str, Any]],
        expected: TaskExpected,
    ) -> EpisodeVerdict:
        checks: Dict[str, CheckResult] = {}

        if expected.status is not None:
            passed = status == expected.status
            checks["status"] = CheckResult(
                passed=passed,
                reason=f"'{status}'"
                if passed
                else f"got '{status}', expected '{expected.status}'",
            )

        phrases = expected.output_must_contain or []
        n = len(phrases)
        for i, phrase in enumerate(phrases):
            key = f"output[{i}]" if n > 1 else "output"
            passed = phrase in output_text
            checks[key] = CheckResult(
                passed=passed,
                reason=f"contains '{phrase}'" if passed else f"missing '{phrase}'",
            )

        for exp_call in expected.tool_calls:
            count = _count_matches(exp_call, tool_calls)
            passed = count >= exp_call.times
            checks[f"tool_call:{exp_call.name}"] = CheckResult(
                passed=passed,
                reason=f"called {count}×"
                if passed
                else f"called {count}×, expected ≥{exp_call.times}×",
            )

        overall = all(c.passed for c in checks.values()) if checks else True
        return EpisodeVerdict(passed=overall, checks=checks)
