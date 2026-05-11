from __future__ import annotations

from typing import Any, Iterable


def build_context_from_events(events: Iterable[dict[str, Any]]) -> str:
    lines: list[str] = []
    for event in events:
        t_ms = event.get("t_ms")
        status = event.get("task_status", "")
        preview = event.get("content_preview", "")
        user_input = event.get("user_input")

        if user_input:
            if t_ms is None:
                lines.append(f"[USER] {user_input}")
            else:
                lines.append(f"[{float(t_ms):.0f}ms | USER] {user_input}")
        else:
            if t_ms is None:
                lines.append(f"[{status}] {preview}")
            else:
                lines.append(f"[{float(t_ms):.0f}ms | {status}] {preview}")

    return "\n".join(lines) if lines else "No events"


def build_expected_desc(
    status: str | None = "completed",
    expected_outcome: str | None = None,
    expected_tool_calls: list[Any] | None = None,
) -> str:
    parts: list[str] = []
    if expected_outcome:
        parts.append(f"Expected outcome: {expected_outcome.strip()}")
    if status is not None:
        parts.append(f"The task should reach final status '{status}'.")
    if expected_tool_calls:
        calls = [
            f"'{c.name}' (at least {c.times}×)" if c.times > 1 else f"'{c.name}'"
            for c in expected_tool_calls
        ]
        parts.append(f"Expected tool calls: {', '.join(calls)}.")
    return " ".join(parts) if parts else "No specific expectations defined."
