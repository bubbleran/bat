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


def build_expected_desc(must_succeed: bool, final_contains: str | None = None) -> str:
    description = f"must_succeed={must_succeed}"
    if final_contains:
        description += f", response should contain: '{final_contains}'"
    return description
