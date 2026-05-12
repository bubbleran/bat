from __future__ import annotations

from typing import Any, Iterable


def build_context_from_events(events: Iterable[dict[str, Any]]) -> str:
    """Build a labelled conversation transcript from trace events.

    Labels:
      [USER]         — explicit user input
      [AGENT OUTPUT] — agent-generated content sent to the user (input-required status);
                       values here are agent proposals, NOT trusted system feedback
      [SYSTEM]       — internal status updates from the runtime
    """
    lines: list[str] = []
    for event in events:
        t_ms = event.get("t_ms")
        status = event.get("task_status", "")
        preview = event.get("content_preview", "")
        user_input = event.get("user_input")

        prefix = f"{float(t_ms):.0f}ms | " if t_ms is not None else ""

        if user_input:
            lines.append(f"[{prefix}USER] {user_input}")
        elif status == "input-required":
            lines.append(f"[{prefix}AGENT OUTPUT] {preview}")
        else:
            lines.append(f"[{prefix}SYSTEM] {preview}")

    return "\n".join(lines) if lines else "No events"


def build_user_facts_summary(events: Iterable[dict[str, Any]]) -> str:
    """Return a bullet list of everything the user explicitly stated across all turns."""
    facts = [
        f"- {event['user_input'].strip()}"
        for event in events
        if event.get("user_input")
    ]
    return "\n".join(facts) if facts else "No explicit user statements recorded."


def build_expected_desc(
    status: str | None = "completed",
    expected_outcome: str | None = None,
    output_must_contain: list[str] | None = None,
    expected_tool_calls: list[Any] | None = None,
) -> str:
    parts: list[str] = []
    if expected_outcome:
        parts.append(f"Expected outcome: {expected_outcome.strip()}")
    if status is not None:
        parts.append(f"The task should reach final status '{status}'.")
    if output_must_contain:
        quoted = ", ".join(f'"{s}"' for s in output_must_contain)
        parts.append(f"Output must contain: {quoted}.")
    if expected_tool_calls:
        calls = [
            f"'{c.name}' (at least {c.times}×)" if c.times > 1 else f"'{c.name}'"
            for c in expected_tool_calls
        ]
        parts.append(f"Expected tool calls: {', '.join(calls)}.")
    return " ".join(parts) if parts else "No specific expectations defined."
