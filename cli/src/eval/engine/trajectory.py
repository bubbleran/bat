"""Condense one conversation's spans into what the agent did, step by step.

A model call keeps only its output: its input is the growing history the
digest already holds. Reads the span dicts bat-adk's JsonFileSpanExporter
writes, without importing the ADK.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict, deque
from itertools import zip_longest
from typing import Any

from .contracts import (
    AgentStep,
    ErrorStep,
    ModelStep,
    Step,
    ToolStep,
    Trajectory,
    TrajectoryTotals,
    TrajectoryTurn,
)

_OPERATION = "gen_ai.operation.name"
_SPAN_KIND = "openinference.span.kind"
_INVOKE_AGENT = "invoke_agent"

# A2A task states, in the eval's own status vocabulary.
TASK_STATES = {
    "TASK_STATE_SUBMITTED": "working",
    "TASK_STATE_WORKING": "working",
    "TASK_STATE_INPUT_REQUIRED": "input-required",
    "TASK_STATE_COMPLETED": "completed",
    "TASK_STATE_FAILED": "error",
    "TASK_STATE_CANCELED": "error",
    "TASK_STATE_REJECTED": "error",
}


# --- building --------------------------------------------------------------


def conversation_roots(
    spans: list[dict[str, Any]], conversation_id: str
) -> list[dict[str, Any]]:
    """The root span of each request the agent served in the conversation.

    Roots have no parent: a called agent's root hangs off its caller's
    CLIENT span, so it is not one of these even though it carries the same
    conversation id.
    """
    return sorted(
        (
            span
            for span in spans
            if span.get("parent_span_id") is None
            and _is_agent_root(span)
            and _attrs(span).get("gen_ai.conversation.id") == conversation_id
        ),
        key=_start,
    )


def build_trajectory(
    spans: list[dict[str, Any]],
    conversation_id: str,
    turns: list[str],
    *,
    max_field_chars: int = 4000,
) -> Trajectory:
    """The digest of conversation ``conversation_id`` in ``spans``.

    Each request the agent served is one trace, whose root carries the
    conversation id; the traces are matched to ``turns`` in order. Spans of
    called agents join their caller's trace through the propagated
    ``traceparent``, so they are found by trace id wherever they were written.
    """
    cap = _Capper(max_field_chars)
    roots = conversation_roots(spans, conversation_id)
    built: list[TrajectoryTurn] = []
    for user, root in zip_longest(turns, roots):
        steps: list[Step] = []
        if root is not None:
            trace = [s for s in spans if s.get("trace_id") == root["trace_id"]]
            steps = _Trace(trace).steps(cap)
        built.append(TrajectoryTurn(user=user, steps=steps))

    trajectory = Trajectory(found=bool(roots), turns=built)
    trajectory.totals = _totals(trajectory)
    trajectory.truncated = cap.truncated
    return trajectory


def tool_calls_from(trajectory: Trajectory) -> list[dict[str, Any]]:
    """Every tool call of the conversation, called agents' included."""
    return [
        {"name": step.name, "args": step.args, "error": step.error}
        for step in all_steps(trajectory)
        if isinstance(step, ToolStep)
    ]


class _Capper:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.truncated = False

    def __call__(self, value: Any) -> Any:
        if isinstance(value, str):
            if len(value) <= self.limit:
                return value
            self.truncated = True
            return f"{value[: self.limit]} …[truncated: {len(value)} chars]"
        if isinstance(value, dict):
            return {key: self(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self(item) for item in value]
        return value


class _Trace:
    """One trace's spans, indexed for walking up and down the tree."""

    def __init__(self, spans: list[dict[str, Any]]) -> None:
        self.spans = sorted(spans, key=_start)
        self.by_id = {span["span_id"]: span for span in self.spans}
        self.children: dict[str | None, list[dict[str, Any]]] = defaultdict(
            list
        )
        for span in self.spans:
            self.children[span.get("parent_span_id")].append(span)
        # Model tool-call decisions, for tools whose span lacks arguments.
        self._args_by_call_id: dict[str, dict[str, Any]] = {}
        self._args_by_name: dict[str, deque[dict[str, Any]]] = defaultdict(
            deque
        )

    def _parent(self, span: dict[str, Any]) -> dict[str, Any] | None:
        return self.by_id.get(span.get("parent_span_id"))

    def _ancestors(self, span: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        current = self._parent(span)
        while current is not None and len(out) < 10_000:
            out.append(current)
            current = self._parent(current)
        return out

    def _has_failed_descendant(self, span: dict[str, Any]) -> bool:
        stack = list(self.children.get(span["span_id"], []))
        while stack:
            child = stack.pop()
            if _failed(child):
                return True
            stack.extend(self.children.get(child["span_id"], []))
        return False

    def _node(self, span: dict[str, Any]) -> str | None:
        """The agent-graph node ``span`` ran in.

        That is the span two levels below the nearest agent root (root ->
        compiled graph -> node). A span hanging off the root directly -- the
        call CallAgentNode makes from a background task -- takes the node
        that was running when it started.
        """
        ancestors = self._ancestors(span)
        for depth, candidate in enumerate(ancestors):
            if _is_agent_root(candidate):
                if depth >= 2:
                    return ancestors[depth - 2].get("name")
                started = _start(span)
                for graph in self.children.get(candidate["span_id"], []):
                    for node in self.children.get(graph["span_id"], []):
                        if _start(node) <= started <= _end(node):
                            return node.get("name")
                break
        return _metadata_node(span)

    def _container(self, span: dict[str, Any]) -> str | None:
        """The nearest enclosing agent call or tool: where a step nests."""
        for ancestor in self._ancestors(span):
            if _is_agent_call(ancestor) or _is_tool(ancestor):
                return ancestor["span_id"]
        return None

    def steps(self, cap: _Capper) -> list[Step]:
        events = [span for span in self.spans if self._is_step(span)]
        for span in events:
            if _is_model(span):
                for call_id, name, args in _model_output(_attrs(span))[1]:
                    if call_id:
                        self._args_by_call_id[call_id] = args
                    if name:
                        self._args_by_name[name].append(args)

        nested: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
        for span in events:
            nested[self._container(span)].append(span)

        def build(container: str | None) -> list[Step]:
            out: list[Step] = []
            for span in nested.get(container, []):
                step = self._step(span, cap)
                if isinstance(step, (AgentStep, ToolStep)):
                    step.steps = build(span["span_id"])
                out.append(step)
            return out

        return build(None)

    def _is_step(self, span: dict[str, Any]) -> bool:
        if _is_model(span) or _is_tool(span) or _is_agent_call(span):
            return True
        return _failed(span) and not self._has_failed_descendant(span)

    def _step(self, span: dict[str, Any], cap: _Capper) -> Step:
        attributes = _attrs(span)
        node = self._node(span)
        error = _error_text(span)
        if _is_model(span):
            texts, calls = _model_output(attributes)
            return ModelStep(
                node=node,
                model=attributes.get("llm.model_name"),
                tokens_in=_int(attributes.get("llm.token_count.prompt")),
                tokens_out=_int(attributes.get("llm.token_count.completion")),
                tokens_cached=_int(
                    attributes.get("llm.token_count.prompt_details.cache_read")
                ),
                reasoning_tokens=_int(
                    attributes.get(
                        "llm.token_count.completion_details.reasoning"
                    )
                ),
                said=cap(_visible_text(texts)),
                called=[name for _, name, _ in calls if name],
                error=cap(error),
            )
        if _is_tool(span):
            name = attributes.get("tool.name") or span.get("name")
            result, call_id = _tool_result(attributes.get("output.value"))
            args = _json_object(attributes.get("input.value"))
            if not args:
                args = self._model_args(call_id, name)
            return ToolStep(
                node=node,
                name=name,
                args=cap(args),
                result=cap(result),
                error=cap(error),
            )
        if _is_agent_call(span):
            return AgentStep(
                node=node,
                agent=attributes.get("gen_ai.agent.name"),
                asked=cap(attributes.get("input.value")),
                answered=cap(attributes.get("output.value")),
                status=TASK_STATES.get(attributes.get("bat.a2a.task_state")),
                error=cap(error),
            )
        return ErrorStep(node=node, span=span.get("name"), error=cap(error))

    def _model_args(self, call_id: str | None, name: str | None) -> dict:
        if call_id and call_id in self._args_by_call_id:
            args = self._args_by_call_id.pop(call_id)
            queue = self._args_by_name.get(name or "")
            if queue and args in queue:
                queue.remove(args)
            return args
        queue = self._args_by_name.get(name or "")
        return queue.popleft() if queue else {}


def _totals(trajectory: Trajectory) -> TrajectoryTotals:
    totals = TrajectoryTotals()
    for step in all_steps(trajectory):
        if isinstance(step, ModelStep):
            totals.model_calls += 1
            totals.tokens_in += step.tokens_in
            totals.tokens_out += step.tokens_out
            totals.tokens_cached += step.tokens_cached
        elif isinstance(step, ToolStep):
            totals.tool_calls += 1
        elif isinstance(step, AgentStep):
            totals.agent_calls += 1
        if step.error:
            totals.errors += 1
    return totals


def all_steps(trajectory: Trajectory) -> list[Step]:
    """Every step of the conversation, called agents' and tools' included."""

    def walk(steps: list[Step]) -> list[Step]:
        out: list[Step] = []
        for step in steps:
            out.append(step)
            out.extend(walk(getattr(step, "steps", [])))
        return out

    return walk([step for turn in trajectory.turns for step in turn.steps])


# --- span helpers ------------------------------------------------------------


def _attrs(span: dict[str, Any]) -> dict[str, Any]:
    return span.get("attributes") or {}


def _start(span: dict[str, Any]) -> float:
    value = span.get("start_time")
    return float(value) if isinstance(value, (int, float)) else 0.0


def _end(span: dict[str, Any]) -> float:
    value = span.get("end_time")
    return float(value) if isinstance(value, (int, float)) else math.inf


def _is_agent_call(span: dict[str, Any]) -> bool:
    return (
        span.get("kind") == "CLIENT"
        and _attrs(span).get(_OPERATION) == _INVOKE_AGENT
    )


def _is_agent_root(span: dict[str, Any]) -> bool:
    """The span an agent opens around each request it serves."""
    return (
        span.get("kind") != "CLIENT"
        and _attrs(span).get(_OPERATION) == _INVOKE_AGENT
    )


def _is_model(span: dict[str, Any]) -> bool:
    return _attrs(span).get(_SPAN_KIND) == "LLM"


def _is_tool(span: dict[str, Any]) -> bool:
    return _attrs(span).get(_SPAN_KIND) == "TOOL"


def _failed(span: dict[str, Any]) -> bool:
    return span.get("status") == "ERROR"


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _error_text(span: dict[str, Any]) -> str | None:
    """Why ``span`` failed, or None when it did not."""
    if not _failed(span):
        return None
    for event in span.get("events") or []:
        if event.get("name") != "exception":
            continue
        attributes = event.get("attributes") or {}
        kind = attributes.get("exception.type")
        message = attributes.get("exception.message")
        if kind and message:
            return f"{kind}: {message}"
        if kind or message:
            return str(kind or message)
    # Older ADK file exports have no events; OpenInference puts the error and
    # its whole traceback in the description.
    description = str(span.get("status_description") or "")
    return description.split("Traceback")[0].strip() or "failed"


def _metadata_node(span: dict[str, Any]) -> str | None:
    metadata = _json_object(_attrs(span).get("metadata"))
    namespace = metadata.get("langgraph_checkpoint_ns") or ""
    if namespace:
        return namespace.split("|")[0].split(":")[0] or None
    return metadata.get("langgraph_node")


def _json_object(raw: Any) -> dict[str, Any]:
    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _model_output(
    attributes: dict[str, Any],
) -> tuple[list[str], list[tuple[str | None, str | None, dict[str, Any]]]]:
    """The texts a model call produced, and the tool calls it decided on."""
    messages: dict[int, dict[str, Any]] = defaultdict(dict)
    for key, value in attributes.items():
        match = re.match(r"^llm\.output_messages\.(\d+)\.message\.(.+)$", key)
        if match:
            messages[int(match.group(1))][match.group(2)] = value

    texts: list[str] = []
    calls: list[tuple[str | None, str | None, dict[str, Any]]] = []
    for index in sorted(messages):
        fields = messages[index]
        content = fields.get("content")
        if isinstance(content, str) and content.strip():
            texts.append(content)
        # Content blocks (e.g. OpenAI's Responses API) instead of a string.
        blocks: dict[int, str] = {}
        for key, value in fields.items():
            match = re.match(r"^contents\.(\d+)\.message_content\.text$", key)
            if match and isinstance(value, str) and value.strip():
                blocks[int(match.group(1))] = value
        texts.extend(blocks[index] for index in sorted(blocks))

        call_fields: dict[int, dict[str, Any]] = defaultdict(dict)
        for key, value in fields.items():
            match = re.match(r"^tool_calls\.(\d+)\.tool_call\.(.+)$", key)
            if match:
                call_fields[int(match.group(1))][match.group(2)] = value
        for call_index in sorted(call_fields):
            call = call_fields[call_index]
            calls.append(
                (
                    call.get("id"),
                    call.get("function.name"),
                    _json_object(call.get("function.arguments")),
                )
            )
    return texts, calls


def _visible_text(texts: list[str]) -> str | None:
    """What the model said, without the thinking some models put first."""
    return "\n".join(texts).rsplit("</think>", 1)[-1].strip() or None


def _tool_result(raw: Any) -> tuple[str | None, str | None]:
    """A tool span's result text, and the tool-call id it answered."""
    if raw is None:
        return None, None
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return raw, None
    if (
        isinstance(parsed, dict)
        and "update" in parsed
        and set(parsed) <= {"graph", "update", "resume", "goto"}
    ):
        # A serialized LangGraph Command.
        return _command_result(parsed["update"]), None
    if isinstance(parsed, dict):
        # A serialized ToolMessage: {"type": "tool", "data": {...}}.
        data = parsed.get("data") if parsed.get("type") == "tool" else parsed
        if isinstance(data, dict) and "content" in data:
            text = data["content"]
            if not isinstance(text, str):
                text = json.dumps(text, ensure_ascii=False)
            return text, data.get("tool_call_id")
    if isinstance(raw, str):
        return raw, None
    return json.dumps(parsed, ensure_ascii=False), None


def _command_result(update: Any) -> str | None:
    """A LangGraph ``Command``'s output: the ToolMessages in its update."""
    texts: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "tool" and isinstance(
                value.get("data"), dict
            ):
                texts.append(_tool_result(value)[0])
            else:
                for item in value.values():
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(update)
    return "\n".join(texts) or None


# --- rendering (for the judge) --------------------------------------------


def render_trajectory(trajectory: Trajectory, *, max_chars: int) -> str:
    """The digest as numbered lines, at most ``max_chars`` long.

    Over budget, long fields are shortened first, then the middle steps of
    each turn are left out -- the first and last steps of a turn are where
    the request is understood and the answer is formed.
    """
    for field_chars in (None, 600, 250, 100):
        text = _render(trajectory, field_chars=field_chars, keep=None)
        if len(text) <= max_chars:
            return text
    for keep in (4, 2, 1):
        text = _render(trajectory, field_chars=100, keep=keep)
        if len(text) <= max_chars:
            return text
    marker = "\n…[rendering cut; the rest of the steps are omitted]"
    return text[: max(0, max_chars - len(marker))] + marker


def _render(
    trajectory: Trajectory, *, field_chars: int | None, keep: int | None
) -> str:
    def short(text: Any) -> str:
        value = "" if text is None else str(text)
        if field_chars is not None and len(value) > field_chars:
            return f"{value[:field_chars]}…"
        return value

    lines: list[str] = []
    for number, turn in enumerate(trajectory.turns, 1):
        lines.append(f"Turn {number} - user: {short(turn.user)}")
        steps = turn.steps
        if not steps:
            lines.append("  (no steps recorded)")
        if keep is not None and len(steps) > 2 * keep:
            _render_steps(steps[:keep], "", short, lines)
            lines.append(f"  … {len(steps) - 2 * keep} steps omitted …")
            tail = steps[-keep:]
            _render_steps(tail, "", short, lines, start=len(steps) - keep + 1)
        else:
            _render_steps(steps, "", short, lines)

    totals = trajectory.totals
    lines.append(
        f"Totals: {totals.model_calls} model calls, {totals.tool_calls} tool "
        f"calls, {totals.agent_calls} agent calls, {totals.errors} failed steps"
    )
    return "\n".join(lines)


def _render_steps(steps, prefix, short, lines, start=1) -> None:
    pad = "  " + "   " * prefix.count(".")
    for offset, step in enumerate(steps):
        label = f"{prefix}{start + offset}"
        head = f"{pad}{label}{'.' if not prefix else ''} "
        if isinstance(step, ModelStep):
            # No model or token counts: no rubric scores cost.
            parts = []
            if step.called:
                parts.append("called: " + ", ".join(step.called))
            if step.said:
                parts.append(f'said: "{short(step.said)}"')
            if not parts:
                parts.append("said nothing")
            if step.error:
                parts.append(f"FAILED: {short(step.error)}")
            label_text = f"[model {step.node or '?'}] "
            lines.append(head + label_text + "; ".join(parts))
        elif isinstance(step, ToolStep):
            outcome = short(step.result) or "(no result)"
            if step.error:
                outcome = f"FAILED: {short(step.error)}"
            args = short(json.dumps(step.args, ensure_ascii=False))
            lines.append(f"{head}[tool {step.name}] args: {args} -> {outcome}")
            _render_steps(step.steps, f"{label}.", short, lines)
        elif isinstance(step, AgentStep):
            asked = "(not recorded)"
            if step.asked is not None:
                asked = f'"{short(step.asked)}"'
            lines.append(f"{head}[agent {step.agent}] asked: {asked}")
            _render_steps(step.steps, f"{label}.", short, lines)
            if step.error:
                outcome = f"FAILED: {short(step.error)}"
            elif step.answered is not None:
                answered = short(step.answered)
                outcome = f'{step.status or "answered"}: "{answered}"'
            else:
                outcome = step.status or "(answer not recorded)"
            lines.append(f"{pad}   -> {outcome}")
        else:
            lines.append(
                f"{head}[error {step.node or step.span or '?'}] "
                f"FAILED: {short(step.error)}"
            )
