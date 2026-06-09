from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import (
    get_artifact_text,
    get_message_text,
    new_text_message,
)
from a2a.types import Role, SendMessageRequest, StreamResponse, TaskState
from httpx import AsyncClient

from .contracts import EpisodeResult, EpisodeTrace, TaskSpec, TraceEvent

TERMINAL_STATUSES = {"completed", "error", "input-required"}


_TASK_STATE_TO_STR = {
    TaskState.TASK_STATE_SUBMITTED: "working",
    TaskState.TASK_STATE_WORKING: "working",
    TaskState.TASK_STATE_INPUT_REQUIRED: "input-required",
    TaskState.TASK_STATE_COMPLETED: "completed",
    TaskState.TASK_STATE_FAILED: "error",
    TaskState.TASK_STATE_CANCELED: "error",
    TaskState.TASK_STATE_REJECTED: "error",
}


# Span attribute keys emitted by the agent: OpenInference (LLM/tool spans) plus
# the ADK manual spans. Usage and tool calls are reconstructed from the spans
# the agent writes to JSON-Lines files (OTEL_TRACES_EXPORTER=file), since they
# are no longer carried in the A2A message metadata.
_ATTR_CONVERSATION_ID = "gen_ai.conversation.id"
_ATTR_TOKEN_PROMPT = "llm.token_count.prompt"
_ATTR_TOKEN_COMPLETION = "llm.token_count.completion"
_ATTR_TOKEN_TOTAL = "llm.token_count.total"
_ATTR_SPAN_KIND = "openinference.span.kind"
_ATTR_TOOL_NAME = "tool.name"
_ATTR_INPUT_VALUE = "input.value"


def _read_spans_dir(directory: str) -> list[dict[str, Any]]:
    """Read every ``*.jsonl`` span file in ``directory`` into one list.

    A directory (not a single file) so multi-agent runs work: each agent
    process writes its own span file there, and they are recomposed by
    ``trace_id`` downstream. Missing directory or unreadable lines are skipped.
    """
    spans: list[dict[str, Any]] = []
    base = Path(directory)
    if not base.is_dir():
        return spans
    for span_file in sorted(base.glob("*.jsonl")):
        try:
            with open(span_file, encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        spans.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue
    return spans


def _tool_call_from_span(
    span: dict[str, Any], attributes: dict[str, Any]
) -> dict[str, Any]:
    name = attributes.get(_ATTR_TOOL_NAME) or attributes.get("gen_ai.tool.name")
    args: dict[str, Any] = {}
    raw = attributes.get(_ATTR_INPUT_VALUE)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                args = parsed
        except json.JSONDecodeError:
            pass
    return {"name": name, "args": args, "id": span.get("span_id")}


def _aggregate_from_spans(
    spans_dir: str, conversation_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """Reconstruct ``(usage, tool_calls, found)`` for one episode from spans.

    Spans are grouped by the trace(s) whose root carries
    ``gen_ai.conversation.id == conversation_id``. Every span sharing one of
    those ``trace_id``\\ s is then aggregated — including spans from remote
    sub-agents, which run in their own process and write their own file but
    share the ``trace_id`` via the propagated W3C ``traceparent``. This is how
    multi-agent usage is recomposed across processes.

    ``found`` is False when no span carries the conversation id yet (the agent
    may not have written the root span); callers can retry to absorb the small
    write race.
    """
    spans = _read_spans_dir(spans_dir)
    trace_ids = {
        span.get("trace_id")
        for span in spans
        if (span.get("attributes") or {}).get(_ATTR_CONVERSATION_ID)
        == conversation_id
    }
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "inference_time": 0.0,
    }
    tool_calls: list[dict[str, Any]] = []
    if not trace_ids:
        return usage, tool_calls, False

    for span in spans:
        if span.get("trace_id") not in trace_ids:
            continue
        attributes = span.get("attributes") or {}
        prompt = attributes.get(_ATTR_TOKEN_PROMPT)
        completion = attributes.get(_ATTR_TOKEN_COMPLETION)
        if (
            attributes.get(_ATTR_SPAN_KIND) == "LLM"
            or prompt is not None
            or completion is not None
        ):
            in_tok = int(prompt or 0)
            out_tok = int(completion or 0)
            usage["input_tokens"] += in_tok
            usage["output_tokens"] += out_tok
            usage["total_tokens"] += int(
                attributes.get(_ATTR_TOKEN_TOTAL) or (in_tok + out_tok)
            )
            start, end = span.get("start_time"), span.get("end_time")
            if isinstance(start, (int, float)) and isinstance(
                end, (int, float)
            ):
                usage["inference_time"] += max(0.0, (end - start) / 1e9)

        if (
            attributes.get(_ATTR_SPAN_KIND) == "TOOL"
            or _ATTR_TOOL_NAME in attributes
        ):
            tool_calls.append(_tool_call_from_span(span, attributes))

    return usage, tool_calls, True


def _extract_status_and_content(
    chunk: StreamResponse,
) -> tuple[str | None, str]:
    if chunk.HasField("message"):
        return "completed", get_message_text(chunk.message)
    if chunk.HasField("artifact_update"):
        return "completed", get_artifact_text(chunk.artifact_update.artifact)
    if chunk.HasField("status_update"):
        state = chunk.status_update.status.state
        status = _TASK_STATE_TO_STR.get(state)
        content = ""
        if chunk.status_update.status.HasField("message"):
            content = get_message_text(chunk.status_update.status.message)
        return status, content
    if chunk.HasField("task"):
        state = chunk.task.status.state
        status = _TASK_STATE_TO_STR.get(state)
        texts = [get_artifact_text(a) for a in chunk.task.artifacts]
        return status, "\n".join(t for t in texts if t)
    return None, ""


class BatA2AAdapter:
    def __init__(
        self,
        agent_url: str,
        request_timeout_s: float = 180.0,
        max_events: int = 200,
        spans_dir: str | None = None,
    ) -> None:
        self.agent_url = agent_url
        self.request_timeout_s = request_timeout_s
        self.max_events = max_events
        # Directory of JSON-Lines span files written by the agent(s)
        # (OTEL_TRACES_EXPORTER=file); usage and tool calls are reconstructed
        # from it per episode, grouped by trace_id (multi-agent aware).
        self.spans_dir = spans_dir

    async def _collect_from_spans(
        self, conversation_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Aggregate usage/tool-calls for an episode from the spans directory.

        Retries briefly: the agent writes spans synchronously, but the root
        span (which carries the conversation id) ends just after the response,
        so it may not be on disk the instant the stream closes.
        """
        assert self.spans_dir is not None
        usage: dict[str, Any] = {}
        tool_calls: list[dict[str, Any]] = []
        for _ in range(20):  # up to ~2s
            usage, tool_calls, found = _aggregate_from_spans(
                self.spans_dir, conversation_id
            )
            if found:
                return usage, tool_calls
            await asyncio.sleep(0.1)
        return usage, tool_calls

    async def run_task(
        self, task: TaskSpec, *, thread_id: str
    ) -> EpisodeResult:
        t0_perf = time.perf_counter()
        trace = EpisodeTrace()

        last_status: str | None = None
        last_content = ""

        async with AsyncClient(timeout=self.request_timeout_s) as httpx_client:
            resolver = A2ACardResolver(
                httpx_client=httpx_client, base_url=self.agent_url
            )
            agent_card = await resolver.get_agent_card()

            client = ClientFactory(
                ClientConfig(httpx_client=httpx_client, streaming=True)
            ).create(card=agent_card)

            try:
                for turn in task.turns:
                    turn_started = False
                    message = new_text_message(
                        text=turn,
                        context_id=thread_id,
                        role=Role.ROLE_USER,
                    )
                    stream = client.send_message(
                        SendMessageRequest(message=message)
                    )

                    async for chunk in stream:
                        status, content = _extract_status_and_content(chunk)
                        if status is None:
                            continue

                        if len(trace.events) < self.max_events:
                            trace.events.append(
                                TraceEvent(
                                    t_ms=(time.perf_counter() - t0_perf)
                                    * 1000.0,
                                    task_status=status,
                                    content_preview=content,
                                    user_input=turn
                                    if not turn_started
                                    else None,
                                )
                            )
                            turn_started = True

                        if status in TERMINAL_STATUSES:
                            last_status = status
                            last_content = content or ""

            except Exception as exc:
                last_status = "error"
                last_content = f"{type(exc).__name__}: {exc}"

        trace.timings["wall_ms"] = (time.perf_counter() - t0_perf) * 1000.0

        # Usage and tool calls now come from the agent's OpenTelemetry spans
        # (written to files), not from A2A message metadata.
        if self.spans_dir:
            usage, tool_calls = await self._collect_from_spans(thread_id)
            trace.usage = usage
            trace.tool_calls = tool_calls

        final_status = last_status or "error"
        final_output = last_content or ""

        return EpisodeResult(
            task_id=task.id,
            final_status=final_status,
            final_output=final_output,
            trace=trace,
            aux={"agent_url": self.agent_url},
        )
