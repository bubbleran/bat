from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path
from typing import Any

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import get_artifact_text, get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest, StreamResponse, TaskState
from bat.logging import create_logger
from httpx import AsyncClient

from .contracts import (
    EpisodeResult,
    EpisodeTrace,
    TaskSpec,
    TraceEvent,
    Trajectory,
)
from .trajectory import (
    build_trajectory,
    count_conversation_roots,
    tool_calls_from,
)

logger = create_logger(__name__, level="info")

_STATUS = {
    TaskState.TASK_STATE_SUBMITTED: "working",
    TaskState.TASK_STATE_WORKING: "working",
    TaskState.TASK_STATE_INPUT_REQUIRED: "input-required",
    TaskState.TASK_STATE_COMPLETED: "completed",
    TaskState.TASK_STATE_FAILED: "error",
    TaskState.TASK_STATE_CANCELED: "error",
    TaskState.TASK_STATE_REJECTED: "error",
}
_FINAL = {"completed", "error", "input-required"}
_MAX_EVENTS = 200


def read_spans(paths: list[str]) -> list[dict[str, Any]]:
    """The spans in these files, and in the ``*.jsonl`` of these directories."""
    spans: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
        for file in files:
            if not file.is_file():
                continue
            for line in file.read_text(encoding="utf-8").splitlines():
                with contextlib.suppress(json.JSONDecodeError):
                    spans.append(json.loads(line))
    return spans


def _status_and_text(chunk: StreamResponse) -> tuple[str | None, str]:
    if chunk.HasField("message"):
        return "completed", get_message_text(chunk.message)
    if chunk.HasField("artifact_update"):
        return "completed", get_artifact_text(chunk.artifact_update.artifact)
    if chunk.HasField("status_update"):
        status = chunk.status_update.status
        text = (
            get_message_text(status.message)
            if status.HasField("message")
            else ""
        )
        return _STATUS.get(status.state), text
    if chunk.HasField("task"):
        texts = [
            get_artifact_text(artifact) for artifact in chunk.task.artifacts
        ]
        return _STATUS.get(chunk.task.status.state), "\n".join(
            t for t in texts if t
        )
    return None, ""


async def _trajectory(
    paths: list[str], conversation_id: str, turns: list[str]
) -> Trajectory:
    """The conversation's trajectory, once each turn's root span is on disk:
    the last one ends just after the answer is streamed."""
    spans = read_spans(paths)
    for _ in range(20):
        if not paths or count_conversation_roots(spans, conversation_id) >= len(
            turns
        ):
            break
        await asyncio.sleep(0.1)
        spans = read_spans(paths)
    trajectory = build_trajectory(spans, conversation_id, turns)
    if paths and not trajectory.found:
        logger.warning(
            "No spans found for conversation '%s' in %s: is telemetry on?",
            conversation_id,
            ", ".join(paths),
        )
    return trajectory


async def run_task(
    agent_url: str, task: TaskSpec, thread_id: str, span_paths: list[str]
) -> EpisodeResult:
    """One conversation with the agent, and what its spans say it did."""
    started = time.perf_counter()
    trace = EpisodeTrace()
    status, output = "error", ""
    async with AsyncClient(timeout=180.0) as http:
        card = await A2ACardResolver(
            httpx_client=http, base_url=agent_url
        ).get_agent_card()
        client = ClientFactory(
            ClientConfig(httpx_client=http, streaming=True)
        ).create(card=card)
        try:
            for turn in task.turns:
                output = ""  # the final output is the last turn's answer
                user_input: str | None = turn
                message = new_text_message(
                    text=turn, context_id=thread_id, role=Role.ROLE_USER
                )
                async for chunk in client.send_message(
                    SendMessageRequest(message=message)
                ):
                    chunk_status, text = _status_and_text(chunk)
                    if chunk_status is None:
                        continue
                    if len(trace.events) < _MAX_EVENTS:
                        trace.events.append(
                            TraceEvent(
                                t_ms=(time.perf_counter() - started) * 1000,
                                task_status=chunk_status,
                                content_preview=text,
                                user_input=user_input,
                            )
                        )
                        user_input = None
                    if chunk_status in _FINAL:
                        status = chunk_status
                        # The stream closes on a status with no text.
                        output = text or output
        except Exception as exc:
            status, output = "error", f"{type(exc).__name__}: {exc}"

    trace.wall_ms = (time.perf_counter() - started) * 1000
    trace.trajectory = await _trajectory(span_paths, thread_id, task.turns)
    trace.tool_calls = tool_calls_from(trace.trajectory)
    return EpisodeResult(
        task_id=task.id,
        final_status=status,
        final_output=output,
        trace=trace,
        aux={"agent_url": agent_url},
    )
