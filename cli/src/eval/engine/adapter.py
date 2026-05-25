from __future__ import annotations

import json
import time
import uuid
from typing import Any

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import (
    get_artifact_text,
    get_message_text,
    new_text_message,
)
from a2a.types import Role, StreamResponse, TaskState
from google.protobuf.json_format import MessageToDict
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


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(by_alias=True)
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _struct_to_dict(metadata_struct: Any) -> dict[str, Any]:
    if metadata_struct is None:
        return {}
    try:
        return MessageToDict(metadata_struct) or {}
    except Exception:
        return {}


def _chunk_key(chunk: StreamResponse) -> str:
    if chunk.HasField("status_update"):
        message_id = chunk.status_update.status.message.message_id
        if message_id:
            return f"status:{message_id}"
    if chunk.HasField("artifact_update"):
        artifact_id = chunk.artifact_update.artifact.artifact_id
        if artifact_id:
            return f"artifact:{artifact_id}"
    if chunk.HasField("message"):
        if chunk.message.message_id:
            return f"message:{chunk.message.message_id}"
    if chunk.HasField("task"):
        if chunk.task.id:
            return f"task:{chunk.task.id}"
    return f"raw:{chunk.SerializeToString().hex()}"


def _extract_status_and_content(chunk: StreamResponse) -> tuple[str | None, str]:
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


def _extract_metadata(chunk: StreamResponse) -> dict[str, Any]:
    metadata_struct: Any = None
    if chunk.HasField("status_update"):
        metadata_struct = chunk.status_update.metadata
    elif chunk.HasField("artifact_update"):
        metadata_struct = chunk.artifact_update.metadata
    elif chunk.HasField("message"):
        metadata_struct = chunk.message.metadata
    elif chunk.HasField("task"):
        metadata_struct = chunk.task.metadata

    metadata = _struct_to_dict(metadata_struct)

    if chunk.HasField("artifact_update"):
        artifact_metadata = _struct_to_dict(chunk.artifact_update.artifact.metadata)
        if artifact_metadata:
            merged = dict(artifact_metadata)
            merged.update(metadata)
            metadata = merged

    return metadata


def _normalize_usage(metadata: dict[str, Any]) -> dict[str, Any]:
    usage = _to_dict(metadata.get("usage"))
    if not usage:
        return {}

    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
    inference_time = float(usage.get("inference_time") or 0.0)

    if input_tokens == 0 and output_tokens == 0 and total_tokens == 0 and inference_time == 0.0:
        return {}

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "inference_time": inference_time,
    }


def _add_usage(total: dict[str, Any], incremental: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_tokens": int(total.get("input_tokens") or 0) + int(incremental.get("input_tokens") or 0),
        "output_tokens": int(total.get("output_tokens") or 0) + int(incremental.get("output_tokens") or 0),
        "total_tokens": int(total.get("total_tokens") or 0) + int(incremental.get("total_tokens") or 0),
        "inference_time": float(total.get("inference_time") or 0.0)
        + float(incremental.get("inference_time") or 0.0),
    }


def _extract_tool_calls(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    trace = _to_dict(metadata.get("trace"))
    tool_calls = trace.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [item for item in tool_calls if isinstance(item, dict)]


def _tool_call_key(tool_call: dict[str, Any]) -> str:
    call_id = tool_call.get("id")
    if isinstance(call_id, str) and call_id:
        return f"id:{call_id}"
    return json.dumps(tool_call, sort_keys=True, ensure_ascii=True, default=str)


class BatA2AAdapter:
    def __init__(self, agent_url: str, request_timeout_s: float = 180.0, max_events: int = 200) -> None:
        self.agent_url = agent_url
        self.request_timeout_s = request_timeout_s
        self.max_events = max_events

    async def run_task(self, task: TaskSpec, *, thread_id: str) -> EpisodeResult:
        t0_perf = time.perf_counter()
        trace = EpisodeTrace()

        usage_total: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "inference_time": 0.0,
        }
        usage_seen: set[str] = set()
        tool_calls_seen: set[str] = set()

        last_status: str | None = None
        last_content = ""

        async with AsyncClient(timeout=self.request_timeout_s) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.agent_url)
            agent_card = await resolver.get_agent_card()

            client = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=True)).create(card=agent_card)

            try:
                for turn in task.turns:
                    turn_started = False
                    message = new_text_message(
                        text=turn,
                        context_id=thread_id,
                        task_id=str(uuid.uuid4()),
                        role=Role.ROLE_USER,
                    )
                    stream = client.send_message(message)

                    async for chunk in stream:
                        metadata = _extract_metadata(chunk)

                        usage = _normalize_usage(metadata)
                        if usage:
                            usage_key = f"{_chunk_key(chunk)}::{json.dumps(usage, sort_keys=True, ensure_ascii=True)}"
                            if usage_key not in usage_seen:
                                usage_seen.add(usage_key)
                                usage_total = _add_usage(usage_total, usage)

                        for tool_call in _extract_tool_calls(metadata):
                            key = _tool_call_key(tool_call)
                            if key in tool_calls_seen:
                                continue
                            tool_calls_seen.add(key)
                            trace.tool_calls.append(tool_call)

                        status, content = _extract_status_and_content(chunk)
                        if status is None:
                            continue

                        if len(trace.events) < self.max_events:
                            trace.events.append(
                                TraceEvent(
                                    t_ms=(time.perf_counter() - t0_perf) * 1000.0,
                                    task_status=status,
                                    content_preview=content,
                                    user_input=turn if not turn_started else None,
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
        trace.usage = usage_total

        final_status = last_status or "error"
        final_output = last_content or ""

        return EpisodeResult(
            task_id=task.id,
            final_status=final_status,
            final_output=final_output,
            trace=trace,
            aux={"agent_url": self.agent_url},
        )
