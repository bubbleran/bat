from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Optional

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, TextPart
from httpx import AsyncClient

from .contracts import EpisodeResult, EpisodeTrace, TaskSpec, TraceEvent


TERMINAL_STATUSES = {"completed", "error", "input-required"}


def _to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(by_alias=True)
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _payload(item: Any) -> Dict[str, Any]:
    if isinstance(item, tuple) and len(item) == 2:
        return _to_dict(item[1])
    return _to_dict(item)


def _event_key(payload: Dict[str, Any]) -> str:
    kind = payload.get("kind")
    if kind == "status-update":
        status = _to_dict(payload.get("status"))
        message = _to_dict(status.get("message"))
        message_id = message.get("messageId")
        if isinstance(message_id, str) and message_id:
            return f"status:{message_id}"
    if kind == "artifact-update":
        artifact = _to_dict(payload.get("artifact"))
        artifact_id = artifact.get("artifactId")
        if isinstance(artifact_id, str) and artifact_id:
            return f"artifact:{artifact_id}"
    message_id = payload.get("messageId")
    if isinstance(message_id, str) and message_id:
        return f"message:{message_id}"
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)


def _extract_text_from_parts(parts: Any) -> str:
    if not isinstance(parts, list):
        return ""

    chunks: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue

        if part.get("kind") == "text" and isinstance(part.get("text"), str):
            chunks.append(part["text"])
            continue

        root = _to_dict(part.get("root"))
        if root.get("kind") == "text" and isinstance(root.get("text"), str):
            chunks.append(root["text"])

    return "\n\n".join(chunk for chunk in chunks if chunk)


def _extract_text(value: Any) -> str:
    data = _to_dict(value)
    if not data:
        return ""

    parts = data.get("parts")
    if isinstance(parts, list):
        return _extract_text_from_parts(parts)

    status = _to_dict(data.get("status"))
    if status:
        message = _to_dict(status.get("message"))
        text = _extract_text(message)
        if text:
            return text

    artifact = _to_dict(data.get("artifact"))
    if artifact:
        text = _extract_text(artifact)
        if text:
            return text

    message = _to_dict(data.get("message"))
    if message:
        text = _extract_text(message)
        if text:
            return text

    return ""


def _normalize_status(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"submitted", "working", "in-progress", "inprogress"}:
        return "working"
    if normalized in {"input-required", "inputrequired"}:
        return "input-required"
    if normalized in {"completed", "complete", "done"}:
        return "completed"
    if normalized in {"failed", "error"}:
        return "error"
    return None


def _extract_status_and_content(item: Any) -> tuple[Optional[str], str]:
    data = _payload(item)
    if not data:
        return None, ""

    kind = data.get("kind")
    if kind == "artifact-update":
        return "completed", _extract_text(_to_dict(data.get("artifact")))
    if kind == "message":
        return "completed", _extract_text(data)

    if kind in {"status-update", "task"}:
        status = _to_dict(data.get("status"))
        return _normalize_status(status.get("state")), _extract_text(_to_dict(status.get("message")))

    status = _to_dict(data.get("status"))
    if status:
        return _normalize_status(status.get("state")), _extract_text(_to_dict(status.get("message")))

    return None, ""


def _extract_metadata(item: Any) -> Dict[str, Any]:
    data = _payload(item)
    metadata = _to_dict(data.get("metadata"))

    artifact = _to_dict(data.get("artifact"))
    artifact_metadata = _to_dict(artifact.get("metadata"))
    if artifact_metadata:
        merged = dict(artifact_metadata)
        merged.update(metadata)
        metadata = merged

    return metadata


def _normalize_usage(metadata: Dict[str, Any]) -> Dict[str, Any]:
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


def _add_usage(total: Dict[str, Any], incremental: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input_tokens": int(total.get("input_tokens") or 0) + int(incremental.get("input_tokens") or 0),
        "output_tokens": int(total.get("output_tokens") or 0) + int(incremental.get("output_tokens") or 0),
        "total_tokens": int(total.get("total_tokens") or 0) + int(incremental.get("total_tokens") or 0),
        "inference_time": float(total.get("inference_time") or 0.0)
        + float(incremental.get("inference_time") or 0.0),
    }


def _extract_tool_calls(metadata: Dict[str, Any]) -> list[Dict[str, Any]]:
    trace = _to_dict(metadata.get("trace"))
    tool_calls = trace.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    return [item for item in tool_calls if isinstance(item, dict)]


def _tool_call_key(tool_call: Dict[str, Any]) -> str:
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

        usage_total: Dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "inference_time": 0.0,
        }
        usage_seen: set[str] = set()
        tool_calls_seen: set[str] = set()

        last_status: Optional[str] = None
        last_content = ""

        async with AsyncClient(timeout=self.request_timeout_s) as httpx_client:
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.agent_url)
            agent_card = await resolver.get_agent_card()

            client = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=True)).create(card=agent_card)

            try:
                for turn in task.turns:
                    turn_started = False
                    stream = client.send_message(
                        request=Message(
                            context_id=thread_id,
                            message_id=str(uuid.uuid4()),
                            role="user",
                            parts=[TextPart(text=turn)],
                        )
                    )

                    async for item in stream:
                        payload = _payload(item)
                        metadata = _extract_metadata(item)

                        usage = _normalize_usage(metadata)
                        if usage:
                            usage_key = f"{_event_key(payload)}::{json.dumps(usage, sort_keys=True, ensure_ascii=True)}"
                            if usage_key not in usage_seen:
                                usage_seen.add(usage_key)
                                usage_total = _add_usage(usage_total, usage)

                        for tool_call in _extract_tool_calls(metadata):
                            key = _tool_call_key(tool_call)
                            if key in tool_calls_seen:
                                continue
                            tool_calls_seen.add(key)
                            trace.tool_calls.append(tool_call)

                        status, content = _extract_status_and_content(item)
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

        status = last_status or "error"
        output_text = last_content or ""

        return EpisodeResult(
            task_id=task.id,
            status=status,
            output_text=output_text,
            trace=trace,
            aux={"agent_url": self.agent_url},
        )
