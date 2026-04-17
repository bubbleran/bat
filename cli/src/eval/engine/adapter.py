from __future__ import annotations

import time
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig

from .contracts import EpisodeResult, EpisodeTrace, TaskSpec, TraceEvent


def _usage_to_dict(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    return {"raw": str(usage)}


def _merge_runnable_config(base: RunnableConfig, thread_id: str) -> RunnableConfig:
    base = base or {}
    cfg = dict(base)
    cfg_conf = dict(cfg.get("configurable") or {})
    cfg_conf["thread_id"] = thread_id
    cfg["configurable"] = cfg_conf
    return cfg


class BatAgentGraphAdapter:
    def __init__(
        self,
        agent_graph: Any,
        base_runnable_config: Optional[RunnableConfig] = None,
        max_events: int = 200,
    ):
        self.agent = agent_graph
        self.base_cfg: RunnableConfig = base_runnable_config or {}
        self.max_events = max_events

    async def run_task(self, task: TaskSpec, *, thread_id: str) -> EpisodeResult:
        t0_perf = time.perf_counter()
        t0_epoch = time.time()

        trace = EpisodeTrace()
        last_terminal_status: Optional[str] = None
        last_terminal_content: str = ""

        cfg = _merge_runnable_config(self.base_cfg, thread_id=thread_id)

        try:
            for turn in task.turns:
                turn_started = False
                async for item in self.agent.astream(query=turn, config=cfg):
                    if len(trace.events) < self.max_events:
                        trace.events.append(
                            TraceEvent(
                                t_ms=(time.perf_counter() - t0_perf) * 1000.0,
                                task_status=item.task_status,
                                content_preview=(item.content or ""),
                                user_input=turn if not turn_started else None,
                            )
                        )
                        turn_started = True

                    if item.task_status in ("completed", "error", "input-required"):
                        last_terminal_status = item.task_status
                        last_terminal_content = item.content or ""

        except Exception as exc:
            last_terminal_status = "error"
            last_terminal_content = f"{type(exc).__name__}: {exc}"

        trace.timings["wall_ms"] = (time.perf_counter() - t0_perf) * 1000.0

        try:
            usage = self.agent._get_usage_metadata(from_timestamp=t0_epoch)
            trace.usage = _usage_to_dict(usage)
        except Exception:
            trace.usage = {}

        status = last_terminal_status or "error"
        output_text = last_terminal_content or ""

        success = True
        if task.expected.must_succeed:
            success = status == "completed"
            if success and task.expected.final_contains:
                success = task.expected.final_contains in output_text

        return EpisodeResult(
            task_id=task.id,
            status=status,
            output_text=output_text,
            success=success,
            trace=trace,
            aux={},
        )
