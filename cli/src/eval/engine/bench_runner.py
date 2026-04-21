from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .contracts import EpisodeResult, TaskSpec


def _safe_task_id(task_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in task_id)


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class RunConfig:
    run_name: str
    out_dir: str = "output"
    k: int = 1
    model: str = "default"
    task_id: str = ""


class BenchRunner:
    def __init__(self, adapter: object, config: RunConfig):
        self.adapter = adapter
        self.config = config
        self.task_dir: Optional[Path] = None
        self.run_dir: Optional[Path] = None

    def _episodes_dir(self) -> Path:
        if self.run_dir is None:
            raise ValueError("run_dir is not initialized")
        return self.run_dir / "episodes"

    def persist_results(self, results: List[EpisodeResult]) -> None:
        episodes_dir = self._episodes_dir()
        episodes_dir.mkdir(parents=True, exist_ok=True)

        for episode in results:
            task_file_id = _safe_task_id(episode.task_id)
            attempt_index = int(episode.aux.get("attempt_index", 0))
            (episodes_dir / f"{task_file_id}__try{attempt_index}.json").write_text(
                episode.model_dump_json(indent=2),
                encoding="utf-8",
            )

    async def run(self, tasks: List[TaskSpec]) -> List[EpisodeResult]:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        task_id = self.config.task_id or stamp
        safe_model_name = self.config.model.replace(":", "-")

        self.task_dir = Path(self.config.out_dir) / task_id
        self.run_dir = self.task_dir / f"{self.config.run_name}_{safe_model_name}"
        self._episodes_dir().mkdir(parents=True, exist_ok=True)

        display_model_name = self.config.model.split(":")[-1] if ":" in self.config.model else self.config.model
        _write_json(
            self.run_dir / "run.json",
            {
                "run_name": self.config.run_name,
                "timestamp_utc": stamp,
                "k": self.config.k,
                "model_name": display_model_name,
            },
        )

        all_attempts: list[EpisodeResult] = []

        for task in tasks:
            task_attempts: list[EpisodeResult] = []

            for i in range(max(1, int(self.config.k))):
                thread_id = f"{task.id}__try{i}"
                episode = await self.adapter.run_task(task=task, thread_id=thread_id)
                episode.model_name = self.config.model
                episode.aux["attempt_index"] = i
                task_attempts.append(episode)
                all_attempts.append(episode)

        self.persist_results(all_attempts)

        per_task = []
        for task in tasks:
            task_attempts = [attempt for attempt in all_attempts if attempt.task_id == task.id]
            task_total = len(task_attempts)
            task_passed = sum(1 for attempt in task_attempts if attempt.success)
            per_task.append(
                {
                    "task_id": task.id,
                    "attempts": task_total,
                    "passed": task_passed,
                    "failed": task_total - task_passed,
                    "success_percentage": (task_passed / task_total) * 100.0 if task_total else 0.0,
                }
            )

        passed = sum(1 for result in all_attempts if result.success)
        failed = len(all_attempts) - passed
        avg_latency = (
            sum(result.trace.timings.get("wall_ms", 0.0) for result in all_attempts) / len(all_attempts)
            if all_attempts
            else 0.0
        )

        _write_json(
            self.run_dir / "summary.json",
            {
                "episodes": len(all_attempts),
                "passed": passed,
                "failed": failed,
                "pass_rate": (passed / len(all_attempts)) if all_attempts else 0.0,
                "avg_latency_ms": avg_latency,
                "k_attempts": self.config.k,
                "total_attempts": len(all_attempts),
                "per_task": per_task,
            },
        )

        return all_attempts
