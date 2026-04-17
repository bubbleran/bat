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
    save_attempts: bool = False
    model: str = "default"
    task_id: str = ""


class BenchRunner:
    def __init__(self, adapter: object, config: RunConfig):
        self.adapter = adapter
        self.config = config
        self.task_dir: Optional[Path] = None
        self.run_dir: Optional[Path] = None

    async def run(self, tasks: List[TaskSpec]) -> List[EpisodeResult]:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        task_id = self.config.task_id or stamp
        safe_model_name = self.config.model.replace(":", "-")

        self.task_dir = Path(self.config.out_dir) / task_id
        self.run_dir = self.task_dir / f"{self.config.run_name}_{safe_model_name}"
        episodes_dir = self.run_dir / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        display_model_name = self.config.model.split(":")[-1] if ":" in self.config.model else self.config.model
        _write_json(
            self.run_dir / "run.json",
            {
                "run_name": self.config.run_name,
                "timestamp_utc": stamp,
                "k": self.config.k,
                "save_attempts": self.config.save_attempts,
                "model_name": display_model_name,
            },
        )

        chosen_results: list[EpisodeResult] = []
        all_attempts: list[EpisodeResult] = []

        for task in tasks:
            task_file_id = _safe_task_id(task.id)
            task_attempts: list[EpisodeResult] = []

            for i in range(max(1, int(self.config.k))):
                thread_id = f"{task.id}__try{i}"
                episode = await self.adapter.run_task(task=task, thread_id=thread_id)
                episode.model_name = self.config.model
                task_attempts.append(episode)
                all_attempts.append(episode)

                if self.config.save_attempts:
                    (episodes_dir / f"{task_file_id}__try{i}.json").write_text(
                        episode.model_dump_json(indent=2),
                        encoding="utf-8",
                    )

            chosen = next((ep for ep in task_attempts if ep.success), task_attempts[0])
            chosen_results.append(chosen)

            if not self.config.save_attempts:
                (episodes_dir / f"{task_file_id}.json").write_text(
                    chosen.model_dump_json(indent=2),
                    encoding="utf-8",
                )

        passed = sum(1 for result in chosen_results if result.success)
        failed = len(chosen_results) - passed
        avg_latency = (
            sum(result.trace.timings.get("wall_ms", 0.0) for result in chosen_results) / len(chosen_results)
            if chosen_results
            else 0.0
        )

        _write_json(
            self.run_dir / "summary.json",
            {
                "episodes": len(chosen_results),
                "passed": passed,
                "failed": failed,
                "pass_rate": (passed / len(chosen_results)) if chosen_results else 0.0,
                "avg_latency_ms": avg_latency,
                "k_attempts": self.config.k,
                "total_attempts": len(all_attempts),
            },
        )

        return all_attempts if self.config.k > 1 else chosen_results
