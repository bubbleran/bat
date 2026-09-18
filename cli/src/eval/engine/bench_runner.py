from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from bat.logging import create_logger

from .contracts import EpisodeResult, EpisodeVerdict, TaskSpec
from .evaluator import EpisodeEvaluator

logger = create_logger(__name__, level="info")

_QUALITATIVE_FIELDS = (
    "response_relevance",
    "task_completion_quality",
    "hallucination_score",
    "tool_call_appropriateness",
)


def _episode_passed(ep: EpisodeResult) -> bool:
    return ep.verdict.passed if ep.verdict is not None else False


def _average_qualitative_scores(
    results: list[EpisodeResult],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for field in _QUALITATIVE_FIELDS:
        values = [
            getattr(r.qualitative_scores, field)
            for r in results
            if r.qualitative_scores is not None
            and getattr(r.qualitative_scores, field) is not None
        ]
        if values:
            out[field] = sum(values) / len(values)
    return out


def _safe_task_id(task_id: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in task_id
    )


def _write_json(path: Path, obj: object) -> None:
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8"
    )


@dataclass
class RunConfig:
    run_name: str
    out_dir: str = "output"
    k: int = 1
    model: str = "default"
    task_id: str = ""


class BenchRunner:
    def __init__(
        self,
        adapter: object,
        config: RunConfig,
        evaluator: EpisodeEvaluator | None = None,
    ):
        self.adapter = adapter
        self.config = config
        self.evaluator = evaluator or EpisodeEvaluator()
        self.task_dir: Path | None = None
        self.run_dir: Path | None = None

    def _episodes_dir(self) -> Path:
        if self.run_dir is None:
            raise ValueError("run_dir is not initialized")
        return self.run_dir / "episodes"

    def persist_results(self, results: list[EpisodeResult]) -> None:
        episodes_dir = self._episodes_dir()
        episodes_dir.mkdir(parents=True, exist_ok=True)

        for episode in results:
            task_file_id = _safe_task_id(episode.task_id)
            attempt_index = int(episode.aux.get("attempt_index", 0))
            (
                episodes_dir / f"{task_file_id}__try{attempt_index}.json"
            ).write_text(
                episode.model_dump_json(indent=2),
                encoding="utf-8",
            )

    async def run(self, tasks: list[TaskSpec]) -> list[EpisodeResult]:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        task_id = self.config.task_id or stamp
        safe_model_name = self.config.model.replace(":", "-")

        self.task_dir = Path(self.config.out_dir) / task_id
        self.run_dir = (
            self.task_dir / f"{self.config.run_name}_{safe_model_name}"
        )
        self._episodes_dir().mkdir(parents=True, exist_ok=True)
        self._run_timestamp = stamp

        all_attempts: list[EpisodeResult] = []

        for task in tasks:
            for i in range(max(1, int(self.config.k))):
                thread_id = f"{task.id}__try{i}"
                try:
                    episode = await self.adapter.run_task(
                        task=task, thread_id=thread_id
                    )
                    episode.verdict = self.evaluator.evaluate(
                        episode.final_status,
                        episode.final_output,
                        episode.trace.tool_calls,
                        task.expected,
                    )
                    episode.expected_outcome = task.expected.expected_outcome
                    episode.model_name = self.config.model
                    episode.aux["attempt_index"] = i
                except Exception as exc:
                    # One bad attempt must not abort the whole run: record it as
                    # a failed episode so the summary still accounts for it.
                    logger.error(
                        "Task '%s' attempt %d failed: %s", task.id, i, exc
                    )
                    episode = EpisodeResult(
                        model_name=self.config.model,
                        task_id=task.id,
                        expected_outcome=task.expected.expected_outcome,
                        final_status="error",
                        final_output=f"<eval error: {exc}>",
                        verdict=EpisodeVerdict(
                            passed=False, reason=f"eval error: {exc}"
                        ),
                        aux={"attempt_index": i, "error": str(exc)},
                    )
                all_attempts.append(episode)

        self.persist_results(all_attempts)
        return all_attempts

    def write_summary(self, results: list[EpisodeResult]) -> None:
        if self.run_dir is None:
            raise ValueError("run_dir is not initialized; call run() first")

        display_model_name = (
            self.config.model.split(":")[-1]
            if ":" in self.config.model
            else self.config.model
        )

        attempts_by_task: dict[str, list[EpisodeResult]] = {}
        for attempt in results:
            attempts_by_task.setdefault(attempt.task_id, []).append(attempt)

        attempts = []
        for task_id, task_attempts in attempts_by_task.items():
            total = len(task_attempts)
            passed = sum(1 for ep in task_attempts if _episode_passed(ep))
            attempts.append(
                {
                    "task_id": task_id,
                    "attempts": total,
                    "passed": passed,
                    "failed": total - passed,
                    "success_percentage": (passed / total) * 100.0
                    if total
                    else 0.0,
                }
            )

        _write_json(
            self.run_dir / "summary.json",
            {
                "run_name": self.config.run_name,
                "timestamp_utc": getattr(self, "_run_timestamp", ""),
                "k": self.config.k,
                "model_name": display_model_name,
                "attempts": attempts,
                "qualitative_scores": _average_qualitative_scores(results),
                "passed": sum(1 for ep in results if _episode_passed(ep)),
            },
        )
