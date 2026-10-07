from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from bat.logging import create_logger

from .adapter import run_task
from .contracts import EpisodeResult, EpisodeVerdict, JudgeSpec, TaskSpec
from .evaluator import verdict
from .judge import score
from .metrics import metrics, summary

logger = create_logger(__name__, level="info")


def load_tasks(path: Path) -> list[TaskSpec]:
    try:
        return [
            TaskSpec.model_validate(task)
            for task in json.loads(path.read_text(encoding="utf-8"))
        ]
    except Exception as exc:
        raise ValueError(f"Dataset not formatted correctly in {path}") from exc


async def _attempt(
    agent_url: str, task: TaskSpec, attempt: int, span_paths: list[str]
) -> EpisodeResult:
    try:
        episode = await run_task(
            agent_url, task, f"{task.id}__try{attempt}", span_paths
        )
        episode.verdict = verdict(episode, task.expected)
    except Exception as exc:
        # One bad attempt must not abort the run.
        logger.error("Task '%s' attempt %d failed: %s", task.id, attempt, exc)
        episode = EpisodeResult(
            task_id=task.id,
            final_status="error",
            final_output=f"<eval error: {exc}>",
            verdict=EpisodeVerdict(passed=False, reason=f"eval error: {exc}"),
            aux={"error": str(exc)},
        )
    episode.expected_outcome = task.expected.expected_outcome
    episode.aux["attempt_index"] = attempt
    return episode


def _write(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_episodes(run_dir: Path, episodes: list[EpisodeResult]) -> None:
    for episode in episodes:
        name = re.sub(r"[^\w.-]", "_", episode.task_id)
        attempt = episode.aux["attempt_index"]
        (run_dir / "episodes" / f"{name}__try{attempt}.json").write_text(
            episode.model_dump_json(indent=2), encoding="utf-8"
        )


async def run_evaluation(
    *,
    agent_url: str,
    model: str,
    dataset: Path,
    out_dir: Path,
    run_name: str,
    k: int,
    span_paths: list[str],
    judge: JudgeSpec | None,
) -> list[EpisodeResult]:
    """Run every task k times against the agent, score the episodes, and
    write them, summary.json and metrics.json under out_dir."""
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    tasks = load_tasks(dataset)
    episodes: list[EpisodeResult] = []
    for task in tasks:
        for attempt in range(k):
            episode = await _attempt(agent_url, task, attempt, span_paths)
            episode.model_name = model
            episodes.append(episode)

    run_dir = out_dir / f"{run_name}_{model.replace(':', '-')}"
    (run_dir / "episodes").mkdir(parents=True, exist_ok=True)
    # Written before judging too: a slow judge must not cost the episodes.
    _write_episodes(run_dir, episodes)
    if judge is not None:
        score(judge, episodes, {task.id: task for task in tasks})
        _write_episodes(run_dir, episodes)
    _write(
        run_dir / "summary.json", summary(episodes, run_name, model, k, stamp)
    )
    _write(run_dir / "metrics.json", metrics(episodes, k))
    logger.info(f"Artifacts written to: {run_dir}")
    return episodes
