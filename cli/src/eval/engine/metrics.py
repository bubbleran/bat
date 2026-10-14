from __future__ import annotations

from typing import Any

from .contracts import EpisodeResult
from .judge import RUBRICS


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 1) if whole else 0.0


def _passed(episode: EpisodeResult) -> bool:
    return episode.verdict is not None and episode.verdict.passed


def _episode(episode: EpisodeResult) -> dict[str, Any]:
    totals = episode.trace.trajectory.totals
    metrics: dict[str, Any] = {
        "task_id": episode.task_id,
        "expected_outcome": episode.expected_outcome,
        "status": episode.final_status,
        "success": _passed(episode),
        "time": {"wall_ms": episode.trace.wall_ms},
        "tokens": {
            "prompt_tokens": totals.tokens_in,
            "completion_tokens": totals.tokens_out,
            "total_tokens": totals.tokens_in + totals.tokens_out,
            "cached_prompt_pct": _pct(totals.tokens_cached, totals.tokens_in),
        },
    }
    if episode.verdict:
        metrics["verdict"] = episode.verdict.model_dump()
    if episode.qualitative_scores:
        metrics["qualitative"] = episode.qualitative_scores.model_dump()
    return metrics


def _scores(episodes: list[EpisodeResult]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for episode in episodes:
        for _, field in RUBRICS.values():
            value = getattr(episode.qualitative_scores, field, None)
            if value is not None:
                values.setdefault(field, []).append(value)
    return values


def metrics(episodes: list[EpisodeResult], k: int) -> dict[str, Any]:
    """metrics.json: every episode, and the run's totals."""
    per_episode = [_episode(episode) for episode in episodes]
    n = len(per_episode)
    times = [m["time"]["wall_ms"] for m in per_episode]
    totals = [m["tokens"]["total_tokens"] for m in per_episode]
    prompt = sum(m["tokens"]["prompt_tokens"] for m in per_episode)
    cached = sum(e.trace.trajectory.totals.tokens_cached for e in episodes)
    passed = sum(1 for m in per_episode if m["success"])
    summary: dict[str, Any] = {
        "episodes": n,
        "k_attempts": k,
        "total_runs": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": passed / n if n else 0.0,
        "time": {
            "total_wall_ms": sum(times),
            "avg_wall_ms": sum(times) / n if n else 0.0,
            "min_wall_ms": min(times, default=0.0),
            "max_wall_ms": max(times, default=0.0),
        },
        "tokens": {
            "prompt_tokens_total": prompt,
            "completion_tokens_total": sum(
                m["tokens"]["completion_tokens"] for m in per_episode
            ),
            "total_tokens_total": sum(totals),
            "cached_prompt_pct": _pct(cached, prompt),
            "avg_total_tokens": sum(totals) / n if n else 0.0,
            "min_total_tokens": min(totals, default=0),
            "max_total_tokens": max(totals, default=0),
        },
    }
    scores = _scores(episodes)
    if scores:
        summary["qualitative"] = {
            field: {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
            for field, values in scores.items()
        }
    return {"per_episode": per_episode, "summary": summary}


def summary(
    episodes: list[EpisodeResult], run_name: str, model: str, k: int, stamp: str
) -> dict[str, Any]:
    """summary.json: pass counts per task, and the averaged judge scores."""
    by_task: dict[str, list[EpisodeResult]] = {}
    for episode in episodes:
        by_task.setdefault(episode.task_id, []).append(episode)
    attempts = []
    for task_id, tries in by_task.items():
        passed = sum(1 for episode in tries if _passed(episode))
        attempts.append(
            {
                "task_id": task_id,
                "attempts": len(tries),
                "passed": passed,
                "failed": len(tries) - passed,
                "success_percentage": 100.0 * passed / len(tries),
            }
        )
    return {
        "run_name": run_name,
        "timestamp_utc": stamp,
        "k": k,
        "model_name": model.split(":", 1)[-1],
        "attempts": attempts,
        "qualitative_scores": {
            field: sum(values) / len(values)
            for field, values in _scores(episodes).items()
        },
        "passed": sum(1 for episode in episodes if _passed(episode)),
    }
