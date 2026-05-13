from __future__ import annotations

from typing import Any

from ..contracts import EpisodeResult


def _extract_wall_ms(ep: EpisodeResult) -> float:
    return float(ep.trace.timings.get("wall_ms", 0.0))


def _extract_tokens_from_usage(usage: dict[str, Any]) -> tuple[int, int, int]:
    if not isinstance(usage, dict) or not usage:
        return 0, 0, 0

    prompt = usage.get("input_tokens")
    completion = usage.get("output_tokens")
    total = usage.get("total_tokens")

    if prompt is not None or completion is not None or total is not None:
        p = int(prompt or 0)
        c = int(completion or 0)
        t = int(total) if total is not None else p + c
        return p, c, t

    by_model = usage.get("by_model") or usage.get("models")
    if isinstance(by_model, dict):
        prompt_sum = 0
        completion_sum = 0
        total_sum = 0
        for _, model_usage in by_model.items():
            if isinstance(model_usage, dict):
                p, c, t = _extract_tokens_from_usage(model_usage)
                prompt_sum += p
                completion_sum += c
                total_sum += t
        return prompt_sum, completion_sum, total_sum

    return 0, 0, 0


def episode_metrics(ep: EpisodeResult) -> dict[str, Any]:
    wall_ms = _extract_wall_ms(ep)
    prompt, completion, total = _extract_tokens_from_usage(ep.trace.usage)

    metrics: dict[str, Any] = {
        "task_id": ep.task_id,
        "expected_outcome": ep.expected_outcome,
        "status": ep.status,
        "success": ep.success,
        "time": {"wall_ms": wall_ms},
        "tokens": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        },
    }

    if ep.verdict:
        metrics["verdict"] = {
            "passed": ep.verdict.passed,
            "checks": {
                k: {"passed": v.passed, "reason": v.reason}
                for k, v in ep.verdict.checks.items()
            },
        }

    if ep.qualitative_scores:
        metrics["qualitative"] = {
            "response_relevance": ep.qualitative_scores.response_relevance,
            "task_completion_quality": ep.qualitative_scores.task_completion_quality,
            "hallucination_score": ep.qualitative_scores.hallucination_score,
            "tool_call_appropriateness": ep.qualitative_scores.tool_call_appropriateness,
        }

    return metrics


def summarize_episode_metrics(results: list[EpisodeResult], k: int = 1) -> dict[str, Any]:
    per_episode = [episode_metrics(result) for result in results]
    n = len(per_episode)

    wall_times = [metric["time"]["wall_ms"] for metric in per_episode]
    total_wall_ms = sum(wall_times)
    avg_wall_ms = (total_wall_ms / n) if n else 0.0

    prompt_tokens = [metric["tokens"]["prompt_tokens"] for metric in per_episode]
    completion_tokens = [metric["tokens"]["completion_tokens"] for metric in per_episode]
    total_tokens = [metric["tokens"]["total_tokens"] for metric in per_episode]

    passed = sum(1 for metric in per_episode if metric["success"])
    failed = n - passed

    summary: dict[str, Any] = {
        "episodes": n,
        "k_attempts": k,
        "total_runs": n,
        "passed": passed,
        "failed": failed,
        "pass_rate": (passed / n) if n else 0.0,
        "time": {
            "total_wall_ms": total_wall_ms,
            "avg_wall_ms": avg_wall_ms,
            "min_wall_ms": min(wall_times) if wall_times else 0.0,
            "max_wall_ms": max(wall_times) if wall_times else 0.0,
        },
        "tokens": {
            "prompt_tokens_total": sum(prompt_tokens),
            "completion_tokens_total": sum(completion_tokens),
            "total_tokens_total": sum(total_tokens),
            "avg_total_tokens": (sum(total_tokens) / n) if n else 0.0,
            "min_total_tokens": min(total_tokens) if total_tokens else 0.0,
            "max_total_tokens": max(total_tokens) if total_tokens else 0.0,
        },
    }

    qualitative_metrics = [metric["qualitative"] for metric in per_episode if metric.get("qualitative")]
    if qualitative_metrics:
        qualitative_summary: dict[str, Any] = {}
        for field in [
            "response_relevance",
            "task_completion_quality",
            "hallucination_score",
            "tool_call_appropriateness",
        ]:
            values = [metric[field] for metric in qualitative_metrics if metric.get(field) is not None]
            if values:
                qualitative_summary[field] = {
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }
        if qualitative_summary:
            summary["qualitative"] = qualitative_summary

    return {"per_episode": per_episode, "summary": summary}
