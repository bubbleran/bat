from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _extract_base_task_id(task_id: str) -> str:
    return re.sub(r"__try\d+$", "", task_id)


def _group_episodes_by_task(per_episode: List[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {}
    for episode in per_episode:
        base_id = _extract_base_task_id(episode["task_id"])
        grouped.setdefault(base_id, []).append(episode)
    return grouped


def _average_episodes(episodes: List[dict]) -> dict:
    if len(episodes) == 1:
        return episodes[0]

    n = len(episodes)
    avg_time = sum(ep["time"]["wall_ms"] for ep in episodes) / n
    avg_prompt = sum(ep["tokens"]["prompt_tokens"] for ep in episodes) / n
    avg_completion = (
        sum(ep["tokens"]["completion_tokens"] for ep in episodes) / n
    )
    avg_total = sum(ep["tokens"]["total_tokens"] for ep in episodes) / n
    success_rate = sum(ep["success"] for ep in episodes) / n

    result = {
        "task_id": _extract_base_task_id(episodes[0]["task_id"]),
        "status": episodes[0]["status"],
        "success": success_rate >= 0.5,
        "success_rate": success_rate,
        "attempts": n,
        "time": {"wall_ms": avg_time},
        "tokens": {
            "prompt_tokens": avg_prompt,
            "completion_tokens": avg_completion,
            "total_tokens": avg_total,
        },
    }

    qual_episodes = [ep for ep in episodes if "qualitative" in ep]
    if qual_episodes:
        relevance_vals = [
            ep["qualitative"]["response_relevance"]
            for ep in qual_episodes
            if ep["qualitative"].get("response_relevance") is not None
        ]
        completion_vals = [
            ep["qualitative"]["task_completion_quality"]
            for ep in qual_episodes
            if ep["qualitative"].get("task_completion_quality") is not None
        ]
        hallucination_vals = [
            ep["qualitative"]["hallucination_score"]
            for ep in qual_episodes
            if ep["qualitative"].get("hallucination_score") is not None
        ]

        result["qualitative"] = {
            "response_relevance": sum(relevance_vals) / len(relevance_vals)
            if relevance_vals
            else 0,
            "task_completion_quality": sum(completion_vals)
            / len(completion_vals)
            if completion_vals
            else 0,
            "hallucination_score": sum(hallucination_vals)
            / len(hallucination_vals)
            if hallucination_vals
            else 0,
        }

    return result


def _get_per_episode_averages(metrics_data: dict) -> List[dict]:
    per_episode = metrics_data.get("per_episode", [])
    if not per_episode:
        return []
    grouped = _group_episodes_by_task(per_episode)
    return [_average_episodes(episodes) for episodes in grouped.values()]


def _qual_score(episode: dict, key: str) -> float:
    """Read a qualitative score from an episode, coalescing missing/null to 0.

    An episode can carry an explicit ``None`` for a qualitative metric (a judge
    that failed or was not run), and with ``k == 1`` ``_average_episodes`` hands
    the raw episode through unchanged. ``dict.get(key, 0)`` does NOT substitute
    the default when the key is present with value ``None`` -- that ``None``
    would then flow into a matplotlib bar list and raise ``TypeError``, killing
    ``bat eval plot`` and leaking every open figure.
    """
    value = episode.get("qualitative", {}).get(key)
    return value if value is not None else 0


def _plot_comparison(metrics: Dict[str, dict]) -> List[Tuple[str, plt.Figure]]:
    """Build summary comparison charts across runs. Returns (name, figure) pairs."""
    if not metrics:
        return []

    run_names = list(metrics.keys())
    display_names = list(run_names)

    has_qualitative = any(
        "qualitative" in m.get("summary", {}) for m in metrics.values()
    )

    times, prompt_tokens, completion_tokens, total_tokens = [], [], [], []
    relevance_scores, completion_quality_scores, hallucination_scores = (
        [],
        [],
        [],
    )

    for name in run_names:
        summary = metrics[name].get("summary", {})
        times.append(summary.get("time", {}).get("total_wall_ms", 0) / 1000)
        tokens = summary.get("tokens", {})
        prompt_tokens.append(tokens.get("prompt_tokens_total", 0))
        completion_tokens.append(tokens.get("completion_tokens_total", 0))
        total_tokens.append(tokens.get("total_tokens_total", 0))
        qual = summary.get("qualitative", {})
        relevance_scores.append(
            qual.get("response_relevance", {}).get("avg", None)
        )
        completion_quality_scores.append(
            qual.get("task_completion_quality", {}).get("avg", None)
        )
        hallucination_scores.append(
            qual.get("hallucination_score", {}).get("avg", None)
        )

    figures: List[Tuple[str, plt.Figure]] = []

    # 1. Total execution time
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars1 = ax1.bar(range(len(run_names)), times, color="steelblue", alpha=0.7)
    ax1.set_xlabel("Run")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Total Execution Time", fontsize=14, fontweight="bold")
    ax1.set_xticks(range(len(run_names)))
    ax1.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, times, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig1.tight_layout()
    figures.append(("execution_time", fig1))

    # 2. Time vs tokens mirror chart
    fig2, ax2 = plt.subplots(figsize=(max(10, len(run_names) * 1.2), 7))
    x2 = range(len(run_names))
    max_time = max(times) if max(times) > 0 else 1
    max_tok = max(total_tokens) if max(total_tokens) > 0 else 1
    times_norm = [t / max_time for t in times]
    tokens_norm = [-t / max_tok for t in total_tokens]
    ax2.bar(
        x2, times_norm, color="steelblue", alpha=0.75, label="Execution Time"
    )
    ax2.bar(
        x2, tokens_norm, color="darkorange", alpha=0.75, label="Total Tokens"
    )
    ax2.axhline(0, color="black", linewidth=0.8)
    for xi, t_n, t_val, tok_n, tok_val in zip(
        x2, times_norm, times, tokens_norm, total_tokens, strict=False
    ):
        ax2.text(
            xi,
            t_n + 0.02,
            f"{t_val:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            color="steelblue",
        )
        ax2.text(
            xi,
            tok_n - 0.02,
            f"{tok_val:,}",
            ha="center",
            va="top",
            fontsize=8,
            color="darkorange",
        )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax2.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax2.set_yticklabels(
        [
            f"max\n({max_tok:,} tok)",
            "50%",
            "0",
            "50%",
            f"max\n({max_time:.1f}s)",
        ],
        fontsize=8,
    )
    ax2.set_title(
        "Execution Time ↑  vs  Total Tokens ↓", fontsize=14, fontweight="bold"
    )
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.2)
    fig2.tight_layout()
    figures.append(("time_vs_total_tokens", fig2))

    # 3. Token usage stacked
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    x3 = range(len(run_names))
    ax3.bar(
        x3,
        prompt_tokens,
        label="Prompt Tokens",
        color="cornflowerblue",
        alpha=0.8,
    )
    ax3.bar(
        x3,
        completion_tokens,
        bottom=prompt_tokens,
        label="Completion Tokens",
        color="lightcoral",
        alpha=0.8,
    )
    ax3.set_xlabel("Run")
    ax3.set_ylabel("Token Count")
    ax3.set_title(
        "Token Usage (Prompt vs Completion)", fontsize=14, fontweight="bold"
    )
    ax3.set_xticks(x3)
    ax3.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)
    fig3.tight_layout()
    figures.append(("token_usage", fig3))

    # 4. Total tokens
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    bars4 = ax4.bar(
        range(len(run_names)), total_tokens, color="mediumpurple", alpha=0.7
    )
    ax4.set_xlabel("Run")
    ax4.set_ylabel("Total Tokens")
    ax4.set_title("Total Tokens", fontsize=14, fontweight="bold")
    ax4.set_xticks(range(len(run_names)))
    ax4.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax4.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars4, total_tokens, strict=False):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig4.tight_layout()
    figures.append(("total_tokens", fig4))

    if not has_qualitative:
        return figures

    # 5. Combined qualitative
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    x_pos = range(len(run_names))
    width = 0.25
    rel_vals = [v if v is not None else 0 for v in relevance_scores]
    comp_vals = [v if v is not None else 0 for v in completion_quality_scores]
    hall_vals = [v if v is not None else 0 for v in hallucination_scores]
    ax5.bar(
        [i - width for i in x_pos],
        rel_vals,
        width,
        label="Response Relevance",
        color="lightblue",
        alpha=0.8,
    )
    ax5.bar(
        x_pos,
        comp_vals,
        width,
        label="Task Completion",
        color="lightgreen",
        alpha=0.8,
    )
    ax5.bar(
        [i + width for i in x_pos],
        hall_vals,
        width,
        label="Groundedness",
        color="khaki",
        alpha=0.8,
    )
    ax5.set_xlabel("Run")
    ax5.set_ylabel("Score (0-1)")
    ax5.set_title(
        "Qualitative Metrics (LLM Judge)", fontsize=14, fontweight="bold"
    )
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
    ax5.set_ylim(0, 1.1)
    ax5.legend(fontsize=8)
    ax5.grid(axis="y", alpha=0.3)
    ax5.axhline(y=0.7, color="orange", linestyle="--", alpha=0.3)
    fig5.tight_layout()
    figures.append(("qualitative_metrics", fig5))

    # 6–8. Individual qualitative charts
    for metric_name, vals, title in [
        ("response_relevance", rel_vals, "Response Relevance"),
        ("task_completion_quality", comp_vals, "Task Completion Quality"),
        (
            "hallucination_score",
            hall_vals,
            "Groundedness (Hallucination Score)",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [
            "green" if s >= 0.8 else "orange" if s >= 0.6 else "red"
            for s in vals
        ]
        bars = ax.bar(range(len(run_names)), vals, color=colors, alpha=0.7)
        ax.set_xlabel("Run")
        ax.set_ylabel("Score (0-1)")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(run_names)))
        ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.3, linewidth=1)
        ax.axhline(
            y=0.6, color="orange", linestyle="--", alpha=0.3, linewidth=1
        )
        for bar, val in zip(bars, vals, strict=False):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        fig.tight_layout()
        figures.append((metric_name, fig))

    return figures


def _plot_per_episode_comparison(
    metrics: Dict[str, dict],
    task_filter: str | None = None,
) -> List[Tuple[str, plt.Figure]]:
    """Build per-task charts comparing all runs. Returns (name, figure) pairs.

    If ``task_filter`` is provided, only tasks whose id contains the substring
    are plotted.
    """
    if not metrics:
        return []

    tasks_by_model: Dict[str, Dict[str, dict]] = {}
    all_task_ids: set = set()

    for run_name, data in metrics.items():
        episodes = _get_per_episode_averages(data)
        tasks_by_model[run_name] = {ep["task_id"]: ep for ep in episodes}
        all_task_ids.update(ep["task_id"] for ep in episodes)

    if not all_task_ids:
        return []

    if task_filter:
        all_task_ids = {tid for tid in all_task_ids if task_filter in tid}
        if not all_task_ids:
            return []

    sorted_tasks = sorted(all_task_ids)
    model_names = list(metrics.keys())
    has_qualitative = any(
        any("qualitative" in ep for ep in _get_per_episode_averages(data))
        for data in metrics.values()
    )

    figures: List[Tuple[str, plt.Figure]] = []

    for task_id in sorted_tasks:
        task_data = []
        available_models = []
        for run_name in model_names:
            if task_id in tasks_by_model[run_name]:
                task_data.append(tasks_by_model[run_name][task_id])
                available_models.append(run_name)

        if not task_data:
            continue

        display_names = list(available_models)
        n_plots = 3 if has_qualitative else 2
        fig, axes = plt.subplots(
            n_plots, 1, figsize=(max(10, len(task_data) * 0.8), 4 * n_plots)
        )
        if n_plots == 1:
            axes = [axes]

        fig.suptitle(f"Task: {task_id}", fontsize=14, fontweight="bold")

        # Time
        times = [ep["time"]["wall_ms"] / 1000 for ep in task_data]
        axes[0].bar(
            range(len(display_names)), times, color="steelblue", alpha=0.7
        )
        axes[0].set_ylabel("Time (s)", fontsize=11)
        axes[0].set_title(
            "Execution Time by Model", fontsize=12, fontweight="bold"
        )
        axes[0].set_xticks(range(len(display_names)))
        axes[0].set_xticklabels(
            display_names, rotation=45, ha="right", fontsize=10
        )
        axes[0].grid(axis="y", alpha=0.3)
        for bar, val in zip(axes[0].patches, times, strict=False):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Tokens
        prompt_tokens = [ep["tokens"]["prompt_tokens"] for ep in task_data]
        completion_tokens = [
            ep["tokens"]["completion_tokens"] for ep in task_data
        ]
        x = range(len(display_names))
        axes[1].bar(
            x, prompt_tokens, label="Prompt", color="cornflowerblue", alpha=0.8
        )
        axes[1].bar(
            x,
            completion_tokens,
            bottom=prompt_tokens,
            label="Completion",
            color="lightcoral",
            alpha=0.8,
        )
        axes[1].set_ylabel("Tokens", fontsize=11)
        axes[1].set_title(
            "Token Usage by Model", fontsize=12, fontweight="bold"
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            display_names, rotation=45, ha="right", fontsize=10
        )
        axes[1].legend(fontsize=9)
        axes[1].grid(axis="y", alpha=0.3)

        # Qualitative
        if has_qualitative:
            width = 0.25
            x_pos = range(len(display_names))
            relevance = [
                _qual_score(ep, "response_relevance") for ep in task_data
            ]
            completion_q = [
                _qual_score(ep, "task_completion_quality")
                for ep in task_data
            ]
            hallucination = [
                _qual_score(ep, "hallucination_score") for ep in task_data
            ]
            axes[2].bar(
                [i - width for i in x_pos],
                relevance,
                width,
                label="Relevance",
                color="lightblue",
                alpha=0.8,
            )
            axes[2].bar(
                x_pos,
                completion_q,
                width,
                label="Completion",
                color="lightgreen",
                alpha=0.8,
            )
            axes[2].bar(
                [i + width for i in x_pos],
                hallucination,
                width,
                label="Groundedness",
                color="khaki",
                alpha=0.8,
            )
            axes[2].set_ylabel("Score", fontsize=11)
            axes[2].set_title(
                "Qualitative Metrics by Model", fontsize=12, fontweight="bold"
            )
            axes[2].set_xticks(x_pos)
            axes[2].set_xticklabels(
                display_names, rotation=45, ha="right", fontsize=10
            )
            axes[2].set_ylim(0, 1.1)
            axes[2].legend(fontsize=9)
            axes[2].grid(axis="y", alpha=0.3)
            axes[2].axhline(y=0.7, color="orange", linestyle="--", alpha=0.3)

        fig.tight_layout()
        safe_task = (
            task_id.replace(":", "-").replace("/", "-").replace(" ", "_")
        )
        figures.append((f"per_task_{safe_task}", fig))

    return figures


def _save_figures(
    figures: List[Tuple[str, plt.Figure]], output_dir: Path
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for name, fig in figures:
        path = output_dir / f"metrics_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved


def generate_and_save_plots(
    metrics: Dict[str, dict],
    output_dir: Path,
    task_filter: str | None = None,
) -> List[Path]:
    """Generate all comparison and per-task charts and save them to output_dir.

    ``task_filter`` is a substring match against ``task_id`` and restricts
    the per-task charts only. Summary/comparison charts always reflect the
    full run.
    """
    figures = _plot_comparison(metrics)
    figures += _plot_per_episode_comparison(metrics, task_filter=task_filter)
    return _save_figures(figures, output_dir)
