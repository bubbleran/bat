from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

QUALITATIVE = (
    ("response_relevance", "Response Relevance", "lightblue"),
    ("task_completion_quality", "Task Completion", "lightgreen"),
    ("hallucination_score", "Groundedness", "khaki"),
)


def _mean(values: list) -> float:
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else 0


def _grade(score: float) -> str:
    if score >= 0.8:
        return "green"
    if score >= 0.6:
        return "orange"
    return "red"


def _style(ax, names: list[str], title: str, ylabel: str, ylim=None) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    if ylim:
        ax.set_ylim(0, ylim)


def _labels(ax, bars, values: list, fmt: str) -> None:
    for bar, value in zip(bars, values, strict=False):
        if value:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=8,
            )


def _bars(ax, names: list[str], values: list, fmt: str, color) -> None:
    bars = ax.bar(range(len(names)), values, color=color, alpha=0.7)
    _labels(ax, bars, values, fmt)


def _tokens(ax, names: list[str], prompt: list, completion: list) -> None:
    x = range(len(names))
    ax.bar(x, prompt, label="Prompt Tokens", color="cornflowerblue", alpha=0.8)
    ax.bar(
        x,
        completion,
        bottom=prompt,
        label="Completion Tokens",
        color="lightcoral",
        alpha=0.8,
    )
    ax.legend(fontsize=8)


def _scores(ax, names: list[str], scores: dict[str, list]) -> None:
    width = 0.25
    for offset, (field, label, color) in zip(
        (-width, 0, width), QUALITATIVE, strict=True
    ):
        x = [index + offset for index in range(len(names))]
        ax.bar(x, scores[field], width, label=label, color=color, alpha=0.8)
    ax.axhline(y=0.7, color="orange", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8)


def _time_vs_tokens(names: list[str], times: list, totals: list):
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 7))
    max_time = max(times) or 1
    max_tokens = max(totals) or 1
    x = range(len(names))
    ax.bar(
        x,
        [t / max_time for t in times],
        color="steelblue",
        alpha=0.75,
        label="Execution Time",
    )
    ax.bar(
        x,
        [-t / max_tokens for t in totals],
        color="darkorange",
        alpha=0.75,
        label="Total Tokens",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    for index, (time_s, tokens) in enumerate(zip(times, totals, strict=True)):
        ax.text(
            index,
            time_s / max_time + 0.02,
            f"{time_s:.1f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            color="steelblue",
        )
        ax.text(
            index,
            -tokens / max_tokens - 0.02,
            f"{tokens:,}",
            ha="center",
            va="top",
            fontsize=8,
            color="darkorange",
        )
    _style(ax, names, "Execution Time ↑  vs  Total Tokens ↓", "")
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticklabels(
        [
            f"max\n({max_tokens:,} tok)",
            "50%",
            "0",
            "50%",
            f"max\n({max_time:.1f}s)",
        ],
        fontsize=8,
    )
    ax.legend(fontsize=9)
    return fig


def _run_charts(metrics: dict[str, dict]) -> list[tuple[str, plt.Figure]]:
    """Charts comparing the runs as a whole."""
    names = list(metrics)
    summaries = [data.get("summary", {}) for data in metrics.values()]
    times = [
        s.get("time", {}).get("total_wall_ms", 0) / 1000 for s in summaries
    ]
    tokens = [s.get("tokens", {}) for s in summaries]
    prompt = [t.get("prompt_tokens_total", 0) for t in tokens]
    completion = [t.get("completion_tokens_total", 0) for t in tokens]
    totals = [t.get("total_tokens_total", 0) for t in tokens]
    figures = []

    fig, ax = plt.subplots(figsize=(10, 6))
    _bars(ax, names, times, "{:.1f}s", "steelblue")
    _style(ax, names, "Total Execution Time", "Time (seconds)")
    figures.append(("execution_time", fig))

    figures.append(
        ("time_vs_total_tokens", _time_vs_tokens(names, times, totals))
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    _tokens(ax, names, prompt, completion)
    _style(ax, names, "Token Usage (Prompt vs Completion)", "Token Count")
    figures.append(("token_usage", fig))

    fig, ax = plt.subplots(figsize=(10, 6))
    _bars(ax, names, totals, "{:,}", "mediumpurple")
    _style(ax, names, "Total Tokens", "Total Tokens")
    figures.append(("total_tokens", fig))

    if not any("qualitative" in s for s in summaries):
        return figures
    scores = {
        field: [
            (s.get("qualitative", {}).get(field) or {}).get("avg") or 0
            for s in summaries
        ]
        for field, _, _ in QUALITATIVE
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    _scores(ax, names, scores)
    _style(ax, names, "Qualitative Metrics (LLM Judge)", "Score (0-1)", 1.1)
    figures.append(("qualitative_metrics", fig))
    for field, label, _ in QUALITATIVE:
        fig, ax = plt.subplots(figsize=(10, 6))
        _bars(
            ax,
            names,
            scores[field],
            "{:.3f}",
            [_grade(v) for v in scores[field]],
        )
        _style(ax, names, label, "Score (0-1)", 1.1)
        ax.axhline(y=0.8, color="green", linestyle="--", alpha=0.3)
        ax.axhline(y=0.6, color="orange", linestyle="--", alpha=0.3)
        figures.append((field, fig))
    return figures


def _per_task(per_episode: list[dict]) -> dict[str, dict]:
    """Each task's attempts, averaged."""
    attempts: dict[str, list[dict]] = {}
    for episode in per_episode:
        attempts.setdefault(episode["task_id"], []).append(episode)
    averages = {}
    for task_id, tries in attempts.items():
        row = {
            "wall_ms": _mean([e["time"]["wall_ms"] for e in tries]),
            "prompt": _mean([e["tokens"]["prompt_tokens"] for e in tries]),
            "completion": _mean(
                [e["tokens"]["completion_tokens"] for e in tries]
            ),
            "qualitative": any("qualitative" in e for e in tries),
        }
        for field, _, _ in QUALITATIVE:
            row[field] = _mean(
                [(e.get("qualitative") or {}).get(field) for e in tries]
            )
        averages[task_id] = row
    return averages


def _task_charts(
    metrics: dict[str, dict], task_filter: str | None
) -> list[tuple[str, plt.Figure]]:
    """One chart per task, comparing the runs on it."""
    runs = {
        name: _per_task(data.get("per_episode", []))
        for name, data in metrics.items()
    }
    task_ids = sorted(
        {
            task_id
            for tasks in runs.values()
            for task_id in tasks
            if not task_filter or task_filter in task_id
        }
    )
    qualitative = any(
        row["qualitative"] for tasks in runs.values() for row in tasks.values()
    )
    figures = []
    for task_id in task_ids:
        names = [name for name, tasks in runs.items() if task_id in tasks]
        rows = [runs[name][task_id] for name in names]
        count = 3 if qualitative else 2
        fig, axes = plt.subplots(
            count, 1, figsize=(max(10, len(rows) * 0.8), 4 * count)
        )
        fig.suptitle(f"Task: {task_id}", fontsize=14, fontweight="bold")
        _bars(
            axes[0],
            names,
            [row["wall_ms"] / 1000 for row in rows],
            "{:.1f}s",
            "steelblue",
        )
        _style(axes[0], names, "Execution Time by Model", "Time (s)")
        _tokens(
            axes[1],
            names,
            [row["prompt"] for row in rows],
            [row["completion"] for row in rows],
        )
        _style(axes[1], names, "Token Usage by Model", "Tokens")
        if qualitative:
            _scores(
                axes[2],
                names,
                {
                    field: [row[field] for row in rows]
                    for field, _, _ in QUALITATIVE
                },
            )
            _style(axes[2], names, "Qualitative Metrics by Model", "Score", 1.1)
        safe = task_id.replace(":", "-").replace("/", "-").replace(" ", "_")
        figures.append((f"per_task_{safe}", fig))
    return figures


def generate_and_save_plots(
    metrics: dict[str, dict], output_dir: Path, task_filter: str | None = None
) -> list[Path]:
    """Save the run charts and the per-task ones (only tasks whose id
    contains task_filter, when given) as PNGs in output_dir."""
    saved = []
    for name, fig in _run_charts(metrics) + _task_charts(metrics, task_filter):
        fig.tight_layout()
        path = output_dir / f"metrics_{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
    return saved
