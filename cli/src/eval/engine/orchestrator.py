from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from .adapter import BatA2AAdapter
from .bench_runner import BenchRunner, RunConfig
from .contracts import EpisodeResult, TaskSpec
from .metrics.llm_evaluators import evaluate_episode_quality
from .metrics.metrics import summarize_episode_metrics
from .metrics.qualitative_helpers import build_context_from_events, build_expected_desc, build_user_facts_summary


logger = logging.getLogger(__name__)


def load_tasks(path: str | Path) -> List[TaskSpec]:
    dataset_path = Path(path)
    try:
        content = dataset_path.read_text(encoding="utf-8").strip()
        objects = json.loads(content)
        if not isinstance(objects, list):
            raise ValueError(f"Expected a JSON array of task objects in {dataset_path}")
        return [TaskSpec.model_validate(obj) for obj in objects]
    except Exception as exc:
        raise ValueError(f"Dataset not formatted correctly in {dataset_path}") from exc


_QUALITATIVE_CONCURRENCY = 8


async def _evaluate_qualitative(results: list[EpisodeResult], tasks_by_id: dict[str, TaskSpec]) -> None:
    sem = asyncio.Semaphore(_QUALITATIVE_CONCURRENCY)

    async def _score(episode: EpisodeResult) -> None:
        task = tasks_by_id.get(episode.task_id)
        if task is None:
            return
        logger.info(f"Evaluating qualitative scores for episode {episode.task_id}")
        query = " -> ".join(task.turns)
        raw_events = [event.model_dump() for event in episode.trace.events]
        context = build_context_from_events(raw_events)
        user_facts = build_user_facts_summary(raw_events)
        tool_calls = json.dumps(episode.trace.tool_calls, ensure_ascii=False, indent=2)
        expected_desc = build_expected_desc(
            status=task.expected.status,
            expected_outcome=task.expected.expected_outcome,
            output_must_contain=task.expected.output_must_contain,
            expected_tool_calls=task.expected.tool_calls or None,
        )

        async with sem:
            episode.qualitative_scores = await asyncio.to_thread(
                evaluate_episode_quality,
                query,
                episode.output_text,
                episode.status,
                context,
                expected_desc,
                tool_calls,
                bool(task.expected.tool_calls),
                user_facts,
            )

    await asyncio.gather(*(_score(ep) for ep in results))

async def run_evaluation(
    agent_url: str,
    model: str,
    model_provider: str,
    input_path: Path,
    run_name: str = "benchmark",
    task_id: str = "",
    enable_scoring: bool = True,
    enable_qualitative_eval: bool = False,
    k: int = 1,
    out_dir: str = "output",
) -> None:
    tasks = load_tasks(input_path)
    tasks_by_id = {task.id: task for task in tasks}

    bench_runner = BenchRunner(
        adapter=BatA2AAdapter(
            agent_url=agent_url,
        ),
        config=RunConfig(
            run_name=run_name,
            out_dir=out_dir,
            k=k,
            model=f"{model_provider}:{model}",
            task_id=task_id,
        ),
    )

    logger.info(f"Running evaluation on dataset: {input_path}")
    results = await bench_runner.run(tasks)
    logger.info(f"Evaluation complete. Collected {len(results)} result(s)")

    if enable_qualitative_eval:
        logger.info("Running qualitative evaluation...")
        await _evaluate_qualitative(results, tasks_by_id)
        bench_runner.persist_results(results)

    if not enable_scoring:
        if bench_runner.run_dir:
            logger.info(f"Artifacts written to: {bench_runner.run_dir}")
        return

    metrics = summarize_episode_metrics(results, k=k)
    if bench_runner.run_dir:
        (bench_runner.run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Artifacts written to: {bench_runner.run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run A2A evaluation in the agent environment")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--agent-url", required=True)
    parser.add_argument("--model-provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-name", default="benchmark")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--qualitative", action="store_true")
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()

    asyncio.run(
        run_evaluation(
            agent_url=args.agent_url,
            model=args.model,
            model_provider=args.model_provider,
            input_path=dataset,
            run_name=args.run_name,
            task_id=args.task_id,
            enable_scoring=True,
            enable_qualitative_eval=args.qualitative,
            k=args.k,
            out_dir=str(output_dir),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
