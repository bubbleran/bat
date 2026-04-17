from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Type

from bat.agent.config import AgentConfig
from bat.agent.graph import AgentGraph
from bat.agent.state import AgentState
from bat.logging import create_logger

from .adapter import BatAgentGraphAdapter
from .bench_runner import BenchRunner, RunConfig
from .contracts import EpisodeResult, TaskSpec
from .metrics.llm_evaluators import evaluate_episode_quality
from .metrics.metrics import summarize_episode_metrics
from .metrics.qualitative_helpers import build_context_from_events, build_expected_desc


logger = create_logger(__name__, level="info")


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


def _discover_graph_and_state_types(agent_root: Path) -> tuple[Type[AgentGraph], Type[AgentState]]:
    
    if str(agent_root) not in sys.path:
        sys.path.insert(0, str(agent_root))

    graph_module = importlib.import_module("src.graph")

    graph_candidates: list[Type[AgentGraph]] = []
    state_candidates: list[Type[AgentState]] = []

    for _, obj in inspect.getmembers(graph_module, inspect.isclass):
        if issubclass(obj, AgentGraph) and obj is not AgentGraph:
            graph_candidates.append(obj)
        if issubclass(obj, AgentState) and obj is not AgentState:
            state_candidates.append(obj)

    if len(graph_candidates) != 1:
        names = ", ".join(sorted(cls.__name__ for cls in graph_candidates)) or "none"
        raise ValueError(
            "Could not auto-discover exactly one AgentGraph subclass in src/graph.py. "
            f"Found: {names}."
        )

    if len(state_candidates) != 1:
        names = ", ".join(sorted(cls.__name__ for cls in state_candidates)) or "none"
        raise ValueError(
            "Could not auto-discover exactly one AgentState subclass in src/graph.py. "
            f"Found: {names}."
        )

    return graph_candidates[0], state_candidates[0]


async def _evaluate_qualitative(results: list[EpisodeResult], tasks_by_id: Dict[str, TaskSpec]) -> None:
    for episode in results:
        task = tasks_by_id.get(episode.task_id)
        if task is None:
            continue

        query = " -> ".join(task.turns)
        context = build_context_from_events([event.model_dump() for event in episode.trace.events])
        expected_desc = build_expected_desc(
            must_succeed=task.expected.must_succeed,
            final_contains=task.expected.final_contains,
        )

        episode.qualitative_scores = await asyncio.to_thread(
            evaluate_episode_quality,
            query,
            episode.output_text,
            episode.status,
            context,
            expected_desc,
        )


def build_agent_graph(
    graph_type: Type[AgentGraph],
    state_type: Type[AgentState],
    config: AgentConfig,
    model_provider: str = "",
    model: str = "",
) -> AgentGraph:
    if model_provider and model:
        os.environ["MODEL_PROVIDER"] = model_provider
        os.environ["MODEL"] = model

    logger.debug(f"Building {graph_type.__name__}...")
    graph = graph_type(config=config, StateType=state_type)
    logger.debug(f"Built {graph_type.__name__} successfully")
    return graph


async def run_agent(
    graph_type: Type[AgentGraph],
    state_type: Type[AgentState],
    model: str,
    model_provider: str,
    input_path: Path,
    run_name: str = "benchmark",
    task_id: str = "",
    enable_scoring: bool = True,
    enable_qualitative_eval: bool = False,
    k: int = 1,
    save_attempts: bool = False,
    out_dir: str = "output",
    agent_config_path: str | None = None,
) -> None:
    if agent_config_path is None:
        raise ValueError("agent_config_path is required")

    logger.debug(f"Loading agent config: {agent_config_path}")
    agent_config = await asyncio.to_thread(AgentConfig.load, agent_config_path)

    agent_graph = build_agent_graph(
        graph_type=graph_type,
        state_type=state_type,
        config=agent_config,
        model_provider=model_provider,
        model=model,
    )

    tasks = load_tasks(input_path)
    tasks_by_id = {task.id: task for task in tasks}

    bench_runner = BenchRunner(
        adapter=BatAgentGraphAdapter(
            agent_graph=agent_graph,
            base_runnable_config={"configurable": {}},
        ),
        config=RunConfig(
            run_name=run_name,
            out_dir=out_dir,
            k=k,
            save_attempts=save_attempts,
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
    parser = argparse.ArgumentParser(description="Run evaluation in the agent environment")
    parser.add_argument("--agent-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-name", default="benchmark")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--qualitative", action="store_true")
    parser.add_argument("--save-attempts", action="store_true")
    args = parser.parse_args()


    agent_root = Path(args.agent_root).resolve()
    dataset = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()

    config_path = agent_root / "config.yaml"
    if not config_path.exists():
        raise ValueError(f"Missing agent config: {config_path}")

    graph_type, state_type = _discover_graph_and_state_types(agent_root)

    asyncio.run(
        run_agent(
            graph_type=graph_type,
            state_type=state_type,
            model=args.model,
            model_provider=args.model_provider,
            input_path=dataset,
            run_name=args.run_name,
            task_id=args.task_id,
            enable_scoring=True,
            enable_qualitative_eval=args.qualitative,
            k=args.k,
            save_attempts=args.save_attempts,
            out_dir=str(output_dir),
            agent_config_path=str(config_path),
        )
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
