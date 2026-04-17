from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .orchestrator import _discover_graph_and_state_types, run_agent


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
            task_id=args.task_id,
            run_name=args.run_name,
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
