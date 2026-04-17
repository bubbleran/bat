from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

import typer

from .engine.eval_config import default_eval_yaml, default_tasks_json, load_eval_config


_ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SIMPLE_ENV_REF = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)$")
_BRACED_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _validate_agent_root(agent_root: Path) -> None:
    required = [
        agent_root / "config.yaml",
        agent_root / "src" / "graph.py",
        agent_root / "agent.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.relative_to(agent_root)) for path in missing)
        raise typer.BadParameter(
            f"Current directory does not look like an agent root. Missing: {missing_text}. Please add this files or run this command from the root of an existing agent.",
        )


def _find_cli_src() -> Path | None:
    candidates = [
        Path(__file__).resolve().parents[1],
        Path(sys.executable).resolve().parent.parent / "src",
    ]
    for candidate in candidates:
        if (candidate / "eval" / "engine" / "orchestrator.py").exists():
            return candidate
    return None


def _find_agent_python(agent_root: Path) -> Path | None:
    candidates = [
        agent_root / ".venv" / "bin" / "python",
        agent_root / ".venv" / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_env_value(
    raw_value: str,
    env: Mapping[str, str],
    *,
    section_name: str,
    env_key: str,
) -> str:
    simple_match = _SIMPLE_ENV_REF.fullmatch(raw_value.strip())
    if simple_match:
        ref_name = simple_match.group(1)
        resolved = env.get(ref_name)
        if resolved is None:
            raise typer.BadParameter(
                f"{section_name}.env.{env_key} references missing environment variable: {ref_name}"
            )
        return resolved

    missing_refs: list[str] = []

    def _substitute(match: re.Match[str]) -> str:
        ref_name = match.group(1)
        resolved = env.get(ref_name)
        if resolved is None:
            missing_refs.append(ref_name)
            return ""
        return resolved

    rendered = _BRACED_ENV_REF.sub(_substitute, raw_value)
    if missing_refs:
        refs = ", ".join(sorted(set(missing_refs)))
        raise typer.BadParameter(
            f"{section_name}.env.{env_key} references missing environment variable(s): {refs}"
        )

    return rendered


def _apply_env_overrides(
    env: dict[str, str],
    overrides: dict[str, str],
    *,
    section_name: str,
) -> None:
    for key, value in overrides.items():
        env_key = key.strip()
        if not env_key:
            continue
        if not _ENV_VAR_PATTERN.fullmatch(env_key):
            raise typer.BadParameter(
                f"{section_name}.env contains invalid variable name: {env_key}"
            )
        env[env_key] = _resolve_env_value(
            value,
            env,
            section_name=section_name,
            env_key=env_key,
        )


def eval_init(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite eval/eval.yaml and eval/input/tasks.json if they already exist.",
    ),
) -> None:
    agent_root = Path.cwd()
    _validate_agent_root(agent_root)

    eval_dir = agent_root / "eval"
    input_dir = eval_dir / "input"
    output_dir = eval_dir / "output"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_yaml_path = eval_dir / "eval.yaml"
    tasks_path = input_dir / "tasks.json"

    if eval_yaml_path.exists() and not force:
        typer.secho(
            f"{eval_yaml_path} already exists. Use --force to overwrite.",
            fg=typer.colors.YELLOW,
        )
    else:
        eval_yaml_path.write_text(default_eval_yaml(), encoding="utf-8")
        typer.secho(f"Created {eval_yaml_path}", fg=typer.colors.GREEN)

    if tasks_path.exists() and not force:
        typer.secho(
            f"{tasks_path} already exists. Use --force to overwrite.",
            fg=typer.colors.YELLOW,
        )
    else:
        tasks_path.write_text(default_tasks_json(), encoding="utf-8")
        typer.secho(f"Created {tasks_path}", fg=typer.colors.GREEN)

    typer.secho("Evaluation scaffold ready in eval/", fg=typer.colors.GREEN)


def eval_run() -> None:
    agent_root = Path.cwd()
    _validate_agent_root(agent_root)

    eval_yaml_path = agent_root / "eval" / "eval.yaml"
    if not eval_yaml_path.exists():
        raise typer.BadParameter("Missing ./eval/eval.yaml. Run 'bat eval init' first.")

    try:
        cfg = load_eval_config(agent_root, eval_yaml_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if not cfg.dataset.exists():
        raise typer.BadParameter(f"Dataset not found: {cfg.dataset}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    cli_src = _find_cli_src()
    print("Percorso della sorgente CLI:", cli_src)
    if not (cli_src / "eval" / "engine" / "orchestrator.py").exists():
        raise typer.BadParameter(
            "Cannot locate eval runner module. Expected cli/src/eval/engine/orchestrator.py"
        )

    agent_python = _find_agent_python(agent_root)
    if agent_python is None:
        raise typer.BadParameter(
            "No agent python found at .venv/bin/python. Create the agent virtual environment first."
        )

    task_id = time.strftime("%Y%m%d_%H%M%S")
    typer.secho(
        f"Running evaluation with {len(cfg.models)} model(s). task_id={task_id}",
        fg=typer.colors.CYAN,
    )

    for idx, model_cfg in enumerate(cfg.models):
        typer.secho(f"- {model_cfg.provider}:{model_cfg.model}", fg=typer.colors.CYAN)

        runner_args = [
            "-m",
            "eval.engine.orchestrator",
            "--agent-root",
            str(agent_root.resolve()),
            "--dataset",
            str(cfg.dataset),
            "--output-dir",
            str(cfg.output_dir),
            "--model-provider",
            model_cfg.provider,
            "--model",
            model_cfg.model,
            "--task-id",
            task_id,
            "--k",
            str(cfg.k),
            "--run-name",
            cfg.run_name,
        ]
        if cfg.qualitative:
            runner_args.append("--qualitative")
        if cfg.save_attempts:
            runner_args.append("--save-attempts")

        child_env = os.environ.copy()
        child_env["MODEL_PROVIDER"] = model_cfg.provider
        child_env["MODEL"] = model_cfg.model
        if model_cfg.base_url:
            child_env["BASE_URL"] = model_cfg.base_url
        else:
            child_env.pop("BASE_URL", None)

        _apply_env_overrides(
            child_env,
            model_cfg.env,
            section_name=f"models[{idx}]",
        )

        if cfg.qualitative:
            if cfg.judge is None:
                raise typer.BadParameter(
                    "When evaluation.qualitative is true, judge.provider and judge.model are required"
                )

            child_env["JUDGE_PROVIDER"] = cfg.judge.provider
            child_env["JUDGE_MODEL"] = cfg.judge.model
            if cfg.judge.base_url:
                child_env["JUDGE_BASE_URL"] = cfg.judge.base_url
            else:
                child_env.pop("JUDGE_BASE_URL", None)

            _apply_env_overrides(
                child_env,
                cfg.judge.env,
                section_name="judge",
            )
        else:
            child_env.pop("JUDGE_PROVIDER", None)
            child_env.pop("JUDGE_MODEL", None)
            child_env.pop("JUDGE_BASE_URL", None)

        child_env["PYTHONPATH"] = str(cli_src)

        cmd = [str(agent_python), *runner_args]
        result = subprocess.run(
            cmd,
            cwd=agent_root,
            env=child_env,
            check=False,
        )
        if result.returncode != 0:
            raise typer.Exit(code=result.returncode)

    typer.secho(
        f"Evaluation completed. Output: {cfg.output_dir / task_id}",
        fg=typer.colors.GREEN,
    )
