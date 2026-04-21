from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

import typer

from .engine.eval_config import default_eval_yaml, default_tasks_json, load_eval_config


_ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SIMPLE_ENV_REF = re.compile(r"^\$([A-Za-z_][A-Za-z0-9_]*)$")
_BRACED_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _validate_agent_root(agent_root: Path) -> None:
    required = [
        agent_root / "config.yaml",
        agent_root / "agent.json",
        agent_root / "pyproject.toml",
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


def _parse_agent_url(agent_url: str) -> tuple[str, int, str]:
    parsed = urlparse(agent_url.strip())
    if not parsed.scheme or not parsed.hostname:
        raise typer.BadParameter(
            "evaluation.agent_url must be a full URL, for example: http://127.0.0.1:9900"
        )

    port = parsed.port
    if port is None:
        if parsed.scheme == "https":
            port = 443
        elif parsed.scheme == "http":
            port = 80
        else:
            raise typer.BadParameter("evaluation.agent_url must use http or https")

    base_url = f"{parsed.scheme}://{parsed.hostname}"
    return parsed.hostname, port, base_url


def _wait_for_agent_port(agent_url: str, timeout_s: int, process: subprocess.Popen) -> None:
    host, port, _ = _parse_agent_url(agent_url)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        if process.poll() is not None:
            raise typer.BadParameter(
                f"Agent process exited before becoming ready (exit code: {process.returncode})."
            )
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)

    raise typer.BadParameter(
        f"Agent did not become ready at {agent_url} within {timeout_s} seconds."
    )


def _start_agent_process(agent_root: Path, env: dict[str, str]) -> subprocess.Popen:
    try:
        return subprocess.Popen(
            ["uv", "run", "."],
            cwd=agent_root,
            env=env,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter("Cannot execute 'uv run .'. Ensure uv is installed and available in PATH.") from exc


def _stop_agent_process(process: subprocess.Popen, timeout_s: int) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_s)


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
    if cli_src is None or not (cli_src / "eval" / "engine" / "orchestrator.py").exists():
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

    _, parsed_port, base_url = _parse_agent_url(cfg.agent_url)

    for idx, model_cfg in enumerate(cfg.models):
        typer.secho(f"- {model_cfg.provider}:{model_cfg.model}", fg=typer.colors.CYAN)

        server_env = os.environ.copy()
        server_env["MODEL_PROVIDER"] = model_cfg.provider
        server_env["MODEL"] = model_cfg.model
        server_env["PORT"] = str(parsed_port)
        server_env["URL"] = base_url

        if model_cfg.base_url:
            server_env["BASE_URL"] = model_cfg.base_url
        else:
            server_env.pop("BASE_URL", None)

        _apply_env_overrides(
            server_env,
            model_cfg.env,
            section_name=f"models[{idx}]",
        )

        process = _start_agent_process(agent_root, server_env)

        try:
            _wait_for_agent_port(
                cfg.agent_url,
                timeout_s=cfg.agent_startup_timeout_s,
                process=process,
            )

            runner_args = [
                "-m",
                "eval.engine.orchestrator",
                "--dataset",
                str(cfg.dataset),
                "--output-dir",
                str(cfg.output_dir),
                "--agent-url",
                cfg.agent_url,
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

            runner_env = server_env.copy()
            if cfg.qualitative:
                if cfg.judge is None:
                    raise typer.BadParameter(
                        "When evaluation.qualitative is true, judge.provider and judge.model are required"
                    )

                runner_env["JUDGE_PROVIDER"] = cfg.judge.provider
                runner_env["JUDGE_MODEL"] = cfg.judge.model
                if cfg.judge.base_url:
                    runner_env["JUDGE_BASE_URL"] = cfg.judge.base_url
                else:
                    runner_env.pop("JUDGE_BASE_URL", None)

                _apply_env_overrides(
                    runner_env,
                    cfg.judge.env,
                    section_name="judge",
                )
            else:
                runner_env.pop("JUDGE_PROVIDER", None)
                runner_env.pop("JUDGE_MODEL", None)
                runner_env.pop("JUDGE_BASE_URL", None)

            runner_env["PYTHONPATH"] = str(cli_src)

            cmd = [str(agent_python), *runner_args]
            result = subprocess.run(
                cmd,
                cwd=agent_root,
                env=runner_env,
                check=False,
            )
            if result.returncode != 0:
                raise typer.Exit(code=result.returncode)
        finally:
            _stop_agent_process(process, timeout_s=cfg.agent_shutdown_timeout_s)

    typer.secho(
        f"Evaluation completed. Output: {cfg.output_dir / task_id}",
        fg=typer.colors.GREEN,
    )
