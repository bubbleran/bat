from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from typing import Iterator, Mapping
from urllib.parse import urlparse

import typer
from dotenv import dotenv_values

from .engine.contracts import JudgeSpec
from .engine.orchestrator import run_evaluation
from .engine.eval_config import default_eval_yaml, default_tasks_json, load_eval_config


# Maps provider name → the env var its SDK reads for the API key.
_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
}


def _inject_judge_api_key(judge: JudgeSpec, agent_root: Path, env: dict[str, str]) -> None:
    """Resolve the judge's API key and inject it into env.

    Priority:
      1. judge.api_key field in eval.yaml
      2. Key already present in env (exported in the shell)
      3. Key found in the agent's .env file
    """
    api_key_var = _PROVIDER_API_KEY_ENV.get(judge.provider.lower())
    if api_key_var is None:
        return  # local/no-key provider (e.g. ollama)

    if judge.api_key:
        env[api_key_var] = judge.api_key
        return

    if api_key_var in env:
        return  # already available from the shell

    agent_env_file = agent_root / ".env"
    if agent_env_file.exists():
        agent_dotenv = dotenv_values(agent_env_file)
        if api_key_var in agent_dotenv:
            env[api_key_var] = agent_dotenv[api_key_var]  # type: ignore[assignment]


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


@contextlib.contextmanager
def _temporary_env(overrides: Mapping[str, str]) -> Iterator[None]:
    original_values: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            original_values[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, original in original_values.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original


def _run_eval_orchestrator(
    *,
    agent_url: str,
    model_provider: str,
    model: str,
    dataset: Path,
    output_dir: Path,
    task_id: str,
    k: int,
    run_name: str,
    qualitative: bool,
    env: Mapping[str, str],
) -> None:
    with _temporary_env(env):
        asyncio.run(
            run_evaluation(
                agent_url=agent_url,
                model=model,
                model_provider=model_provider,
                input_path=dataset,
                run_name=run_name,
                task_id=task_id,
                enable_scoring=True,
                enable_qualitative_eval=qualitative,
                k=k,
                out_dir=str(output_dir),
            )
        )


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


def _print_eval_show(cfg) -> None:
    judge_model = f"{cfg.judge.provider}:{cfg.judge.model}" if cfg.judge is not None else "not configured"

    typer.secho("============================", fg=typer.colors.BLUE)
    typer.secho("  EVALUATION CONFIGURATION", fg=typer.colors.BLUE, bold=True)
    typer.secho("============================", fg=typer.colors.BLUE)

    typer.secho("Dataset", fg=typer.colors.BRIGHT_BLUE, bold=True, nl=False)
    typer.echo(f"     : {cfg.dataset}")

    typer.secho("k", fg=typer.colors.BRIGHT_BLUE, bold=True, nl=False)
    typer.echo(f"           : {cfg.k}")

    typer.secho("Qualitative", fg=typer.colors.BRIGHT_BLUE, bold=True, nl=False)
    typer.echo(f" : {'yes' if cfg.qualitative else 'no'}")

    typer.secho("", nl=True)
    typer.secho("Models:", fg=typer.colors.CYAN, bold=True)
    for idx, model in enumerate(cfg.models, start=1):
        typer.echo(f"  [{idx}] {model.provider}:{model.model}")

    typer.secho("", nl=True)
    typer.secho("Judge model", fg=typer.colors.MAGENTA, bold=True, nl=False)
    typer.echo(f" : {judge_model}")
    typer.secho("============================", fg=typer.colors.BLUE)


def eval_show() -> None:
    agent_root = Path.cwd()
    _validate_agent_root(agent_root)

    eval_yaml_path = agent_root / "eval" / "eval.yaml"
    if not eval_yaml_path.exists():
        raise typer.BadParameter("Missing ./eval/eval.yaml. Run 'bat eval init' first.")

    try:
        cfg = load_eval_config(agent_root, eval_yaml_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_eval_show(cfg)


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

                _inject_judge_api_key(cfg.judge, agent_root, runner_env)

                _apply_env_overrides(
                    runner_env,
                    cfg.judge.env,
                    section_name="judge",
                )
            else:
                runner_env.pop("JUDGE_PROVIDER", None)
                runner_env.pop("JUDGE_MODEL", None)
                runner_env.pop("JUDGE_BASE_URL", None)

            _run_eval_orchestrator(
                agent_url=cfg.agent_url,
                model_provider=model_cfg.provider,
                model=model_cfg.model,
                dataset=cfg.dataset,
                output_dir=cfg.output_dir,
                task_id=task_id,
                k=cfg.k,
                run_name=cfg.run_name,
                qualitative=cfg.qualitative,
                env=runner_env,
            )
        finally:
            _stop_agent_process(process, timeout_s=cfg.agent_shutdown_timeout_s)

    typer.secho(
        f"Evaluation completed. Output: {cfg.output_dir / task_id}",
        fg=typer.colors.GREEN,
    )


def eval_plot(
    folder: Path = typer.Option(
        ...,
        "--folder",
        "-f",
        help="Path to an evaluation output folder. Each sub-folder containing a metrics.json is treated as one run.",
    ),
) -> None:
    folder = folder.resolve()

    if not folder.is_dir():
        raise typer.BadParameter(f"Folder not found: {folder}")

    metrics: dict[str, dict] = {}
    for sub in sorted(folder.iterdir()):
        if sub.is_dir():
            metrics_file = sub / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, encoding="utf-8") as f:
                    metrics[sub.name] = json.load(f)

    if not metrics:
        raise typer.BadParameter(
            f"No valid evaluation results found in {folder}. "
            "A sub-folder is a valid run only if it contains a metrics.json file."
        )

    typer.secho(
        f"Found {len(metrics)} run(s): {', '.join(metrics)}",
        fg=typer.colors.CYAN,
    )

    from .engine.plotter import generate_and_save_plots

    saved = generate_and_save_plots(metrics, folder)

    for path in saved:
        typer.secho(f"  {path.relative_to(folder)}", fg=typer.colors.GREEN)

    typer.secho(
        f"\nSaved {len(saved)} chart(s) to {folder}",
        fg=typer.colors.GREEN,
        bold=True,
    )
