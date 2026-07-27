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
from .engine.eval_config import (
    default_eval_yaml,
    default_tasks_json,
    load_eval_config,
)
from .engine.orchestrator import run_evaluation

# Maps provider name → the env var its SDK reads for the API key.
_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
}


def _inject_judge_api_key(
    judge: JudgeSpec, agent_root: Path, env: dict[str, str]
) -> None:
    """Resolve the judge's API key and inject it into env.

    If judge.api_key_env is set, the agent's .env file is the ONLY source: the CLI
    reads that variable name from <agent_root>/.env and uses it. Nothing else is
    consulted (not the shell, not other files).

    If judge.api_key_env is NOT set, fall back to:
      1. Key already present in env (exported in the shell)
      2. Key found in the agent's .env file under the provider's standard name
    """
    api_key_var = _PROVIDER_API_KEY_ENV.get(judge.provider.lower())
    if api_key_var is None:
        return  # local/no-key provider (e.g. ollama)

    if judge.api_key_env:
        agent_env_file = agent_root / ".env"
        if not agent_env_file.exists():
            typer.secho(
                f"Warning: judge.api_key_env='{judge.api_key_env}' was set but no .env file "
                f"exists at {agent_env_file}; the judge will likely fail when called.",
                fg=typer.colors.YELLOW,
                err=True,
            )
            return
        agent_dotenv = dotenv_values(agent_env_file)
        raw_value = agent_dotenv.get(judge.api_key_env)
        value = (raw_value or "").strip()
        if not value:
            typer.secho(
                f"Warning: judge.api_key_env='{judge.api_key_env}' was set but the variable is "
                f"missing or empty in {agent_env_file}; the judge will likely fail when called.",
                fg=typer.colors.YELLOW,
                err=True,
            )
            return
        env[api_key_var] = value
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
        missing_text = ", ".join(
            str(path.relative_to(agent_root)) for path in missing
        )
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
            raise typer.BadParameter(
                "evaluation.agent_url must use http or https"
            )

    base_url = f"{parsed.scheme}://{parsed.hostname}"
    return parsed.hostname, port, base_url


def _wait_until_port_open(
    agent_url: str,
    timeout_s: int,
    *,
    liveness=None,
    what: str = "Agent",
) -> None:
    """Block until the agent's host:port accepts a TCP connection.

    ``liveness`` is an optional callable invoked each iteration; it should raise
    if the backing process/container died before the port opened.
    """
    host, port, _ = _parse_agent_url(agent_url)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        if liveness is not None:
            liveness()
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError:
            time.sleep(0.2)

    raise typer.BadParameter(
        f"{what} did not become ready at {agent_url} within {timeout_s} seconds."
    )


def _wait_for_agent_port(
    agent_url: str,
    timeout_s: int,
    process: subprocess.Popen | None,
) -> None:
    def _liveness() -> None:
        if process is not None and process.poll() is not None:
            raise typer.BadParameter(
                f"Agent process exited before becoming ready (exit code: {process.returncode})."
            )

    _wait_until_port_open(agent_url, timeout_s, liveness=_liveness)


def _wait_for_remote_agent(agent_url: str, timeout_s: int) -> None:
    _wait_until_port_open(agent_url, timeout_s, what="Remote agent")


def _start_agent_process(
    agent_root: Path, env: dict[str, str]
) -> subprocess.Popen:
    try:
        return subprocess.Popen(
            ["uv", "run", "."],
            cwd=agent_root,
            env=env,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            "Cannot execute 'uv run .'. Ensure uv is installed and available in PATH."
        ) from exc


def _stop_agent_process(process: subprocess.Popen, timeout_s: int) -> None:
    if process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_s)


# --- docker target -------------------------------------------------------

# Env keys that select the agent's model; these are passed to the container
# explicitly so they override anything coming from its baked-in --env-file.
_AGENT_MODEL_ENV_KEYS = ("MODEL_PROVIDER", "MODEL", "PORT", "URL", "BASE_URL")


def _resolve_eval_image(agent_root: Path, cfg) -> str:
    if cfg.image:
        return cfg.image
    # Fall back to the same reference build/push produce.
    from image_defaults import resolve_registry, resolve_repo_name

    registry = resolve_registry(agent_root, None)
    repo = resolve_repo_name(agent_root, None)
    return f"{registry}/{repo}:{cfg.image_version}"


def _container_name(task_id: str, idx: int) -> str:
    return f"bat-eval-{task_id}-{idx}"


def _container_explicit_env(
    server_env: Mapping[str, str], model_cfg
) -> dict[str, str]:
    keys = list(_AGENT_MODEL_ENV_KEYS) + list(model_cfg.env.keys())
    return {key: server_env[key] for key in keys if key in server_env}


def _start_agent_container(
    image: str,
    *,
    network: str,
    agent_root: Path,
    explicit_env: Mapping[str, str],
    container_name: str,
    port: int,
) -> None:
    # Clear any stale container left by a previous interrupted run.
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        text=True,
    )

    command = ["docker", "run", "-d", "--name", container_name]
    if network:
        command += ["--network", network]
    if network != "host":
        command += ["-p", f"{port}:{port}"]

    # Forward the agent's own secrets (API keys, etc.) the same way the source
    # mode does: from the agent root's .env. Explicit -e below overrides these.
    env_file = agent_root / ".env"
    if env_file.is_file():
        command += ["--env-file", str(env_file)]
    for key, value in explicit_env.items():
        command += ["-e", f"{key}={value}"]

    command.append(image)

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            "Cannot execute 'docker run'. Ensure Docker is installed and on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise typer.BadParameter(
            f"Failed to start agent container from image '{image}': {detail}"
        ) from exc


def _container_is_running(container_name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _container_logs_tail(container_name: str, lines: int = 20) -> str:
    result = subprocess.run(
        ["docker", "logs", "--tail", str(lines), container_name],
        capture_output=True,
        text=True,
    )
    return ((result.stdout or "") + (result.stderr or "")).strip()


def _wait_for_agent_container(
    agent_url: str, timeout_s: int, container_name: str
) -> None:
    def _liveness() -> None:
        if not _container_is_running(container_name):
            logs = _container_logs_tail(container_name)
            raise typer.BadParameter(
                f"Agent container '{container_name}' exited before becoming "
                f"ready.\n--- container logs ---\n{logs}"
            )

    _wait_until_port_open(
        agent_url, timeout_s, liveness=_liveness, what="Agent container"
    )


def _stop_agent_container(container_name: str, timeout_s: int) -> None:
    subprocess.run(
        ["docker", "stop", "-t", str(timeout_s), container_name],
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        capture_output=True,
        text=True,
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


def _print_eval_show(cfg) -> None:
    judge_model = (
        f"{cfg.judge.provider}:{cfg.judge.model}"
        if cfg.judge is not None
        else "not configured"
    )

    typer.secho("============================", fg=typer.colors.BLUE)
    typer.secho("  EVALUATION CONFIGURATION", fg=typer.colors.BLUE, bold=True)
    typer.secho("============================", fg=typer.colors.BLUE)

    typer.secho("Target", fg=typer.colors.BRIGHT_BLUE, bold=True, nl=False)
    typer.echo(f"      : {cfg.target}")

    if cfg.target == "docker":
        image_label = cfg.image or f"(resolved, tag {cfg.image_version})"
        typer.secho("Image", fg=typer.colors.BRIGHT_BLUE, bold=True, nl=False)
        typer.echo(f"       : {image_label}")

    typer.secho("Agent URL", fg=typer.colors.BRIGHT_BLUE, bold=True, nl=False)
    typer.echo(f"   : {cfg.agent_url}")

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
        raise typer.BadParameter(
            "Missing ./eval/eval.yaml. Run 'bat eval init' first."
        )

    try:
        cfg = load_eval_config(agent_root, eval_yaml_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_eval_show(cfg)


def _build_server_env(
    cfg, model_cfg, parsed_port: int, base_url: str, idx: int
) -> dict[str, str]:
    """Assemble the environment that selects the agent's model for one run."""
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
    return server_env


def _build_runner_env(cfg, agent_root: Path, server_env) -> dict[str, str]:
    """Derive the orchestrator env (judge config runs locally in the CLI)."""
    runner_env = dict(server_env)
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

        for prompt_key in (
            "relevance",
            "task_completion",
            "hallucination",
            "tool_call",
        ):
            env_name = f"JUDGE_PROMPT_{prompt_key.upper()}"
            text = cfg.judge.prompts.get(prompt_key)
            if text:
                runner_env[env_name] = text
            else:
                runner_env.pop(env_name, None)

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
        for prompt_key in (
            "relevance",
            "task_completion",
            "hallucination",
            "tool_call",
        ):
            runner_env.pop(f"JUDGE_PROMPT_{prompt_key.upper()}", None)

    return runner_env


def eval_run() -> None:
    agent_root = Path.cwd()

    eval_yaml_path = agent_root / "eval" / "eval.yaml"
    if not eval_yaml_path.exists():
        raise typer.BadParameter(
            "Missing ./eval/eval.yaml. Run 'bat eval init' first."
        )

    try:
        cfg = load_eval_config(agent_root, eval_yaml_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    # Only the source-launched target needs the full agent project + venv.
    # docker/remote targets just need eval.yaml + the dataset.
    if cfg.target == "local":
        _validate_agent_root(agent_root)
        if _find_agent_python(agent_root) is None:
            raise typer.BadParameter(
                "No agent python found at .venv/bin/python. Create the agent virtual environment first."
            )

    if not cfg.dataset.exists():
        raise typer.BadParameter(f"Dataset not found: {cfg.dataset}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    image = None
    if cfg.target == "docker":
        image = _resolve_eval_image(agent_root, cfg)

    task_id = time.strftime("%Y%m%d_%H%M%S")
    typer.secho(
        f"Running evaluation [{cfg.target}] with {len(cfg.models)} model(s). "
        f"task_id={task_id}",
        fg=typer.colors.CYAN,
    )
    if cfg.target == "remote":
        typer.secho(
            "  remote target: the agent is evaluated as-deployed; the model "
            "entry is used only as the result label.",
            fg=typer.colors.YELLOW,
        )

    _, parsed_port, base_url = _parse_agent_url(cfg.agent_url)

    for idx, model_cfg in enumerate(cfg.models):
        typer.secho(
            f"- {model_cfg.provider}:{model_cfg.model}", fg=typer.colors.CYAN
        )

        server_env = _build_server_env(
            cfg, model_cfg, parsed_port, base_url, idx
        )
        runner_env = _build_runner_env(cfg, agent_root, server_env)

        process: subprocess.Popen | None = None
        container_name: str | None = None
        try:
            if cfg.target == "local":
                process = _start_agent_process(agent_root, server_env)
                _wait_for_agent_port(
                    cfg.agent_url,
                    timeout_s=cfg.agent_startup_timeout_s,
                    process=process,
                )
            elif cfg.target == "docker":
                container_name = _container_name(task_id, idx)
                _start_agent_container(
                    image,
                    network=cfg.docker_network,
                    agent_root=agent_root,
                    explicit_env=_container_explicit_env(server_env, model_cfg),
                    container_name=container_name,
                    port=parsed_port,
                )
                _wait_for_agent_container(
                    cfg.agent_url,
                    timeout_s=cfg.agent_startup_timeout_s,
                    container_name=container_name,
                )
            else:  # remote
                _wait_for_remote_agent(
                    cfg.agent_url, timeout_s=cfg.agent_startup_timeout_s
                )

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
            if process is not None:
                _stop_agent_process(
                    process, timeout_s=cfg.agent_shutdown_timeout_s
                )
            elif container_name is not None:
                _stop_agent_container(
                    container_name, timeout_s=cfg.agent_shutdown_timeout_s
                )

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
    filter: str | None = typer.Option(
        None,
        "--filter",
        "-F",
        help="Substring match on task_id. Restricts the per-task charts to tasks whose id contains this substring. Summary charts are not affected.",
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

    if filter:
        typer.secho(
            f"Per-task filter active: only task ids containing '{filter}' will be plotted",
            fg=typer.colors.CYAN,
        )

    from .engine.plotter import generate_and_save_plots

    saved = generate_and_save_plots(metrics, folder, task_filter=filter)

    for path in saved:
        typer.secho(f"  {path.relative_to(folder)}", fg=typer.colors.GREEN)

    typer.secho(
        f"\nSaved {len(saved)} chart(s) to {folder}",
        fg=typer.colors.GREEN,
        bold=True,
    )
