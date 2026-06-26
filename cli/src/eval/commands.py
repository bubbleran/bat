from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import urlparse

import typer
import yaml
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


def _agent_url_from_config(agent_root: Path) -> str:
    """Build the URL the eval connects to from the agent's own ``config.yaml``.

    The agent binds to ``endpoint.url`` + ``endpoint.port`` (see
    ``AgentApplication``); the eval reads the *same* values so it connects
    exactly where the agent listens, instead of overriding them. Missing
    values fall back to the same defaults the agent uses
    (``http://localhost`` / ``9900``).
    """
    config_path = agent_root / "config.yaml"
    data: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
    endpoint = data.get("endpoint") or {}
    url = endpoint.get("url") or "http://localhost"
    port = endpoint.get("port") or 9900
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    return f"{url.rstrip('/')}:{port}"


def _patch_agent_config(
    agent_root: Path, overrides: Mapping[str, Any]
) -> str | None:
    """Merge ``overrides`` into the agent's ``./config.yaml`` for one run.

    The agent reads ``config.yaml`` as the source of truth for telemetry; the
    eval injects per-run values (enable the local file span exporter at the
    per-run path) by patching that file. Returns the original file contents
    (or ``None`` if absent) so the caller can restore it via
    :func:`_restore_agent_config`.
    """
    config_path = agent_root / "config.yaml"
    original = (
        config_path.read_text(encoding="utf-8")
        if config_path.exists()
        else None
    )
    data: dict[str, Any] = {}
    if original is not None:
        loaded = yaml.safe_load(original)
        if isinstance(loaded, dict):
            data = loaded
    for key, value in overrides.items():
        data[key] = value
    config_path.write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    return original


def _restore_agent_config(agent_root: Path, original: str | None) -> None:
    """Restore ``config.yaml`` to the contents captured by ``_patch_agent_config``."""
    config_path = agent_root / "config.yaml"
    if original is None:
        config_path.unlink(missing_ok=True)
    else:
        config_path.write_text(original, encoding="utf-8")


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
    spans_dir: str | None = None,
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
                spans_dir=spans_dir,
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
            "The agent's config.yaml endpoint must resolve to a full URL, "
            "for example: http://127.0.0.1:9900"
        )

    port = parsed.port
    if port is None:
        if parsed.scheme == "https":
            port = 443
        elif parsed.scheme == "http":
            port = 80
        else:
            raise typer.BadParameter(
                "The agent's config.yaml endpoint URL must use http or https"
            )

    base_url = f"{parsed.scheme}://{parsed.hostname}"
    return parsed.hostname, port, base_url


def _wait_for_agent_port(
    agent_url: str,
    timeout_s: int,
    process: subprocess.Popen | None,
) -> None:
    host, port, _ = _parse_agent_url(agent_url)
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        if process is not None and process.poll() is not None:
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


def _start_agent_process(
    agent_root: Path, env: dict[str, str]
) -> subprocess.Popen:
    try:
        return subprocess.Popen(
            ["uv", "run", "."],
            cwd=agent_root,
            env=env,
            # Own session/process group so teardown can signal the whole tree:
            # `uv run .` forks the actual agent server as a child, and signaling
            # only `uv` would orphan that server (leaking its port).
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            "Cannot execute 'uv run .'. Ensure uv is installed and available in PATH."
        ) from exc


def _signal_agent_tree(process: subprocess.Popen, sig: int) -> None:
    """Send ``sig`` to the agent's whole process group, falling back to the
    launched process alone if the group can't be addressed (e.g. non-POSIX)."""
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except (ProcessLookupError, PermissionError, OSError, AttributeError):
        with contextlib.suppress(ProcessLookupError, OSError):
            process.send_signal(sig)


def _stop_agent_process(process: subprocess.Popen, timeout_s: int) -> None:
    if process.poll() is not None:
        return

    _signal_agent_tree(process, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass

    _signal_agent_tree(process, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
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
    judge_model = (
        f"{cfg.judge.provider}:{cfg.judge.model}"
        if cfg.judge is not None
        else "not configured"
    )

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
        raise typer.BadParameter(
            "Missing ./eval/eval.yaml. Run 'bat eval init' first."
        )

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
        raise typer.BadParameter(
            "Missing ./eval/eval.yaml. Run 'bat eval init' first."
        )

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

    # The agent's endpoint is its own config.yaml's responsibility; the eval
    # reads it (rather than patching it) and connects there.
    agent_url = _agent_url_from_config(agent_root)

    failed_models: list[tuple[str, str]] = []

    for idx, model_cfg in enumerate(cfg.models):
        typer.secho(
            f"- {model_cfg.provider}:{model_cfg.model}", fg=typer.colors.CYAN
        )

        # Model selection rides on env vars: the agent lets MODEL /
        # MODEL_PROVIDER / BASE_URL override the config.yaml `model` section, so
        # the eval can sweep models without rewriting config per model.
        server_env = os.environ.copy()
        server_env["MODEL_PROVIDER"] = model_cfg.provider
        server_env["MODEL"] = model_cfg.model

        if model_cfg.base_url:
            server_env["BASE_URL"] = model_cfg.base_url
        else:
            server_env.pop("BASE_URL", None)

        # Per-run telemetry spans directory: the agent writes ``agent.jsonl``
        # here and the eval reads every ``*.jsonl`` in it, grouping spans by
        # trace_id. For multi-agent runs, point each remote sub-agent's local
        # output (telemetry.output[].file_path) at a distinct file in this same
        # directory (the shared trace_id, carried by the propagated W3C
        # traceparent, ties them together).
        spans_dir = (cfg.output_dir / task_id / f"spans-{idx}").resolve()
        spans_dir.mkdir(parents=True, exist_ok=True)
        spans_dir_str = str(spans_dir)

        _apply_env_overrides(
            server_env,
            model_cfg.env,
            section_name=f"models[{idx}]",
        )

        # Telemetry is read from config.yaml (not env), so patch in for this
        # run a single local (JSONL file) span exporter at the per-run path --
        # the eval reconstructs usage/tool-calls from these files and never
        # needs a remote collector. The endpoint is NOT patched: the agent
        # binds to its own config.yaml and the eval reads it. Always restored
        # in the `finally` below, even if the agent fails to start.
        original_config = _patch_agent_config(
            agent_root,
            {
                "telemetry": {
                    "output": [
                        {
                            "type": "local",
                            "file_path": str(spans_dir / "agent.jsonl"),
                        }
                    ],
                },
            },
        )

        process: subprocess.Popen | None = None
        try:
            process = _start_agent_process(agent_root, server_env)
            _wait_for_agent_port(
                agent_url,
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

            _run_eval_orchestrator(
                agent_url=agent_url,
                model_provider=model_cfg.provider,
                model=model_cfg.model,
                dataset=cfg.dataset,
                output_dir=cfg.output_dir,
                task_id=task_id,
                k=cfg.k,
                run_name=cfg.run_name,
                qualitative=cfg.qualitative,
                env=runner_env,
                spans_dir=spans_dir_str,
            )
        except Exception as exc:
            # One model's failure (startup timeout, orchestrator error, ...)
            # must not abort the whole sweep: record it and move on. Teardown
            # still runs in the `finally` below. KeyboardInterrupt is a
            # BaseException, so Ctrl-C still stops the run.
            label = f"{model_cfg.provider}:{model_cfg.model}"
            failed_models.append((label, str(exc)))
            typer.secho(
                f"  Model {label} failed: {exc}",
                fg=typer.colors.RED,
                err=True,
            )
        finally:
            if process is not None:
                _stop_agent_process(
                    process, timeout_s=cfg.agent_shutdown_timeout_s
                )
            _restore_agent_config(agent_root, original_config)

    completed = len(cfg.models) - len(failed_models)
    output_path = cfg.output_dir / task_id
    if failed_models:
        typer.secho(
            f"Evaluation finished: {completed}/{len(cfg.models)} model(s) "
            f"completed, {len(failed_models)} failed. Output: {output_path}",
            fg=typer.colors.YELLOW,
        )
        for label, err in failed_models:
            typer.secho(f"  - {label}: {err}", fg=typer.colors.RED, err=True)
        if completed == 0:
            raise typer.Exit(code=1)
    else:
        typer.secho(
            f"Evaluation completed. Output: {output_path}",
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
