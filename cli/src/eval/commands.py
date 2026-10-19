from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from string import Template
from unittest.mock import patch
from urllib.parse import urlparse

import typer
import yaml
from dotenv import dotenv_values

from create.agent import PROVIDER_API_KEY_VAR
from project import (
    AgentTarget,
    ProjectError,
    fail,
    privacy_floors,
    resolve_agent_target,
    unwired_agent_warning,
)

from .engine.contracts import EvalConfig, JudgeSpec
from .engine.eval_config import (
    DEFAULT_EVAL_YAML,
    DEFAULT_TASKS_JSON,
    load_eval_config,
)
from .engine.orchestrator import load_tasks, run_evaluation

_AGENT_ARGUMENT = typer.Argument(
    None,
    metavar="[AGENT]",
    help=(
        "Which agent of the enclosing blueprint to act on, so the command "
        "can be run from the blueprint root. Omit it to use the agent "
        "directory you are in (the only form a standalone agent has)."
    ),
)


def _resolve_target(agent: str | None = None) -> AgentTarget:
    try:
        return resolve_agent_target(Path.cwd(), agent)
    except ProjectError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_config(target: AgentTarget) -> EvalConfig:
    path = target.agent_dir / "eval" / "eval.yaml"
    if not path.exists():
        raise typer.BadParameter(
            "Missing ./eval/eval.yaml. Run 'bat eval init' first."
        )
    try:
        return load_eval_config(target.agent_dir, path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _refuse_redacted_spans(target: AgentTarget) -> None:
    """Stop when the agent's privacy floor redacts the spans the eval reads."""
    name = target.agent_dir.name
    for floor in privacy_floors(target):
        if floor.level is None:
            typer.secho(
                "Warning: can't read the telemetry privacy floor at "
                f"{floor.location}.",
                fg=typer.colors.YELLOW,
                err=True,
            )
        elif floor.level != "none":
            fail(
                f"Can't evaluate {name}: its telemetry privacy floor is "
                f'"{floor.level}" ({floor.location}), which redacts the spans '
                'the eval reads. Set telemetry_privacy_floor="none" there '
                "while you evaluate."
            )


def _agent_url(config_path: Path) -> str:
    """Where the agent listens, its config.yaml's endpoint read as the ADK
    reads it."""
    from bat.agent.config import EndpointConfig

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    try:
        endpoint = EndpointConfig(**(config.get("endpoint") or {}))
    except ValueError as exc:
        raise typer.BadParameter(f"{config_path}: {exc}") from exc
    url = (endpoint.url or "http://localhost").rstrip("/")
    if "://" not in url:
        url = "http://" + url
    return f"{url}:{endpoint.port or 9900}"


@contextlib.contextmanager
def _spans_written_to(config_path: Path, spans_file: Path):
    """The agent's config.yaml, for the run, exporting its spans to
    spans_file and nowhere else."""
    original = config_path.read_text(encoding="utf-8")
    config = yaml.safe_load(original) or {}
    config["telemetry"] = {
        "output": [{"type": "local", "file_path": str(spans_file)}]
    }
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    try:
        yield
    finally:
        config_path.write_text(original, encoding="utf-8")


def _with_vars(env: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    """env plus extra, whose values may refer to env as $VAR or ${VAR}."""
    return env | {
        key: Template(value).safe_substitute(env)
        for key, value in extra.items()
    }


def _judge_env(judge: JudgeSpec, env: dict[str, str]) -> dict[str, str]:
    """env, with the judge's API key where its provider reads it, and the
    judge's own variables."""
    env = dict(env)
    key_var = PROVIDER_API_KEY_VAR.get(judge.provider)
    if judge.api_key_env and key_var:
        if env.get(judge.api_key_env):
            env[key_var] = env[judge.api_key_env]
        else:
            typer.secho(
                f"Warning: judge.api_key_env='{judge.api_key_env}' is not set "
                "in the shell or the project's .env; the judge will likely "
                "fail.",
                fg=typer.colors.YELLOW,
                err=True,
            )
    return _with_vars(env, judge.env)


def _wait_for_agent_port(
    agent_url: str, timeout_s: int, process: subprocess.Popen
) -> None:
    url = urlparse(agent_url)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise typer.BadParameter(
                "Agent process exited before becoming ready "
                f"(exit code: {process.returncode})."
            )
        try:
            socket.create_connection(
                (url.hostname, url.port or 80), 1.0
            ).close()
            return
        except OSError:
            time.sleep(0.2)
    raise typer.BadParameter(
        f"Agent did not become ready at {agent_url} within {timeout_s} seconds."
    )


def _why_it_stopped(target: AgentTarget, code: int, log_path: Path) -> str:
    """The agent's own last words, from its log, and what to do about a
    remote agent or MCP server it requires that doesn't answer."""
    log = log_path.read_text(encoding="utf-8", errors="replace").strip()
    last = log.splitlines()[-1].strip() if log else ""
    name = target.agent_dir.name
    reason = f"{name} stopped before it was ready (exit code {code})"
    if last:
        reason += f":\n    {last}"
    if "agent card" in last or "MCP server" in last:
        config = target.config_path.relative_to(target.project_root)
        reason += (
            "\n  It can't reach a remote agent or MCP server it requires: "
            f"start it first, or mark it `required: false` in {config}."
        )
    return reason + f"\n  Agent log: {log_path}"


def _stop_agent_process(process: subprocess.Popen, timeout_s: int) -> None:
    """SIGTERM the agent's process group, then SIGKILL it if it lingers:
    `uv run` forks the server, so signaling uv alone would orphan it."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if process.poll() is not None:
            return
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(process.pid), sig)
        try:
            process.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass


def eval_init(
    agent: str | None = _AGENT_ARGUMENT,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite eval/eval.yaml and eval/input/tasks.json if they already exist.",
    ),
) -> None:
    eval_dir = _resolve_target(agent).agent_dir / "eval"
    (eval_dir / "input").mkdir(parents=True, exist_ok=True)
    (eval_dir / "output").mkdir(parents=True, exist_ok=True)
    for path, content in (
        (eval_dir / "eval.yaml", DEFAULT_EVAL_YAML),
        (eval_dir / "input" / "tasks.json", DEFAULT_TASKS_JSON),
    ):
        if path.exists() and not force:
            typer.secho(
                f"{path} already exists. Use --force to overwrite.",
                fg=typer.colors.YELLOW,
            )
        else:
            path.write_text(content, encoding="utf-8")
            typer.secho(f"Created {path}", fg=typer.colors.GREEN)
    typer.secho(
        f"Evaluation scaffold ready in {eval_dir}", fg=typer.colors.GREEN
    )


def eval_show(agent: str | None = _AGENT_ARGUMENT) -> None:
    cfg = _load_config(_resolve_target(agent))
    judge = (
        f"{cfg.judge.provider}:{cfg.judge.model}"
        if cfg.judge
        else "not configured"
    )
    rule = "============================"
    typer.secho(rule, fg=typer.colors.BLUE)
    typer.secho("  EVALUATION CONFIGURATION", fg=typer.colors.BLUE, bold=True)
    typer.secho(rule, fg=typer.colors.BLUE)
    typer.echo(f"Dataset     : {cfg.dataset}")
    typer.echo(f"k           : {cfg.k}")
    typer.echo(f"Qualitative : {'yes' if cfg.qualitative else 'no'}")
    typer.echo("\nModels:")
    for index, model in enumerate(cfg.models, start=1):
        typer.echo(f"  [{index}] {model.provider}:{model.model}")
    typer.echo(f"\nJudge model : {judge}")
    typer.secho(rule, fg=typer.colors.BLUE)


def eval_run(agent: str | None = _AGENT_ARGUMENT) -> None:
    target = _resolve_target(agent)
    warning = unwired_agent_warning(target)
    if warning is not None:
        typer.secho(f"Warning: {warning}", fg=typer.colors.YELLOW, err=True)
    _refuse_redacted_spans(target)
    cfg = _load_config(target)
    if not cfg.dataset.exists():
        raise typer.BadParameter(f"Dataset not found: {cfg.dataset}")
    try:
        tasks = load_tasks(cfg.dataset)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.output_dir / run_id
    agent_url = _agent_url(target.config_path)
    judge = cfg.judge if cfg.qualitative else None
    # As under `make <agent>`: the project's .env, the shell winning.
    dotenv = dotenv_values(target.project_root / ".env")
    env = {k: v for k, v in dotenv.items() if v is not None} | dict(os.environ)
    judge_env = _judge_env(judge, env) if judge else env

    typer.secho(
        f"Running evaluation with {len(cfg.models)} model(s). task_id={run_id}",
        fg=typer.colors.CYAN,
    )
    failed: list[str] = []
    for index, model in enumerate(cfg.models):
        label = f"{model.provider}:{model.model}"
        typer.secho(f"- {label}", fg=typer.colors.CYAN)
        # The agent reads MODEL / MODEL_PROVIDER / BASE_URL over its config.
        agent_env = {
            **env,
            "MODEL_PROVIDER": model.provider,
            "MODEL": model.model,
            # Its log in the order printed: stdout buffered after stderr
            # buries the error under the startup banner.
            "PYTHONUNBUFFERED": "1",
        }
        agent_env.pop("BASE_URL", None)
        if model.base_url:
            agent_env["BASE_URL"] = model.base_url
        agent_env = _with_vars(agent_env, model.env) | target.run_env
        spans_dir = out_dir / f"spans-{index}"
        spans_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"agent-{index}.log"
        process = None
        try:
            with (
                _spans_written_to(
                    target.config_path, spans_dir / "agent.jsonl"
                ),
                log_path.open("w", encoding="utf-8") as log,
            ):
                # Its own process group, so teardown reaches the forked server.
                process = subprocess.Popen(
                    target.run_command,
                    cwd=target.project_root,
                    env=agent_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                _wait_for_agent_port(
                    agent_url, cfg.agent_startup_timeout_s, process
                )
                with patch.dict(os.environ, judge_env):
                    asyncio.run(
                        run_evaluation(
                            agent_url=agent_url,
                            model=label,
                            tasks=tasks,
                            out_dir=out_dir,
                            run_name=cfg.run_name,
                            k=cfg.k,
                            span_paths=[
                                str(spans_dir),
                                *map(str, cfg.extra_spans),
                            ],
                            judge=judge,
                        )
                    )
        except Exception as exc:
            # One model failing must not abort the sweep; Ctrl-C still does.
            reason = str(exc)
            if process is not None and process.poll() is not None:
                reason = _why_it_stopped(target, process.returncode, log_path)
            failed.append(label)
            typer.secho(
                f"  Model {label} failed: {reason}",
                fg=typer.colors.RED,
                err=True,
            )
        finally:
            if process is not None:
                _stop_agent_process(process, cfg.agent_shutdown_timeout_s)

    if not failed:
        typer.secho(
            f"Evaluation completed. Output: {out_dir}", fg=typer.colors.GREEN
        )
        return
    done = len(cfg.models) - len(failed)
    typer.secho(
        f"Evaluation finished: {done}/{len(cfg.models)} model(s) completed, "
        f"{len(failed)} failed ({', '.join(failed)}). Output: {out_dir}",
        fg=typer.colors.YELLOW,
    )
    if done == 0:
        raise typer.Exit(code=1)


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
    metrics = {
        run.name: json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        for run in sorted(folder.iterdir())
        if (run / "metrics.json").is_file()
    }
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
