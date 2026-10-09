import asyncio
import json
import os
import signal
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from cli import app
from eval.engine.eval_config import (
    EvalConfig,
    JudgeSpec,
    ModelSpec,
    load_eval_config,
)

runner = CliRunner()


def _write_minimal_agent_root(root: Path) -> None:
    (root / "config.yaml").write_text(
        "name: test\nendpoint:\n  url: http://127.0.0.1\n  port: 9900\n",
        encoding="utf-8",
    )
    (root / "agent.json").write_text("{}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='0.1.0'\n", encoding="utf-8"
    )


def test_eval_init_requires_agent_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["eval", "init"])

    assert result.exit_code != 0
    assert "does not look like an agent root" in result.output


def test_eval_init_creates_scaffold(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    result = runner.invoke(app, ["eval", "init"])

    assert result.exit_code == 0
    assert (root / "eval" / "eval.yaml").exists()
    assert (root / "eval" / "input" / "tasks.json").exists()
    assert (root / "eval" / "output").is_dir()


def test_eval_run_requires_eval_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    result = runner.invoke(app, ["eval", "run"])

    assert result.exit_code != 0
    assert "Missing ./eval/eval.yaml" in result.output


def test_eval_show_requires_eval_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    result = runner.invoke(app, ["eval", "show"])

    assert result.exit_code != 0
    assert "Missing ./eval/eval.yaml" in result.output


def test_eval_show_prints_resolved_config(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)
    (root / "eval").mkdir()
    (root / "eval" / "eval.yaml").write_text(
        "evaluation:\n"
        "  k: 2\n"
        "  qualitative: true\n"
        "judge:\n"
        "  provider: ollama\n"
        "  model: judge-model\n"
        "models:\n"
        "  - provider: openai\n"
        "    model: gpt-4.1-mini\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval", "show"])

    assert result.exit_code == 0
    dataset = root / "eval" / "input" / "tasks.json"
    assert "EVALUATION CONFIGURATION" in result.output
    assert f"Dataset     : {dataset.resolve()}" in result.output
    assert "k           : 2" in result.output
    assert "Qualitative : yes" in result.output
    assert "Models:" in result.output
    assert "  [1] openai:gpt-4.1-mini" in result.output
    assert "Judge model : ollama:judge-model" in result.output


def test_load_eval_config_allows_missing_judge_when_qualitative_is_false(
    tmp_path,
) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n"
        "  qualitative: false\n"
        "models:\n"
        "  - provider: openai\n"
        "    model: gpt-4.1-mini\n",
        encoding="utf-8",
    )

    config = load_eval_config(tmp_path, eval_yaml)

    assert config.qualitative is False
    assert config.judge is None


def test_load_eval_config_requires_judge_when_qualitative_is_true(
    tmp_path,
) -> None:
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n"
        "  qualitative: true\n"
        "models:\n"
        "  - provider: openai\n"
        "    model: gpt-4.1-mini\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="judge.provider and judge.model"):
        load_eval_config(tmp_path, eval_yaml)


def test_eval_run_starts_agent_and_runs_orchestrator(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    eval_input = root / "eval" / "input"
    eval_output = root / "eval" / "output"
    eval_input.mkdir(parents=True, exist_ok=True)

    eval_yaml = root / "eval" / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n  dataset: eval/input/tasks.json\n", encoding="utf-8"
    )

    dataset = eval_input / "tasks.json"
    dataset.write_text("[]\n", encoding="utf-8")

    config = EvalConfig(
        dataset=dataset.resolve(),
        output_dir=eval_output.resolve(),
        agent_startup_timeout_s=15,
        agent_shutdown_timeout_s=5,
        k=2,
        qualitative=True,
        run_name="benchmark",
        models=[
            ModelSpec(
                provider="openai",
                model="gpt-4.1-mini",
                base_url="http://model.local",
                env={
                    "EXTRA_FLAG": "enabled",
                    "MODEL_ALIAS": "$MODEL",
                },
            )
        ],
        judge=JudgeSpec(
            provider="ollama",
            model="judge-model",
            base_url="http://judge.local",
            env={"JUDGE_MODE": "strict"},
        ),
    )

    captured: dict[str, object] = {}

    def fake_load_eval_config(
        agent_root: Path, config_path: Path
    ) -> EvalConfig:
        assert agent_root == root
        assert config_path == eval_yaml
        return config

    def fake_strftime(_fmt: str) -> str:
        return "20260101_000000"

    class _FakeProcess:
        pid = 12345

        def __init__(self) -> None:
            self.returncode = None

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

    def fake_popen(cmd, cwd, env, **kwargs):
        captured["popen_cmd"] = cmd
        captured["popen_cwd"] = cwd
        captured["popen_kwargs"] = kwargs
        captured["popen_env"] = env
        return _FakeProcess()

    def fake_wait_for_agent_port(agent_url: str, timeout_s: int, process):
        captured["wait_agent_url"] = agent_url
        captured["wait_timeout_s"] = timeout_s
        assert process is not None

    async def fake_run_evaluation(**kwargs):
        captured["runner_kwargs"] = kwargs
        captured["judge_mode"] = os.environ.get("JUDGE_MODE")

    monkeypatch.setattr("eval.commands.load_eval_config", fake_load_eval_config)
    monkeypatch.setattr("eval.commands.time.strftime", fake_strftime)
    monkeypatch.setattr(
        "eval.commands._wait_for_agent_port", fake_wait_for_agent_port
    )
    monkeypatch.setattr("eval.commands.subprocess.Popen", fake_popen)
    # Teardown signals the process group; keep it off the real OS in tests.
    monkeypatch.setattr("eval.commands.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("eval.commands.os.killpg", lambda pgid, sig: None)
    monkeypatch.setattr("eval.commands.run_evaluation", fake_run_evaluation)

    result = runner.invoke(app, ["eval", "run"])

    assert result.exit_code == 0, result.output

    popen_cmd = captured["popen_cmd"]
    assert popen_cmd == ["uv", "run", "."]
    assert captured["popen_cwd"] == root
    # S4-C1: agent launched in its own session so teardown can kill the group.
    assert captured["popen_kwargs"].get("start_new_session") is True

    popen_env = captured["popen_env"]
    assert popen_env["MODEL_PROVIDER"] == "openai"
    assert popen_env["MODEL"] == "gpt-4.1-mini"
    assert popen_env["BASE_URL"] == "http://model.local"
    assert popen_env["MODEL_ALIAS"] == "gpt-4.1-mini"

    # agent_url is derived from the agent's config.yaml endpoint, not from
    # the eval config or env vars.
    assert captured["wait_agent_url"] == "http://127.0.0.1:9900"
    assert captured["wait_timeout_s"] == 15

    runner_kwargs = captured["runner_kwargs"]
    assert runner_kwargs["agent_url"] == "http://127.0.0.1:9900"
    assert runner_kwargs["model"] == "openai:gpt-4.1-mini"
    assert runner_kwargs["tasks"] == []
    assert runner_kwargs["out_dir"] == eval_output.resolve() / "20260101_000000"
    assert runner_kwargs["k"] == 2
    assert runner_kwargs["run_name"] == "benchmark"
    assert runner_kwargs["judge"] == config.judge
    # The judge runs in this process, with its own variables set.
    assert captured["judge_mode"] == "strict"
    assert "JUDGE_MODE" not in os.environ


def _write_eval_yaml_with_prompts(root: Path, prompts_block: str) -> Path:
    eval_yaml = root / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n"
        "  qualitative: true\n"
        "judge:\n"
        "  provider: openai\n"
        "  model: gpt-4.1-mini\n" + prompts_block + "models:\n"
        "  - provider: openai\n"
        "    model: gpt-4.1-mini\n",
        encoding="utf-8",
    )
    return eval_yaml


def test_judge_prompts_parsed(tmp_path) -> None:
    eval_yaml = _write_eval_yaml_with_prompts(
        tmp_path,
        "  prompts:\n"
        "    relevance: tune relevance\n"
        "    task_completion: tune completion\n"
        "    hallucination: tune hallucination\n"
        "    tool_call: tune tool calls\n",
    )

    config = load_eval_config(tmp_path, eval_yaml)

    assert config.judge is not None
    assert config.judge.prompts == {
        "relevance": "tune relevance",
        "task_completion": "tune completion",
        "hallucination": "tune hallucination",
        "tool_call": "tune tool calls",
    }


def test_judge_prompts_partial_ok(tmp_path) -> None:
    eval_yaml = _write_eval_yaml_with_prompts(
        tmp_path,
        "  prompts:\n    relevance: only relevance set\n",
    )

    config = load_eval_config(tmp_path, eval_yaml)

    assert config.judge is not None
    assert config.judge.prompts == {"relevance": "only relevance set"}


def test_judge_prompts_overflow_errors(tmp_path) -> None:
    overflow = "x" * 1001
    eval_yaml = _write_eval_yaml_with_prompts(
        tmp_path,
        f"  prompts:\n    task_completion: {overflow}\n",
    )

    with pytest.raises(ValueError) as exc:
        load_eval_config(tmp_path, eval_yaml)

    message = str(exc.value)
    assert "judge.prompts.task_completion" in message
    assert "1000-character limit" in message
    assert "got 1001" in message


def test_judge_prompts_unknown_key_errors(tmp_path) -> None:
    eval_yaml = _write_eval_yaml_with_prompts(
        tmp_path,
        "  prompts:\n    bogus: nope\n",
    )

    with pytest.raises(ValueError, match="bogus.*relevance"):
        load_eval_config(tmp_path, eval_yaml)


def test_judge_prompts_absent_is_empty_dict(tmp_path) -> None:
    eval_yaml = _write_eval_yaml_with_prompts(tmp_path, "")

    config = load_eval_config(tmp_path, eval_yaml)

    assert config.judge is not None
    assert config.judge.prompts == {}


def _write_run_metrics(run_dir: Path, task_ids: list[str]) -> None:
    run_dir.mkdir(parents=True)
    per_episode = [
        {
            "task_id": tid,
            "status": "completed",
            "success": True,
            "time": {"wall_ms": 500},
            "tokens": {
                "prompt_tokens": 50,
                "completion_tokens": 25,
                "total_tokens": 75,
            },
        }
        for tid in task_ids
    ]
    metrics = {
        "summary": {
            "time": {"total_wall_ms": 500 * len(task_ids)},
            "tokens": {
                "prompt_tokens_total": 50 * len(task_ids),
                "completion_tokens_total": 25 * len(task_ids),
                "total_tokens_total": 75 * len(task_ids),
            },
        },
        "per_episode": per_episode,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_eval_plot_filter_restricts_per_task_charts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = Path.cwd() / "out"
    _write_run_metrics(out / "run_a", ["foo_alpha", "bar_beta", "foo_gamma"])

    result = runner.invoke(
        app, ["eval", "plot", "--folder", str(out), "--filter", "foo"]
    )

    assert result.exit_code == 0, result.output
    assert "Per-task filter active" in result.output

    png_names = {p.name for p in out.iterdir() if p.suffix == ".png"}
    assert "metrics_per_task_foo_alpha.png" in png_names
    assert "metrics_per_task_foo_gamma.png" in png_names
    assert "metrics_per_task_bar_beta.png" not in png_names


def test_eval_plot_without_filter_keeps_all_per_task_charts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = Path.cwd() / "out"
    _write_run_metrics(out / "run_a", ["foo_alpha", "bar_beta"])

    result = runner.invoke(app, ["eval", "plot", "--folder", str(out)])

    assert result.exit_code == 0, result.output
    assert "Per-task filter active" not in result.output
    png_names = {p.name for p in out.iterdir() if p.suffix == ".png"}
    assert "metrics_per_task_foo_alpha.png" in png_names
    assert "metrics_per_task_bar_beta.png" in png_names


def _write_qualitative_run_metrics(run_dir: Path, task_id: str) -> None:
    """A single-attempt run whose qualitative scores are all null.

    Mirrors a real metrics.json where the LLM judge failed/returned no score:
    the qualitative block exists but its values are ``None``. With k=1 the
    plotter passes the raw episode straight to the bar chart (EV-C1).
    """
    run_dir.mkdir(parents=True)
    metrics = {
        "summary": {
            "time": {"total_wall_ms": 500},
            "tokens": {
                "prompt_tokens_total": 50,
                "completion_tokens_total": 25,
                "total_tokens_total": 75,
            },
            "qualitative": {
                "response_relevance": {"avg": None},
                "task_completion_quality": {"avg": None},
                "hallucination_score": {"avg": None},
            },
        },
        "per_episode": [
            {
                "task_id": task_id,
                "status": "completed",
                "success": True,
                "time": {"wall_ms": 500},
                "tokens": {
                    "prompt_tokens": 50,
                    "completion_tokens": 25,
                    "total_tokens": 75,
                },
                "qualitative": {
                    "response_relevance": None,
                    "task_completion_quality": None,
                    "hallucination_score": None,
                },
            }
        ],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_eval_plot_survives_null_qualitative_scores(
    tmp_path, monkeypatch
) -> None:
    """EV-C1: a null qualitative score must not crash `bat eval plot`."""
    monkeypatch.chdir(tmp_path)
    out = Path.cwd() / "out"
    _write_qualitative_run_metrics(out / "run_a", "foo_alpha")

    result = runner.invoke(app, ["eval", "plot", "--folder", str(out)])

    assert result.exit_code == 0, result.output
    png_names = {p.name for p in out.iterdir() if p.suffix == ".png"}
    assert "metrics_per_task_foo_alpha.png" in png_names
    assert "metrics_qualitative_metrics.png" in png_names


def test_a_failed_judge_leaves_its_error_in_the_reasoning(monkeypatch) -> None:
    """EV-M1: a failed judge must leave its error in judge_reasoning."""
    from eval.engine import judge
    from eval.engine.contracts import EpisodeResult, JudgeSpec, TaskSpec

    def ask(self, rubric, prompt):
        if rubric == "relevance":
            return {"reasoning": "Error: bad api key", "score": None}
        return {"reasoning": "looks good", "score": 0.75}

    monkeypatch.setattr(judge.Judge, "ask", ask)
    episode = EpisodeResult(
        task_id="t", final_status="completed", final_output="ok"
    )

    judge.score(
        JudgeSpec(provider="openai", model="m"),
        [episode],
        {"t": TaskSpec(id="t", turns=["hi"])},
    )

    scores = episode.qualitative_scores
    assert scores.response_relevance is None
    assert scores.judge_reasoning["relevance"] == "Error: bad api key"
    assert scores.task_completion_quality == 0.75
    assert scores.judge_reasoning["tool_call"].startswith("skipped")


def test_load_eval_config_rejects_unsupported_provider(tmp_path) -> None:
    """S4-M5: a provider the adk client can't accept fails at config load."""
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n"
        "  qualitative: false\n"
        "models:\n"
        "  - provider: groq\n"
        "    model: llama-3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="groq.*Valid providers"):
        load_eval_config(tmp_path, eval_yaml)


def test_load_eval_config_rejects_unsupported_judge_provider(tmp_path) -> None:
    """S4-M5: judge provider is validated too."""
    eval_yaml = tmp_path / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n"
        "  qualitative: true\n"
        "judge:\n"
        "  provider: azure\n"
        "  model: gpt-4o\n"
        "models:\n"
        "  - provider: openai\n"
        "    model: gpt-4.1-mini\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="azure"):
        load_eval_config(tmp_path, eval_yaml)


def test_judge_client_does_not_inherit_model_base_url(monkeypatch) -> None:
    """S4-C2: the judge's base_url is its own, never the model's BASE_URL."""
    from eval.engine import judge
    from eval.engine.contracts import JudgeSpec

    captured: dict[str, object] = {}

    class _FakeConfig:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(judge, "ChatModelClientConfig", _FakeConfig)
    monkeypatch.setattr(judge, "ChatModelClient", lambda **kwargs: None)
    monkeypatch.setenv("BASE_URL", "http://model-under-test:11434")

    judge.Judge(JudgeSpec(provider="openai", model="m")).client(
        "relevance", True
    )

    assert captured["base_url"] is None


def test_stop_agent_process_signals_whole_group(monkeypatch) -> None:
    """S4-C1: teardown signals the agent's process group, not just `uv`."""
    from eval import commands as cmd

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(cmd.os, "getpgid", lambda pid: 4242)
    monkeypatch.setattr(
        cmd.os, "killpg", lambda pgid, sig: signals.append((pgid, sig))
    )

    class _FakeProc:
        pid = 999

        def poll(self) -> None:
            return None

        def wait(self, timeout=None) -> int:
            return 0

    cmd._stop_agent_process(_FakeProc(), timeout_s=1)

    assert (4242, signal.SIGTERM) in signals


def test_agent_url_reads_the_endpoint_as_the_adk_does(tmp_path) -> None:
    """A port written in endpoint.url is the port; two that disagree are
    refused, as the agent itself would refuse to start."""
    from eval import commands as cmd

    config = tmp_path / "config.yaml"
    config.write_text("endpoint:\n  url: localhost:9300\n", encoding="utf-8")
    assert cmd._agent_url(config) == "http://localhost:9300"

    config.write_text(
        "endpoint:\n  url: http://localhost:9300\n  port: 9301\n",
        encoding="utf-8",
    )
    with pytest.raises(typer.BadParameter, match="Conflicting ports"):
        cmd._agent_url(config)


def test_one_failing_task_does_not_stop_the_run(tmp_path, monkeypatch) -> None:
    """S4-M6: one task raising is recorded as an error, run still finishes."""
    from eval.engine import orchestrator
    from eval.engine.contracts import EpisodeResult, TaskSpec

    async def run_task(agent_url, task, thread_id, span_paths):
        if task.id == "boom":
            raise RuntimeError("kaboom")
        return EpisodeResult(
            task_id=task.id, final_status="completed", final_output="ok"
        )

    monkeypatch.setattr(orchestrator, "run_task", run_task)

    results = asyncio.run(
        orchestrator.run_evaluation(
            agent_url="http://agent",
            model="openai:m",
            tasks=[
                TaskSpec(id="boom", turns=["x"]),
                TaskSpec(id="fine", turns=["y"]),
            ],
            out_dir=tmp_path,
            run_name="r",
            k=1,
            span_paths=[],
            judge=None,
        )
    )

    by_id = {r.task_id: r for r in results}
    assert by_id["boom"].final_status == "error"
    assert by_id["boom"].verdict.passed is False
    assert "kaboom" in by_id["boom"].aux["error"]
    assert by_id["fine"].final_status == "completed"
    assert (tmp_path / "r_openai-m" / "metrics.json").is_file()


def test_eval_run_continues_after_one_model_fails(
    tmp_path, monkeypatch
) -> None:
    """S4-M1: one model failing must not abort the rest of the sweep."""
    root = tmp_path
    _write_minimal_agent_root(root)
    eval_input = root / "eval" / "input"
    eval_input.mkdir(parents=True)
    (eval_input / "tasks.json").write_text("[]\n", encoding="utf-8")
    eval_yaml = root / "eval" / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n  dataset: eval/input/tasks.json\n", encoding="utf-8"
    )

    config = EvalConfig(
        dataset=(eval_input / "tasks.json").resolve(),
        output_dir=(root / "eval" / "output").resolve(),
        agent_startup_timeout_s=5,
        agent_shutdown_timeout_s=5,
        k=1,
        qualitative=False,
        run_name="benchmark",
        models=[
            ModelSpec(provider="openai", model="m1"),
            ModelSpec(provider="openai", model="m2"),
        ],
        judge=None,
    )

    ran_models: list[str] = []
    wait_calls = {"n": 0}

    class _FakeProcess:
        pid = 222

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    def fake_wait(agent_url, timeout_s, process):
        wait_calls["n"] += 1
        if wait_calls["n"] == 1:
            raise RuntimeError("agent did not start")

    monkeypatch.setattr("eval.commands.load_eval_config", lambda r, c: config)
    monkeypatch.setattr("eval.commands.time.strftime", lambda fmt: "T0")
    monkeypatch.setattr(
        "eval.commands.subprocess.Popen",
        lambda *a, **k: _FakeProcess(),
    )
    monkeypatch.setattr("eval.commands.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("eval.commands.os.killpg", lambda pgid, sig: None)
    monkeypatch.setattr("eval.commands._wait_for_agent_port", fake_wait)

    async def fake_run_evaluation(**kw):
        ran_models.append(kw["model"])

    monkeypatch.setattr("eval.commands.run_evaluation", fake_run_evaluation)

    monkeypatch.chdir(root)
    result = runner.invoke(app, ["eval", "run"])

    # First model failed, but the sweep continued and ran the second.
    assert result.exit_code == 0, result.output
    assert ran_models == ["openai:m2"]
    assert "1 failed" in result.output
    assert "openai:m1" in result.output
