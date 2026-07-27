import json
from pathlib import Path

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
    (root / "config.yaml").write_text("name: test\n", encoding="utf-8")
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


def test_eval_show_prints_resolved_config(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    eval_input = root / "eval" / "input"
    eval_output = root / "eval" / "output"
    eval_input.mkdir(parents=True, exist_ok=True)
    eval_output.mkdir(parents=True, exist_ok=True)

    eval_yaml = root / "eval" / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n  dataset: eval/input/tasks.json\n", encoding="utf-8"
    )

    dataset = eval_input / "tasks.json"
    dataset.write_text("[]\n", encoding="utf-8")

    config = EvalConfig(
        dataset=dataset.resolve(),
        output_dir=eval_output.resolve(),
        agent_url="http://127.0.0.1:9900",
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
                env={"EXTRA_FLAG": "enabled"},
            )
        ],
        judge=JudgeSpec(
            provider="ollama",
            model="judge-model",
            base_url="http://judge.local",
            env={"JUDGE_MODE": "strict"},
        ),
    )

    def fake_load_eval_config(
        agent_root: Path, config_path: Path
    ) -> EvalConfig:
        assert agent_root == root
        assert config_path == eval_yaml
        return config

    monkeypatch.setattr("eval.commands.load_eval_config", fake_load_eval_config)

    result = runner.invoke(app, ["eval", "show"])

    assert result.exit_code == 0

    assert "EVALUATION CONFIGURATION" in result.output
    assert f"Dataset     : {dataset.resolve()}" in result.output
    assert "k           : 2" in result.output
    assert "Qualitative : yes" in result.output
    assert "Models:" in result.output
    assert "  [1] openai:gpt-4.1-mini" in result.output
    assert "Judge model : ollama:judge-model" in result.output


def test_load_eval_config_allows_missing_judge_when_qualitative_is_false(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    eval_input = root / "eval" / "input"
    eval_input.mkdir(parents=True)
    dataset = eval_input / "tasks.json"
    dataset.write_text("[]\n", encoding="utf-8")

    eval_yaml = root / "eval" / "eval.yaml"
    eval_yaml.parent.mkdir(exist_ok=True)
    eval_yaml.write_text(
        "evaluation:\n"
        "  qualitative: false\n"
        "models:\n"
        "  - provider: openai\n"
        "    model: gpt-4.1-mini\n",
        encoding="utf-8",
    )

    config = load_eval_config(root, eval_yaml)

    assert config.qualitative is False
    assert config.judge is None


def test_load_eval_config_requires_judge_when_qualitative_is_true(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    eval_input = root / "eval" / "input"
    eval_input.mkdir(parents=True)
    dataset = eval_input / "tasks.json"
    dataset.write_text("[]\n", encoding="utf-8")

    eval_yaml = root / "eval" / "eval.yaml"
    eval_yaml.parent.mkdir(exist_ok=True)
    eval_yaml.write_text(
        "evaluation:\n"
        "  qualitative: true\n"
        "models:\n"
        "  - provider: openai\n"
        "    model: gpt-4.1-mini\n",
        encoding="utf-8",
    )

    try:
        load_eval_config(root, eval_yaml)
    except ValueError as exc:
        assert "judge.provider and judge.model" in str(exc)
    else:
        raise AssertionError("Expected missing qualitative judge to fail")


def test_eval_run_starts_agent_and_runs_orchestrator(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    eval_input = root / "eval" / "input"
    eval_output = root / "eval" / "output"
    eval_input.mkdir(parents=True, exist_ok=True)
    eval_output.mkdir(parents=True, exist_ok=True)

    eval_yaml = root / "eval" / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n  dataset: eval/input/tasks.json\n", encoding="utf-8"
    )

    dataset = eval_input / "tasks.json"
    dataset.write_text("[]\n", encoding="utf-8")

    config = EvalConfig(
        dataset=dataset.resolve(),
        output_dir=eval_output.resolve(),
        agent_url="http://127.0.0.1:9900",
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

    def fake_find_agent_python(agent_root: Path) -> Path:
        assert agent_root == root
        return Path("/tmp/agent/.venv/bin/python")

    def fake_strftime(_fmt: str) -> str:
        return "20260101_000000"

    class _FakeProcess:
        def __init__(self) -> None:
            self.returncode = None

        def poll(self):  # noqa: ANN001
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def wait(self, timeout=None):  # noqa: ANN001
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def fake_popen(cmd, cwd, env):  # noqa: ANN001
        captured["popen_cmd"] = cmd
        captured["popen_cwd"] = cwd
        captured["popen_env"] = env
        return _FakeProcess()

    def fake_wait_for_agent_port(agent_url: str, timeout_s: int, process):  # noqa: ANN001
        captured["wait_agent_url"] = agent_url
        captured["wait_timeout_s"] = timeout_s
        assert process is not None

    def fake_run_eval_orchestrator(**kwargs):  # noqa: ANN003
        captured["runner_kwargs"] = kwargs

    monkeypatch.setattr("eval.commands.load_eval_config", fake_load_eval_config)
    monkeypatch.setattr(
        "eval.commands._find_agent_python", fake_find_agent_python
    )
    monkeypatch.setattr("eval.commands.time.strftime", fake_strftime)
    monkeypatch.setattr(
        "eval.commands._wait_for_agent_port", fake_wait_for_agent_port
    )
    monkeypatch.setattr("eval.commands.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "eval.commands._run_eval_orchestrator", fake_run_eval_orchestrator
    )

    result = runner.invoke(app, ["eval", "run"])

    assert result.exit_code == 0

    popen_cmd = captured["popen_cmd"]
    assert popen_cmd == ["uv", "run", "."]
    assert captured["popen_cwd"] == root

    popen_env = captured["popen_env"]
    assert popen_env["MODEL_PROVIDER"] == "openai"
    assert popen_env["MODEL"] == "gpt-4.1-mini"
    assert popen_env["BASE_URL"] == "http://model.local"
    assert popen_env["MODEL_ALIAS"] == "gpt-4.1-mini"
    assert popen_env["PORT"] == "9900"
    assert popen_env["URL"] == "http://127.0.0.1"

    assert captured["wait_agent_url"] == "http://127.0.0.1:9900"
    assert captured["wait_timeout_s"] == 15

    runner_kwargs = captured["runner_kwargs"]
    assert runner_kwargs["agent_url"] == "http://127.0.0.1:9900"
    assert runner_kwargs["model_provider"] == "openai"
    assert runner_kwargs["model"] == "gpt-4.1-mini"
    assert runner_kwargs["dataset"] == dataset.resolve()
    assert runner_kwargs["output_dir"] == eval_output.resolve()
    assert runner_kwargs["task_id"] == "20260101_000000"
    assert runner_kwargs["k"] == 2
    assert runner_kwargs["run_name"] == "benchmark"
    assert runner_kwargs["qualitative"] is True

    run_env = runner_kwargs["env"]
    assert run_env["MODEL_PROVIDER"] == "openai"
    assert run_env["MODEL"] == "gpt-4.1-mini"
    assert run_env["BASE_URL"] == "http://model.local"
    assert run_env["MODEL_ALIAS"] == "gpt-4.1-mini"
    assert run_env["JUDGE_PROVIDER"] == "ollama"
    assert run_env["JUDGE_MODEL"] == "judge-model"
    assert run_env["JUDGE_BASE_URL"] == "http://judge.local"
    assert run_env["JUDGE_MODE"] == "strict"


def _write_eval_yaml_with_prompts(root: Path, prompts_block: str) -> Path:
    eval_input = root / "eval" / "input"
    eval_input.mkdir(parents=True)
    (eval_input / "tasks.json").write_text("[]\n", encoding="utf-8")

    eval_yaml = root / "eval" / "eval.yaml"
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


def test_judge_prompts_parsed(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)
    eval_yaml = _write_eval_yaml_with_prompts(
        root,
        "  prompts:\n"
        "    relevance: tune relevance\n"
        "    task_completion: tune completion\n"
        "    hallucination: tune hallucination\n"
        "    tool_call: tune tool calls\n",
    )

    config = load_eval_config(root, eval_yaml)

    assert config.judge is not None
    assert config.judge.prompts == {
        "relevance": "tune relevance",
        "task_completion": "tune completion",
        "hallucination": "tune hallucination",
        "tool_call": "tune tool calls",
    }


def test_judge_prompts_partial_ok(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)
    eval_yaml = _write_eval_yaml_with_prompts(
        root,
        "  prompts:\n    relevance: only relevance set\n",
    )

    config = load_eval_config(root, eval_yaml)

    assert config.judge is not None
    assert config.judge.prompts == {"relevance": "only relevance set"}


def test_judge_prompts_overflow_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)
    overflow = "x" * 1001
    eval_yaml = _write_eval_yaml_with_prompts(
        root,
        f"  prompts:\n    task_completion: {overflow}\n",
    )

    try:
        load_eval_config(root, eval_yaml)
    except ValueError as exc:
        message = str(exc)
        assert "judge.prompts.task_completion" in message
        assert "1000-character limit" in message
        assert "got 1001" in message
    else:
        raise AssertionError("Expected overflow to fail load")


def test_judge_prompts_unknown_key_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)
    eval_yaml = _write_eval_yaml_with_prompts(
        root,
        "  prompts:\n    bogus: nope\n",
    )

    try:
        load_eval_config(root, eval_yaml)
    except ValueError as exc:
        message = str(exc)
        assert "bogus" in message
        assert "relevance" in message
    else:
        raise AssertionError("Expected unknown key to fail load")


def test_judge_prompts_absent_is_empty_dict(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)
    eval_yaml = _write_eval_yaml_with_prompts(root, "")

    config = load_eval_config(root, eval_yaml)

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
