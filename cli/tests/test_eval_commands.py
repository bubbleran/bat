import json
import signal
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
    # The eval reads the agent's endpoint from config.yaml to know where to
    # connect (it no longer patches/forces it).
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
def test_eval_init_requires_agent_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["eval", "init"])

    assert result.exit_code != 0
    assert "does not look like an agent root" in result.output
    assert result.exit_code != 0
    assert "does not look like an agent root" in result.output


def test_eval_init_creates_scaffold(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)
def test_eval_init_creates_scaffold(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    result = runner.invoke(app, ["eval", "init"])
    result = runner.invoke(app, ["eval", "init"])

    assert result.exit_code == 0
    assert (root / "eval" / "eval.yaml").exists()
    assert (root / "eval" / "input" / "tasks.json").exists()
    assert (root / "eval" / "output").is_dir()
    assert result.exit_code == 0
    assert (root / "eval" / "eval.yaml").exists()
    assert (root / "eval" / "input" / "tasks.json").exists()
    assert (root / "eval" / "output").is_dir()


def test_eval_run_requires_eval_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)
def test_eval_run_requires_eval_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    result = runner.invoke(app, ["eval", "run"])
    result = runner.invoke(app, ["eval", "run"])

    assert result.exit_code != 0
    assert "Missing ./eval/eval.yaml" in result.output
    assert result.exit_code != 0
    assert "Missing ./eval/eval.yaml" in result.output


def test_eval_show_requires_eval_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)
def test_eval_show_requires_eval_yaml(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    result = runner.invoke(app, ["eval", "show"])
    result = runner.invoke(app, ["eval", "show"])

    assert result.exit_code != 0
    assert "Missing ./eval/eval.yaml" in result.output
    assert result.exit_code != 0
    assert "Missing ./eval/eval.yaml" in result.output


def test_eval_show_prints_resolved_config(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    eval_input = root / "eval" / "input"
    eval_output = root / "eval" / "output"
    eval_input.mkdir(parents=True, exist_ok=True)
    eval_output.mkdir(parents=True, exist_ok=True)
    eval_input = root / "eval" / "input"
    eval_output = root / "eval" / "output"
    eval_input.mkdir(parents=True, exist_ok=True)
    eval_output.mkdir(parents=True, exist_ok=True)

    eval_yaml = root / "eval" / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n  dataset: eval/input/tasks.json\n", encoding="utf-8"
    )
    eval_yaml = root / "eval" / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n  dataset: eval/input/tasks.json\n", encoding="utf-8"
    )

    dataset = eval_input / "tasks.json"
    dataset.write_text("[]\n", encoding="utf-8")
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
    def fake_load_eval_config(
        agent_root: Path, config_path: Path
    ) -> EvalConfig:
        assert agent_root == root
        assert config_path == eval_yaml
        return config

    monkeypatch.setattr("eval.commands.load_eval_config", fake_load_eval_config)
    monkeypatch.setattr("eval.commands.load_eval_config", fake_load_eval_config)

    result = runner.invoke(app, ["eval", "show"])
    result = runner.invoke(app, ["eval", "show"])

    assert result.exit_code == 0
    assert result.exit_code == 0

    assert "EVALUATION CONFIGURATION" in result.output
    assert f"Dataset     : {dataset.resolve()}" in result.output
    assert "k           : 2" in result.output
    assert "Qualitative : yes" in result.output
    assert "Models:" in result.output
    assert "  [1] openai:gpt-4.1-mini" in result.output
    assert "Judge model : ollama:judge-model" in result.output
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
    config = load_eval_config(root, eval_yaml)

    assert config.qualitative is False
    assert config.judge is None
    assert config.qualitative is False
    assert config.judge is None


def test_load_eval_config_requires_judge_when_qualitative_is_true(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)
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
    try:
        load_eval_config(root, eval_yaml)
    except ValueError as exc:
        assert "judge.provider and judge.model" in str(exc)
    else:
        raise AssertionError("Expected missing qualitative judge to fail")


def test_eval_run_starts_agent_and_runs_orchestrator(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)

    eval_input = root / "eval" / "input"
    eval_output = root / "eval" / "output"
    eval_input.mkdir(parents=True, exist_ok=True)
    eval_output.mkdir(parents=True, exist_ok=True)
    eval_input = root / "eval" / "input"
    eval_output = root / "eval" / "output"
    eval_input.mkdir(parents=True, exist_ok=True)
    eval_output.mkdir(parents=True, exist_ok=True)

    eval_yaml = root / "eval" / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n  dataset: eval/input/tasks.json\n", encoding="utf-8"
    )
    eval_yaml = root / "eval" / "eval.yaml"
    eval_yaml.write_text(
        "evaluation:\n  dataset: eval/input/tasks.json\n", encoding="utf-8"
    )

    dataset = eval_input / "tasks.json"
    dataset.write_text("[]\n", encoding="utf-8")
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
    captured: dict[str, object] = {}

    def fake_load_eval_config(
        agent_root: Path, config_path: Path
    ) -> EvalConfig:
        assert agent_root == root
        assert config_path == eval_yaml
        return config
    def fake_load_eval_config(
        agent_root: Path, config_path: Path
    ) -> EvalConfig:
        assert agent_root == root
        assert config_path == eval_yaml
        return config

    def fake_find_agent_python(agent_root: Path) -> Path:
        assert agent_root == root
        return Path("/tmp/agent/.venv/bin/python")
    def fake_find_agent_python(agent_root: Path) -> Path:
        assert agent_root == root
        return Path("/tmp/agent/.venv/bin/python")

    def fake_strftime(_fmt: str) -> str:
        return "20260101_000000"
    def fake_strftime(_fmt: str) -> str:
        return "20260101_000000"

    class _FakeProcess:
        pid = 12345

        def __init__(self) -> None:
            self.returncode = None

        def poll(self):  # noqa: ANN001
            return self.returncode
        def poll(self):  # noqa: ANN001
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0
        def terminate(self) -> None:
            self.returncode = 0

        def wait(self, timeout=None):  # noqa: ANN001
            if self.returncode is None:
                self.returncode = 0
            return self.returncode
        def wait(self, timeout=None):  # noqa: ANN001
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9
        def kill(self) -> None:
            self.returncode = -9

    def fake_popen(cmd, cwd, env, **kwargs):  # noqa: ANN001, ANN003
        captured["popen_cmd"] = cmd
        captured["popen_cwd"] = cwd
        captured["popen_kwargs"] = kwargs
        captured["popen_env"] = env
        return _FakeProcess()

    def fake_wait_for_agent_port(agent_url: str, timeout_s: int, process):  # noqa: ANN001
        captured["wait_agent_url"] = agent_url
        captured["wait_timeout_s"] = timeout_s
        assert process is not None
    def fake_wait_for_agent_port(agent_url: str, timeout_s: int, process):  # noqa: ANN001
        captured["wait_agent_url"] = agent_url
        captured["wait_timeout_s"] = timeout_s
        assert process is not None

    def fake_run_eval_orchestrator(**kwargs):  # noqa: ANN003
        captured["runner_kwargs"] = kwargs
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
    # Teardown signals the process group; keep it off the real OS in tests.
    monkeypatch.setattr("eval.commands.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("eval.commands.os.killpg", lambda pgid, sig: None)
    monkeypatch.setattr(
        "eval.commands._run_eval_orchestrator", fake_run_eval_orchestrator
    )

    result = runner.invoke(app, ["eval", "run"])
    result = runner.invoke(app, ["eval", "run"])

    assert result.exit_code == 0
    assert result.exit_code == 0

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
    assert runner_kwargs["model_provider"] == "openai"
    assert runner_kwargs["model"] == "gpt-4.1-mini"
    assert runner_kwargs["dataset"] == dataset.resolve()
    assert runner_kwargs["output_dir"] == eval_output.resolve()
    assert runner_kwargs["task_id"] == "20260101_000000"
    assert runner_kwargs["k"] == 2
    assert runner_kwargs["run_name"] == "benchmark"
    assert runner_kwargs["qualitative"] is True
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
    config = load_eval_config(root, eval_yaml)

    assert config.judge is not None
    assert config.judge.prompts == {
        "relevance": "tune relevance",
        "task_completion": "tune completion",
        "hallucination": "tune hallucination",
        "tool_call": "tune tool calls",
    }
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
def test_judge_prompts_partial_ok(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)
    eval_yaml = _write_eval_yaml_with_prompts(
        root,
        "  prompts:\n    relevance: only relevance set\n",
    )

    config = load_eval_config(root, eval_yaml)
    config = load_eval_config(root, eval_yaml)

    assert config.judge is not None
    assert config.judge.prompts == {"relevance": "only relevance set"}
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
def test_judge_prompts_absent_is_empty_dict(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = Path.cwd()
    _write_minimal_agent_root(root)
    eval_yaml = _write_eval_yaml_with_prompts(root, "")

    config = load_eval_config(root, eval_yaml)
    config = load_eval_config(root, eval_yaml)

    assert config.judge is not None
    assert config.judge.prompts == {}
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
def test_eval_plot_filter_restricts_per_task_charts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = Path.cwd() / "out"
    _write_run_metrics(out / "run_a", ["foo_alpha", "bar_beta", "foo_gamma"])

    result = runner.invoke(
        app, ["eval", "plot", "--folder", str(out), "--filter", "foo"]
    )
    result = runner.invoke(
        app, ["eval", "plot", "--folder", str(out), "--filter", "foo"]
    )

    assert result.exit_code == 0, result.output
    assert "Per-task filter active" in result.output
    assert result.exit_code == 0, result.output
    assert "Per-task filter active" in result.output

    png_names = {p.name for p in out.iterdir() if p.suffix == ".png"}
    assert "metrics_per_task_foo_alpha.png" in png_names
    assert "metrics_per_task_foo_gamma.png" in png_names
    assert "metrics_per_task_bar_beta.png" not in png_names
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
def test_eval_plot_without_filter_keeps_all_per_task_charts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out = Path.cwd() / "out"
    _write_run_metrics(out / "run_a", ["foo_alpha", "bar_beta"])

    result = runner.invoke(app, ["eval", "plot", "--folder", str(out)])
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


def test_apply_judge_result_records_reasoning_when_score_is_none() -> None:
    """EV-M1: a failed judge must leave its error in judge_reasoning."""
    from concurrent.futures import Future

    from eval.engine.contracts import QualitativeScores
    from eval.engine.metrics.llm_evaluators import _apply_judge_result

    scores = QualitativeScores()

    failed: Future = Future()
    failed.set_result({"score": None, "reasoning": "Error: bad api key"})
    _apply_judge_result(
        scores,
        failed,
        score_attr="response_relevance",
        reasoning_key="relevance",
        label="Response relevance",
    )
    assert scores.response_relevance is None
    assert scores.judge_reasoning["relevance"] == "Error: bad api key"

    raised: Future = Future()
    raised.set_exception(RuntimeError("boom"))
    _apply_judge_result(
        scores,
        raised,
        score_attr="hallucination_score",
        reasoning_key="hallucination",
        label="Hallucination",
    )
    assert scores.hallucination_score is None
    assert "boom" in scores.judge_reasoning["hallucination"]

    ok: Future = Future()
    ok.set_result({"score": 0.75, "reasoning": "looks good"})
    _apply_judge_result(
        scores,
        ok,
        score_attr="task_completion_quality",
        reasoning_key="completion",
        label="Task completion",
    )
    assert scores.task_completion_quality == 0.75
    assert scores.judge_reasoning["completion"] == "looks good"


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
    try:
        load_eval_config(tmp_path, eval_yaml)
    except ValueError as exc:
        assert "groq" in str(exc)
        assert "Valid providers" in str(exc)
    else:
        raise AssertionError("Expected unsupported provider to fail load")


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
    try:
        load_eval_config(tmp_path, eval_yaml)
    except ValueError as exc:
        assert "azure" in str(exc)
    else:
        raise AssertionError("Expected unsupported judge provider to fail")


def test_judge_client_does_not_inherit_model_base_url(monkeypatch) -> None:
    """S4-C2: judge base_url comes only from JUDGE_BASE_URL, never BASE_URL."""
    from eval.engine.metrics import llm_evaluators as le

    captured: dict[str, object] = {}

    class _FakeConfig:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            captured.update(kwargs)

    class _FakeClient:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            pass

    monkeypatch.setattr(le, "ChatModelClientConfig", _FakeConfig)
    monkeypatch.setattr(le, "ChatModelClient", _FakeClient)
    monkeypatch.setattr(le, "_judge_clients", {})
    monkeypatch.setenv("BASE_URL", "http://model-under-test:11434")
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.setenv("JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("JUDGE_MODEL", "gpt-4.1-mini")

    le._get_judge_client("relevance")

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

        def wait(self, timeout=None) -> int:  # noqa: ANN001
            return 0

    cmd._stop_agent_process(_FakeProc(), timeout_s=1)

    assert (4242, signal.SIGTERM) in signals


def test_bench_runner_isolates_failing_task(tmp_path) -> None:
    """S4-M6: one task raising is recorded as an error, run still finishes."""
    import asyncio

    from eval.engine.bench_runner import BenchRunner, RunConfig
    from eval.engine.contracts import EpisodeResult, EpisodeTrace, TaskSpec

    class _FakeAdapter:
        async def run_task(self, task, thread_id):  # noqa: ANN001
            if task.id == "boom":
                raise RuntimeError("kaboom")
            return EpisodeResult(
                task_id=task.id,
                final_status="completed",
                final_output="ok",
                trace=EpisodeTrace(),
            )

    tasks = [
        TaskSpec(id="boom", turns=["x"]),
        TaskSpec(id="fine", turns=["y"]),
    ]
    runner = BenchRunner(
        _FakeAdapter(),
        RunConfig(run_name="r", out_dir=str(tmp_path), k=1, model="m"),
    )

    results = asyncio.run(runner.run(tasks))

    by_id = {r.task_id: r for r in results}
    assert len(results) == 2
    assert by_id["boom"].final_status == "error"
    assert by_id["boom"].verdict is not None
    assert by_id["boom"].verdict.passed is False
    assert "kaboom" in by_id["boom"].aux["error"]
    assert by_id["fine"].final_status == "completed"


def test_eval_run_continues_after_one_model_fails(
    tmp_path, monkeypatch
) -> None:
    """S4-M1: one model failing must not abort the rest of the sweep."""
    root = tmp_path
    _write_minimal_agent_root(root)
    eval_input = root / "eval" / "input"
    eval_input.mkdir(parents=True)
    (eval_input / "tasks.json").write_text("[]\n", encoding="utf-8")
    (root / "eval" / "output").mkdir(parents=True)
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

        def poll(self):  # noqa: ANN201
            return None

        def wait(self, timeout=None):  # noqa: ANN001, ANN201
            return 0

    def fake_wait(agent_url, timeout_s, process):  # noqa: ANN001
        wait_calls["n"] += 1
        if wait_calls["n"] == 1:
            raise RuntimeError("agent did not start")

    monkeypatch.setattr("eval.commands.load_eval_config", lambda r, c: config)
    monkeypatch.setattr(
        "eval.commands._find_agent_python", lambda r: Path("/tmp/x/python")
    )
    monkeypatch.setattr("eval.commands.time.strftime", lambda fmt: "T0")
    monkeypatch.setattr(
        "eval.commands.subprocess.Popen",
        lambda *a, **k: _FakeProcess(),
    )
    monkeypatch.setattr("eval.commands.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("eval.commands.os.killpg", lambda pgid, sig: None)
    monkeypatch.setattr("eval.commands._wait_for_agent_port", fake_wait)
    monkeypatch.setattr(
        "eval.commands._run_eval_orchestrator",
        lambda **kw: ran_models.append(kw["model"]),
    )

    monkeypatch.chdir(root)
    result = runner.invoke(app, ["eval", "run"])

    # First model failed, but the sweep continued and ran the second.
    assert result.exit_code == 0, result.output
    assert ran_models == ["m2"]
    assert "1 failed" in result.output
    assert "openai:m1" in result.output
