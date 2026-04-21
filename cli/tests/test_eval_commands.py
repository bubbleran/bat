from pathlib import Path

from typer.testing import CliRunner

from cli import app
from eval.engine.eval_config import EvalConfig, JudgeSpec, ModelSpec


runner = CliRunner()


def _write_minimal_agent_root(root: Path) -> None:
    (root / "config.yaml").write_text("name: test\n", encoding="utf-8")
    (root / "agent.json").write_text("{}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='agent'\nversion='0.1.0'\n", encoding="utf-8")


def test_eval_init_requires_agent_root() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["eval", "init"])

        assert result.exit_code != 0
        assert "does not look like an agent root" in result.output


def test_eval_init_creates_scaffold() -> None:
    with runner.isolated_filesystem():
        root = Path.cwd()
        _write_minimal_agent_root(root)

        result = runner.invoke(app, ["eval", "init"])

        assert result.exit_code == 0
        assert (root / "eval" / "eval.yaml").exists()
        assert (root / "eval" / "input" / "tasks.json").exists()
        assert (root / "eval" / "output").is_dir()


def test_eval_run_requires_eval_yaml() -> None:
    with runner.isolated_filesystem():
        root = Path.cwd()
        _write_minimal_agent_root(root)

        result = runner.invoke(app, ["eval", "run"])

        assert result.exit_code != 0
        assert "Missing ./eval/eval.yaml" in result.output


def test_eval_run_starts_agent_and_runs_orchestrator(monkeypatch) -> None:
    with runner.isolated_filesystem():
        root = Path.cwd()
        _write_minimal_agent_root(root)

        eval_input = root / "eval" / "input"
        eval_output = root / "eval" / "output"
        eval_input.mkdir(parents=True, exist_ok=True)
        eval_output.mkdir(parents=True, exist_ok=True)

        eval_yaml = root / "eval" / "eval.yaml"
        eval_yaml.write_text("evaluation:\n  dataset: eval/input/tasks.json\n", encoding="utf-8")

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

        def fake_load_eval_config(agent_root: Path, config_path: Path) -> EvalConfig:
            assert agent_root == root
            assert config_path == eval_yaml
            return config

        def fake_find_cli_src() -> Path:
            return Path("/tmp/cli/src")

        def fake_find_agent_python(agent_root: Path) -> Path:
            assert agent_root == root
            return Path("/tmp/agent/.venv/bin/python")

        def fake_strftime(_fmt: str) -> str:
            return "20260101_000000"

        class _Result:
            returncode = 0

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

        def fake_run(cmd, cwd, env, check):  # noqa: ANN001
            captured["run_cmd"] = cmd
            captured["run_cwd"] = cwd
            captured["run_env"] = env
            captured["run_check"] = check
            return _Result()

        monkeypatch.setattr("eval.commands.load_eval_config", fake_load_eval_config)
        monkeypatch.setattr("eval.commands._find_cli_src", fake_find_cli_src)
        monkeypatch.setattr("eval.commands._find_agent_python", fake_find_agent_python)
        monkeypatch.setattr("eval.commands.time.strftime", fake_strftime)
        monkeypatch.setattr("eval.commands._wait_for_agent_port", fake_wait_for_agent_port)
        monkeypatch.setattr("eval.commands.subprocess.Popen", fake_popen)
        monkeypatch.setattr("eval.commands.subprocess.run", fake_run)

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

        run_cmd = captured["run_cmd"]
        assert run_cmd[0] == "/tmp/agent/.venv/bin/python"
        assert "-m" in run_cmd
        assert "eval.engine.orchestrator" in run_cmd
        assert "--agent-url" in run_cmd
        assert "http://127.0.0.1:9900" in run_cmd
        assert "--task-id" in run_cmd
        assert "20260101_000000" in run_cmd
        assert "--qualitative" in run_cmd

        assert captured["run_cwd"] == root
        assert captured["run_check"] is False

        run_env = captured["run_env"]
        assert run_env["MODEL_PROVIDER"] == "openai"
        assert run_env["MODEL"] == "gpt-4.1-mini"
        assert run_env["BASE_URL"] == "http://model.local"
        assert run_env["MODEL_ALIAS"] == "gpt-4.1-mini"
        assert run_env["JUDGE_PROVIDER"] == "ollama"
        assert run_env["JUDGE_MODEL"] == "judge-model"
        assert run_env["JUDGE_BASE_URL"] == "http://judge.local"
        assert run_env["JUDGE_MODE"] == "strict"
        assert run_env["PYTHONPATH"].startswith("/tmp/cli/src")
