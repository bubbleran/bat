from pathlib import Path

from typer.testing import CliRunner

from cli import app
from eval.engine.eval_config import EvalConfig, JudgeSpec, ModelSpec


runner = CliRunner()


def _write_minimal_agent_root(root: Path) -> None:
    (root / "config.yaml").write_text("name: test\n", encoding="utf-8")
    (root / "agent.json").write_text("{}\n", encoding="utf-8")
    src_dir = root / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "graph.py").write_text("# graph placeholder\n", encoding="utf-8")


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


def test_eval_run_uses_agent_environment(monkeypatch) -> None:
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
            k=2,
            qualitative=True,
            save_attempts=True,
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

        def fake_run(cmd, cwd, env, check):  # noqa: ANN001
            captured["cmd"] = cmd
            captured["cwd"] = cwd
            captured["env"] = env
            captured["check"] = check
            return _Result()

        monkeypatch.setattr("eval.commands.load_eval_config", fake_load_eval_config)
        monkeypatch.setattr("eval.commands._find_cli_src", fake_find_cli_src)
        monkeypatch.setattr("eval.commands._find_agent_python", fake_find_agent_python)
        monkeypatch.setattr("eval.commands.time.strftime", fake_strftime)
        monkeypatch.setattr("eval.commands.subprocess.run", fake_run)

        result = runner.invoke(app, ["eval", "run"])

        assert result.exit_code == 0

        cmd = captured["cmd"]
        assert cmd[0] == "/tmp/agent/.venv/bin/python"
        assert "-m" in cmd
        assert "eval.engine.agent_runner" in cmd
        assert "--task-id" in cmd
        assert "20260101_000000" in cmd
        assert "--save-attempts" in cmd
        assert "--qualitative" in cmd

        assert captured["cwd"] == root
        assert captured["check"] is False

        env = captured["env"]
        assert env["MODEL_PROVIDER"] == "openai"
        assert env["MODEL"] == "gpt-4.1-mini"
        assert env["BASE_URL"] == "http://model.local"
        assert env["MODEL_ALIAS"] == "gpt-4.1-mini"
        assert env["JUDGE_PROVIDER"] == "ollama"
        assert env["JUDGE_MODEL"] == "judge-model"
        assert env["JUDGE_BASE_URL"] == "http://judge.local"
        assert env["JUDGE_MODE"] == "strict"
        assert env["PYTHONPATH"].startswith("/tmp/cli/src")
