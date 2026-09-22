"""`bat eval run` from an agent directory inside a blueprint.

The agent is not the project: `uv run .` happens at the blueprint root, the
agent is named as an argument, and CONFIG_PATH points at the agent's own
config.yaml rather than the blueprint's.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from cli import app
from eval.engine.eval_config import EvalConfig, ModelSpec

runner = CliRunner()


class _FakeProcess:
    pid = 4242

    def __init__(self) -> None:
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


def _write_blueprint_with_agent(root: Path) -> Path:
    (root / "blueprint.yaml").write_text(
        "name: demo\nagents:\n  netops: {}\n", encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='1.0.0'\n", encoding="utf-8"
    )
    (root / "config.yaml").write_text(
        "endpoint:\n  url: http://localhost\n  port: 9900\n", encoding="utf-8"
    )
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    (venv_bin / "python").write_text("", encoding="utf-8")

    agent = root / "netops"
    (agent / "eval" / "input").mkdir(parents=True, exist_ok=True)
    (agent / "eval" / "output").mkdir(parents=True, exist_ok=True)
    (agent / "agent.json").write_text("{}\n", encoding="utf-8")
    (agent / "config.yaml").write_text(
        "agent_card: netops/agent.json\n"
        "endpoint:\n  url: http://127.0.0.1\n  port: 9309\n"
        "telemetry:\n  privacy: content\n  output: []\n",
        encoding="utf-8",
    )
    (agent / "eval" / "eval.yaml").write_text(
        "evaluation:\n  dataset: eval/input/tasks.json\n", encoding="utf-8"
    )
    (agent / "eval" / "input" / "tasks.json").write_text(
        "[]\n", encoding="utf-8"
    )
    return agent


def _patch_eval(monkeypatch, captured: dict, agent: Path) -> None:
    config = EvalConfig(
        dataset=(agent / "eval" / "input" / "tasks.json").resolve(),
        output_dir=(agent / "eval" / "output").resolve(),
        agent_startup_timeout_s=15,
        agent_shutdown_timeout_s=5,
        k=1,
        qualitative=False,
        run_name="bench",
        models=[ModelSpec(provider="openai", model="gpt-4.1-mini")],
        judge=None,
    )

    def fake_popen(cmd, cwd, env, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["env"] = env
        captured["config_during_run"] = (agent / "config.yaml").read_text(
            encoding="utf-8"
        )
        return _FakeProcess()

    monkeypatch.setattr("eval.commands.load_eval_config", lambda r, c: config)
    monkeypatch.setattr("eval.commands.subprocess.Popen", fake_popen)
    monkeypatch.setattr("eval.commands.time.strftime", lambda fmt: "T0")
    monkeypatch.setattr("eval.commands.os.getpgid", lambda pid: pid)
    monkeypatch.setattr("eval.commands.os.killpg", lambda pgid, sig: None)
    monkeypatch.setattr(
        "eval.commands._wait_for_agent_port",
        lambda agent_url, timeout_s, process: captured.__setitem__(
            "agent_url", agent_url
        ),
    )
    monkeypatch.setattr(
        "eval.commands._run_eval_orchestrator",
        lambda **kwargs: captured.__setitem__("runner_kwargs", kwargs),
    )


def test_eval_run_starts_the_blueprint_agent(tmp_path, monkeypatch) -> None:
    agent = _write_blueprint_with_agent(tmp_path)
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(agent)

    result = runner.invoke(app, ["eval", "run"])

    assert result.exit_code == 0, result.output
    assert captured["cmd"] == ["uv", "run", ".", "netops"]
    assert captured["cwd"] == tmp_path
    assert captured["env"]["CONFIG_PATH"] == "netops/config.yaml"
    # The URL comes from the agent's config.yaml, not the blueprint's.
    assert captured["agent_url"] == "http://127.0.0.1:9309"


def test_eval_run_patches_and_restores_the_agents_config(
    tmp_path, monkeypatch
) -> None:
    agent = _write_blueprint_with_agent(tmp_path)
    blueprint_config_before = (tmp_path / "config.yaml").read_text(
        encoding="utf-8"
    )
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(agent)

    runner.invoke(app, ["eval", "run"])

    during = yaml.safe_load(captured["config_during_run"])
    assert during["telemetry"]["output"][0]["type"] == "local"
    # The blueprint's shared config is none of the eval's business.
    assert (tmp_path / "config.yaml").read_text(
        encoding="utf-8"
    ) == blueprint_config_before
    # And the agent's is put back.
    after = yaml.safe_load((agent / "config.yaml").read_text(encoding="utf-8"))
    assert after["telemetry"]["output"] == []


def test_eval_run_accepts_the_agent_name_from_the_blueprint_root(
    tmp_path, monkeypatch
) -> None:
    agent = _write_blueprint_with_agent(tmp_path)
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    # No cd into the agent: the name is what selects it.
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["eval", "run", "netops"])

    assert result.exit_code == 0, result.output
    assert captured["cmd"] == ["uv", "run", ".", "netops"]
    assert captured["cwd"] == tmp_path
    assert captured["env"]["CONFIG_PATH"] == "netops/config.yaml"
    assert captured["agent_url"] == "http://127.0.0.1:9309"


def test_eval_run_from_the_blueprint_root_without_a_name_explains_itself(
    tmp_path, monkeypatch
) -> None:
    agent = _write_blueprint_with_agent(tmp_path)
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["eval", "run"])

    assert result.exit_code != 0
    assert "root of a blueprint" in result.output


def test_eval_run_with_an_unknown_agent_lists_the_known_ones(
    tmp_path, monkeypatch
) -> None:
    agent = _write_blueprint_with_agent(tmp_path)
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["eval", "run", "nope"])

    assert result.exit_code != 0
    assert "netops" in result.output


def test_eval_show_accepts_the_agent_name(tmp_path, monkeypatch) -> None:
    agent = _write_blueprint_with_agent(tmp_path)
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["eval", "show", "netops"])

    assert result.exit_code == 0, result.output
