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
from eval.commands import _wait_for_agent_port
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


def _write_blueprint_with_agent(root: Path, selector: str = "netops") -> Path:
    """A blueprint laid out like the real ones: no manifest and no root
    config.yaml, just the project and a dispatcher accepting ``selector``."""
    (root / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='1.0.0'\n", encoding="utf-8"
    )
    (root / "__main__.py").write_text(
        f'APPS = {{"{selector}"}}\n', encoding="utf-8"
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

    async def fake_run_evaluation(**kwargs):
        captured["runner_kwargs"] = kwargs

    monkeypatch.setattr("eval.commands.run_evaluation", fake_run_evaluation)


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
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(agent)

    runner.invoke(app, ["eval", "run"])

    during = yaml.safe_load(captured["config_during_run"])
    assert during["telemetry"]["output"][0]["type"] == "local"
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


def test_eval_run_warns_when_the_dispatcher_does_not_name_the_agent(
    tmp_path, monkeypatch
) -> None:
    """The agent directory is the selector, but the dispatcher decides what it
    accepts: a mismatch means the agent never starts, which should not first
    surface as a startup timeout."""
    agent = _write_blueprint_with_agent(tmp_path, selector="ops")
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(agent)

    result = runner.invoke(app, ["eval", "run"])

    assert "never names 'netops'" in result.output


def test_eval_run_hands_the_blueprints_dotenv_to_the_agent(
    tmp_path, monkeypatch
) -> None:
    """`make netops` passes the root .env as UV_ENV_FILE; the eval starts the
    same agent, so it has to give it the same variables."""
    agent = _write_blueprint_with_agent(tmp_path)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=from-dotenv\n", encoding="utf-8"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["eval", "run", "netops"])

    assert result.exit_code == 0, result.output
    assert captured["env"]["OPENAI_API_KEY"] == "from-dotenv"


def test_eval_run_lets_the_shell_override_the_dotenv(
    tmp_path, monkeypatch
) -> None:
    """As with uv's env file: a variable already exported wins."""
    agent = _write_blueprint_with_agent(tmp_path)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=from-dotenv\n", encoding="utf-8"
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from-shell")
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(agent)

    result = runner.invoke(app, ["eval", "run"])

    assert result.exit_code == 0, result.output
    assert captured["env"]["OPENAI_API_KEY"] == "from-shell"


def _write_app(agent: Path, floor: str) -> None:
    (agent / "app.py").write_text(
        "from bat.agent import AgentApplication\n"
        "\n"
        "def run():\n"
        f"    AgentApplication(telemetry_privacy_floor={floor}).run()\n",
        encoding="utf-8",
    )


def test_eval_run_refuses_an_agent_whose_privacy_floor_is_above_none(
    tmp_path, monkeypatch
) -> None:
    """At `full` the spans hide prompts, results and tool names: the run
    would score answers only, with nothing to say why."""
    agent = _write_blueprint_with_agent(tmp_path)
    _write_app(agent, '"full"')
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["eval", "run", "netops"])

    assert result.exit_code == 1
    assert "Can't evaluate netops" in result.output
    assert '"full" (netops/app.py:4)' in result.output
    assert 'telemetry_privacy_floor="none"' in result.output
    assert "cmd" not in captured  # the agent was never started


def test_eval_run_accepts_a_privacy_floor_of_none(
    tmp_path, monkeypatch
) -> None:
    agent = _write_blueprint_with_agent(tmp_path)
    _write_app(agent, '"none"')
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["eval", "run", "netops"])

    assert result.exit_code == 0, result.output
    assert captured["cmd"] == ["uv", "run", ".", "netops"]


def test_eval_run_warns_when_it_cannot_read_the_privacy_floor(
    tmp_path, monkeypatch
) -> None:
    agent = _write_blueprint_with_agent(tmp_path)
    _write_app(agent, "FLOOR")
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["eval", "run", "netops"])

    assert result.exit_code == 0, result.output
    assert "netops/app.py:4" in result.output
    assert "cmd" in captured


# What bat-adk prints last when a required remote agent does not answer.
CARD_ERROR = (
    "ValueError: Failed to load and validate agent configuration: Network "
    "communication error fetching agent card from "
    "http://localhost:9301/.well-known/agent-card.json: All connection "
    "attempts failed"
)


def _run_an_agent_that_stops(tmp_path, monkeypatch, last_line: str):
    """`bat eval run netops` on an agent that prints a traceback and exits."""
    agent = _write_blueprint_with_agent(tmp_path)
    _patch_eval(monkeypatch, {}, agent)

    class _Stopped(_FakeProcess):
        def __init__(self) -> None:
            self.returncode = 1

    def fake_popen(cmd, cwd, env, stdout, **kwargs):
        stdout.write(f"Traceback (most recent call last):\n  ...\n{last_line}\n")
        return _Stopped()

    monkeypatch.setattr("eval.commands.subprocess.Popen", fake_popen)
    monkeypatch.setattr(
        "eval.commands._wait_for_agent_port", _wait_for_agent_port
    )
    monkeypatch.chdir(tmp_path)
    return agent, runner.invoke(app, ["eval", "run", "netops"])


def test_eval_run_explains_an_agent_that_stops_on_a_missing_dependency(
    tmp_path, monkeypatch
) -> None:
    agent, result = _run_an_agent_that_stops(tmp_path, monkeypatch, CARD_ERROR)

    assert result.exit_code == 1
    assert "netops stopped before it was ready (exit code 1)" in result.output
    assert "http://localhost:9301/.well-known/agent-card.json" in result.output
    assert "`required: false` in netops/config.yaml" in result.output
    log = agent / "eval" / "output" / "T0" / "agent-0.log"
    assert str(log) in result.output
    assert CARD_ERROR in log.read_text(encoding="utf-8")


def test_eval_run_reports_any_other_startup_error_without_the_hint(
    tmp_path, monkeypatch
) -> None:
    error = "ModuleNotFoundError: No module named 'netops.src'"

    _, result = _run_an_agent_that_stops(tmp_path, monkeypatch, error)

    assert result.exit_code == 1
    assert error in result.output
    assert "required: false" not in result.output
