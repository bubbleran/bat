"""Tests for `bat add agent`: creating an agent inside a blueprint and
registering it everywhere the blueprint tracks agents."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def _blueprint(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "blueprint", "demo"])
    assert result.exit_code == 0, result.output
    root = tmp_path / "demo"
    monkeypatch.chdir(root)
    return root


def test_add_agent_creates_the_agent_package(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    result = runner.invoke(app, ["add", "agent", "netops"])

    assert result.exit_code == 0, result.output
    agent = root / "netops"
    for name in [
        "__init__.py",
        "app.py",
        "agent.json",
        "config.yaml",
        "src/__init__.py",
        "src/graph.py",
        "src/llm_clients/__init__.py",
    ]:
        assert (agent / name).exists(), f"Missing agent file: {name}"
    # Everything that belongs to the blueprint must NOT be duplicated here.
    for name in ["pyproject.toml", "Dockerfile", "Makefile", "__main__.py"]:
        assert not (agent / name).exists(), f"{name} belongs to the blueprint"


def test_add_agent_config_points_at_its_own_card_and_port(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    runner.invoke(app, ["add", "agent", "netops", "--port", "9309"])

    config = yaml.safe_load((root / "netops" / "config.yaml").read_text())
    # The process runs from the blueprint root, so the card path is relative
    # to it, not to the agent directory.
    assert config["agent_card"] == "netops/agent.json"
    assert config["endpoint"]["port"] == 9309


def test_add_agent_registers_in_blueprint_yaml(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    runner.invoke(app, ["add", "agent", "netops"])

    blueprint = yaml.safe_load((root / "blueprint.yaml").read_text())
    assert "netops" in blueprint["agents"]


def test_add_agent_registers_in_the_dispatcher(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    runner.invoke(app, ["add", "agent", "netops"])

    main = (root / "__main__.py").read_text(encoding="utf-8")
    assert 'APPS: set[str] = {"netops"}' in main
    # Static and lazy: PyInstaller has to see the import, and it must not run
    # until the branch is taken.
    assert '    if app == "netops":' in main
    assert "        from netops import run" in main
    assert "importlib" not in main


def test_add_agent_registers_a_compose_service(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    runner.invoke(app, ["add", "agent", "netops"])

    compose = yaml.safe_load((root / "docker-compose.yaml").read_text())
    service = compose["services"]["netops"]
    assert service["command"] == ["netops"]
    assert service["environment"]["CONFIG_PATH"] == "netops/config.yaml"


def test_adding_a_second_agent_keeps_the_first(tmp_path, monkeypatch) -> None:
    """The failure mode of a marker rewrite is dropping or duplicating what
    was already there, so add two and check every registry."""
    root = _blueprint(tmp_path, monkeypatch)

    runner.invoke(app, ["add", "agent", "netops"])
    result = runner.invoke(app, ["add", "agent", "hermes"])

    assert result.exit_code == 0, result.output

    blueprint = yaml.safe_load((root / "blueprint.yaml").read_text())
    assert sorted(blueprint["agents"]) == ["hermes", "netops"]

    main = (root / "__main__.py").read_text(encoding="utf-8")
    assert 'APPS: set[str] = {"hermes", "netops"}' in main
    assert main.count("from netops import run") == 1
    assert main.count("from hermes import run") == 1
    assert main.count("# bat:agents:begin") == 1

    compose = yaml.safe_load((root / "docker-compose.yaml").read_text())
    assert sorted(compose["services"]) == ["hermes", "netops"]

    assert (root / "netops" / "app.py").exists()
    assert (root / "hermes" / "app.py").exists()


def test_second_agent_takes_the_next_free_port(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    runner.invoke(app, ["add", "agent", "netops", "--port", "9309"])
    runner.invoke(app, ["add", "agent", "hermes"])

    config = yaml.safe_load((root / "hermes" / "config.yaml").read_text())
    assert config["endpoint"]["port"] == 9310


def test_add_agent_outside_a_blueprint_is_rejected(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["add", "agent", "netops"])

    assert result.exit_code != 0
    assert "blueprint.yaml" in result.output


def test_add_agent_rejects_a_duplicate_name(tmp_path, monkeypatch) -> None:
    _blueprint(tmp_path, monkeypatch)
    runner.invoke(app, ["add", "agent", "netops"])

    result = runner.invoke(app, ["add", "agent", "netops"])

    assert result.exit_code != 0
    assert "already" in result.output


def test_add_agent_rejects_a_name_that_is_not_importable(
    tmp_path, monkeypatch
) -> None:
    """The directory becomes a Python package the dispatcher imports, so a
    dash would produce a module name nothing can import."""
    _blueprint(tmp_path, monkeypatch)

    result = runner.invoke(app, ["add", "agent", "cluster-view"])

    assert result.exit_code != 0
