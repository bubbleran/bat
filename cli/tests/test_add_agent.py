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


def test_compose_services_share_one_image(tmp_path, monkeypatch) -> None:
    """One binary serves every agent, so every service runs the same image;
    only the command and the config differ."""
    root = _blueprint(tmp_path, monkeypatch)

    runner.invoke(app, ["add", "agent", "netops"])
    runner.invoke(app, ["add", "agent", "hermes"])

    services = yaml.safe_load((root / "docker-compose.yaml").read_text())[
        "services"
    ]
    assert services["netops"]["image"] == "${IMAGE_TAG:-demo:dev}"
    assert services["hermes"]["image"] == "${IMAGE_TAG:-demo:dev}"


def test_compose_service_mounts_the_agents_own_config(
    tmp_path, monkeypatch
) -> None:
    """The image carries no config.yaml; without the mount the agent has no
    endpoint to bind and refuses to start."""
    root = _blueprint(tmp_path, monkeypatch)

    runner.invoke(app, ["add", "agent", "netops"])

    service = yaml.safe_load((root / "docker-compose.yaml").read_text())[
        "services"
    ]["netops"]
    assert service["volumes"] == [
        "./netops/config.yaml:/app/netops/config.yaml:ro"
    ]


def test_compose_healthcheck_probes_the_agents_port(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    runner.invoke(app, ["add", "agent", "netops", "--port", "9309"])

    service = yaml.safe_load((root / "docker-compose.yaml").read_text())[
        "services"
    ]["netops"]
    assert "http://localhost:9309/ping" in " ".join(
        service["healthcheck"]["test"]
    )


def test_add_agent_registers_an_agent_directory_added_by_hand(
    tmp_path, monkeypatch
) -> None:
    """The agents on disk are the registry: one created without the CLI is
    wired in the next time the dispatcher is regenerated."""
    root = _blueprint(tmp_path, monkeypatch)
    hermes = root / "hermes"
    hermes.mkdir()
    (hermes / "config.yaml").write_text(
        "endpoint:\n  port: 9302\n", encoding="utf-8"
    )
    (hermes / "agent.json").write_text("{}\n", encoding="utf-8")

    runner.invoke(app, ["add", "agent", "netops"])

    main = (root / "__main__.py").read_text(encoding="utf-8")
    assert 'APPS: set[str] = {"hermes", "netops"}' in main
    compose = yaml.safe_load((root / "docker-compose.yaml").read_text())
    assert sorted(compose["services"]) == ["hermes", "netops"]


def test_add_agent_leaves_a_nested_project_out_of_the_dispatcher(
    tmp_path, monkeypatch
) -> None:
    """A directory with its own pyproject.toml is a separate project (like
    automation's cluster-view); importing it from the dispatcher would fail."""
    root = _blueprint(tmp_path, monkeypatch)
    nested = root / "cluster_view"
    nested.mkdir()
    for name, content in [
        ("pyproject.toml", "[project]\nname='cluster-view'\n"),
        ("config.yaml", "endpoint:\n  port: 9901\n"),
        ("agent.json", "{}\n"),
    ]:
        (nested / name).write_text(content, encoding="utf-8")

    runner.invoke(app, ["add", "agent", "netops"])

    main = (root / "__main__.py").read_text(encoding="utf-8")
    assert 'APPS: set[str] = {"netops"}' in main


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
    assert "bat init blueprint" in result.output


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


def test_blueprint_agent_run_from_source_has_no_floor(
    tmp_path, monkeypatch, floor_given_to_the_application
) -> None:
    """Outside the image (make, bat eval) nothing is baked in, so
    config.yaml alone decides."""
    root = _blueprint(tmp_path, monkeypatch)
    runner.invoke(app, ["add", "agent", "netops"])

    assert floor_given_to_the_application(root, "netops") == "none"


def test_every_blueprint_agent_takes_the_floor_baked_into_the_image(
    tmp_path, monkeypatch, floor_given_to_the_application
) -> None:
    """The Dockerfile writes telemetry_floor.py right before freezing: one
    floor, and every agent of the binary starts with it."""
    root = _blueprint(tmp_path, monkeypatch)
    runner.invoke(app, ["add", "agent", "netops"])
    runner.invoke(app, ["add", "agent", "hermes"])
    (root / "telemetry_floor.py").write_text(
        'TELEMETRY_PRIVACY_FLOOR = "names"\n', encoding="utf-8"
    )

    assert floor_given_to_the_application(root, "netops") == "names"
    assert floor_given_to_the_application(root, "hermes") == "names"


def test_add_agent_refuses_a_blueprint_it_cannot_register_in(
    tmp_path, monkeypatch
) -> None:
    """A hand-made blueprint (like orama's supervisor) is recognised by its
    layout, but its __main__.py has no managed region to register the agent
    in: refuse before writing, not after leaving a half-created agent."""
    root = tmp_path / "supervisor"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname='supervisor'\n", encoding="utf-8"
    )
    (root / "__main__.py").write_text(
        'APPS = {"supervisor"}\n', encoding="utf-8"
    )
    (root / "docker-compose.yaml").write_text(
        "services: {}\n", encoding="utf-8"
    )
    monkeypatch.chdir(root)

    result = runner.invoke(app, ["add", "agent", "netops"])

    assert result.exit_code != 0
    assert "bat:agents" in result.output
    assert not (root / "netops").exists()
