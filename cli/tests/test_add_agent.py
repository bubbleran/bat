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

    compose = yaml.safe_load((root / "docker-compose.yaml").read_text())
    assert sorted(compose["services"]) == ["netops"]


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
    assert "hyphens are not allowed" in result.output
    assert "Use 'cluster_view'" in result.output


def test_add_agent_suggests_a_name_for_one_starting_with_a_digit(
    tmp_path, monkeypatch
) -> None:
    _blueprint(tmp_path, monkeypatch)

    result = runner.invoke(app, ["add", "agent", "5gcore"])

    assert result.exit_code != 0
    assert "can't start with a digit" in result.output


def test_add_agent_rejects_a_python_keyword(tmp_path, monkeypatch) -> None:
    """`class` is an identifier, but `from class import run` is a syntax
    error in the dispatcher."""
    root = _blueprint(tmp_path, monkeypatch)

    result = runner.invoke(app, ["add", "agent", "class"])

    assert result.exit_code != 0
    assert "is a Python keyword" in result.output
    assert not (root / "class").exists()


def test_blueprint_agent_passes_no_floor_by_default(
    tmp_path, monkeypatch, floor_given_to_the_application
) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    runner.invoke(app, ["add", "agent", "netops"])

    assert floor_given_to_the_application(root, "netops") == "none"


def test_each_blueprint_agent_has_its_own_floor(
    tmp_path, monkeypatch, floor_given_to_the_application
) -> None:
    """The floor is an argument of each agent's AgentApplication, so agents
    sharing one binary still have their own."""
    root = _blueprint(tmp_path, monkeypatch)
    runner.invoke(
        app, ["add", "agent", "netops", "--privacy", "names"]
    )
    runner.invoke(app, ["add", "agent", "hermes"])

    assert floor_given_to_the_application(root, "netops") == "names"
    assert floor_given_to_the_application(root, "hermes") == "none"


def test_add_agent_rejects_an_unknown_floor_before_writing(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    result = runner.invoke(
        app, ["add", "agent", "netops", "--privacy", "secret"]
    )

    assert result.exit_code != 0
    assert "secret" in result.output
    assert not (root / "netops").exists()


def test_add_agent_keeps_the_services_written_by_hand(
    tmp_path, monkeypatch
) -> None:
    """Only the new agent's service is added: other services, and edits to
    an agent's own, are left as they are."""
    root = _blueprint(tmp_path, monkeypatch)
    runner.invoke(app, ["add", "agent", "netops"])
    compose_path = root / "docker-compose.yaml"
    compose = yaml.safe_load(compose_path.read_text())
    compose["services"]["phoenix"] = {"image": "arizephoenix/phoenix"}
    compose["services"]["netops"]["environment"]["LOG_LEVEL"] = "debug"
    compose_path.write_text(yaml.safe_dump(compose, sort_keys=False))

    runner.invoke(app, ["add", "agent", "hermes"])

    services = yaml.safe_load(compose_path.read_text())["services"]
    assert services["phoenix"] == {"image": "arizephoenix/phoenix"}
    assert services["netops"]["environment"]["LOG_LEVEL"] == "debug"
    assert list(services) == ["netops", "phoenix", "hermes"]
