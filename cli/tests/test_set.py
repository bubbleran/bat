"""Tests for `bat set config` (an agent's config.yaml) and `bat set image`
(the project's Makefile) inside a blueprint."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from cli import app

runner = CliRunner()

needs_make = pytest.mark.skipif(
    shutil.which("make") is None, reason="make is not installed"
)


@pytest.fixture(autouse=True)
def _no_image_settings_from_the_shell(monkeypatch) -> None:
    monkeypatch.delenv("DOCKER_REGISTRY", raising=False)
    monkeypatch.delenv("REPO", raising=False)


def _blueprint(tmp_path: Path, monkeypatch, *agents: str) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "blueprint", "demo"]).exit_code == 0
    root = tmp_path / "demo"
    monkeypatch.chdir(root)
    for agent in agents:
        assert runner.invoke(app, ["add", "agent", agent]).exit_code == 0
    return root


def _makefile_value(root: Path, name: str) -> str:
    match = re.search(
        rf"^{name}[ \t]*\?=[ \t]*(.*)$",
        (root / "Makefile").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, name
    return match.group(1)


def _port(agent_dir: Path) -> int:
    config = yaml.safe_load(
        (agent_dir / "config.yaml").read_text(encoding="utf-8")
    )
    return config["endpoint"]["port"]


def test_set_env_is_gone(tmp_path, monkeypatch) -> None:
    _blueprint(tmp_path, monkeypatch)

    result = runner.invoke(app, ["set", "env", "--port", "9999"])

    assert result.exit_code != 0
    assert "No such command" in result.output


# -- bat set config -----------------------------------------------------------


def test_set_config_from_an_agent_folder(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch, "netops")
    monkeypatch.chdir(root / "netops")

    result = runner.invoke(app, ["set", "config", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert _port(root / "netops") == 9999


def test_set_config_names_an_agent_from_the_blueprint_root(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch, "netops", "hermes")
    hermes_port = _port(root / "hermes")

    result = runner.invoke(app, ["set", "config", "netops", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert _port(root / "netops") == 9999
    assert _port(root / "hermes") == hermes_port


def test_set_config_at_the_blueprint_root_needs_an_agent(
    tmp_path, monkeypatch
) -> None:
    _blueprint(tmp_path, monkeypatch, "netops")

    result = runner.invoke(app, ["set", "config", "--port", "9999"])

    assert result.exit_code == 1
    assert "root of a blueprint" in result.output
    assert "netops" in result.output


def test_set_config_rejects_an_unknown_agent(tmp_path, monkeypatch) -> None:
    _blueprint(tmp_path, monkeypatch, "netops")

    result = runner.invoke(app, ["set", "config", "kpi", "--port", "9999"])

    assert result.exit_code == 1
    assert "no agent named 'kpi'" in result.output


# -- bat set image ------------------------------------------------------------


@needs_make
def test_set_image_writes_the_registry_into_the_makefile(
    tmp_path, monkeypatch
) -> None:
    """At the blueprint root: what make reads, so `make build` agrees."""
    root = _blueprint(tmp_path, monkeypatch)
    env_before = (root / ".env").read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "set",
            "image",
            "--docker-registry",
            "hub.bubbleran.com",
            "--repo",
            "orama/demo",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _makefile_value(root, "DOCKER_REGISTRY") == "hub.bubbleran.com"
    assert _makefile_value(root, "REPO") == "orama/demo"
    assert (root / ".env").read_text(encoding="utf-8") == env_before
    dry_run = subprocess.run(
        ["make", "-n", "build"], cwd=root, capture_output=True, text=True
    )
    assert "--tag hub.bubbleran.com/orama/demo:dev" in dry_run.stdout


def test_set_image_from_an_agent_folder_writes_the_blueprints_makefile(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch, "netops")
    monkeypatch.chdir(root / "netops")

    result = runner.invoke(
        app, ["set", "image", "--docker-registry", "hub.bubbleran.com"]
    )

    assert result.exit_code == 0, result.output
    assert _makefile_value(root, "DOCKER_REGISTRY") == "hub.bubbleran.com"
    assert not (root / "netops" / "Makefile").exists()


def test_set_image_replaces_an_old_placeholder(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    makefile = root / "Makefile"
    makefile.write_text(
        "DOCKER_REGISTRY ?= INSERT_YOUR_DOCKER_REGISTRY_HERE\n"
        "REPO ?= YOUR_REPOSITORY/demo\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["set", "image", "--docker-registry", "hub.bubbleran.com", "--repo", "x"],
    )

    assert result.exit_code == 0, result.output
    assert makefile.read_text(encoding="utf-8") == (
        "DOCKER_REGISTRY ?= hub.bubbleran.com\nREPO ?= x\n"
    )


def test_set_image_needs_a_line_to_set(tmp_path, monkeypatch) -> None:
    """A hand-written Makefile may name its image some other way: nothing
    is guessed, and nothing is written."""
    root = _blueprint(tmp_path, monkeypatch)
    makefile = root / "Makefile"
    makefile.write_text("build:\n\tdocker build -t demo .\n", encoding="utf-8")

    result = runner.invoke(
        app, ["set", "image", "--docker-registry", "hub.x.com", "--repo", "x"]
    )

    assert result.exit_code == 1
    assert "DOCKER_REGISTRY" in result.output
    assert makefile.read_text(encoding="utf-8") == (
        "build:\n\tdocker build -t demo .\n"
    )


def test_set_image_needs_a_makefile(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app, ["set", "image", "--docker-registry", "hub.bubbleran.com"]
    )

    assert result.exit_code == 1
    assert "No Makefile in" in result.output


def test_set_image_requires_at_least_one_option(tmp_path, monkeypatch) -> None:
    _blueprint(tmp_path, monkeypatch)

    result = runner.invoke(app, ["set", "image"])

    assert result.exit_code == 1
    assert "--docker-registry" in result.output
