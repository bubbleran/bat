"""Tests for resolving a working directory to the agent a command acts on."""

from __future__ import annotations

from pathlib import Path

import pytest

from project import (
    ProjectError,
    find_blueprint_root,
    resolve_agent_target,
    unwired_agent_warning,
)


def _write_standalone(root: Path) -> None:
    (root / "config.yaml").write_text(
        "endpoint:\n  port: 9900\n", encoding="utf-8"
    )
    (root / "agent.json").write_text("{}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='0.1.0'\n", encoding="utf-8"
    )


def _write_blueprint(root: Path, *agents: str) -> None:
    """A blueprint as the real ones are laid out: no manifest, just one uv
    project whose ``__main__.py`` dispatches to agent packages below it."""
    (root / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0.1.0'\n", encoding="utf-8"
    )
    apps = ", ".join(f'"{agent}"' for agent in agents)
    (root / "__main__.py").write_text(f"APPS = {{{apps}}}\n", encoding="utf-8")
    for agent in agents:
        agent_dir = root / agent
        agent_dir.mkdir(parents=True, exist_ok=True)
        (agent_dir / "config.yaml").write_text(
            "endpoint:\n  port: 9900\n", encoding="utf-8"
        )
        (agent_dir / "agent.json").write_text("{}\n", encoding="utf-8")


def test_standalone_agent_is_its_own_project_root(tmp_path: Path) -> None:
    _write_standalone(tmp_path)

    target = resolve_agent_target(tmp_path)

    assert target.project_root == tmp_path
    assert target.agent_dir == tmp_path
    assert target.agent_name is None
    assert target.is_blueprint is False
    assert target.run_command == ["uv", "run", "."]
    assert target.run_env == {}
    assert target.config_path == tmp_path / "config.yaml"


def test_blueprint_agent_carries_selector_and_config_path(
    tmp_path: Path,
) -> None:
    _write_blueprint(tmp_path, "netops")

    target = resolve_agent_target(tmp_path / "netops")

    assert target.project_root == tmp_path
    assert target.agent_dir == tmp_path / "netops"
    assert target.agent_name == "netops"
    assert target.is_blueprint is True
    # Both halves of what the blueprint Makefile does by hand.
    assert target.run_command == ["uv", "run", ".", "netops"]
    assert target.run_env == {"CONFIG_PATH": "netops/config.yaml"}
    assert target.config_path == tmp_path / "netops" / "config.yaml"


def test_blueprint_root_itself_is_not_an_agent(tmp_path: Path) -> None:
    _write_blueprint(tmp_path, "netops")

    with pytest.raises(ProjectError, match="root of a blueprint"):
        resolve_agent_target(tmp_path)


def test_blueprint_directory_without_agent_files_is_rejected(
    tmp_path: Path,
) -> None:
    _write_blueprint(tmp_path)
    (tmp_path / "docs").mkdir()

    with pytest.raises(ProjectError, match="agent.json"):
        resolve_agent_target(tmp_path / "docs")


def test_directory_that_is_neither_shape_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="does not look like an agent root"):
        resolve_agent_target(tmp_path)


def test_named_agent_resolves_from_the_blueprint_root(tmp_path: Path) -> None:
    _write_blueprint(tmp_path, "netops", "hermes")

    target = resolve_agent_target(tmp_path, agent_name="netops")

    assert target.project_root == tmp_path
    assert target.agent_dir == tmp_path / "netops"
    assert target.agent_name == "netops"


def test_named_agent_wins_over_the_working_directory(tmp_path: Path) -> None:
    _write_blueprint(tmp_path, "netops", "hermes")

    target = resolve_agent_target(tmp_path / "netops", agent_name="hermes")

    assert target.agent_name == "hermes"
    assert target.agent_dir == tmp_path / "hermes"


def test_unknown_named_agent_lists_the_known_ones(tmp_path: Path) -> None:
    _write_blueprint(tmp_path, "netops", "hermes")

    with pytest.raises(ProjectError, match="hermes, netops"):
        resolve_agent_target(tmp_path, agent_name="nope")


def test_naming_an_agent_outside_a_blueprint_is_rejected(
    tmp_path: Path,
) -> None:
    _write_standalone(tmp_path)

    with pytest.raises(ProjectError, match="only inside a blueprint"):
        resolve_agent_target(tmp_path, agent_name="netops")


def test_uv_project_without_a_dispatcher_is_not_a_blueprint(
    tmp_path: Path,
) -> None:
    """Without ``__main__.py`` nothing can select the agent, so
    ``uv run . <agent>`` would not start it."""
    _write_blueprint(tmp_path, "netops")
    (tmp_path / "__main__.py").unlink()

    with pytest.raises(ProjectError, match="pyproject.toml"):
        resolve_agent_target(tmp_path / "netops")


def test_agent_with_its_own_project_stays_standalone_inside_a_blueprint(
    tmp_path: Path,
) -> None:
    """A nested uv project (like automation's cluster-view) is its own
    project, even though it sits in a blueprint's folder."""
    _write_blueprint(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_standalone(nested)

    target = resolve_agent_target(nested)

    assert target.project_root == nested
    assert target.agent_name is None


def test_agent_named_like_its_blueprint_resolves(tmp_path: Path) -> None:
    """blueprints/supervisor/supervisor/: the agent directory and the
    blueprint share a name."""
    root = tmp_path / "supervisor"
    root.mkdir()
    _write_blueprint(root, "supervisor")

    target = resolve_agent_target(root / "supervisor")

    assert target.project_root == root
    assert target.run_command == ["uv", "run", ".", "supervisor"]


def test_blueprint_root_is_found_from_one_of_its_agents(
    tmp_path: Path,
) -> None:
    _write_blueprint(tmp_path, "netops")

    assert find_blueprint_root(tmp_path / "netops") == tmp_path
    assert find_blueprint_root(tmp_path) == tmp_path


def test_standalone_agent_is_not_a_blueprint_root(tmp_path: Path) -> None:
    """A standalone agent has a pyproject.toml and a __main__.py too; its
    agent.json is what tells it apart from a blueprint root."""
    _write_standalone(tmp_path)
    (tmp_path / "__main__.py").write_text("", encoding="utf-8")

    assert find_blueprint_root(tmp_path) is None


def test_agent_the_dispatcher_never_names_is_reported(tmp_path: Path) -> None:
    """automation's logs_agent/ is selected as ``logs``: the directory name
    is not a selector ``__main__.py`` accepts."""
    _write_blueprint(tmp_path, "logs")
    agent = tmp_path / "logs_agent"
    agent.mkdir()
    (agent / "config.yaml").write_text("endpoint: {}\n", encoding="utf-8")
    (agent / "agent.json").write_text("{}\n", encoding="utf-8")

    warning = unwired_agent_warning(resolve_agent_target(agent))

    assert warning is not None
    assert "logs_agent" in warning


def test_agent_the_dispatcher_names_is_not_reported(tmp_path: Path) -> None:
    _write_blueprint(tmp_path, "netops")

    assert (
        unwired_agent_warning(resolve_agent_target(tmp_path / "netops")) is None
    )


def test_standalone_agent_has_no_dispatcher_to_check(tmp_path: Path) -> None:
    _write_standalone(tmp_path)

    assert unwired_agent_warning(resolve_agent_target(tmp_path)) is None
