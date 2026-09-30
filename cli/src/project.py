"""Where am I: resolve a working directory to the agent a command acts on.

Two project shapes exist. A *standalone* agent keeps ``config.yaml``,
``agent.json`` and ``pyproject.toml`` in one directory and starts with
``uv run .``. A *blueprint* is one uv project holding several agents: the
pyproject and the ``__main__.py`` dispatcher sit at the blueprint root, each
agent owns a directory right below it with its own ``config.yaml`` and
``agent.json`` (but no pyproject), and starting one takes both a selector
(``uv run . <agent>``) and ``CONFIG_PATH`` naming that agent's config -- the
SDK would otherwise look for ``./config.yaml`` at the blueprint root.

A blueprint is recognised by that layout alone, with no manifest file: the
real ones carry none, and a freshly initialised one has no agents yet.

Commands resolve an :class:`AgentTarget` once and read the difference off it,
so the two shapes stay in one place instead of spreading through every
command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_STANDALONE_REQUIRED = ("config.yaml", "agent.json", "pyproject.toml")
_BLUEPRINT_AGENT_REQUIRED = ("config.yaml", "agent.json")


class ProjectError(Exception):
    """The working directory is not an agent the CLI can act on."""


@dataclass(frozen=True)
class AgentTarget:
    """One agent, located.

    Attributes:
        project_root (Path): Where ``uv run .`` works -- the directory holding
            ``pyproject.toml`` and the virtualenv.
        agent_dir (Path): Where this agent's ``config.yaml`` and ``agent.json``
            live. Equal to ``project_root`` for a standalone agent.
        agent_name (str | None): The dispatcher selector, or ``None`` when the
            agent is standalone and there is nothing to select.
    """

    project_root: Path
    agent_dir: Path
    agent_name: str | None

    @property
    def is_blueprint(self) -> bool:
        return self.agent_name is not None

    @property
    def config_path(self) -> Path:
        return self.agent_dir / "config.yaml"

    @property
    def run_command(self) -> list[str]:
        """The argv that starts this agent, run from :attr:`project_root`."""
        if self.agent_name is None:
            return ["uv", "run", "."]
        return ["uv", "run", ".", self.agent_name]

    @property
    def run_env(self) -> dict[str, str]:
        """Env vars the launch needs on top of the caller's own.

        Inside a blueprint the process runs from the blueprint root, where
        ``./config.yaml`` is not the agent's, so ``CONFIG_PATH`` has to name
        the agent's explicitly. The SDK reads it at startup.
        """
        if self.agent_name is None:
            return {}
        return {"CONFIG_PATH": f"{self.agent_name}/config.yaml"}


def is_blueprint_root(path: Path) -> bool:
    """Whether ``path`` is the root of a blueprint.

    A uv project with a ``__main__.py`` to dispatch from -- and no
    ``agent.json``, which is what a standalone agent, holding the same two
    files, has on top.
    """
    return (
        (path / "pyproject.toml").is_file()
        and (path / "__main__.py").is_file()
        and not (path / "agent.json").is_file()
    )


def find_blueprint_root(start: Path) -> Path | None:
    """The blueprint ``start`` belongs to, if any.

    That is ``start`` itself, or its parent when ``start`` is a directory of
    that blueprint -- unless it holds a pyproject of its own, which makes it
    a separate project that merely sits in the blueprint's folder.
    """
    start = start.resolve()
    if is_blueprint_root(start):
        return start
    if (
        is_blueprint_root(start.parent)
        and not (start / "pyproject.toml").is_file()
    ):
        return start.parent
    return None


def blueprint_agent_names_on_disk(blueprint_root: Path) -> list[str]:
    """The agent directories actually present under ``blueprint_root``.

    Discovered from the filesystem -- the directories right below the root
    that hold a ``config.yaml`` and an ``agent.json`` and are not projects of
    their own -- so there is no registry to fall out of date.
    """
    return sorted(
        child.name
        for child in blueprint_root.iterdir()
        if child.is_dir()
        and all((child / name).is_file() for name in _BLUEPRINT_AGENT_REQUIRED)
        and not (child / "pyproject.toml").is_file()
    )


def resolve_agent_target(
    cwd: Path, agent_name: str | None = None
) -> AgentTarget:
    """Resolve the agent a command should act on.

    Without ``agent_name`` the agent is wherever ``cwd`` is. With one, the
    agent is that named directory of the enclosing blueprint, so a command
    can be run from the blueprint root without cd-ing first; the name wins
    over ``cwd``.

    Raises:
        ProjectError: When ``cwd`` is neither a standalone agent root nor an
            agent directory inside a blueprint, or when ``agent_name`` names
            something that is not an agent of the enclosing blueprint.
    """
    cwd = cwd.resolve()
    blueprint_root = find_blueprint_root(cwd)

    if agent_name is not None:
        if blueprint_root is None:
            raise ProjectError(
                f"Naming an agent ('{agent_name}') works only inside a "
                "blueprint, where several agents share one project. A "
                "standalone agent is the directory you are in."
            )
        return _resolve_named_agent(blueprint_root, agent_name)

    if blueprint_root is None:
        return _resolve_standalone(cwd)
    return _resolve_blueprint_agent(blueprint_root, cwd)


def unwired_agent_warning(target: AgentTarget) -> str | None:
    """Why ``uv run . <agent>`` would likely be refused, if it would.

    The selector is the agent's directory name, but only the blueprint's
    ``__main__.py`` decides what it accepts -- automation's ``logs_agent/``
    is selected as ``logs``. A dispatcher that never spells the name out is
    almost certainly one that will reject it; the check stays a warning
    because a dispatcher is free to compute its names.
    """
    if target.agent_name is None:
        return None
    main = (target.project_root / "__main__.py").read_text(encoding="utf-8")
    if re.search(rf"""["']{re.escape(target.agent_name)}["']""", main):
        return None
    return (
        f"{target.project_root.name}/__main__.py never names "
        f"'{target.agent_name}', so `uv run . {target.agent_name}` will "
        "probably be rejected. Add it to the dispatcher (`bat add agent` "
        "does this for the blueprints it creates)."
    )


def _resolve_named_agent(blueprint_root: Path, agent_name: str) -> AgentTarget:
    agent_dir = blueprint_root / agent_name
    required = [
        name
        for name in _BLUEPRINT_AGENT_REQUIRED
        if not (agent_dir / name).is_file()
    ]
    if required:
        known = ", ".join(blueprint_agent_names_on_disk(blueprint_root))
        raise ProjectError(
            f"Blueprint {blueprint_root.name} has no agent named "
            f"'{agent_name}'. Known agents: {known or '(none yet)'}."
        )
    return AgentTarget(
        project_root=blueprint_root,
        agent_dir=agent_dir,
        agent_name=agent_name,
    )


def _resolve_standalone(cwd: Path) -> AgentTarget:
    missing = [
        name for name in _STANDALONE_REQUIRED if not (cwd / name).is_file()
    ]
    if missing:
        raise ProjectError(
            "Current directory does not look like an agent root. Missing: "
            f"{', '.join(missing)}. Run this command from the root of an "
            "existing agent, or from an agent directory inside a blueprint "
            "(whose parent holds pyproject.toml and __main__.py)."
        )
    return AgentTarget(project_root=cwd, agent_dir=cwd, agent_name=None)


def _resolve_blueprint_agent(blueprint_root: Path, cwd: Path) -> AgentTarget:
    if cwd == blueprint_root:
        known = ", ".join(blueprint_agent_names_on_disk(blueprint_root))
        raise ProjectError(
            f"{cwd} is the root of a blueprint, not an agent. cd into one of "
            "its agent directories, or name the agent as an argument. Known "
            f"agents: {known or '(none yet -- run `bat add agent`)'}."
        )
    missing = [
        name
        for name in _BLUEPRINT_AGENT_REQUIRED
        if not (cwd / name).is_file()
    ]
    if missing:
        raise ProjectError(
            f"{cwd.name} is not an agent of blueprint {blueprint_root.name}. "
            f"Missing: {', '.join(missing)}."
        )
    return AgentTarget(
        project_root=blueprint_root,
        agent_dir=cwd,
        agent_name=cwd.name,
    )
