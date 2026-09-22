"""Where am I: resolve a working directory to the agent a command acts on.

Two project shapes exist. A *standalone* agent keeps ``config.yaml``,
``agent.json`` and ``pyproject.toml`` in one directory and starts with
``uv run .``. A *blueprint* is one uv project holding several agents: the
pyproject and the ``__main__.py`` dispatcher sit at the blueprint root, each
agent owns a directory with its own ``config.yaml`` and ``agent.json``, and
starting one takes both a selector (``uv run . <agent>``) and ``CONFIG_PATH``
naming that agent's config -- the SDK would otherwise read the blueprint's
shared ``./config.yaml``.

Commands resolve an :class:`AgentTarget` once and read the difference off it,
so the two shapes stay in one place instead of spreading through every
command.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

BLUEPRINT_FILE = "blueprint.yaml"

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

        Inside a blueprint ``./config.yaml`` is the blueprint's, not the
        agent's, so ``CONFIG_PATH`` has to name the agent's explicitly. The
        SDK reads it at startup.
        """
        if self.agent_name is None:
            return {}
        return {"CONFIG_PATH": f"{self.agent_name}/config.yaml"}


def find_blueprint_root(start: Path) -> Path | None:
    """Nearest ancestor of ``start`` (inclusive) holding a blueprint.yaml."""
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / BLUEPRINT_FILE).is_file():
            return candidate
    return None


def blueprint_agent_names_on_disk(blueprint_root: Path) -> list[str]:
    """The agent directories actually present under ``blueprint_root``.

    Discovered from the filesystem -- a directory holding a ``config.yaml`` --
    the same way the generated Makefile does, so a stale ``blueprint.yaml``
    never makes a real agent unreachable.
    """
    return sorted(
        child.name
        for child in blueprint_root.iterdir()
        if child.is_dir() and (child / "config.yaml").is_file()
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
            "existing agent, or from an agent directory inside a blueprint."
        )
    return AgentTarget(project_root=cwd, agent_dir=cwd, agent_name=None)


def _resolve_blueprint_agent(blueprint_root: Path, cwd: Path) -> AgentTarget:
    relative = cwd.relative_to(blueprint_root)
    if not relative.parts:
        known = ", ".join(blueprint_agent_names_on_disk(blueprint_root))
        raise ProjectError(
            f"{cwd} is the root of a blueprint, not an agent. cd into one of "
            "its agent directories, or name the agent as an argument. Known "
            f"agents: {known or '(none yet -- run `bat add agent`)'}."
        )
    if len(relative.parts) > 1:
        raise ProjectError(
            f"{cwd} is nested inside blueprint {blueprint_root.name} but is "
            "not one of its agent directories, which sit one level down."
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
        agent_name=relative.parts[0],
    )
