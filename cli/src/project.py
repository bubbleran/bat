"""Where am I: resolve a working directory to the agent a command acts on.

A standalone agent keeps config.yaml, agent.json and pyproject.toml in one
directory and starts with ``uv run .``. A blueprint is one uv project (the
pyproject and the ``__main__.py`` dispatcher at its root) with one agent per
directory right below it, started with ``uv run . <agent>`` and CONFIG_PATH.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import typer

PRIVACY_LEVELS = ("none", "content", "names", "full")
_AGENT_FILES = ("config.yaml", "agent.json")


class ProjectError(Exception):
    """The working directory is not an agent the CLI can act on."""


def fail(message: str) -> NoReturn:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@dataclass(frozen=True)
class AgentTarget:
    project_root: Path  # where `uv run .` works
    agent_dir: Path  # where config.yaml and agent.json live
    agent_name: str | None  # the dispatcher selector; None when standalone

    @property
    def config_path(self) -> Path:
        return self.agent_dir / "config.yaml"

    @property
    def run_command(self) -> list[str]:
        if self.agent_name is None:
            return ["uv", "run", "."]
        return ["uv", "run", ".", self.agent_name]

    @property
    def run_env(self) -> dict[str, str]:
        """A blueprint runs from its root, where ./config.yaml is not the
        agent's."""
        if self.agent_name is None:
            return {}
        return {"CONFIG_PATH": f"{self.agent_name}/config.yaml"}


def is_blueprint_root(path: Path) -> bool:
    """A uv project with a ``__main__.py`` and, unlike a standalone agent,
    no ``agent.json``."""
    return (
        (path / "pyproject.toml").is_file()
        and (path / "__main__.py").is_file()
        and not (path / "agent.json").is_file()
    )


def find_blueprint_root(start: Path) -> Path | None:
    """``start`` itself, or its parent when ``start`` is one of its agent
    folders (not a project of its own)."""
    start = start.resolve()
    if is_blueprint_root(start):
        return start
    if (
        is_blueprint_root(start.parent)
        and not (start / "pyproject.toml").is_file()
    ):
        return start.parent
    return None


def _missing(directory: Path, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not (directory / name).is_file()]


def blueprint_agents(root: Path) -> list[str]:
    """The agent folders right below ``root``, read off the disk."""
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir()
        and not _missing(child, _AGENT_FILES)
        and not (child / "pyproject.toml").is_file()
    )


def _known_agents(root: Path) -> str:
    return ", ".join(blueprint_agents(root)) or (
        "(none yet -- run `bat add agent`)"
    )


def resolve_agent_target(
    cwd: Path, agent_name: str | None = None
) -> AgentTarget:
    """The agent named ``agent_name`` in the enclosing blueprint, else the
    one ``cwd`` is.

    Raises:
        ProjectError: When there is no such agent.
    """
    cwd = cwd.resolve()
    root = find_blueprint_root(cwd)

    if root is None:
        if agent_name is not None:
            raise ProjectError(
                f"Naming an agent ('{agent_name}') works only inside a "
                "blueprint, where several agents share one project. A "
                "standalone agent is the directory you are in."
            )
        missing = _missing(cwd, (*_AGENT_FILES, "pyproject.toml"))
        if missing:
            raise ProjectError(
                "Current directory does not look like an agent root. Missing: "
                f"{', '.join(missing)}. Run this command from the root of an "
                "existing agent, or from an agent directory inside a blueprint "
                "(whose parent holds pyproject.toml and __main__.py)."
            )
        return AgentTarget(cwd, cwd, None)

    if agent_name is not None:
        if _missing(root / agent_name, _AGENT_FILES):
            raise ProjectError(
                f"Blueprint {root.name} has no agent named '{agent_name}'. "
                f"Known agents: {_known_agents(root)}."
            )
        return AgentTarget(root, root / agent_name, agent_name)

    if cwd == root:
        raise ProjectError(
            f"{cwd} is the root of a blueprint, not an agent. cd into one of "
            "its agent directories, or name the agent as an argument. Known "
            f"agents: {_known_agents(root)}."
        )
    missing = _missing(cwd, _AGENT_FILES)
    if missing:
        raise ProjectError(
            f"{cwd.name} is not an agent of blueprint {root.name}. "
            f"Missing: {', '.join(missing)}."
        )
    return AgentTarget(root, cwd, cwd.name)


def unwired_agent_warning(target: AgentTarget) -> str | None:
    """Why ``uv run . <agent>`` would likely be refused, if it would.

    Only the blueprint's ``__main__.py`` decides which selectors it accepts;
    one that never spells the name out almost certainly rejects it.
    """
    if target.agent_name is None:
        return None
    main = (target.project_root / "__main__.py").read_text(encoding="utf-8")
    # The scaffolded dispatcher imports whichever agent it is given.
    if "import_module(" in main:
        return None
    if re.search(rf"""["']{re.escape(target.agent_name)}["']""", main):
        return None
    return (
        f"{target.project_root.name}/__main__.py never names "
        f"'{target.agent_name}', so `uv run . {target.agent_name}` will "
        "probably be rejected. Add it to the dispatcher (the one "
        "`bat init blueprint` writes finds every agent folder by itself)."
    )


@dataclass(frozen=True)
class PrivacyFloor:
    location: str
    level: str | None  # None when the value is computed at runtime


def privacy_floors(target: AgentTarget) -> list[PrivacyFloor]:
    """Every ``telemetry_privacy_floor=`` in the agent's code, its
    virtualenv and tests left out."""
    floors: list[PrivacyFloor] = []
    for path in sorted(target.agent_dir.rglob("*.py")):
        folders = path.relative_to(target.agent_dir).parts[:-1]
        if any(part.startswith(".") or part == "tests" for part in folders):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        location = path.relative_to(target.project_root).as_posix()
        for node in ast.walk(tree):
            for keyword in getattr(node, "keywords", []):
                if keyword.arg == "telemetry_privacy_floor":
                    floors.append(
                        PrivacyFloor(
                            f"{location}:{keyword.value.lineno}",
                            _privacy_level(keyword.value),
                        )
                    )
    return floors


def _privacy_level(value: ast.expr) -> str | None:
    """Read as bat-adk does: a level name or ``TelemetryPrivacy`` member,
    and ``none`` for a literal it doesn't know."""
    if isinstance(value, ast.Attribute):
        name = value.attr.lower()
        return name if name in PRIVACY_LEVELS else None
    if isinstance(value, ast.Constant):
        name = str(value.value).strip().lower()
        return name if name in PRIVACY_LEVELS else "none"
    return None
