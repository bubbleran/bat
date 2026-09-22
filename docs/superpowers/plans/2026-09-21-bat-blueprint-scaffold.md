# Blueprint Scaffolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach `bat-cli` the blueprint project shape — one uv project holding several agents — through `bat init blueprint`, `bat add agent`, and a `bat eval` that can start an agent inside one.

**Architecture:** A new `cli/src/project.py` resolves the working directory to an `AgentTarget` (project root, agent directory, dispatcher name). Every command that finds, starts or reconfigures an agent reads the standalone/blueprint difference off that object instead of assuming one directory holds everything. Scaffolding lives in a new `cli/src/create/blueprint.py` beside the existing `agent.py`, reusing its template renderer.

**Tech Stack:** Python 3.12+, Typer, PyYAML, pytest with `typer.testing.CliRunner`.

**Spec:** `docs/superpowers/specs/2026-09-21-bat-blueprint-scaffold-design.md`

## Global Constraints

- Python `>=3.12`; the CLI package is src-layout (`cli/src`), tests run from `cli/` with `.venv/bin/python -m pytest tests -q`.
- **Commits are on hold.** Enrico asked for no commits for now. Do every commit step as a staged-but-uncommitted checkpoint, or run the commit only after he lifts the hold. Do not push.
- A scaffolded project pins `bat-adk[<provider>,telemetry]>=2026.9.10a0`. Reuse `create.agent._bat_adk_extras` — never hand-write the extras list, and never drop `telemetry` (without it the SDK's tracers are no-ops and `bat eval` reports zero tokens).
- `bat init agent` and the standalone agent shape keep working exactly as today. Every change to `cli/src/eval/commands.py` must leave the existing tests in `cli/tests/test_eval_commands.py` green without edits.
- The managed regions in generated files are delimited by `# bat:agents:begin` / `# bat:agents:end`. Generators rewrite only what is between them.
- New templates go under `cli/src/create/templates/blueprint/`; `cli/src/create/templates/agent/` is the standalone shape and is not moved or renamed.
- The blueprint's PyInstaller binary needs static, explicit imports. The dispatcher must never use `importlib` or a dynamic lookup table.

---

### Task 1: The `AgentTarget` resolver

**Files:**
- Create: `cli/src/project.py`
- Test: `cli/tests/test_project.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `BLUEPRINT_FILE: str = "blueprint.yaml"`
  - `class ProjectError(Exception)`
  - `@dataclass(frozen=True) class AgentTarget` with fields `project_root: Path`, `agent_dir: Path`, `agent_name: str | None` and properties `is_blueprint: bool`, `config_path: Path`, `run_command: list[str]`, `run_env: dict[str, str]`
  - `find_blueprint_root(start: Path) -> Path | None`
  - `resolve_agent_target(cwd: Path) -> AgentTarget`

- [ ] **Step 1: Write the failing tests**

Create `cli/tests/test_project.py`:

```python
"""Tests for resolving a working directory to the agent a command acts on."""

from __future__ import annotations

from pathlib import Path

import pytest

from project import AgentTarget, ProjectError, resolve_agent_target


def _write_standalone(root: Path) -> None:
    (root / "config.yaml").write_text("endpoint:\n  port: 9900\n", encoding="utf-8")
    (root / "agent.json").write_text("{}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname='agent'\nversion='0.1.0'\n", encoding="utf-8"
    )


def _write_blueprint(root: Path, *agents: str) -> None:
    (root / "blueprint.yaml").write_text("name: demo\nagents: {}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0.1.0'\n", encoding="utf-8"
    )
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


def test_blueprint_agent_carries_selector_and_config_path(tmp_path: Path) -> None:
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


def test_blueprint_directory_without_agent_files_is_rejected(tmp_path: Path) -> None:
    _write_blueprint(tmp_path)
    (tmp_path / "docs").mkdir()

    with pytest.raises(ProjectError, match="agent.json"):
        resolve_agent_target(tmp_path / "docs")


def test_directory_that_is_neither_shape_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProjectError, match="does not look like an agent root"):
        resolve_agent_target(tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd cli && .venv/bin/python -m pytest tests/test_project.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'project'`.

- [ ] **Step 3: Write the implementation**

Create `cli/src/project.py`:

```python
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
    """The nearest ancestor of ``start`` (inclusive) holding a blueprint.yaml."""
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / BLUEPRINT_FILE).is_file():
            return candidate
    return None


def resolve_agent_target(cwd: Path) -> AgentTarget:
    """Resolve ``cwd`` to the agent a command should act on.

    Raises:
        ProjectError: When ``cwd`` is neither a standalone agent root nor an
            agent directory inside a blueprint.
    """
    cwd = cwd.resolve()
    blueprint_root = find_blueprint_root(cwd)
    if blueprint_root is None:
        return _resolve_standalone(cwd)
    return _resolve_blueprint_agent(blueprint_root, cwd)


def _resolve_standalone(cwd: Path) -> AgentTarget:
    missing = [name for name in _STANDALONE_REQUIRED if not (cwd / name).is_file()]
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
        raise ProjectError(
            f"{cwd} is the root of a blueprint, not an agent. cd into one of "
            "its agent directories and run the command again."
        )
    if len(relative.parts) > 1:
        raise ProjectError(
            f"{cwd} is nested inside blueprint {blueprint_root.name} but is "
            "not one of its agent directories, which sit one level down."
        )
    missing = [
        name for name in _BLUEPRINT_AGENT_REQUIRED if not (cwd / name).is_file()
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd cli && .venv/bin/python -m pytest tests/test_project.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run the whole suite**

Run: `cd cli && .venv/bin/python -m pytest tests -q`
Expected: all previously passing tests still pass (64 total).

- [ ] **Step 6: Checkpoint (commit held — see Global Constraints)**

```bash
git add cli/src/project.py cli/tests/test_project.py
# git commit -m "feat(cli): resolve a working directory to an AgentTarget"
```

---

### Task 2: `bat init blueprint`

**Files:**
- Create: `cli/src/create/rendering.py`
- Modify: `cli/src/create/agent.py:64-73` (delegate `_render_template` to the shared helper)
- Create: `cli/src/create/blueprint.py`
- Create: `cli/src/create/templates/blueprint/blueprint.yaml`
- Create: `cli/src/create/templates/blueprint/pyproject.toml.template`
- Create: `cli/src/create/templates/blueprint/__main__.py`
- Create: `cli/src/create/templates/blueprint/config.yaml`
- Create: `cli/src/create/templates/blueprint/Makefile`
- Create: `cli/src/create/templates/blueprint/docker-compose.yaml`
- Create: `cli/src/create/templates/blueprint/Dockerfile`
- Create: `cli/src/create/templates/blueprint/blueprint.spec`
- Create: `cli/src/create/templates/blueprint/.env.template`
- Create: `cli/src/create/templates/blueprint/README.md`
- Create: `cli/src/create/templates/blueprint/.gitignore` — copy `cli/src/create/templates/agent/.gitignore` verbatim
- Create: `cli/src/create/templates/blueprint/.dockerignore` — copy `cli/src/create/templates/agent/.dockerignore` verbatim
- Create: `cli/src/create/templates/blueprint/.python-version` — copy `cli/src/create/templates/agent/.python-version` verbatim
- Modify: `cli/src/cli.py` (add the `init blueprint` command after `create_new_agent`, which ends at line 206)
- Modify: `cli/docs/bat-cli.md` (command tree, around line 64)
- Test: `cli/tests/test_create_blueprint.py`

**Interfaces:**
- Consumes: `create.agent._bat_adk_extras(model_provider: str) -> str`, `create.agent.BAT_ADK_VERSION: str`, `create.agent._PROVIDER_API_KEY_VAR: dict[str, str]`.
- Produces:
  - `create.rendering.render_template(templates_dir: Path, template_file: str, replacements: dict[str, str]) -> str`
  - `create.blueprint.BLUEPRINT_TEMPLATES_DIR: Path`
  - `create.blueprint.AGENTS_BEGIN: str = "# bat:agents:begin"`, `create.blueprint.AGENTS_END: str = "# bat:agents:end"`
  - `create.blueprint.create_blueprint_scaffold(target_dir: Path, *, force: bool = False, model_provider: str = "openai", namespace: str = "default", provider: str = "bubbleran", version: str = "v0.1.0") -> list[Path]`

- [ ] **Step 1: Write the failing tests**

Create `cli/tests/test_create_blueprint.py`:

```python
"""Tests for `bat init blueprint`: the shape of a freshly scaffolded blueprint."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def test_init_blueprint_creates_expected_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", "blueprint", "demo"])

    assert result.exit_code == 0, result.output
    root = Path("demo")
    for name in [
        "blueprint.yaml",
        "pyproject.toml",
        "__main__.py",
        "config.yaml",
        "Makefile",
        "docker-compose.yaml",
        "Dockerfile",
        "demo.spec",
        ".env",
        ".gitignore",
        ".dockerignore",
        ".python-version",
        "README.md",
    ]:
        assert (root / name).exists(), f"Missing blueprint file: {name}"


def test_init_blueprint_starts_with_no_agents(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["init", "blueprint", "demo"])

    blueprint = (Path("demo") / "blueprint.yaml").read_text(encoding="utf-8")
    assert "name: demo" in blueprint
    assert "namespace: default" in blueprint
    assert "provider: bubbleran" in blueprint
    assert "agents: {}" in blueprint


def test_init_blueprint_pins_adk_with_telemetry_extra(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["init", "blueprint", "demo", "--model-provider", "ollama"])

    pyproject = (Path("demo") / "pyproject.toml").read_text(encoding="utf-8")
    assert '"bat-adk[ollama,telemetry]>=2026.9.10a0"' in pyproject


def test_init_blueprint_dispatcher_has_an_empty_managed_region(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["init", "blueprint", "demo"])

    main = (Path("demo") / "__main__.py").read_text(encoding="utf-8")
    assert "# bat:agents:begin" in main
    assert "# bat:agents:end" in main
    assert "APPS: set[str] = set()" in main
    # The frozen binary needs static imports, so the dispatcher must never
    # reach for importlib.
    assert "importlib" not in main


def test_init_blueprint_makefile_discovers_agents(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["init", "blueprint", "demo"])

    makefile = (Path("demo") / "Makefile").read_text(encoding="utf-8")
    # Auto-discovery is what keeps `bat add agent` from having to edit this file.
    assert "AGENTS := $(patsubst %/config.yaml,%,$(wildcard */config.yaml))" in makefile
    assert "CONFIG_PATH=$@/config.yaml uv run . $@" in makefile


def test_init_blueprint_refuses_a_non_empty_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    existing = Path("demo")
    existing.mkdir()
    (existing / "keep.txt").write_text("mine\n", encoding="utf-8")

    result = runner.invoke(app, ["init", "blueprint", "demo"])

    assert result.exit_code != 0
    assert "already exists" in result.output
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd cli && .venv/bin/python -m pytest tests/test_create_blueprint.py -v`
Expected: FAIL — `No such command 'blueprint'`.

- [ ] **Step 3: Extract the shared template renderer**

Create `cli/src/create/rendering.py`:

```python
"""Rendering shared by the agent and blueprint scaffolds."""

from __future__ import annotations

from pathlib import Path


def render_template(
    templates_dir: Path, template_file: str, replacements: dict[str, str]
) -> str:
    """Read ``template_file`` from ``templates_dir``, substituting ``__KEY__``."""
    template_path = templates_dir / template_file
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    rendered = template_path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        rendered = rendered.replace(f"__{key}__", value)

    return rendered
```

In `cli/src/create/agent.py`, replace the body of `_render_template` with a delegation, keeping its signature:

```python
def _render_template(template_file: str, replacements: dict[str, str]) -> str:
    return render_template(TEMPLATES_DIR, template_file, replacements)
```

and add `from .rendering import render_template` to its imports.

- [ ] **Step 4: Write the blueprint templates**

`cli/src/create/templates/blueprint/blueprint.yaml`:

```yaml
# What this blueprint is, and which agents it ships. `bat add agent` appends
# to `agents`, and the same list drives the dispatcher in __main__.py and the
# services in docker-compose.yaml.
name: __BLUEPRINT_NAME__
namespace: __BLUEPRINT_NAMESPACE__
provider: __BLUEPRINT_PROVIDER__
version: __BLUEPRINT_VERSION__
agents: {}
```

`cli/src/create/templates/blueprint/pyproject.toml.template`:

```toml
[project]
name = "__BLUEPRINT_NAME__"
version = "1.0.0"
description = "__BLUEPRINT_DESCRIPTION__"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    # One project for every agent in this blueprint: one venv, one lockfile,
    # one frozen binary. `telemetry` pulls in OpenTelemetry/OpenInference --
    # without it the SDK's tracers are no-ops, nothing is exported, and
    # `bat eval` reports zero tokens for every episode. The other extra is the
    # provider's LangChain integration, which `langchain` itself does not ship.
    "bat-adk[__BAT_ADK_EXTRAS__]>=__BAT_ADK_VERSION__"
]
```

`cli/src/create/templates/blueprint/__main__.py`:

```python
"""Blueprint entrypoint: picks which agent this process runs.

Two ways in, because the two runtimes disagree on how an agent is selected.
Locally (`uv run . <agent>`, `make <agent>`, `docker run ... <agent>`) it is
argv[1]. On a cluster the Deployment is rendered with no command and no args,
so AGENT_MODE is the only selector available. argv wins when present, so a
local run always overrides a stale env var.

The block between the `bat:agents` markers is generated by `bat add agent`
from blueprint.yaml. Edit the rest of this file freely; that block will be
rewritten. Its imports are static and lazy on purpose: PyInstaller has to see
them to freeze them into the binary, and importing inside the branch keeps one
agent's dependencies off another agent's startup path.
"""

import os
import sys


# bat:agents:begin -- generated by `bat add agent`; do not edit by hand
APPS: set[str] = set()


def _load(app: str):
    raise SystemExit(f"Unknown agent: {app}")


# bat:agents:end


def main() -> None:
    if len(sys.argv) > 1:
        app = sys.argv[1]
        sys.argv = sys.argv[1:]
    else:
        app = os.getenv("AGENT_MODE", "")

    if app not in APPS:
        known = "|".join(sorted(APPS)) or "none yet -- run `bat add agent`"
        print(f"Usage: agent <{known}>")
        print("(or set AGENT_MODE to one of the same values)")
        sys.exit(1)

    _load(app)()


if __name__ == "__main__":
    main()
```

`cli/src/create/templates/blueprint/config.yaml`:

```yaml
# The blueprint's own config, read when nothing points CONFIG_PATH elsewhere.
# Each agent has its own config.yaml next to its code, and that is the one that
# matters: `make <agent>` and `bat eval` both set CONFIG_PATH to it.
endpoint:
  url: http://localhost
  port: 9900

checkpoints: false
```

`cli/src/create/templates/blueprint/Makefile`:

```make
DOCKER_REGISTRY ?= INSERT_YOUR_DOCKER_REGISTRY_HERE
REPO ?= YOUR_REPOSITORY/__BLUEPRINT_NAME__
VERSION ?= $(shell git describe --tags --always --abbrev --dirty)
IMAGE_TAG := $(DOCKER_REGISTRY)/$(REPO):$(VERSION)

# Every directory holding a config.yaml is an agent, so `bat add agent` never
# has to edit this file.
AGENTS := $(patsubst %/config.yaml,%,$(wildcard */config.yaml))

.PHONY: build push clean $(AGENTS)

# Run ONE agent locally against its own config.yaml. Both halves matter: the
# SDK reads CONFIG_PATH at startup (the blueprint's ./config.yaml is not the
# agent's), and the name selects which agent the shared entrypoint runs.
#   make <agent>
$(AGENTS):
	CONFIG_PATH=$@/config.yaml uv run . $@

build:
	docker build $(if $(NO_CACHE),--no-cache) \
		--build-arg VERSION=$(VERSION) \
		--tag $(IMAGE_TAG) .

push: build
	docker push $(IMAGE_TAG)

clean:
	-docker rmi $(IMAGE_TAG)
```

`cli/src/create/templates/blueprint/docker-compose.yaml`:

```yaml
# One image, one service per agent, selected by the service's command.
# `bat add agent` rewrites the block between the markers; everything outside
# it is yours.
name: __BLUEPRINT_NAME__

# bat:agents:begin -- generated by `bat add agent`; do not edit by hand
services: {}
# bat:agents:end
```

`cli/src/create/templates/blueprint/Dockerfile`:

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base
FROM debian:bookworm-slim AS runtime

# Stage 1: Builder
FROM base AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    binutils

WORKDIR /app
COPY . .

RUN uv lock
RUN uv sync --locked

RUN uv add pyinstaller
RUN uv run pyinstaller __BLUEPRINT_NAME__.spec

RUN strip dist/__BLUEPRINT_NAME__

# Collect each agent's runtime files, keeping the directory layout the
# CONFIG_PATH values in the compose file and the Deployment expect. Discovered
# rather than listed, so adding an agent needs no edit here.
RUN mkdir -p /out && \
    for config in */config.yaml; do \
        agent=$(dirname "$config"); \
        mkdir -p "/out/$agent"; \
        cp "$agent/config.yaml" "/out/$agent/"; \
        if [ -f "$agent/agent.json" ]; then cp "$agent/agent.json" "/out/$agent/"; fi; \
    done

# Stage 2: Runtime
FROM runtime

COPY --from=builder /app/dist/__BLUEPRINT_NAME__ /app/
COPY --from=builder /app/config.yaml /app/
COPY --from=builder /out/ /app/

WORKDIR /app
# The agent is the first argument, or AGENT_MODE when there is none:
#   docker run <image> netops
ENTRYPOINT ["./__BLUEPRINT_NAME__"]
```

`cli/src/create/templates/blueprint/blueprint.spec`:

```python
# -*- mode: python ; coding: utf-8 -*-

# No `datas`: each agent's config.yaml and agent.json are read from the
# filesystem at runtime (the Dockerfile copies them next to the binary), not
# frozen in. The agents themselves are reached through the static imports in
# __main__.py, which is what PyInstaller follows.
datas = []

a = Analysis(
	['__main__.py'],
	pathex=[],
	binaries=[],
	datas=datas,
	hiddenimports=[],
	hookspath=[],
	hooksconfig={},
	runtime_hooks=[],
	excludes=[],
	noarchive=False,
	optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
	pyz,
	a.scripts,
	a.binaries,
	a.datas,
	[],
	name='__BLUEPRINT_NAME__',
	debug=False,
	bootloader_ignore_signals=False,
	strip=False,
	upx=True,
	upx_exclude=[],
	runtime_tmpdir=None,
	console=True,
	disable_windowed_traceback=False,
	argv_emulation=False,
	target_arch=None,
	codesign_identity=None,
	entitlements_file=None,
)
```

`cli/src/create/templates/blueprint/.env.template`:

```
# Secrets only. Everything else (endpoints, models, telemetry) lives in the
# blueprint's config.yaml and in each agent's own config.yaml.
__API_KEY_LINE__
```

`cli/src/create/templates/blueprint/README.md`:

```markdown
# __BLUEPRINT_NAME__

A BAT blueprint: one uv project, one frozen binary, several agents.

## Layout

- `blueprint.yaml` — identity and the list of agents
- `__main__.py` — shared entrypoint; picks the agent from `argv[1]`, or from
  `AGENT_MODE` when there is none
- `config.yaml` — the blueprint's own config; each agent has its own
- `<agent>/` — one directory per agent, added with `bat add agent`

## Running an agent

```bash
make <agent>            # CONFIG_PATH=<agent>/config.yaml uv run . <agent>
docker compose up       # every agent at once
```

## Adding one

```bash
bat add agent <name>
```
```

- [ ] **Step 5: Write the scaffold module**

Create `cli/src/create/blueprint.py`:

```python
"""Scaffold a blueprint: one uv project holding several agents.

The standalone agent scaffold in :mod:`create.agent` writes a project per
agent. A blueprint inverts that: the project, the entrypoint, the PyInstaller
spec and the packaging live once at the root, and each agent is a package
inside it. `bat add agent` (Task 3) fills it in; `bat init blueprint` creates
it empty.
"""

from __future__ import annotations

from pathlib import Path

from .agent import (
    BAT_ADK_VERSION,
    _PROVIDER_API_KEY_VAR,
    _bat_adk_extras,
)
from .rendering import render_template

BLUEPRINT_TEMPLATES_DIR = (
    Path(__file__).resolve().parent / "templates" / "blueprint"
)

AGENTS_BEGIN = "# bat:agents:begin"
AGENTS_END = "# bat:agents:end"

# Templates copied byte for byte, with no substitution.
_STATIC_FILES = (".gitignore", ".dockerignore", ".python-version")


def _render(template_file: str, replacements: dict[str, str]) -> str:
    return render_template(
        BLUEPRINT_TEMPLATES_DIR, template_file, replacements
    )


def _api_key_line(model_provider: str) -> str:
    key_var = _PROVIDER_API_KEY_VAR.get(model_provider.lower())
    if key_var:
        return f"{key_var}=your-api-key-here"
    return f"# The '{model_provider}' provider needs no API key."


def create_blueprint_scaffold(
    target_dir: Path,
    *,
    force: bool = False,
    model_provider: str = "openai",
    namespace: str = "default",
    provider: str = "bubbleran",
    version: str = "v0.1.0",
) -> list[Path]:
    """Write an empty blueprint into ``target_dir``.

    Empty means no agents: `bat add agent` adds those. The provider is fixed
    here rather than per agent because the extras live in the one shared
    pyproject.
    """
    name = target_dir.name.lower()

    if target_dir.exists() and not target_dir.is_dir():
        raise FileExistsError(
            f"Target path '{target_dir}' already exists and is not a "
            "directory. Choose a different blueprint name or remove the file."
        )
    if target_dir.is_dir() and any(target_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Target directory '{target_dir}' already exists and is not "
            "empty. Use --force to overwrite files."
        )

    target_dir.mkdir(parents=True, exist_ok=True)

    substitutions = {
        "BLUEPRINT_NAME": name,
        "BLUEPRINT_NAMESPACE": namespace,
        "BLUEPRINT_PROVIDER": provider,
        "BLUEPRINT_VERSION": version,
        "BLUEPRINT_DESCRIPTION": f"{name.upper()} blueprint",
        "BAT_ADK_EXTRAS": _bat_adk_extras(model_provider),
        "BAT_ADK_VERSION": BAT_ADK_VERSION,
        "API_KEY_LINE": _api_key_line(model_provider),
    }

    # (written name, template name). The spec is named after the blueprint so
    # the binary is, too.
    rendered: list[tuple[str, str]] = [
        ("blueprint.yaml", "blueprint.yaml"),
        ("pyproject.toml", "pyproject.toml.template"),
        ("__main__.py", "__main__.py"),
        ("config.yaml", "config.yaml"),
        ("Makefile", "Makefile"),
        ("docker-compose.yaml", "docker-compose.yaml"),
        ("Dockerfile", "Dockerfile"),
        (f"{name}.spec", "blueprint.spec"),
        (".env", ".env.template"),
        ("README.md", "README.md"),
    ]

    created: list[Path] = []
    for written_name, template_name in rendered:
        path = target_dir / written_name
        if path.exists() and not force:
            continue
        path.write_text(_render(template_name, substitutions), encoding="utf-8")
        created.append(path)

    for static_name in _STATIC_FILES:
        path = target_dir / static_name
        if path.exists() and not force:
            continue
        path.write_text(
            (BLUEPRINT_TEMPLATES_DIR / static_name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        created.append(path)

    return created
```

- [ ] **Step 6: Add the command**

In `cli/src/cli.py`, import the scaffold next to the existing create imports:

```python
from create.blueprint import create_blueprint_scaffold
```

and add this command after `create_new_agent` (which ends at line 206):

```python
@init_app.command("blueprint")
def create_new_blueprint(
    name: str = typer.Argument(
        help="Name of the blueprint directory to create."
    ),
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        "-o",
        help="Directory where the blueprint folder will be created.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing files when the target directory exists.",
    ),
    model_provider: str = typer.Option(
        "openai",
        "--model-provider",
        "--model_provider",
        help="Model provider for every agent in this blueprint; selects the bat-adk extra in pyproject.toml.",
    ),
    namespace: str = typer.Option(
        "default", "--namespace", help="Value written to blueprint.yaml."
    ),
    provider: str = typer.Option(
        "bubbleran", "--provider", help="Value written to blueprint.yaml."
    ),
) -> None:
    blueprint_name = _validate_agent_name(name)
    target_dir = output_dir / blueprint_name.lower()

    try:
        created_files = create_blueprint_scaffold(
            target_dir,
            force=force,
            model_provider=model_provider,
            namespace=namespace,
            provider=provider,
        )
    except FileExistsError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(
        f"Created BAT blueprint in: {target_dir.resolve()}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Files written: {len(created_files)}")
    typer.echo("Next: cd into it and run `bat add agent <name>`.")
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cd cli && .venv/bin/python -m pytest tests/test_create_blueprint.py -v`
Expected: 6 passed.

- [ ] **Step 8: Run the whole suite**

Run: `cd cli && .venv/bin/python -m pytest tests -q`
Expected: all green, including the untouched `test_create_new_agent.py`.

- [ ] **Step 9: Document it**

In `cli/docs/bat-cli.md`, add `blueprint` under `init` in the command tree (the tree starts around line 64) and a short section explaining the blueprint shape: one uv project, one binary, agents as directories, and `make <agent>` / `CONFIG_PATH` for running one.

- [ ] **Step 10: Checkpoint (commit held — see Global Constraints)**

```bash
git add cli/src/create/rendering.py cli/src/create/blueprint.py \
        cli/src/create/templates/blueprint cli/src/create/agent.py \
        cli/src/cli.py cli/tests/test_create_blueprint.py cli/docs/bat-cli.md
# git commit -m "feat(cli): scaffold a blueprint with `bat init blueprint`"
```

---

### Task 3: `bat add agent`

**Files:**
- Create: `cli/src/create/templates/blueprint/agent/__init__.py`
- Create: `cli/src/create/templates/blueprint/agent/app.py`
- Create: `cli/src/create/templates/blueprint/agent/config.yaml`
- Modify: `cli/src/create/blueprint.py` (append the region helper, the generators and `add_agent_to_blueprint`)
- Modify: `cli/src/create/templates/blueprint/blueprint.yaml` (wrap `agents` in the managed region)
- Modify: `cli/src/cli.py` (add the `add agent` command after `add_new_client`)
- Modify: `cli/docs/bat-cli.md`
- Test: `cli/tests/test_add_agent.py`

**Interfaces:**
- Consumes: `create.blueprint.AGENTS_BEGIN`, `AGENTS_END`, `BLUEPRINT_TEMPLATES_DIR`, `_render`; `create.agent._agent_class_name(agent_dir_name: str) -> str`, `_build_agent_json_content(agent_dir_name: str) -> str`, `_build_src_init_content(agent_dir_name: str) -> str`, `_build_graph_content(agent_dir_name: str, clients: list[str] | None) -> str`, `_write_llm_clients(llm_clients_dir: Path, *, clients: list[str] | None, force: bool) -> list[Path]`; `project.find_blueprint_root`.
- Produces:
  - `create.blueprint.replace_managed_region(text: str, body: str, *, begin: str = AGENTS_BEGIN, end: str = AGENTS_END) -> str`
  - `create.blueprint.blueprint_agent_names(blueprint_root: Path) -> list[str]`
  - `create.blueprint.add_agent_to_blueprint(blueprint_root: Path, name: str, *, port: int | None = None, model: str = "gpt-4o-mini", model_provider: str = "openai", clients: list[str] | None = None, force: bool = False, class_name_source: str | None = None) -> list[Path]`

- [ ] **Step 1: Wrap `agents` in a managed region**

Change the last line of `cli/src/create/templates/blueprint/blueprint.yaml` from `agents: {}` to:

```yaml
# bat:agents:begin -- generated by `bat add agent`; do not edit by hand
agents: {}
# bat:agents:end
```

The `test_init_blueprint_starts_with_no_agents` assertion (`"agents: {}" in blueprint`) still holds.

- [ ] **Step 2: Write the failing tests**

Create `cli/tests/test_add_agent.py`:

```python
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


def test_add_agent_outside_a_blueprint_is_rejected(tmp_path, monkeypatch) -> None:
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd cli && .venv/bin/python -m pytest tests/test_add_agent.py -v`
Expected: FAIL — `No such command 'agent'` for the `add` group.

- [ ] **Step 4: Write the agent templates**

`cli/src/create/templates/blueprint/agent/__init__.py`:

```python
from .app import run

__all__ = ["run"]
```

`cli/src/create/templates/blueprint/agent/app.py`:

```python
"""Entrypoint for the __AGENT_NAME__ agent.

`run` is what the blueprint's __main__.py imports once this agent is
selected, which is why the import of the graph lives inside this module
rather than in the dispatcher.
"""

from bat.agent import AgentApplication

from .src.graph import (
    __AGENT_CLASS_NAME__AgentGraph,
    __AGENT_CLASS_NAME__AgentState,
)


def run() -> None:
    AgentApplication(
        AgentGraphType=__AGENT_CLASS_NAME__AgentGraph,
        AgentStateType=__AGENT_CLASS_NAME__AgentState,
    ).run()
```

`cli/src/create/templates/blueprint/agent/config.yaml`:

```yaml
# This agent's own config. The process runs from the blueprint root, so every
# path here is relative to that root -- and CONFIG_PATH must name this file
# (`make __AGENT_NAME__` does it) or the SDK reads the blueprint's instead.
agent_card: __AGENT_NAME__/agent.json

endpoint:                              # where this agent binds (the eval reads this too)
  url: http://localhost
  port: __PORT__

model:
  provider: __MODEL_PROVIDER__
  name: __MODEL__
  # base_url: http://localhost:11434   # optional (e.g. for ollama)

checkpoints: false

telemetry:
  # service_name: __AGENT_NAME__       # optional (defaults to the agent name)
  # project_name: __BLUEPRINT_NAME__   # Phoenix project; agents sharing a trace must share this
  # privacy: none                      # none (default) | content | names | full
  output: []                           # telemetry is ON when output has >=1 entry

# remote-agents:
#   - name: Other Agent
#     url: http://localhost:9901
#     protocol: a2a

# mcp-servers:
#   - name: my-mcp
#     url: http://localhost:8000
```

- [ ] **Step 5: Write the region helper and generators**

Append to `cli/src/create/blueprint.py`:

```python
import yaml

from .agent import (
    _agent_class_name,
    _build_agent_json_content,
    _build_graph_content,
    _build_src_init_content,
    _write_llm_clients,
)

_DEFAULT_PORT = 9900


def replace_managed_region(
    text: str, body: str, *, begin: str = AGENTS_BEGIN, end: str = AGENTS_END
) -> str:
    """Return ``text`` with everything between the markers replaced by ``body``.

    The marker lines themselves are kept verbatim, so their trailing comment
    ("generated by ...") survives, and anything outside them is untouched.
    """
    lines = text.splitlines()
    begin_index: int | None = None
    end_index: int | None = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if begin_index is None:
            if stripped.startswith(begin):
                begin_index = index
        elif stripped.startswith(end):
            end_index = index
            break

    if begin_index is None or end_index is None:
        raise ValueError(
            f"Managed region {begin!r}..{end!r} not found: the file no longer "
            "has the markers `bat add agent` writes between."
        )

    new_lines = lines[: begin_index + 1] + body.splitlines() + lines[end_index:]
    return "\n".join(new_lines) + "\n"


def blueprint_agent_names(blueprint_root: Path) -> list[str]:
    """The agents registered in ``blueprint.yaml``, sorted."""
    data = yaml.safe_load(
        (blueprint_root / "blueprint.yaml").read_text(encoding="utf-8")
    )
    agents = (data or {}).get("agents") or {}
    return sorted(agents)


def _dispatcher_region(agent_names: list[str]) -> str:
    """The dispatcher's APPS set and `_load`, rebuilt from the agent list.

    Explicit branches with the import inside: PyInstaller follows static
    imports to decide what to freeze, and keeping each one in its branch means
    selecting one agent never imports another's dependencies.
    """
    if not agent_names:
        return (
            "APPS: set[str] = set()\n"
            "\n"
            "\n"
            "def _load(app: str):\n"
            '    raise SystemExit(f"Unknown agent: {app}")\n'
        )

    literal = ", ".join(f'"{name}"' for name in agent_names)
    lines = [f"APPS: set[str] = {{{literal}}}", "", "", "def _load(app: str):"]
    for index, name in enumerate(agent_names):
        keyword = "if" if index == 0 else "elif"
        lines.append(f'    {keyword} app == "{name}":')
        lines.append(f"        from {name} import run")
        lines.append("")
        lines.append("        return run")
    lines.append('    raise SystemExit(f"Unknown agent: {app}")')
    return "\n".join(lines) + "\n"


def _compose_region(agent_names: list[str]) -> str:
    """The compose `services` block, rebuilt from the agent list.

    One image for the whole blueprint; the service's command is the selector,
    and CONFIG_PATH names the agent's config the same way the Makefile does.
    `network_mode: host` keeps the localhost ports in each config.yaml valid
    between agents.
    """
    if not agent_names:
        return "services: {}"

    lines = ["services:"]
    for name in agent_names:
        lines.extend(
            [
                f"  {name}:",
                "    build: .",
                f'    command: ["{name}"]',
                "    environment:",
                f"      CONFIG_PATH: {name}/config.yaml",
                "    env_file:",
                "      - .env",
                "    network_mode: host",
            ]
        )
    return "\n".join(lines)


def _next_free_port(blueprint_root: Path, agent_names: list[str]) -> int:
    ports: list[int] = []
    for name in agent_names:
        config_path = blueprint_root / name / "config.yaml"
        if not config_path.is_file():
            continue
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        port = (data.get("endpoint") or {}).get("port")
        if isinstance(port, int):
            ports.append(port)
    return max(ports) + 1 if ports else _DEFAULT_PORT


def _rewrite_region(path: Path, body: str) -> None:
    path.write_text(
        replace_managed_region(path.read_text(encoding="utf-8"), body),
        encoding="utf-8",
    )


def add_agent_to_blueprint(
    blueprint_root: Path,
    name: str,
    *,
    port: int | None = None,
    model: str = "gpt-4o-mini",
    model_provider: str = "openai",
    clients: list[str] | None = None,
    force: bool = False,
    class_name_source: str | None = None,
) -> list[Path]:
    """Create agent ``name`` inside ``blueprint_root`` and register it.

    Registration means three files: the ``agents`` map in blueprint.yaml, the
    dispatcher's managed region in __main__.py, and the services block in
    docker-compose.yaml. The Makefile discovers agents from the filesystem and
    needs no edit.

    Raises:
        ValueError: If ``name`` is not importable as a Python module, or an
            agent with that name is already registered.
    """
    agent_dir_name = name.lower()
    if not agent_dir_name.isidentifier():
        raise ValueError(
            f"'{name}' cannot be an agent name: the directory becomes a Python "
            "package the blueprint's entrypoint imports, so it must be a valid "
            "identifier (letters, digits and underscores, not starting with a "
            "digit)."
        )

    existing = blueprint_agent_names(blueprint_root)
    if agent_dir_name in existing and not force:
        raise ValueError(
            f"Blueprint '{blueprint_root.name}' already has an agent named "
            f"'{agent_dir_name}'."
        )

    resolved_port = (
        port if port is not None else _next_free_port(blueprint_root, existing)
    )
    class_source = class_name_source if class_name_source is not None else name

    agent_dir = blueprint_root / agent_dir_name
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "src").mkdir(parents=True, exist_ok=True)
    (agent_dir / "src" / "llm_clients").mkdir(parents=True, exist_ok=True)

    substitutions = {
        "AGENT_NAME": agent_dir_name,
        "AGENT_CLASS_NAME": _agent_class_name(class_source),
        "BLUEPRINT_NAME": blueprint_root.name,
        "PORT": str(resolved_port),
        "MODEL": model,
        "MODEL_PROVIDER": model_provider,
    }

    created: list[Path] = []

    def _write(relative: str, content: str) -> None:
        path = agent_dir / relative
        if path.exists() and not force:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(path)

    _write("__init__.py", _render("agent/__init__.py", substitutions))
    _write("app.py", _render("agent/app.py", substitutions))
    _write("config.yaml", _render("agent/config.yaml", substitutions))
    _write("agent.json", _build_agent_json_content(agent_dir_name))
    _write("src/__init__.py", _build_src_init_content(class_source))
    _write("src/graph.py", _build_graph_content(class_source, clients))

    created.extend(
        _write_llm_clients(
            agent_dir / "src" / "llm_clients", clients=clients, force=force
        )
    )

    # --- registration -----------------------------------------------------
    names = sorted({*existing, agent_dir_name})

    blueprint_path = blueprint_root / "blueprint.yaml"
    agents_body = yaml.safe_dump(
        {"agents": {agent: {} for agent in names}}, sort_keys=True
    ).rstrip("\n")
    _rewrite_region(blueprint_path, agents_body)
    created.append(blueprint_path)

    main_path = blueprint_root / "__main__.py"
    _rewrite_region(main_path, _dispatcher_region(names))
    created.append(main_path)

    compose_path = blueprint_root / "docker-compose.yaml"
    _rewrite_region(compose_path, _compose_region(names))
    created.append(compose_path)

    return created
```

Note on `_build_src_init_content` and `_build_graph_content`: both take the name they derive the class name from, so pass `class_source` (the name as typed), matching how `create_agent_scaffold` uses `class_name_source`.

- [ ] **Step 6: Add the command**

In `cli/src/cli.py`, extend the import and add the command after `add_new_client`:

```python
from create.blueprint import add_agent_to_blueprint, create_blueprint_scaffold
from project import find_blueprint_root
```

```python
@add_app.command("agent")
def add_new_agent(
    name: str = typer.Argument(help="Name of the agent to add to the blueprint."),
    clients: str | None = typer.Option(
        None,
        "--clients",
        "-c",
        help="Optional comma-separated LLM client names to generate.",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help="Port written to the agent's config.yaml. Defaults to one past the highest already used in the blueprint.",
    ),
    model: str = typer.Option(
        "gpt-4o-mini", "--model", help="Model written to the agent's config.yaml."
    ),
    model_provider: str = typer.Option(
        "openai",
        "--model-provider",
        "--model_provider",
        help="Model provider written to the agent's config.yaml.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing files for this agent."
    ),
) -> None:
    blueprint_root = find_blueprint_root(Path.cwd())
    if blueprint_root is None:
        typer.secho(
            "No blueprint.yaml found here or in any parent directory. Run this "
            "command inside a blueprint, or create one with `bat init blueprint`.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    agent_name = _validate_agent_name(name)

    try:
        created_files = add_agent_to_blueprint(
            blueprint_root,
            agent_name,
            port=port,
            model=model,
            model_provider=model_provider,
            clients=_parse_clients_option(clients),
            force=force,
        )
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(
        f"Added agent '{agent_name.lower()}' to blueprint "
        f"'{blueprint_root.name}'.",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Files written: {len(created_files)}")
    typer.echo(f"Run it with: make {agent_name.lower()}")
```

If `_validate_agent_name` rejects a dash before `add_agent_to_blueprint` sees it, `test_add_agent_rejects_a_name_that_is_not_importable` still passes — check which layer rejects it and keep both guards; the one in `add_agent_to_blueprint` is what protects direct callers.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cd cli && .venv/bin/python -m pytest tests/test_add_agent.py -v`
Expected: 10 passed.

- [ ] **Step 8: Run the whole suite**

Run: `cd cli && .venv/bin/python -m pytest tests -q`
Expected: all green.

- [ ] **Step 9: Document it**

In `cli/docs/bat-cli.md`, add `agent` under `add` in the command tree and show the two-step flow: `bat init blueprint demo`, `cd demo`, `bat add agent netops`, `make netops`.

- [ ] **Step 10: Checkpoint (commit held — see Global Constraints)**

```bash
git add cli/src/create/blueprint.py cli/src/create/templates/blueprint \
        cli/src/cli.py cli/tests/test_add_agent.py cli/docs/bat-cli.md
# git commit -m "feat(cli): add an agent to a blueprint with `bat add agent`"
```

---

### Task 4: `bat eval` inside a blueprint

**Files:**
- Modify: `cli/src/eval/commands.py` — `_validate_agent_root` (95-108), `_agent_url_from_config` (111-131), `_patch_agent_config` (134-162), `_restore_agent_config` (164-170), `_inject_judge_api_key` (39-93), `_find_agent_python` (221-230), `_start_agent_process` (338-354), `eval_init` (383-...), `eval_show` (456-...), `eval_run` (474-...)
- Test: `cli/tests/test_eval_blueprint.py`
- Modify: `cli/docs/bat-cli.md`

**Interfaces:**
- Consumes: `project.AgentTarget`, `project.ProjectError`, `project.resolve_agent_target`.
- Produces: nothing new for later tasks; the private helpers change shape:
  - `_resolve_target() -> AgentTarget` (new, wraps `resolve_agent_target(Path.cwd())` and re-raises `ProjectError` as `typer.BadParameter`)
  - `_agent_url_from_config(config_path: Path) -> str`
  - `_patch_agent_config(config_path: Path, overrides: Mapping[str, Any]) -> str | None`
  - `_restore_agent_config(config_path: Path, original: str | None) -> None`
  - `_start_agent_process(target: AgentTarget, env: dict[str, str]) -> subprocess.Popen`

- [ ] **Step 1: Write the failing test**

Create `cli/tests/test_eval_blueprint.py`. It mirrors `test_eval_run_starts_agent_and_runs_orchestrator` in `cli/tests/test_eval_commands.py:198`, whose `fake_popen` already captures the command, cwd and env:

```python
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
    venv_python = root / ".venv" / "bin"
    venv_python.mkdir(parents=True, exist_ok=True)
    (venv_python / "python").write_text("", encoding="utf-8")

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
    (agent / "eval" / "input" / "tasks.json").write_text("[]\n", encoding="utf-8")
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
    blueprint_config_before = (tmp_path / "config.yaml").read_text(encoding="utf-8")
    captured: dict = {}
    _patch_eval(monkeypatch, captured, agent)
    monkeypatch.chdir(agent)

    runner.invoke(app, ["eval", "run"])

    during = yaml.safe_load(captured["config_during_run"])
    assert during["telemetry"]["output"][0]["type"] == "local"
    # The blueprint's shared config is none of the eval's business.
    assert (tmp_path / "config.yaml").read_text(encoding="utf-8") == (
        blueprint_config_before
    )
    # And the agent's is put back.
    after = yaml.safe_load((agent / "config.yaml").read_text(encoding="utf-8"))
    assert after["telemetry"]["output"] == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd cli && .venv/bin/python -m pytest tests/test_eval_blueprint.py -v`
Expected: FAIL — the agent-root check rejects the directory ("does not look like an agent root. Missing: pyproject.toml").

- [ ] **Step 3: Thread `AgentTarget` through the eval**

In `cli/src/eval/commands.py`:

Add the import and replace `_validate_agent_root`:

```python
from project import AgentTarget, ProjectError, resolve_agent_target
```

```python
def _resolve_target() -> AgentTarget:
    """The agent this command acts on, standalone or inside a blueprint."""
    try:
        return resolve_agent_target(Path.cwd())
    except ProjectError as exc:
        raise typer.BadParameter(str(exc)) from exc
```

Change the four config helpers to take the config path directly:

```python
def _agent_url_from_config(config_path: Path) -> str:
    ...
    data: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        ...
```

```python
def _patch_agent_config(
    config_path: Path, overrides: Mapping[str, Any]
) -> str | None:
    ...


def _restore_agent_config(config_path: Path, original: str | None) -> None:
    ...
```

(remove the `config_path = agent_root / "config.yaml"` line from each body).

Change the launcher:

```python
def _start_agent_process(
    target: AgentTarget, env: dict[str, str]
) -> subprocess.Popen:
    """Start the agent, from wherever `uv run .` works for its shape.

    Inside a blueprint that is the blueprint root, the agent is named as an
    argument, and CONFIG_PATH has to point at the agent's own config -- the
    SDK would otherwise read the blueprint's shared one.
    """
    process_env = {**env, **target.run_env}
    try:
        return subprocess.Popen(
            target.run_command,
            cwd=target.project_root,
            env=process_env,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            "Cannot execute 'uv run'. Ensure uv is installed and available in PATH."
        ) from exc
```

In `eval_init`, `eval_show` and `eval_run`, replace

```python
agent_root = Path.cwd()
_validate_agent_root(agent_root)
```

with

```python
target = _resolve_target()
```

and then map each former `agent_root` use:

| was | becomes | why |
|---|---|---|
| `agent_root / "eval"` | `target.agent_dir / "eval"` | the eval scaffold belongs to the agent |
| `load_eval_config(agent_root, ...)` | `load_eval_config(target.agent_dir, ...)` | its paths are relative to the agent |
| `_find_agent_python(agent_root)` | `_find_agent_python(target.project_root)` | the venv is the blueprint's |
| `_agent_url_from_config(agent_root)` | `_agent_url_from_config(target.config_path)` | |
| `_patch_agent_config(agent_root, ...)` | `_patch_agent_config(target.config_path, ...)` | |
| `_restore_agent_config(agent_root, ...)` | `_restore_agent_config(target.config_path, ...)` | |
| `_inject_judge_api_key(cfg.judge, agent_root, env)` | `_inject_judge_api_key(cfg.judge, target.project_root, env)` | `.env` sits with the project |
| `_start_agent_process(agent_root, server_env)` | `_start_agent_process(target, server_env)` | |

- [ ] **Step 4: Run the new test**

Run: `cd cli && .venv/bin/python -m pytest tests/test_eval_blueprint.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the whole suite**

Run: `cd cli && .venv/bin/python -m pytest tests -q`
Expected: all green. `cli/tests/test_eval_commands.py` must pass **unedited** — it is the standalone-shape regression test (`["uv", "run", "."]`, no `CONFIG_PATH`).

- [ ] **Step 6: Document it**

In `cli/docs/bat-cli.md`, in the evaluation section, note that `bat eval` runs from the agent directory in both shapes, and that inside a blueprint it starts the agent as `CONFIG_PATH=<agent>/config.yaml uv run . <agent>` from the blueprint root.

- [ ] **Step 7: Checkpoint (commit held — see Global Constraints)**

```bash
git add cli/src/eval/commands.py cli/tests/test_eval_blueprint.py cli/docs/bat-cli.md
# git commit -m "feat(cli): run `bat eval` against an agent inside a blueprint"
```

---

## Out of scope, on purpose

`bat build` and `bat push` still build a per-agent image and do not know about the blueprint's single `.spec`; running them from a blueprint agent directory is undefined and should be left alone until that work is planned. Cluster descriptors (`aifabric.yaml`, `composition-model.yaml`, per-agent RBAC `rules`) are not generated. Nested MCP-server projects like `cluster-view` keep their own pyproject and are not managed by `bat add agent`.
