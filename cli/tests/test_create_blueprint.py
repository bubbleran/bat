"""Tests for `bat init blueprint`: the shape of a fresh blueprint."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app
from project import ProjectError, resolve_agent_target

runner = CliRunner()


def test_init_blueprint_creates_expected_files(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["init", "blueprint", "demo"])

    assert result.exit_code == 0, result.output
    # Exactly these: no manifest and no root config.yaml, like the real
    # blueprints -- each agent brings its own config.
    assert sorted(path.name for path in Path("demo").iterdir()) == [
        ".dockerignore",
        ".env",
        ".gitignore",
        ".python-version",
        "Dockerfile",
        "Makefile",
        "README.md",
        "__main__.py",
        "demo.spec",
        "docker-compose.yaml",
        "pyproject.toml",
    ]


def test_init_blueprint_is_recognised_as_a_blueprint(
    tmp_path, monkeypatch
) -> None:
    """There is no manifest to find, so the layout itself has to be what
    `bat add agent` and `bat eval` recognise -- before any agent exists."""
    root = _fresh_blueprint(tmp_path, monkeypatch)

    with pytest.raises(ProjectError, match="root of a blueprint"):
        resolve_agent_target(root)


def test_init_blueprint_pins_adk_with_telemetry_extra(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(
        app, ["init", "blueprint", "demo", "--model-provider", "ollama"]
    )

    pyproject = (Path("demo") / "pyproject.toml").read_text(encoding="utf-8")
    assert '"bat-adk[ollama,telemetry]>=2026.10.9a0"' in pyproject


def test_init_blueprint_installs_pyinstaller_with_the_project(
    tmp_path, monkeypatch
) -> None:
    """The Dockerfile only runs `uv sync --frozen`: pyinstaller has to come
    from the project's own dev group, or the freeze step has no tool."""
    root = _fresh_blueprint(tmp_path, monkeypatch)

    pyproject = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )
    dev = pyproject["dependency-groups"]["dev"]
    assert any(req.startswith("pyinstaller") for req in dev)


def _stub_agent(root: Path, name: str) -> None:
    """An agent folder whose run() prints how it was started."""
    (root / name).mkdir()
    (root / name / "agent.json").write_text("{}", encoding="utf-8")
    (root / name / "__init__.py").write_text(
        "import os\nimport sys\n\n\ndef run():\n"
        "    print('ran', __name__, sys.argv)\n"
        "    print('config', os.environ.get('CONFIG_PATH'))\n",
        encoding="utf-8",
    )


def _dispatch(
    root: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run the blueprint's entrypoint the way `uv run . <agent>` does."""
    base = {k: v for k, v in os.environ.items() if k != "CONFIG_PATH"}
    return subprocess.run(
        [sys.executable, "__main__.py", *args],
        cwd=root,
        capture_output=True,
        text=True,
        env={**base, **(env or {})},
    )


def _fresh_blueprint(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "blueprint", "demo"])
    return tmp_path / "demo"


def test_the_dispatcher_runs_the_agent_named_first(
    tmp_path, monkeypatch
) -> None:
    """No list of agents to keep in sync: any agent folder can be run, and
    the rest of the arguments are handed on to it."""
    root = _fresh_blueprint(tmp_path, monkeypatch)
    _stub_agent(root, "netops")

    result = _dispatch(root, "netops", "--verbose")

    assert result.returncode == 0, result.stderr
    assert "ran netops ['netops', '--verbose']" in result.stdout


def test_the_dispatcher_points_the_agent_at_its_own_config(
    tmp_path, monkeypatch
) -> None:
    """`uv run . netops` works on its own: run from the blueprint root, the
    agent's config is netops/config.yaml, not ./config.yaml."""
    root = _fresh_blueprint(tmp_path, monkeypatch)
    _stub_agent(root, "netops")

    result = _dispatch(root, "netops")

    assert "config netops/config.yaml" in result.stdout, result.stderr


def test_a_config_path_already_set_is_kept(tmp_path, monkeypatch) -> None:
    """docker-compose.yaml and `bat eval` set CONFIG_PATH themselves."""
    root = _fresh_blueprint(tmp_path, monkeypatch)
    _stub_agent(root, "netops")

    result = _dispatch(root, "netops", env={"CONFIG_PATH": "/etc/n.yaml"})

    assert "config /etc/n.yaml" in result.stdout, result.stderr


def test_a_config_at_the_root_is_left_to_the_sdk(
    tmp_path, monkeypatch
) -> None:
    """The operator mounts the agent's rendered config at /app/config.yaml
    and sets no CONFIG_PATH: the SDK's ./config.yaml has to stay the one
    read."""
    root = _fresh_blueprint(tmp_path, monkeypatch)
    _stub_agent(root, "netops")
    (root / "config.yaml").write_text("{}", encoding="utf-8")

    result = _dispatch(root, "netops")

    assert "config None" in result.stdout, result.stderr


def test_the_dispatcher_lists_the_agents_it_can_run(
    tmp_path, monkeypatch
) -> None:
    root = _fresh_blueprint(tmp_path, monkeypatch)
    _stub_agent(root, "netops")
    _stub_agent(root, "hermes")

    result = _dispatch(root)

    assert result.returncode == 1
    assert "Usage: demo <hermes|netops>" in result.stdout


def test_the_dispatcher_runs_nothing_that_is_not_an_agent(
    tmp_path, monkeypatch
) -> None:
    """Only a folder with an agent card is an agent: not a stdlib module,
    not a path."""
    root = _fresh_blueprint(tmp_path, monkeypatch)
    _stub_agent(root, "netops")

    for name in ("json", "../netops", ""):
        result = _dispatch(root, name)
        assert result.returncode == 1, name
        assert "Usage: demo <netops>" in result.stdout


def test_a_fresh_dispatcher_says_there_are_no_agents_yet(
    tmp_path, monkeypatch
) -> None:
    root = _fresh_blueprint(tmp_path, monkeypatch)

    result = _dispatch(root)

    assert result.returncode == 1
    assert "none yet" in result.stdout


def test_the_spec_bundles_every_agent_folder(tmp_path, monkeypatch) -> None:
    """The dispatcher imports agents by name, which PyInstaller cannot see:
    the spec finds the agent folders itself and bundles them."""
    root = _fresh_blueprint(tmp_path, monkeypatch)
    monkeypatch.chdir(root)
    runner.invoke(app, ["add", "agent", "netops"])
    runner.invoke(app, ["add", "agent", "hermes"])
    captured: dict = {}

    def analysis(scripts, **options):
        captured.update(options)
        return types.SimpleNamespace(
            pure=[], scripts=[], binaries=[], datas=[]
        )

    hooks = types.ModuleType("PyInstaller.utils.hooks")
    hooks.copy_metadata = lambda distribution: []
    for name in ("PyInstaller", "PyInstaller.utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)

    spec = (root / "demo.spec").read_text(encoding="utf-8")
    exec(
        compile(spec, "demo.spec", "exec"),
        {
            "Analysis": analysis,
            "PYZ": lambda *args, **kwargs: None,
            "EXE": lambda *args, **kwargs: None,
            "SPECPATH": str(root),
        },
    )

    assert sorted(captured["hiddenimports"]) == ["hermes", "netops"]


def _make_dry_run(root: Path, target: str) -> str:
    result = subprocess.run(
        ["make", "-n", target],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_makefile_runs_an_agent_it_was_never_told_about(
    tmp_path, monkeypatch
) -> None:
    """Agents are discovered, so `bat add agent` never edits the Makefile;
    and the .env goes to uv explicitly, which an editable bat-adk needs."""
    root = _fresh_blueprint(tmp_path, monkeypatch)
    monkeypatch.chdir(root)
    runner.invoke(app, ["add", "agent", "netops"])

    commands = _make_dry_run(root, "netops")

    assert (
        "CONFIG_PATH=netops/config.yaml UV_ENV_FILE=.env uv run . netops"
        in commands
    )


def test_makefile_build_locks_before_building(tmp_path, monkeypatch) -> None:
    """The image installs from the lockfile (`uv sync --frozen`), and a fresh
    blueprint has none yet."""
    root = _fresh_blueprint(tmp_path, monkeypatch)

    commands = _make_dry_run(root, "build").splitlines()

    lock = commands.index("uv lock")
    build = next(i for i, line in enumerate(commands) if "docker build" in line)
    assert lock < build


def test_init_blueprint_refuses_a_non_empty_directory(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    existing = Path("demo")
    existing.mkdir()
    (existing / "keep.txt").write_text("mine\n", encoding="utf-8")

    result = runner.invoke(app, ["init", "blueprint", "demo"])

    assert result.exit_code != 0
    assert "already exists" in result.output


# A template placeholder (``__BLUEPRINT_NAME__``), or one an editor's
# Markdown formatter turned into bold (``**BLUEPRINT_NAME**``). Python's own
# dunders (``__main__``) are lowercase, so they do not match.
_PLACEHOLDER = re.compile(r"(__|\*\*)[A-Z][A-Z_]*[A-Z](__|\*\*)")


def test_no_placeholder_survives_in_a_scaffolded_blueprint(
    tmp_path, monkeypatch
) -> None:
    root = _fresh_blueprint(tmp_path, monkeypatch)
    monkeypatch.chdir(root)
    runner.invoke(app, ["add", "agent", "netops"])

    leftovers = {
        str(path.relative_to(tmp_path)): _PLACEHOLDER.findall(
            path.read_text(encoding="utf-8")
        )
        for path in root.rglob("*")
        if path.is_file()
    }

    assert {name: found for name, found in leftovers.items() if found} == {}
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# demo\n")


def test_a_blueprint_cannot_be_created_inside_a_blueprint(
    tmp_path, monkeypatch
) -> None:
    """Its agents are the folders right below its root, so a blueprint
    created there would sit among them as a stray project."""
    root = _fresh_blueprint(tmp_path, monkeypatch)
    monkeypatch.chdir(root)

    result = runner.invoke(app, ["init", "blueprint", "inner"])

    assert result.exit_code != 0
    assert "inside blueprint 'demo'" in result.output
    assert not (root / "inner").exists()


def test_a_blueprint_cannot_be_created_inside_one_of_its_agents(
    tmp_path, monkeypatch
) -> None:
    root = _fresh_blueprint(tmp_path, monkeypatch)
    monkeypatch.chdir(root)
    runner.invoke(app, ["add", "agent", "netops"])
    monkeypatch.chdir(root / "netops")

    result = runner.invoke(app, ["init", "blueprint", "inner"])

    assert result.exit_code != 0
    assert not (root / "netops" / "inner").exists()
