"""Tests for `bat init blueprint`: the shape of a fresh blueprint."""

from __future__ import annotations

import subprocess
import tomllib
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
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["init", "blueprint", "demo"])

    with pytest.raises(ProjectError, match="root of a blueprint"):
        resolve_agent_target(Path("demo"))


def test_init_blueprint_pins_adk_with_telemetry_extra(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(
        app, ["init", "blueprint", "demo", "--model-provider", "ollama"]
    )

    pyproject = (Path("demo") / "pyproject.toml").read_text(encoding="utf-8")
    assert '"bat-adk[ollama,telemetry]>=2026.9.29a0"' in pyproject


def test_init_blueprint_installs_pyinstaller_with_the_project(
    tmp_path, monkeypatch
) -> None:
    """The Dockerfile only runs `uv sync --frozen`: pyinstaller has to come
    from the project's own dev group, or the freeze step has no tool."""
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["init", "blueprint", "demo"])

    pyproject = tomllib.loads(
        (Path("demo") / "pyproject.toml").read_text(encoding="utf-8")
    )
    dev = pyproject["dependency-groups"]["dev"]
    assert any(req.startswith("pyinstaller") for req in dev)


def test_init_blueprint_dispatcher_has_an_empty_managed_region(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["init", "blueprint", "demo"])

    main = (Path("demo") / "__main__.py").read_text(encoding="utf-8")
    assert "# bat:agents:begin" in main
    assert "# bat:agents:end" in main
    assert "APPS: set[str] = set()" in main
    # The frozen binary needs static imports, so the dispatcher must
    # never reach for importlib.
    assert "importlib" not in main


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
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "blueprint", "demo"])
    monkeypatch.chdir(tmp_path / "demo")
    runner.invoke(app, ["add", "agent", "netops"])

    commands = _make_dry_run(tmp_path / "demo", "netops")

    assert (
        "CONFIG_PATH=netops/config.yaml UV_ENV_FILE=.env uv run . netops"
        in commands
    )


def test_makefile_build_locks_before_building(tmp_path, monkeypatch) -> None:
    """The image installs from the lockfile (`uv sync --frozen`), and a fresh
    blueprint has none yet."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "blueprint", "demo"])

    commands = _make_dry_run(tmp_path / "demo", "build").splitlines()

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


def test_init_blueprint_image_has_no_floor_by_default(
    tmp_path, monkeypatch, image_floor
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["init", "blueprint", "demo"])

    assert image_floor(Path("demo")) == "none"


def test_init_blueprint_sets_one_floor_for_the_whole_image(
    tmp_path, monkeypatch, image_floor
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app, ["init", "blueprint", "demo", "--telemetry-privacy", "Content"]
    )

    assert result.exit_code == 0, result.output
    assert image_floor(Path("demo")) == "content"


def test_init_blueprint_rejects_an_unknown_floor_before_writing(
    tmp_path, monkeypatch
) -> None:
    """A typo must not silently degrade to ``none`` -- the ADK itself falls
    back to it -- nor leave a half-created blueprint behind."""
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app, ["init", "blueprint", "demo", "--telemetry-privacy", "secret"]
    )

    assert result.exit_code != 0
    assert "secret" in result.output
    assert not Path("demo").exists()
