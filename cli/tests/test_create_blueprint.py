"""Tests for `bat init blueprint`: the shape of a fresh blueprint."""

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


def test_init_blueprint_pins_adk_with_telemetry_extra(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(
        app, ["init", "blueprint", "demo", "--model-provider", "ollama"]
    )

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
    # The frozen binary needs static imports, so the dispatcher must
    # never reach for importlib.
    assert "importlib" not in main


def test_init_blueprint_makefile_discovers_agents(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)

    runner.invoke(app, ["init", "blueprint", "demo"])

    makefile = (Path("demo") / "Makefile").read_text(encoding="utf-8")
    # Auto-discovery keeps `bat add agent` from having to edit this file.
    assert (
        "AGENTS := $(patsubst %/config.yaml,%,$(wildcard */config.yaml))"
        in makefile
    )
    assert "CONFIG_PATH=$@/config.yaml uv run . $@" in makefile


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
