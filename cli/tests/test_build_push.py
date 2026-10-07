"""Tests for building and pushing images: the Makefiles the scaffolds write,
and `bat build` / `bat push`, which run them."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app

runner = CliRunner()

needs_make = pytest.mark.skipif(
    shutil.which("make") is None, reason="make is not installed"
)


@pytest.fixture(autouse=True)
def _no_image_settings_from_the_shell(monkeypatch) -> None:
    """make reads DOCKER_REGISTRY and REPO from the environment."""
    monkeypatch.delenv("DOCKER_REGISTRY", raising=False)
    monkeypatch.delenv("REPO", raising=False)


def _blueprint(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "blueprint", "demo"]).exit_code == 0
    return tmp_path / "demo"


def _standalone(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init", "agent", "solo"]).exit_code == 0
    return tmp_path / "solo"


def _dry_run(root: Path, *args: str) -> subprocess.CompletedProcess:
    """What `make` would run, without running it (and outside any git
    checkout: tmp_path is not one)."""
    return subprocess.run(
        ["make", "-n", *args], cwd=root, capture_output=True, text=True
    )


def _docker_line(output: str, verb: str) -> str:
    """The `docker <verb>` command in make's output, on one line."""
    for line in output.replace("\\\n", " ").splitlines():
        if line.strip().startswith(f"docker {verb}"):
            return " ".join(line.split())
    raise AssertionError(f"no docker {verb} in:\n{output}")


@needs_make
@pytest.mark.parametrize("scaffold", [_blueprint, _standalone])
def test_make_build_works_outside_git_with_no_settings(
    tmp_path, monkeypatch, scaffold
) -> None:
    """Out of the box the tag is <project>:dev, a valid local image -- the
    one docker-compose.yaml runs -- and no version is stamped on the cards."""
    root = scaffold(tmp_path, monkeypatch)

    result = _dry_run(root, "build")

    assert result.returncode == 0, result.stderr
    build = _docker_line(result.stdout, "build")
    assert f"--tag {root.name}:dev" in build
    assert "--build-arg VERSION= " in build
    assert "fatal" not in result.stderr


@needs_make
@pytest.mark.parametrize("scaffold", [_blueprint, _standalone])
def test_make_build_takes_registry_repo_and_version(
    tmp_path, monkeypatch, scaffold
) -> None:
    root = scaffold(tmp_path, monkeypatch)

    result = _dry_run(
        root,
        "build",
        "DOCKER_REGISTRY=hub.bubbleran.com",
        "REPO=orama/labs/demo",
        "VERSION=1.2.3",
    )

    assert result.returncode == 0, result.stderr
    build = _docker_line(result.stdout, "build")
    assert "--tag hub.bubbleran.com/orama/labs/demo:1.2.3" in build
    assert "--build-arg VERSION=1.2.3" in build


@needs_make
@pytest.mark.parametrize("scaffold", [_blueprint, _standalone])
def test_make_push_needs_a_registry(tmp_path, monkeypatch, scaffold) -> None:
    """Without one, `docker push demo:dev` would go to Docker Hub."""
    root = scaffold(tmp_path, monkeypatch)

    result = _dry_run(root, "push")

    assert result.returncode != 0
    assert "DOCKER_REGISTRY" in result.stderr
    assert "docker" not in result.stdout


_real_run = subprocess.run


def _is_dry_run(cmd) -> bool:  # noqa: ANN001
    return list(cmd[:2]) == ["make", "-n"]


def _record_runs(monkeypatch, module: str) -> list[dict]:
    """Record what would really run; dry runs (`make -n`) still run."""
    runs: list[dict] = []

    def fake_run(cmd, *args, cwd, **kwargs):  # noqa: ANN001
        if _is_dry_run(cmd):
            return _real_run(cmd, *args, cwd=cwd, **kwargs)
        runs.append({"cmd": cmd, "cwd": Path(cwd)})

    monkeypatch.setattr(f"{module}.subprocess.run", fake_run)
    return runs


def test_bat_build_runs_make_build(tmp_path, monkeypatch) -> None:
    """With nothing to pass, `bat build` is exactly `make build`: the
    Makefile's defaults apply."""
    root = _blueprint(tmp_path, monkeypatch)
    runs = _record_runs(monkeypatch, "image")
    monkeypatch.chdir(root)

    result = runner.invoke(app, ["build"])

    assert result.exit_code == 0, result.output
    assert runs == [{"cmd": ["make", "build"], "cwd": root}]


def test_bat_build_passes_what_it_was_given(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    runs = _record_runs(monkeypatch, "image")
    monkeypatch.chdir(root)

    result = runner.invoke(
        app,
        [
            "build",
            "--docker-registry",
            "hub.bubbleran.com",
            "--repo",
            "orama/labs/demo",
            "--version",
            "1.2.3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert runs[0]["cmd"] == [
        "make",
        "build",
        "DOCKER_REGISTRY=hub.bubbleran.com",
        "REPO=orama/labs/demo",
        "VERSION=1.2.3",
    ]


def test_bat_build_has_no_no_cache_flag(tmp_path, monkeypatch) -> None:
    """`make build NO_CACHE=1` still does it, for whoever needs it."""
    root = _blueprint(tmp_path, monkeypatch)
    monkeypatch.chdir(root)

    result = runner.invoke(app, ["build", "--no-cache"])

    assert result.exit_code != 0
    assert "No such option" in result.output


@needs_make
def test_a_registry_exported_in_the_shell_counts(
    tmp_path, monkeypatch
) -> None:
    """make reads it itself, so `bat build` has nothing to pass."""
    root = _blueprint(tmp_path, monkeypatch)
    # build and push share the one subprocess module: one recorder sees both.
    runs = _record_runs(monkeypatch, "image")
    monkeypatch.chdir(root)
    monkeypatch.setenv("DOCKER_REGISTRY", "hub.bubbleran.com")

    built = runner.invoke(app, ["build"])
    pushed = runner.invoke(app, ["push"])

    assert built.exit_code == 0, built.output
    assert "built successfully: hub.bubbleran.com/demo:dev" in built.output
    assert pushed.exit_code == 0, pushed.output
    assert [run["cmd"] for run in runs] == [["make", "build"], ["make", "push"]]


@needs_make
def test_bat_docker_variables_are_not_read(tmp_path, monkeypatch) -> None:
    """The registry lives in the Makefile now, not in .env."""
    root = _blueprint(tmp_path, monkeypatch)
    _record_runs(monkeypatch, "image")
    monkeypatch.chdir(root)
    with (root / ".env").open("a", encoding="utf-8") as env:
        env.write("BAT_DOCKER_REGISTRY=hub.bubbleran.com\n")
    monkeypatch.setenv("BAT_DOCKER_REGISTRY", "hub.bubbleran.com")

    result = runner.invoke(app, ["push"])

    assert result.exit_code == 1
    assert "no Docker registry set" in result.output


def test_bat_build_from_an_agent_folder_builds_the_blueprint(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    monkeypatch.chdir(root)
    runner.invoke(app, ["add", "agent", "netops"])
    runs = _record_runs(monkeypatch, "image")
    monkeypatch.chdir(root / "netops")

    result = runner.invoke(app, ["build"])

    assert result.exit_code == 0, result.output
    assert runs == [{"cmd": ["make", "build"], "cwd": root}]


def test_bat_build_runs_a_standalone_agents_make_build(
    tmp_path, monkeypatch
) -> None:
    root = _standalone(tmp_path, monkeypatch)
    runs = _record_runs(monkeypatch, "image")
    monkeypatch.chdir(root)

    result = runner.invoke(app, ["build"])

    assert result.exit_code == 0, result.output
    assert runs == [{"cmd": ["make", "build"], "cwd": root}]


def test_bat_build_reports_a_failed_make(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    monkeypatch.chdir(root)

    def failing_run(cmd, *args, cwd, **kwargs):  # noqa: ANN001
        if _is_dry_run(cmd):
            return _real_run(cmd, *args, cwd=cwd, **kwargs)
        raise subprocess.CalledProcessError(2, cmd)

    monkeypatch.setattr("image.subprocess.run", failing_run)

    result = runner.invoke(app, ["build"])

    assert result.exit_code == 2
    assert "make build failed" in result.output


def test_bat_push_runs_make_push(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    runs = _record_runs(monkeypatch, "image")
    monkeypatch.chdir(root)

    result = runner.invoke(
        app,
        [
            "push",
            "--docker-registry",
            "hub.bubbleran.com",
            "--version",
            "1.2.3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert runs == [
        {
            "cmd": [
                "make",
                "push",
                "DOCKER_REGISTRY=hub.bubbleran.com",
                "VERSION=1.2.3",
            ],
            "cwd": root,
        }
    ]


# -- the registry ------------------------------------------------------------


@pytest.mark.parametrize(
    ("image", "registry"),
    [
        ("hub.bubbleran.com/orama/demo:1.2.3", "hub.bubbleran.com"),
        ("localhost:5000/demo:dev", "localhost:5000"),
        ("localhost/demo:dev", "localhost"),
        ("demo:dev", None),
        ("orama/demo:1.2.3", None),
        ("INSERT_YOUR_DOCKER_REGISTRY_HERE/YOUR_REPOSITORY/demo:", None),
    ],
)
def test_registry_of_an_image(image, registry) -> None:
    """Docker's rule: the first part is a registry only with a dot, a port
    or as localhost -- anything else is a path on Docker Hub."""
    from image import registry_of

    assert registry_of(image) == registry


@needs_make
def test_bat_build_without_a_registry_builds_for_this_machine(
    tmp_path, monkeypatch
) -> None:
    """Building never needs a registry; only pushing does."""
    root = _blueprint(tmp_path, monkeypatch)
    runs = _record_runs(monkeypatch, "image")
    monkeypatch.chdir(root)

    result = runner.invoke(app, ["build"])

    assert result.exit_code == 0, result.output
    assert runs == [{"cmd": ["make", "build"], "cwd": root}]
    assert "built successfully: demo:dev" in result.output
    assert "registry" not in result.output.lower()


@needs_make
def test_bat_build_names_the_image_it_built(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    _record_runs(monkeypatch, "image")
    monkeypatch.chdir(root)

    result = runner.invoke(
        app,
        ["build", "--docker-registry", "hub.bubbleran.com", "--version", "1"],
    )

    assert result.exit_code == 0, result.output
    assert "built successfully: hub.bubbleran.com/demo:1" in result.output


@needs_make
def test_a_registry_in_the_makefile_counts(tmp_path, monkeypatch) -> None:
    """A Makefile with its own default registry, like the supervisor's."""
    root = _blueprint(tmp_path, monkeypatch)
    makefile = root / "Makefile"
    makefile.write_text(
        makefile.read_text(encoding="utf-8").replace(
            "DOCKER_REGISTRY ?=\n", "DOCKER_REGISTRY ?= hub.bubbleran.com\n"
        ),
        encoding="utf-8",
    )
    _record_runs(monkeypatch, "image")
    monkeypatch.chdir(root)

    result = runner.invoke(app, ["build"])

    assert result.exit_code == 0, result.output


def test_bat_build_without_a_makefile_or_registry_builds(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "Dockerfile").write_text(
        "FROM scratch\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path / "agent")
    runs = _record_runs(monkeypatch, "image")

    result = runner.invoke(app, ["build"])

    assert result.exit_code == 0, result.output
    assert runs[0]["cmd"][-3:] == ["--tag", "agent:latest", "."]


def test_bat_push_without_a_makefile_or_registry_is_refused(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "agent").mkdir()
    monkeypatch.chdir(tmp_path / "agent")
    runs = _record_runs(monkeypatch, "image")

    result = runner.invoke(app, ["push"])

    assert result.exit_code == 1
    assert runs == []
    assert "no Docker registry set" in result.output
    assert "BubbleRAN can provide" in result.output


@needs_make
def test_bat_push_without_a_registry_is_refused(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    runs = _record_runs(monkeypatch, "image")
    monkeypatch.chdir(root)

    result = runner.invoke(app, ["push"])

    assert result.exit_code == 1
    assert runs == []
    assert "Can't push demo: no Docker registry set" in result.output
    assert "demo:dev" in result.output
    assert "set DOCKER_REGISTRY in the Makefile" in result.output
    assert "BubbleRAN can provide" in result.output
    assert "--force" not in result.output


@needs_make
def test_a_makefile_pushing_to_no_registry_is_refused(
    tmp_path, monkeypatch
) -> None:
    """An older Makefile has no guard of its own: its placeholder is a path,
    not a registry, so `docker push` would go to Docker Hub."""
    root = _blueprint(tmp_path, monkeypatch)
    (root / "Makefile").write_text(
        "IMAGE := INSERT_YOUR_DOCKER_REGISTRY_HERE/YOUR_REPOSITORY/demo:dev\n"
        "build:\n\tdocker build --tag $(IMAGE) .\n"
        "push:\n\tdocker push $(IMAGE)\n",
        encoding="utf-8",
    )
    runs = _record_runs(monkeypatch, "image")
    monkeypatch.chdir(root)

    result = runner.invoke(app, ["push"])

    assert result.exit_code == 1
    assert runs == []
    assert "no Docker registry set" in result.output
