"""`bat build` / `bat push`: run the project's own Makefile.

The Makefile holds the recipe and the image settings (DOCKER_REGISTRY, REPO,
VERSION); the CLI passes only the flags it was given, so `bat build` and
`make build` build the same image. A dry run (`make -n`) shows the image a
target would tag or push, which is how a push without a registry is caught.
Without a Makefile the CLI runs docker itself, reading the settings as make
would.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import typer

from project import fail, find_blueprint_root

# The image in a dry run's `docker build ... --tag <image>` / `docker push`.
_PLANNED_IMAGE = {
    "build": re.compile(r"docker build\b[^\n]*?(?:--tag|-t)[ =](\S+)"),
    "push": re.compile(r"docker push (\S+)"),
}

_REGISTRY = typer.Option(
    None,
    "--docker-registry",
    help=(
        "Docker registry hostname. Default: DOCKER_REGISTRY in the shell, "
        "then the Makefile's DOCKER_REGISTRY (`bat set image "
        "--docker-registry` sets it)."
    ),
)
_REPO = typer.Option(
    None,
    "--repo",
    help=(
        "Image repository path. Default: REPO in the shell, then the "
        "Makefile's REPO (`bat set image --repo` sets it), else the "
        "project's name."
    ),
)
_VERSION = typer.Option(
    None,
    "--version",
    help=(
        "Image version, used as the tag (and the VERSION build arg). "
        "Default: the Makefile's (the git tag or commit), or latest "
        "without a Makefile."
    ),
)


def project_dir(start: Path) -> Path:
    """The project ``start`` builds: the blueprint's root from one of its
    agent folders, since the blueprint has the one image."""
    start = start.resolve()
    return find_blueprint_root(start) or start


def registry_of(image: str) -> str | None:
    """The registry ``image`` names. Per Docker, the first part is one only
    with a dot or a port, or as localhost; anything else is a Docker Hub
    path."""
    first, slash, _ = image.partition("/")
    if slash and ("." in first or ":" in first or first == "localhost"):
        return first
    return None


def make_variables(
    docker_registry: str | None, repo: str | None, version: str | None
) -> list[str]:
    """The ``NAME=value`` overrides for the flags given."""
    return [
        f"{name}={value}"
        for name, value in (
            ("DOCKER_REGISTRY", docker_registry),
            ("REPO", repo),
            ("VERSION", version),
        )
        if value
    ]


@dataclass(frozen=True)
class MakePlan:
    image: str | None  # None when the dry run shows none, or failed
    error: str = ""  # make's complaint when the dry run failed


def plan(target: str, directory: Path, variables: list[str]) -> MakePlan:
    """Dry-run ``make <target>``: nothing is built, pushed or locked."""
    try:
        dry = subprocess.run(
            ["make", "-n", target, *variables],
            cwd=directory,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return MakePlan(None)  # _run reports the missing make
    if dry.returncode != 0:
        return MakePlan(None, dry.stderr)
    found = _PLANNED_IMAGE[target].findall(dry.stdout.replace("\\\n", " "))
    return MakePlan(found[-1] if found else None)


def _run(command: list[str], directory: Path) -> None:
    typer.echo(f"Running in {directory}: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, cwd=directory)
    except FileNotFoundError:
        fail(f"{command[0]} not found in PATH.")
    except subprocess.CalledProcessError as exc:
        typer.secho(
            f"{' '.join(command[:2])} failed.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=exc.returncode) from exc


def _plain_image(
    directory: Path, registry: str | None, repo: str | None, version: str
) -> str:
    """The image of a project without a Makefile."""
    registry = registry or os.environ.get("DOCKER_REGISTRY", "").strip()
    repo = (
        repo
        or os.environ.get("REPO", "").strip()
        or re.sub(r"[^a-z0-9]+", "-", directory.name.lower()).strip("-")
        or "agent"
    )
    image = f"{repo}:{version}"
    return f"{registry}/{image}" if registry else image


def _refuse_push(
    directory: Path, image: str | None, *, makefile: bool
) -> NoReturn:
    where = (
        "set DOCKER_REGISTRY in the Makefile (bat set image --docker-registry)"
        if makefile
        else "export DOCKER_REGISTRY"
    )
    fail(
        f"Can't push {directory.name}: no Docker registry set, so "
        f"{image or 'the image'} would go to Docker Hub.\n"
        f"  Pass --docker-registry, or {where}.\n"
        "  BubbleRAN can provide a Docker registry for your images: ask your "
        "BubbleRAN contact."
    )


def _succeeded(verb: str, image: str | None) -> None:
    typer.secho(
        f"Docker image {verb} successfully" + (f": {image}" if image else "."),
        fg=typer.colors.GREEN,
    )


def build_image(
    docker_registry: str | None = _REGISTRY,
    repo: str | None = _REPO,
    version: str | None = _VERSION,
) -> None:
    """Build the Docker image of the project you are in (the blueprint,
    from one of its agent folders): runs its `make build`."""
    directory = project_dir(Path.cwd())
    if (directory / "Makefile").is_file():
        variables = make_variables(docker_registry, repo, version)
        image = plan("build", directory, variables).image
        _run(["make", "build", *variables], directory)
    else:
        dockerfile = directory / "Dockerfile"
        if not dockerfile.is_file():
            fail(f"Dockerfile not found in context: {dockerfile}")
        version = version or "latest"
        image = _plain_image(directory, docker_registry, repo, version)
        build = ["docker", "build", "--build-arg", f"VERSION={version}"]
        _run([*build, "--tag", image, "."], directory)
    _succeeded("built", image)


def push_image(
    docker_registry: str | None = _REGISTRY,
    repo: str | None = _REPO,
    version: str | None = _VERSION,
) -> None:
    """Push the Docker image of the project you are in to a registry: runs
    its `make push`."""
    directory = project_dir(Path.cwd())
    if (directory / "Makefile").is_file():
        variables = make_variables(docker_registry, repo, version)
        planned = plan("push", directory, variables)
        # The scaffolded Makefile stops without a registry before naming
        # the image; an older one would push to Docker Hub.
        if "DOCKER_REGISTRY" in planned.error:
            built = plan("build", directory, variables).image
            _refuse_push(directory, built, makefile=True)
        image = planned.image
        if image is not None and registry_of(image) is None:
            _refuse_push(directory, image, makefile=True)
        _run(["make", "push", *variables], directory)
    else:
        image = _plain_image(
            directory, docker_registry, repo, version or "latest"
        )
        if registry_of(image) is None:
            _refuse_push(directory, image, makefile=False)
        _run(["docker", "push", image], directory)
    _succeeded("pushed", image)
