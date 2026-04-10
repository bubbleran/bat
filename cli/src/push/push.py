import subprocess
from pathlib import Path

import typer

from image_defaults import resolve_registry, resolve_repo_name


def push_image(
    context: Path = typer.Option(
        Path("."),
        "--context",
        "-C",
        help="Directory used to infer the default repository name.",
    ),
    docker_registry: str | None = typer.Option(
        None,
        "--docker-registry",
        help=(
            "Docker registry hostname. Precedence: --docker-registry > "
            "BAT_DOCKER_REGISTRY env var (or .env in current directory) > default_registry."
        ),
    ),
    repo: str | None = typer.Option(
        None,
        "--repo",
        help=(
            "Image repository path. Precedence: --repo > BAT_DOCKER_REPO env var "
            "(or .env in current directory) > default-repository/<project-name>."
        ),
    ),
    tag: str = typer.Option(
        "latest",
        "--tag",
        help="Image tag.",
    ),
) -> None:
    context_dir = context.resolve()
    if not context_dir.is_dir():
        typer.secho(f"Context directory not found: {context_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    resolved_registry = resolve_registry(context_dir, docker_registry)
    resolved_repo = resolve_repo_name(context_dir, repo)
    image = f"{resolved_registry}/{resolved_repo}:{tag}"
    command = ["docker", "push", image]

    typer.echo(f"Pushing Docker image: {image}")
    try:
        subprocess.run(command, check=True, cwd=context_dir)
    except FileNotFoundError as exc:
        typer.secho("Docker executable not found in PATH.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except subprocess.CalledProcessError as exc:
        typer.secho("Docker push failed.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=exc.returncode or 1) from exc

    typer.secho(f"Docker image pushed successfully: {image}", fg=typer.colors.GREEN)