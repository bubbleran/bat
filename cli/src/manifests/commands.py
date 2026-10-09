from functools import partial
from itertools import takewhile
from pathlib import Path
from typing import Any, Callable

import typer
import yaml

from create.rendering import dump_yaml
from image import make_variables, plan
from project import fail, require_blueprint_root

from .aifabric import ManifestUpdate, update_aifabric
from .composition_model import render, update_composition_model


def _update(
    path: Path,
    update: Callable[[dict[str, Any] | None], ManifestUpdate],
    render: Callable[[dict[str, Any]], str] = dump_yaml,
) -> None:
    """Rewrite the manifest at ``path`` with ``update`` and report what
    changed. PyYAML drops comments, so the leading block, where a manifest
    is documented, is carried over by hand."""
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    try:
        result = update(yaml.safe_load(text))
    except (ValueError, yaml.YAMLError) as exc:
        fail(f"{path}: {exc}")
    header = takewhile(
        lambda line: line.startswith("#") or not line.strip(),
        text.splitlines(keepends=True),
    )
    path.write_text("".join(header) + render(result.document), encoding="utf-8")

    for warning in result.warnings:
        typer.secho(f"Warning: {warning}", fg=typer.colors.YELLOW)
    typer.secho(
        f"{'Updated' if text else 'Wrote'} {path}", fg=typer.colors.GREEN
    )
    if result.added:
        typer.echo(f"  added: {', '.join(result.added)}")
    if text:  # on a new file every entry is new
        for change in result.changes:
            typer.echo(f"  {change}")
        if not result.added and not result.changes:
            typer.echo("  No changes.")


def generate_aifabric(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="The AIFabric file. Default: aifabric.yaml at the blueprint root.",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help="metadata.name. Default: the file's, else <blueprint>-fabric.",
    ),
    namespace: str | None = typer.Option(
        None,
        "--namespace",
        help=(
            "metadata.namespace. Default: the file's, else the "
            "CompositionModel's, else 'default'."
        ),
    ),
    image_pull_secret: list[str] = typer.Option(
        [],
        "--image-pull-secret",
        help="Add a spec.imagePullSecrets entry (repeatable).",
    ),
    telemetry_endpoint: str | None = typer.Option(
        None,
        "--telemetry-endpoint",
        help=(
            "The OTLP collector the agents export to in the cluster "
            "(spec.telemetry.endpoint), e.g. "
            "http://phoenix-svc.phoenix.svc.cluster.local:6006."
        ),
    ),
) -> None:
    """Write or update the blueprint's AIFabric from its agents' configs.

    Each run refreshes what the agents' config.yaml say (LLMs, dependencies,
    MCP servers) and keeps what was written by hand, and other blueprints'
    agents.
    """
    root = require_blueprint_root()
    _update(
        output or root / "aifabric.yaml",
        partial(
            update_aifabric,
            root,
            name=name,
            namespace=namespace,
            image_pull_secrets=image_pull_secret,
            telemetry_endpoint=telemetry_endpoint,
        ),
    )


def generate_composition_model(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "The CompositionModel file. Default: composition-model.yaml at "
            "the blueprint root, where `bat manifests aifabric` reads it."
        ),
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help=(
            "metadata.name, which an AIFabric's models start with. Default: "
            "the file's, else the blueprint's."
        ),
    ),
    namespace: str | None = typer.Option(
        None,
        "--namespace",
        help="metadata.namespace. Default: the file's, else 'default'.",
    ),
    docker_registry: str | None = typer.Option(
        None,
        "--docker-registry",
        help=(
            "Docker registry of the image. Default: as `bat build`: "
            "DOCKER_REGISTRY in the shell, then the Makefile's."
        ),
    ),
    repo: str | None = typer.Option(
        None,
        "--repo",
        help="Image repository path. Default: as `bat build`.",
    ),
    version: str | None = typer.Option(
        None,
        "--version",
        help="Image version. Default: as `bat build` (the git tag or commit).",
    ),
) -> None:
    """Write or update the blueprint's CompositionModel: one deployment mode
    per agent, running the image `bat build` tags with the same options.

    Each run adds the agents that have no mode and points the blueprint's
    modes at that image; RBAC rules, resources, probes, env and MCP modes
    written by hand are kept.
    """
    root = require_blueprint_root()
    dry_run = plan(
        "build", root, make_variables(docker_registry, repo, version)
    )
    if dry_run.image is None:
        error = dry_run.error.strip()
        fail(
            f"Can't tell which image `make build` tags in {root}"
            + (f":\n{error}" if error else ".")
        )
    # An older Makefile outside git: VERSION is empty and has no default.
    if dry_run.image.endswith(":"):
        fail(
            f"`make build` would tag {dry_run.image}, with no version: "
            "pass --version."
        )

    _update(
        output or root / "composition-model.yaml",
        partial(
            update_composition_model,
            root,
            image=dry_run.image,
            name=name,
            namespace=namespace,
        ),
        render,
    )
    typer.echo(f"  image: {dry_run.image} (push it: `bat push`, same options)")
