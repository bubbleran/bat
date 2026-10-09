from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import typer
from typer.core import TyperGroup

from create.agent import create_agent_scaffold, write_llm_clients
from create.blueprint import add_agent_to_blueprint, create_blueprint_scaffold
from eval.commands import eval_init, eval_plot, eval_run, eval_show
from image import build_image, project_dir, push_image
from manifests.commands import generate_aifabric, generate_composition_model
from project import (
    ProjectError,
    fail,
    require_blueprint_root,
    resolve_agent_target,
)
from set.settings import set_config, set_image

_BANNER = r"""
 ____    _  _____    ____ _     ___
| __ )  / \|_   _|  / ___| |   |_ _|
|  _ \ / _ \ | |   | |   | |    | |
| |_) / ___ \| |   | |___| |___ | |
|____/_/   \_\_|    \____|_____|___|
"""


def _gradient(line: str) -> str:
    if not line:
        return ""
    colors = (51, 45, 39, 63, 99, 135)
    return (
        "".join(
            f"\033[38;5;{colors[i * len(colors) // len(line)]}m{char}"
            for i, char in enumerate(line)
        )
        + "\033[0m"
    )


class BannerGroup(TyperGroup):
    def format_help(self, ctx, formatter):
        art = "\n".join(_gradient(line) for line in _BANNER.splitlines())
        typer.echo(
            f"{art}\n\nWelcome to BubbleRAN Agentic Toolkit CLI tool.\n\n"
            "Scaffold, build, push, and evaluate BAT agents from one place.\n"
        )
        super().format_help(ctx, formatter)


app = typer.Typer(cls=BannerGroup)
init_app = typer.Typer(help="Create new BAT resources.")
add_app = typer.Typer(help="Add new components to existing BAT agents.")
set_app = typer.Typer(
    help="Set an agent's config.yaml, or the image settings in the Makefile."
)
eval_app = typer.Typer(
    help="Run local evaluation workflows for existing BAT agents."
)
manifests_app = typer.Typer(help="Generate cluster manifests for a blueprint.")

app.add_typer(init_app, name="init")
app.add_typer(add_app, name="add")
app.add_typer(set_app, name="set")
app.add_typer(eval_app, name="eval")
app.add_typer(manifests_app, name="manifests")

app.command("build")(build_image)
app.command("push")(push_image)
eval_app.command("init", help="Initialize local evaluation scaffold.")(
    eval_init
)
eval_app.command("run", help="Run evaluation using eval/eval.yaml.")(eval_run)
eval_app.command("show", help="Show the resolved evaluation configuration.")(
    eval_show
)
eval_app.command(
    "plot", help="Generate metric charts from an evaluation output folder."
)(eval_plot)
manifests_app.command("aifabric")(generate_aifabric)
manifests_app.command("composition-model")(generate_composition_model)

_CLIENTS_EXAMPLE = "reformulator,planner,executor"
_REASONING_EFFORT = typer.Option(
    None,
    "--reasoning-effort",
    help=(
        "model.reasoning_effort written to config.yaml, e.g. low. Supported "
        "by gpt-5 and later. Default: none."
    ),
)
_SERVICE_TIER = typer.Option(
    None,
    "--service-tier",
    help=(
        "model.service_tier written to config.yaml, e.g. flex. Default: "
        "none, the provider's own."
    ),
)


@app.command("version")
def show_version() -> None:
    """Show the installed bat-cli version."""
    try:
        typer.echo(f"bat-cli {version('bat-cli')}")
    except PackageNotFoundError:
        fail("bat-cli is not installed as a package; version unavailable.")


def _directory_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise typer.BadParameter("Name must not be empty.")
    if name in {".", ".."} or any(char in name for char in "/\\\x00"):
        raise typer.BadParameter(
            "Name must be a single directory name, not a path (no '/', "
            "'\\', '..', or absolute paths). Use --output-dir to choose "
            "where the folder is created."
        )
    return name


def _parse_clients(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    clients = [name.strip() for name in raw.split(",") if name.strip()]
    if not clients:
        raise typer.BadParameter(
            f"Provide at least one client name, for example: {_CLIENTS_EXAMPLE}"
        )
    return clients


@init_app.command("agent")
def create_new_agent(
    name: str = typer.Argument(help="Name of the agent directory to create."),
    clients: str | None = typer.Option(
        None,
        "--clients",
        "-c",
        help=f"Comma-separated LLM client names, e.g. {_CLIENTS_EXAMPLE}",
    ),
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        "-o",
        help="Directory where the agent folder will be created.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite the files of an existing, non-empty target directory.",
    ),
    port: int = typer.Option(
        9900, "--port", help="endpoint.port written to config.yaml."
    ),
    model: str = typer.Option(
        "gpt-4o-mini", "--model", help="model.name written to config.yaml."
    ),
    model_provider: str = typer.Option(
        "openai",
        "--model-provider",
        "--model_provider",
        help="model.provider written to config.yaml.",
    ),
    telemetry_privacy: str | None = typer.Option(
        None,
        "--privacy",
        help=(
            "Lowest telemetry privacy level this agent allows "
            "(none|content|names|full), passed to its AgentApplication as "
            "telemetry_privacy_floor. Default: none. config.yaml can raise "
            "it but never lower it."
        ),
    ),
    reasoning_effort: str | None = _REASONING_EFFORT,
    service_tier: str | None = _SERVICE_TIER,
) -> None:
    name = _directory_name(name)
    # The folder is lowercased; the class names keep the casing typed.
    target_dir = output_dir / name.lower()
    try:
        created = create_agent_scaffold(
            target_dir,
            force=force,
            clients=_parse_clients(clients),
            port=port,
            model=model,
            model_provider=model_provider,
            class_name_source=name,
            telemetry_privacy=telemetry_privacy,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
        )
    except (FileExistsError, ValueError) as exc:
        fail(str(exc))

    typer.secho(
        f"Created BAT agent skeleton in: {target_dir.resolve()}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Files written: {len(created)}")


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
        help=(
            "Model provider for every agent in this blueprint; selects "
            "the bat-adk extra in pyproject.toml."
        ),
    ),
) -> None:
    target_dir = output_dir / _directory_name(name).lower()
    try:
        created = create_blueprint_scaffold(
            target_dir, force=force, model_provider=model_provider
        )
    except (FileExistsError, ValueError) as exc:
        fail(str(exc))

    typer.secho(
        f"Created BAT blueprint in: {target_dir.resolve()}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Files written: {len(created)}")
    typer.echo("Next: cd into it and run `bat add agent <name>`.")


@add_app.command("client")
def add_new_client(
    clients: str = typer.Argument(
        help=f"Comma-separated LLM client names, e.g. {_CLIENTS_EXAMPLE}"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite client files that already exist.",
    ),
) -> None:
    llm_clients_dir = Path.cwd() / "src" / "llm_clients"
    if not llm_clients_dir.is_dir():
        fail(
            "Current directory must contain src/llm_clients. Run this "
            "command from the root of an existing agent."
        )

    created = write_llm_clients(
        llm_clients_dir, clients=_parse_clients(clients), force=force
    )
    typer.secho(
        f"Updated LLM clients in: {llm_clients_dir.resolve()}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Files written: {len(created)}")


@add_app.command("agent")
def add_new_agent(
    name: str = typer.Argument(
        help="Name of the agent to add to the blueprint."
    ),
    clients: str | None = typer.Option(
        None,
        "--clients",
        "-c",
        help="Comma-separated LLM client names to generate.",
    ),
    port: int | None = typer.Option(
        None,
        "--port",
        help=(
            "Port written to the agent's config.yaml. Defaults to one "
            "past the highest already used in the blueprint."
        ),
    ),
    model: str = typer.Option(
        "gpt-4o-mini", "--model", help="Model written to the config.yaml."
    ),
    model_provider: str = typer.Option(
        "openai",
        "--model-provider",
        "--model_provider",
        help="Model provider written to the agent's config.yaml.",
    ),
    telemetry_privacy: str | None = typer.Option(
        None,
        "--privacy",
        help=(
            "Lowest telemetry privacy level this agent allows "
            "(none|content|names|full), passed to its AgentApplication in "
            "app.py. Default: none. Each agent of a blueprint has its own."
        ),
    ),
    reasoning_effort: str | None = _REASONING_EFFORT,
    service_tier: str | None = _SERVICE_TIER,
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing files for this agent."
    ),
) -> None:
    blueprint_root = require_blueprint_root()
    name = _directory_name(name)
    try:
        created = add_agent_to_blueprint(
            blueprint_root,
            name,
            port=port,
            model=model,
            model_provider=model_provider,
            clients=_parse_clients(clients),
            force=force,
            telemetry_privacy=telemetry_privacy,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
        )
    except ValueError as exc:
        fail(str(exc))

    typer.secho(
        f"Added agent '{name.lower()}' to blueprint '{blueprint_root.name}'.",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Files written: {len(created)}")
    typer.echo(f"Run it with: make {name.lower()}")


@set_app.command("config")
def set_agent_config(
    agent: str | None = typer.Argument(
        None,
        help=(
            "The agent to set, inside a blueprint. Default: the agent folder "
            "you are in."
        ),
    ),
    port: int | None = typer.Option(
        None, "--port", help="Set endpoint.port in config.yaml."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Set model.name in config.yaml."
    ),
    model_provider: str | None = typer.Option(
        None,
        "--model-provider",
        "--model_provider",
        help="Set model.provider in config.yaml.",
    ),
    reasoning_effort: str | None = typer.Option(
        None,
        "--reasoning-effort",
        help=(
            "Set model.reasoning_effort in config.yaml, e.g. low. Supported "
            "by gpt-5 and later."
        ),
    ),
    service_tier: str | None = typer.Option(
        None,
        "--service-tier",
        help=(
            "Set model.service_tier in config.yaml, e.g. flex. Values depend "
            "on the provider."
        ),
    ),
) -> None:
    values = {
        "endpoint.port": port,
        "model.name": model,
        "model.provider": model_provider,
        "model.reasoning_effort": reasoning_effort,
        "model.service_tier": service_tier,
    }
    if all(value is None for value in values.values()):
        fail(
            "Provide at least one option to set: --port, --model, "
            "--model-provider, --reasoning-effort, --service-tier"
        )
    try:
        target = resolve_agent_target(Path.cwd(), agent)
    except ProjectError as exc:
        fail(str(exc))

    updated = set_config(target.agent_dir, values)
    typer.secho(
        f"Updated: {target.config_path.resolve()}", fg=typer.colors.GREEN
    )
    typer.echo(f"Keys updated: {', '.join(updated)}")


@set_app.command("image")
def set_image_settings(
    docker_registry: str | None = typer.Option(
        None,
        "--docker-registry",
        help="Set DOCKER_REGISTRY: the registry bat build / bat push use.",
    ),
    repo: str | None = typer.Option(
        None,
        "--repo",
        help="Set REPO: the image repository bat build / bat push use.",
    ),
) -> None:
    """Set the image settings in the Makefile (the blueprint's, from one of
    its agent folders), where make reads them as well as bat build."""
    if docker_registry is None and repo is None:
        fail("Provide at least one option to set: --docker-registry, --repo")

    makefile = project_dir(Path.cwd()) / "Makefile"
    if not makefile.is_file():
        fail(
            f"No Makefile in {makefile.parent}: pass --docker-registry and "
            "--repo to bat build and bat push instead."
        )
    try:
        updated = set_image(
            makefile, docker_registry=docker_registry, repo=repo
        )
    except ValueError as exc:
        fail(str(exc))

    typer.secho(f"Updated: {makefile.resolve()}", fg=typer.colors.GREEN)
    typer.echo(f"Keys updated: {', '.join(updated)}")


if __name__ == "__main__":
    app()
