import shutil
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click
import typer
from typer.core import TyperGroup

from add.client import add_clients_to_existing_agent
from build.build import build_image
from create.agent import create_agent_scaffold
from eval.commands import eval_init, eval_plot, eval_run, eval_show
from push.push import push_image
from set.env import set_agent_settings

_BANNER_COLORS = (51, 45, 39, 63, 99, 135)

_FALLBACK_BANNER = r"""
 ____    _  _____    ____ _     ___
| __ )  / \|_   _|  / ___| |   |_ _|
|  _ \ / _ \ | |   | |   | |    | |
| |_) / ___ \| |   | |___| |___ | |
|____/_/   \_\_|    \____|_____|___|
"""

_BANNER_MOTD = """
Welcome to BubbleRAN Agentic Toolkit CLI tool.

Scaffold, build, push, and evaluate BAT agents from one place.

"""


def _figlet_banner(text: str) -> str | None:
    if shutil.which("figlet") is None:
        return None
    try:
        return subprocess.check_output(
            ["figlet", "-f", "standard", text],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _colorize_line(line: str) -> str:
    n = len(line)
    if n == 0:
        return ""
    palette = _BANNER_COLORS
    pieces = []
    for i, ch in enumerate(line):
        idx = min(int(i * len(palette) / n), len(palette) - 1)
        pieces.append(f"\033[38;5;{palette[idx]}m{ch}")
    pieces.append("\033[0m")
    return "".join(pieces)


def _render_banner() -> str:
    art = _figlet_banner("BAT-CLI") or _FALLBACK_BANNER
    colored = "\n".join(_colorize_line(line) for line in art.splitlines())
    return f"{colored}\n{_BANNER_MOTD}"


class BannerGroup(TyperGroup):
    def format_help(self, ctx, formatter):
        click.echo(_render_banner().rstrip("\n") + "\n", nl=False)
        super().format_help(ctx, formatter)


app = typer.Typer(cls=BannerGroup)
init_app = typer.Typer(help="Create new BAT resources.")
add_app = typer.Typer(help="Add new components to existing BAT agents.")
set_app = typer.Typer(help="Set configuration values for existing BAT agents.")
eval_app = typer.Typer(
    help="Run local evaluation workflows for existing BAT agents."
)

app.add_typer(init_app, name="init")
app.add_typer(add_app, name="add")
app.add_typer(set_app, name="set")
app.add_typer(eval_app, name="eval")
app.command("build", help="Build the Docker image for the agent.")(build_image)
app.command("push", help="Push the Docker image to a registry.")(push_image)
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


@app.command("version", help="Show the installed bat-cli version.")
def show_version() -> None:
    try:
        installed_version = version("bat-cli")
    except PackageNotFoundError:
        typer.secho(
            "bat-cli is not installed as a package; version unavailable.",
            fg=typer.colors.YELLOW,
            err=True,
        )
        raise typer.Exit(code=1) from None

    typer.echo(f"bat-cli {installed_version}")


def _validate_agent_name(name: str) -> str:
    candidate = name.strip()
    if not candidate:
        raise typer.BadParameter("Agent name must not be empty.")
    if (
        candidate in {".", ".."}
        or "/" in candidate
        or "\\" in candidate
        or "\x00" in candidate
        or Path(candidate).is_absolute()
    ):
        raise typer.BadParameter(
            "Agent name must be a single directory name, not a path "
            "(no '/', '\\', '..', or absolute paths). Use --output-dir to "
            "choose where the agent folder is created."
        )
    return candidate


def _parse_clients_option(raw_clients: str | None) -> list[str] | None:
    if raw_clients is None:
        return None
    parsed_clients = [
        client.strip() for client in raw_clients.split(",") if client.strip()
    ]
    if not parsed_clients:
        raise typer.BadParameter(
            "Provide at least one client name, for example: reformulator,planner,executor"
        )
    return parsed_clients


@init_app.command("agent")
def create_new_agent(
    name: str = typer.Argument(help="Name of the agent directory to create."),
    clients: str | None = typer.Option(
        None,
        "--clients",
        "-c",
        help="Optional comma-separated LLM client names to generate, for example: reformulator,planner,executor",
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
        help="Overwrite existing files when the target directory exists. Use with caution as this will delete existing files in the target directory.",
    ),
    port: int = typer.Option(
        9900,
        "--port",
        help="Port value written to config.yaml (endpoint.port).",
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model",
        help="Model value written to config.yaml (model.name).",
    ),
    model_provider: str = typer.Option(
        "openai",
        "--model-provider",
        "--model_provider",
        help="Model provider written to config.yaml (model.provider).",
    ),
) -> None:
    agent_name = _validate_agent_name(name)
    # The directory (and every name derived from it) is lowercased, while the
    # State/Graph class names keep the casing the user typed.
    target_dir = output_dir / agent_name.lower()
    parsed_clients = _parse_clients_option(clients)

    try:
        created_files = create_agent_scaffold(
            target_dir,
            force=force,
            clients=parsed_clients,
            port=port,
            model=model,
            model_provider=model_provider,
            class_name_source=agent_name,
        )
    except FileExistsError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(
        f"Created BAT agent skeleton in: {target_dir.resolve()}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Files written: {len(created_files)}")


@add_app.command("client")
def add_new_client(
    clients: str = typer.Argument(
        ...,
        help="Comma-separated LLM client names to generate, for example: reformulator,planner,executor",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing client files if they already exist. Use with caution as this will delete existing client files with the same name.",
    ),
) -> None:
    current_dir = Path.cwd()
    llm_clients_dir = current_dir / "src" / "llm_clients"
    if not llm_clients_dir.is_dir():
        typer.secho(
            "Current directory must contain src/llm_clients. Run this command from the root of an existing agent.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    parsed_clients = _parse_clients_option(clients) or []

    try:
        created_files = add_clients_to_existing_agent(
            current_dir, clients=parsed_clients, force=force
        )
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(
        f"Updated LLM clients in: {llm_clients_dir.resolve()}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Files written: {len(created_files)}")


@set_app.command("env")
def set_agent_env(
    port: int | None = typer.Option(
        None,
        "--port",
        help="Set endpoint.port in config.yaml.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Set model.name in config.yaml.",
    ),
    model_provider: str | None = typer.Option(
        None,
        "--model-provider",
        "--model_provider",
        help="Set model.provider in config.yaml.",
    ),
    docker_registry: str | None = typer.Option(
        None,
        "--docker-registry",
        help="Set BAT_DOCKER_REGISTRY in .env for build/push defaults.",
    ),
    repo: str | None = typer.Option(
        None,
        "--repo",
        help="Set BAT_DOCKER_REPO in .env for build/push defaults.",
    ),
) -> None:
    current_dir = Path.cwd()
    config = current_dir / "config.yaml"
    if not config.is_file():
        typer.secho(
            "Current directory must contain config.yaml. Run this command from the root of an existing agent.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    if all(
        value is None
        for value in [port, model, model_provider, docker_registry, repo]
    ):
        typer.secho(
            "Provide at least one option to set: --port, --model, --model-provider, --docker-registry, --repo",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    written, updated_keys = set_agent_settings(
        current_dir,
        port=port,
        model=model,
        model_provider=model_provider,
        docker_registry=docker_registry,
        repo=repo,
    )

    typer.secho(
        f"Updated: {', '.join(str(p.resolve()) for p in written)}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Keys updated: {', '.join(updated_keys)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
