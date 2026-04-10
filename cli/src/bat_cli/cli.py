from pathlib import Path

import typer

from bat_cli.add.client import add_clients_to_existing_agent
from bat_cli.build.build import build_image
from bat_cli.create.agent import create_agent_scaffold
from bat_cli.push.push import push_image


app = typer.Typer(help="CLI tool to create, build, and manage AI agents developed with the BubbleRAN Agentic Toolkit.")
init_app = typer.Typer(help="Create new BAT resources.")
add_app = typer.Typer(help="Add new components to existing BAT agents.")

app.add_typer(init_app, name="init")
app.add_typer(add_app, name="add")
app.command("build", help="Build the Docker image for the agent.")(build_image)
app.command("push", help="Push the Docker image to a registry.")(push_image)


def _parse_clients_option(raw_clients: str | None) -> list[str] | None:
    if raw_clients is None:
        return None
    parsed_clients = [client.strip() for client in raw_clients.split(",") if client.strip()]
    if not parsed_clients:
        raise typer.BadParameter("Provide at least one client name, for example: reformulator,planner,executor")
    return parsed_clients


@init_app.command("agent")
def create_new_agent(
    name: str = typer.Argument("default", help="Name of the agent directory to create."),
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
) -> None:
    target_dir = output_dir / name
    parsed_clients = _parse_clients_option(clients)

    try:
        created_files = create_agent_scaffold(target_dir, force=force, clients=parsed_clients)
    except FileExistsError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"Created BAT agent skeleton in: {target_dir.resolve()}", fg=typer.colors.GREEN)
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
        created_files = add_clients_to_existing_agent(current_dir, clients=parsed_clients, force=force)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"Updated LLM clients in: {llm_clients_dir.resolve()}", fg=typer.colors.GREEN)
    typer.echo(f"Files written: {len(created_files)}")



def main() -> None:
    app()


if __name__ == "__main__":
    main()
