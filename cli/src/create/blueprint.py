"""Scaffold a blueprint: one uv project holding several agents.

The project, the dispatcher, the PyInstaller spec and the packaging live once
at the root; each agent is a package right below it. ``bat init blueprint``
creates it empty and ``bat add agent`` fills it in.
"""

from __future__ import annotations

import keyword
from pathlib import Path

import yaml

from project import blueprint_agents, find_blueprint_root

from .agent import (
    BAT_ADK_VERSION,
    agent_class_name,
    api_key_line,
    bat_adk_extras,
    graph_substitutions,
    resolve_telemetry_privacy,
    write_llm_clients,
)
from .agent import TEMPLATES_DIR as AGENT_TEMPLATES_DIR
from .rendering import (
    dump_yaml,
    ensure_empty_dir,
    render,
    template_files,
    write_files,
)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates" / "blueprint"
_AGENT_TEMPLATES = ("agent.json.template", "src/__init__.py", "src/graph.py")
_DEFAULT_PORT = 9900


def create_blueprint_scaffold(
    target_dir: Path, *, force: bool = False, model_provider: str = "openai"
) -> list[Path]:
    """Write an empty blueprint into ``target_dir``. The provider is set
    here because its extra lives in the one shared pyproject."""
    enclosing = find_blueprint_root(target_dir.parent)
    if enclosing is not None:
        raise ValueError(
            f"'{target_dir.parent}' is inside blueprint '{enclosing.name}'. "
            "A blueprint cannot hold another one: add an agent to it with "
            "`bat add agent`, or create the blueprint somewhere else."
        )
    ensure_empty_dir(target_dir, force=force)

    name = target_dir.name.lower()
    substitutions = {
        "BLUEPRINT_NAME": name,
        "BLUEPRINT_DESCRIPTION": f"{name.upper()} blueprint",
        "BAT_ADK_EXTRAS": bat_adk_extras(model_provider),
        "BAT_ADK_VERSION": BAT_ADK_VERSION,
        "API_KEY_LINE": api_key_line(model_provider),
    }
    files = {}
    for template in template_files(TEMPLATES_DIR):
        if template.startswith("agent/"):
            continue
        # The spec is named after the blueprint, so the binary is too.
        output = (
            f"{name}.spec"
            if template == "blueprint.spec"
            else template.removesuffix(".template")
        )
        files[output] = render(TEMPLATES_DIR / template, substitutions)
    return write_files(target_dir, files, force=True)


def _agent_port(blueprint_root: Path, name: str) -> int | None:
    config_path = blueprint_root / name / "config.yaml"
    if not config_path.is_file():
        return None
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    port = (data.get("endpoint") or {}).get("port")
    return port if isinstance(port, int) else None


def _compose_service(blueprint_root: Path, name: str) -> dict:
    """One image for the whole blueprint; the command selects the agent.

    The image carries no config.yaml, so the service mounts the agent's own;
    ``network_mode: host`` keeps the localhost ports between agents valid.
    """
    service: dict = {
        "image": f"${{IMAGE_TAG:-{blueprint_root.name.lower()}:dev}}",
        "build": {"context": ".", "args": {"VERSION": "${VERSION:-}"}},
        "command": [name],
        "environment": {
            "CONFIG_PATH": f"{name}/config.yaml",
            "LOG_LEVEL": "${LOG_LEVEL:-info}",
        },
        "env_file": [".env"],
        "volumes": [f"./{name}/config.yaml:/app/{name}/config.yaml:ro"],
        "network_mode": "host",
    }
    port = _agent_port(blueprint_root, name)
    if port is not None:
        service["healthcheck"] = {
            "test": ["CMD-SHELL", f"curl -fsS http://localhost:{port}/ping"],
            "interval": "5s",
            "timeout": "2s",
            "retries": 5,
        }
    service["restart"] = "unless-stopped"
    return service


def _add_compose_services(
    blueprint_root: Path, agent: str, *, replace: bool
) -> Path:
    """Give every agent without one a compose service (``agent`` a new one
    on ``replace``); every other service is kept as it is."""
    path = blueprint_root / "docker-compose.yaml"
    compose = (
        yaml.safe_load(path.read_text(encoding="utf-8"))
        if path.is_file()
        else None
    )
    if not isinstance(compose, dict):
        compose = {"name": blueprint_root.name.lower()}
    services = compose.get("services") or {}
    for name in blueprint_agents(blueprint_root):
        if name not in services or (replace and name == agent):
            services[name] = _compose_service(blueprint_root, name)
    compose["services"] = services
    path.write_text(dump_yaml(compose), encoding="utf-8")
    return path


def _name_problem(name: str) -> str | None:
    """Why the dispatcher could not ``import`` ``name``, if it could not."""
    if "-" in name:
        return f"hyphens are not allowed. Use '{name.replace('-', '_')}'"
    if name[:1].isdigit():
        return "it can't start with a digit"
    if keyword.iskeyword(name):
        return "it is a Python keyword"
    if not name.isidentifier():
        return "use only letters, digits and underscores"
    return None


def add_agent_to_blueprint(
    blueprint_root: Path,
    name: str,
    *,
    port: int | None = None,
    model: str = "gpt-4o-mini",
    model_provider: str = "openai",
    clients: list[str] | None = None,
    force: bool = False,
    telemetry_privacy: str | None = None,
) -> list[Path]:
    """Create agent ``name`` in the blueprint. Everything but
    docker-compose.yaml finds agents by their folder.

    Raises:
        ValueError: If ``name`` is not importable as a Python module, or an
            agent with that name already exists.
    """
    directory = name.lower()
    problem = _name_problem(directory)
    if problem:
        raise ValueError(f"'{name}' can't be an agent name: {problem}.")
    floor = resolve_telemetry_privacy(telemetry_privacy)

    existing = blueprint_agents(blueprint_root)
    if directory in existing and not force:
        raise ValueError(
            f"Blueprint '{blueprint_root.name}' already has an agent named "
            f"'{directory}'."
        )
    if port is None:
        ports = [_agent_port(blueprint_root, agent) for agent in existing]
        port = max((p for p in ports if p), default=_DEFAULT_PORT - 1) + 1

    substitutions = {
        "AGENT_NAME": directory,
        "AGENT_CLASS_NAME": agent_class_name(name),
        "BLUEPRINT_NAME": blueprint_root.name,
        "PORT": str(port),
        "MODEL": model,
        "MODEL_PROVIDER": model_provider,
        "TELEMETRY_PRIVACY_FLOOR": floor,
        **graph_substitutions(clients),
    }
    files = {
        template.removeprefix("agent/"): render(
            TEMPLATES_DIR / template, substitutions
        )
        for template in template_files(TEMPLATES_DIR)
        if template.startswith("agent/")
    }
    for template in _AGENT_TEMPLATES:
        files[template.removesuffix(".template")] = render(
            AGENT_TEMPLATES_DIR / template, substitutions
        )

    agent_dir = blueprint_root / directory
    return [
        *write_files(agent_dir, files, force=force),
        *write_llm_clients(
            agent_dir / "src" / "llm_clients", clients=clients, force=force
        ),
        _add_compose_services(blueprint_root, directory, replace=force),
    ]
