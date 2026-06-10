import re
from pathlib import Path
from typing import Any

import yaml


def _upsert_env_var(content: str, key: str, value: str) -> str:
    pattern = re.compile(
        rf"^\s*(?:export\s+)?{re.escape(key)}\s*=.*$", flags=re.MULTILINE
    )
    replacement = f"{key}={value}"
    if pattern.search(content):
        return pattern.sub(replacement, content, count=1)

    if content and not content.endswith("\n"):
        content += "\n"
    return f"{content}{replacement}\n"


def _set_nested(data: dict[str, Any], section: str, key: str, value: Any) -> None:
    """Set ``data[section][key] = value``, creating the section if needed."""
    current = data.get(section)
    if not isinstance(current, dict):
        current = {}
    current[key] = value
    data[section] = current


def _patch_config_yaml(
    agent_dir: Path,
    *,
    port: int | None,
    model: str | None,
    model_provider: str | None,
) -> tuple[Path | None, list[str]]:
    """Write endpoint/model values into config.yaml.

    Loads the existing config.yaml (the agent-root marker), merges the given
    values into the ``endpoint`` / ``model`` sections, and writes it back. Note
    that comments in config.yaml are not preserved (YAML round-trip).
    """
    if port is None and model is None and model_provider is None:
        return None, []

    config_path = agent_dir / "config.yaml"
    data: dict[str, Any] = {}
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded

    updated: list[str] = []
    if port is not None:
        _set_nested(data, "endpoint", "port", port)
        updated.append("endpoint.port")
    if model is not None:
        _set_nested(data, "model", "name", model)
        updated.append("model.name")
    if model_provider is not None:
        _set_nested(data, "model", "provider", model_provider)
        updated.append("model.provider")

    config_path.write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    return config_path, updated


def _patch_env_file(
    agent_dir: Path,
    *,
    docker_registry: str | None,
    repo: str | None,
) -> tuple[Path | None, list[str]]:
    """Write build/push defaults into .env (these stay environment variables)."""
    if docker_registry is None and repo is None:
        return None, []

    env_path = agent_dir / ".env"
    content = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""

    updated: list[str] = []
    if docker_registry is not None:
        content = _upsert_env_var(
            content, "BAT_DOCKER_REGISTRY", docker_registry
        )
        updated.append("BAT_DOCKER_REGISTRY")
    if repo is not None:
        content = _upsert_env_var(content, "BAT_DOCKER_REPO", repo)
        updated.append("BAT_DOCKER_REPO")

    env_path.write_text(content, encoding="utf-8")
    return env_path, updated


def set_agent_settings(
    agent_dir: Path,
    *,
    port: int | None = None,
    model: str | None = None,
    model_provider: str | None = None,
    docker_registry: str | None = None,
    repo: str | None = None,
) -> tuple[list[Path], list[str]]:
    """Apply agent settings: endpoint/model to config.yaml, docker defaults to .env.

    Returns the list of files written and the list of updated keys (config keys
    are dotted, e.g. ``model.name``; env keys are bare, e.g.
    ``BAT_DOCKER_REGISTRY``).
    """
    config_path, config_keys = _patch_config_yaml(
        agent_dir, port=port, model=model, model_provider=model_provider
    )
    env_path, env_keys = _patch_env_file(
        agent_dir, docker_registry=docker_registry, repo=repo
    )

    written = [p for p in (config_path, env_path) if p is not None]
    return written, config_keys + env_keys
