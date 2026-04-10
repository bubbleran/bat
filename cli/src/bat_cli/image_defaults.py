import os
import re
from pathlib import Path


REPO_ENV_VAR = "BAT_DOCKER_REPO"
REGISTRY_ENV_VAR = "BAT_DOCKER_REGISTRY"
DEFAULT_REGISTRY = "default_registry"


def default_repo_name(context_dir: Path) -> str:
    project = re.sub(r"[^a-z0-9]+", "-", context_dir.name.lower()).strip("-") or "agent"
    return f"default-repository/{project}"


def _read_dotenv_value(key: str) -> str | None:
    dotenv_path = Path.cwd() / ".env"
    if not dotenv_path.is_file():
        return None

    try:
        content = dotenv_path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.search(
        rf"^\s*(?:export\s+)?{re.escape(key)}\s*=\s*(.*?)\s*$",
        content,
        flags=re.MULTILINE,
    )
    if not match:
        return None

    value = match.group(1).strip()
    if value.startswith("#"):
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1].strip()

    if value:
        return value

    return None


def _resolve_image_value(
    cli_value: str | None,
    *,
    env_key: str,
    default_value: str,
) -> str:
    if cli_value:
        return cli_value

    env_value = os.environ.get(env_key, "").strip()
    if env_value:
        return env_value

    dotenv_value = _read_dotenv_value(env_key)
    if dotenv_value:
        return dotenv_value

    return default_value


def resolve_repo_name(context_dir: Path, repo: str | None) -> str:
    return _resolve_image_value(
        repo,
        env_key=REPO_ENV_VAR,
        default_value=default_repo_name(context_dir),
    )


def resolve_registry(context_dir: Path, registry: str | None) -> str:
    return _resolve_image_value(
        registry,
        env_key=REGISTRY_ENV_VAR,
        default_value=DEFAULT_REGISTRY,
    )
