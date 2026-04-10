import re
from pathlib import Path


def _upsert_env_var(content: str, key: str, value: str) -> str:
    pattern = re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}\s*=.*$", flags=re.MULTILINE)
    replacement = f"{key}={value}"
    if pattern.search(content):
        return pattern.sub(replacement, content, count=1)

    if content and not content.endswith("\n"):
        content += "\n"
    return f"{content}{replacement}\n"


def set_env_values(
    agent_dir: Path,
    *,
    port: int | None = None,
    model: str | None = None,
    model_provider: str | None = None,
    docker_registry: str | None = None,
    repo: str | None = None,
) -> tuple[Path, list[str]]:
    env_path = agent_dir / ".env"
    if env_path.is_file():
        content = env_path.read_text(encoding="utf-8")
    else:
        content = ""

    updated_keys: list[str] = []

    if port is not None:
        content = _upsert_env_var(content, "PORT", str(port))
        updated_keys.append("PORT")
    if model is not None:
        content = _upsert_env_var(content, "MODEL", model)
        updated_keys.append("MODEL")
    if model_provider is not None:
        content = _upsert_env_var(content, "MODEL_PROVIDER", model_provider)
        updated_keys.append("MODEL_PROVIDER")
    if docker_registry is not None:
        content = _upsert_env_var(content, "BAT_DOCKER_REGISTRY", docker_registry)
        updated_keys.append("BAT_DOCKER_REGISTRY")
    if repo is not None:
        content = _upsert_env_var(content, "BAT_DOCKER_REPO", repo)
        updated_keys.append("BAT_DOCKER_REPO")

    env_path.write_text(content, encoding="utf-8")
    return env_path, updated_keys
