import json
import re
from pathlib import Path
from typing import Any


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


def _format_yaml_scalar(value: Any) -> str:
    """Render ``value`` as a YAML scalar, quoting it only when necessary."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    needs_quotes = (
        text == ""
        or text != text.strip()
        or bool(re.search(r"""[:#\[\]{}&*!|>%@`,"']""", text))
    )
    # json.dumps yields an ASCII, double-quoted string that is also valid YAML.
    return json.dumps(text) if needs_quotes else text


def _set_yaml_value(text: str, section: str, key: str, value: Any) -> str:
    """Set ``section.key`` to ``value`` in YAML ``text`` in place.

    Edits only the target line so surrounding comments, blank lines and key
    ordering survive (unlike a ``safe_load``/``safe_dump`` round-trip). Inserts
    the key (or the whole section) when missing.
    """
    scalar = _format_yaml_scalar(value)
    lines = text.splitlines(keepends=True)
    section_re = re.compile(rf"^{re.escape(section)}\s*:")
    key_re = re.compile(rf"^(\s+){re.escape(key)}\s*:(.*)$")

    section_start = next(
        (i for i, line in enumerate(lines) if section_re.match(line)), None
    )

    if section_start is None:
        suffix = "" if (not text or text.endswith("\n")) else "\n"
        return f"{text}{suffix}{section}:\n  {key}: {scalar}\n"

    for i in range(section_start + 1, len(lines)):
        raw = lines[i]
        line = raw.rstrip("\r\n")
        eol = raw[len(line):] or "\n"
        stripped = line.strip()
        # A non-indented, non-blank, non-comment line ends the section block.
        if stripped and not line[:1].isspace() and not stripped.startswith("#"):
            break
        match = key_re.match(line)
        if match:
            indent, after = match.group(1), match.group(2)
            comment_match = re.search(r"\s#", after)
            comment = after[comment_match.start():] if comment_match else ""
            lines[i] = f"{indent}{key}: {scalar}{comment}{eol}"
            return "".join(lines)

    # Section present but key absent: insert it as the first child.
    header = lines[section_start]
    if not header.endswith("\n"):
        lines[section_start] = header + "\n"
    lines.insert(section_start + 1, f"  {key}: {scalar}\n")
    return "".join(lines)


def _patch_config_yaml(
    agent_dir: Path,
    *,
    port: int | None,
    model: str | None,
    model_provider: str | None,
) -> tuple[Path | None, list[str]]:
    """Write endpoint/model values into config.yaml, preserving comments.

    Edits the existing config.yaml (the agent-root marker) line by line so
    comments and structure are kept intact, then writes it back.
    """
    if port is None and model is None and model_provider is None:
        return None, []

    config_path = agent_dir / "config.yaml"
    content = (
        config_path.read_text(encoding="utf-8")
        if config_path.is_file()
        else ""
    )

    updated: list[str] = []
    if port is not None:
        content = _set_yaml_value(content, "endpoint", "port", port)
        updated.append("endpoint.port")
    if model is not None:
        content = _set_yaml_value(content, "model", "name", model)
        updated.append("model.name")
    if model_provider is not None:
        content = _set_yaml_value(content, "model", "provider", model_provider)
        updated.append("model.provider")

    config_path.write_text(content, encoding="utf-8")
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
