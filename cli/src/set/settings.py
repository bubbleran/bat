"""`bat set`: edit config.yaml and the Makefile line by line, so their
comments and layout survive."""

import json
import re
from pathlib import Path


def _set_make_variable(content: str, name: str, value: str) -> str:
    """``content`` with the existing ``NAME ?=`` (``:=``, ``=``) assignment
    set to ``value``.

    Raises:
        ValueError: If the Makefile assigns no ``name``.
    """
    pattern = re.compile(
        rf"^({re.escape(name)}[ \t]*(?:\?|::?)?=)[^\n]*$", re.MULTILINE
    )
    if not pattern.search(content):
        raise ValueError(
            f"The Makefile sets no {name} (no `{name} ?= ...` line), so it "
            "names its image some other way: edit it by hand."
        )
    return pattern.sub(lambda match: f"{match.group(1)} {value}", content, 1)


def _yaml_scalar(value: int | str) -> str:
    if isinstance(value, int):
        return str(value)
    plain = value and value == value.strip()
    if plain and not re.search(r"""[:#\[\]{}&*!|>%@`,"']""", value):
        return value
    return json.dumps(value)  # a double-quoted string is valid YAML too


def _set_yaml_value(text: str, section: str, key: str, value: int | str) -> str:
    """``text`` with ``section.key`` set to ``value``, the key (or the whole
    section) added when missing."""
    scalar = _yaml_scalar(value)
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^{re.escape(section)}\s*:", line):
            start = i
            break
    if start is None:
        separator = "" if not text or text.endswith("\n") else "\n"
        return f"{text}{separator}{section}:\n  {key}: {scalar}\n"

    key_re = re.compile(rf"^(\s+){re.escape(key)}\s*:(.*)$")
    for i in range(start + 1, len(lines)):
        line = lines[i].rstrip("\r\n")
        stripped = line.strip()
        # A non-indented line that is no comment ends the section.
        if stripped and not line[:1].isspace() and not stripped.startswith("#"):
            break
        match = key_re.match(line)
        if match:
            comment = re.search(r"\s#.*", match.group(2))
            eol = lines[i][len(line) :] or "\n"
            lines[i] = (
                f"{match.group(1)}{key}: {scalar}"
                f"{comment.group(0) if comment else ''}{eol}"
            )
            return "".join(lines)

    if not lines[start].endswith("\n"):
        lines[start] += "\n"
    lines.insert(start + 1, f"  {key}: {scalar}\n")
    return "".join(lines)


def set_config(
    agent_dir: Path,
    *,
    port: int | None = None,
    model: str | None = None,
    model_provider: str | None = None,
) -> list[str]:
    """Write the given values into the agent's config.yaml; returns the
    dotted keys updated."""
    path = agent_dir / "config.yaml"
    text = path.read_text(encoding="utf-8")
    updated: list[str] = []
    for section, key, value in (
        ("endpoint", "port", port),
        ("model", "name", model),
        ("model", "provider", model_provider),
    ):
        if value is not None:
            text = _set_yaml_value(text, section, key, value)
            updated.append(f"{section}.{key}")
    path.write_text(text, encoding="utf-8")
    return updated


def set_image(
    makefile: Path,
    *,
    docker_registry: str | None = None,
    repo: str | None = None,
) -> list[str]:
    """Write DOCKER_REGISTRY / REPO into the Makefile, where `make build`
    reads them as well as `bat build`; returns the variables updated.

    Raises:
        ValueError: If the Makefile assigns one of them nowhere. Nothing is
            written then.
    """
    content = makefile.read_text(encoding="utf-8")
    updated: list[str] = []
    for name, value in (("DOCKER_REGISTRY", docker_registry), ("REPO", repo)):
        if value is not None:
            content = _set_make_variable(content, name, value)
            updated.append(name)
    makefile.write_text(content, encoding="utf-8")
    return updated
