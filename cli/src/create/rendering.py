"""Rendering and writing shared by the agent and blueprint scaffolds."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def render(template: Path, substitutions: dict[str, str]) -> str:
    """``template`` with every ``__KEY__`` replaced by its value."""
    text = template.read_text(encoding="utf-8")
    for key, value in substitutions.items():
        text = text.replace(f"__{key}__", value)
    return text


def template_files(templates_dir: Path) -> list[str]:
    """Every file under ``templates_dir``, relative, bytecode left out."""
    return sorted(
        path.relative_to(templates_dir).as_posix()
        for path in templates_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    )


def ensure_empty_dir(target_dir: Path, *, force: bool) -> None:
    """Raises FileExistsError unless ``target_dir`` is new, empty, or
    ``force`` lets it be overwritten."""
    if target_dir.exists() and not target_dir.is_dir():
        raise FileExistsError(
            f"Target path '{target_dir}' already exists and is not a "
            "directory. Choose a different name or remove the file."
        )
    if target_dir.is_dir() and any(target_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Target directory '{target_dir}' already exists and is not "
            "empty. Use --force to overwrite files."
        )


def write_files(
    root: Path, files: dict[str, str], *, force: bool
) -> list[Path]:
    """Write ``files`` (relative path -> content) under ``root``; an
    existing file is kept unless ``force``."""
    written: list[Path] = []
    for relative, content in files.items():
        path = root / relative
        if path.exists() and not force:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


class _IndentedDumper(yaml.SafeDumper):
    """Indents lists under their keys, as kubectl and docker compose do, so
    a rewrite only shows in a diff what changed."""

    def increase_indent(self, flow: bool = False, indentless: bool = False):
        return super().increase_indent(flow, False)


def dump_yaml(document: Any) -> str:
    return yaml.dump(
        document, Dumper=_IndentedDumper, sort_keys=False, allow_unicode=True
    )
