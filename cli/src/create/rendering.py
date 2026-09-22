"""Rendering shared by the agent and blueprint scaffolds."""

from __future__ import annotations

from pathlib import Path


def render_template(
    templates_dir: Path, template_file: str, replacements: dict[str, str]
) -> str:
    """Read ``template_file`` from ``templates_dir``, substituting keys."""
    template_path = templates_dir / template_file
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    rendered = template_path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        rendered = rendered.replace(f"__{key}__", value)

    return rendered
