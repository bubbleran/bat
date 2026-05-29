import re
from pathlib import Path
from typing import Literal

BAT_ADK_VERSION = "2026.4.23"

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates" / "agent"
_DYNAMIC_TEMPLATE_FILES = {
    ".env.template",
    "agent.json.template",
    "agent.spec",
    "Dockerfile",
    "Makefile",
    "llm_client.py.template",
    "pyproject.toml.template",
    "src/graph.py",
    "src/__init__.py",
    "__main__.py",
}


def _load_static_templates() -> dict[str, str]:
    templates: dict[str, str] = {}
    for template_path in sorted(TEMPLATES_DIR.rglob("*")):
        if not template_path.is_file():
            continue

        relative_path = template_path.relative_to(TEMPLATES_DIR).as_posix()
        if (
            relative_path in _DYNAMIC_TEMPLATE_FILES
            or "__pycache__" in template_path.parts
            or template_path.suffix == ".pyc"
        ):
            continue

        templates[relative_path] = template_path.read_text(encoding="utf-8")

    return templates


def _render_template(template_file: str, replacements: dict[str, str]) -> str:
    template_path = TEMPLATES_DIR / template_file
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")

    rendered = template_path.read_text(encoding="utf-8")
    for key, value in replacements.items():
        rendered = rendered.replace(f"__{key}__", value)

    return rendered


def _normalize_name(
    raw: str, style: Literal["project", "snake", "pascal"]
) -> str:
    if style == "pascal":
        name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", raw)
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        name = re.sub(r"[^A-Za-z0-9]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        parts = [part for part in name.split("_") if part]
        if parts and parts[-1].lower() == "agent":
            parts = parts[:-1]
        if not parts:
            return ""
        return "".join(part[:1].upper() + part[1:] for part in parts)

    separator = "-" if style == "project" else "_"
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", rf"\1{separator}\2", raw)
    name = re.sub(r"([a-z0-9])([A-Z])", rf"\1{separator}\2", name)
    name = re.sub(r"[^A-Za-z0-9]+", separator, name)
    collapsed_separator = re.escape(separator)
    name = re.sub(rf"{collapsed_separator}+", separator, name).strip(separator)

    if style == "project":
        project_name = (name or "agent").lower()
        if project_name.endswith("-agent"):
            project_name = project_name[: -len("-agent")]
        return project_name or "agent"
    return name.lower()


def _build_pyproject_content(agent_dir_name: str) -> str:
    project_name = _normalize_name(agent_dir_name, "project")
    return _render_template(
        "pyproject.toml.template",
        {
            "BAT_ADK_VERSION": BAT_ADK_VERSION,
            "PROJECT_DESCRIPTION": f"{project_name.upper()} Agent",
            "PROJECT_NAME": project_name,
        },
    )


def _build_agent_spec_content(agent_dir_name: str) -> str:
    return _render_template(
        "agent.spec",
        {
            "PROJECT_NAME": _normalize_name(agent_dir_name, "project"),
        },
    )


def _build_dockerfile_content(agent_dir_name: str) -> str:
    return _render_template(
        "Dockerfile",
        {
            "PROJECT_NAME": _normalize_name(agent_dir_name, "project"),
        },
    )


def _build_makefile_content(agent_dir_name: str) -> str:
    return _render_template(
        "Makefile",
        {
            "PROJECT_NAME": _normalize_name(agent_dir_name, "project"),
        },
    )


def _agent_class_name(agent_dir_name: str) -> str:
    return _normalize_name(agent_dir_name, "pascal") or "Agent"


def _build_main_content(agent_dir_name: str) -> str:
    return _render_template(
        "__main__.py",
        {
            "AGENT_CLASS_NAME": _agent_class_name(agent_dir_name),
        },
    )


def _build_src_init_content(agent_dir_name: str) -> str:
    return _render_template(
        "src/__init__.py",
        {
            "AGENT_CLASS_NAME": _agent_class_name(agent_dir_name),
        },
    )


def _build_graph_content(agent_dir_name: str, clients: list[str] | None) -> str:
    resolved_clients = _resolve_client_specs(clients)
    agent_class_name = _agent_class_name(agent_dir_name)

    client_imports = "\n".join(
        f"from .llm_clients.{file_stem} import {class_name}"
        for file_stem, class_name in resolved_clients
    )

    setup_blocks: list[str] = []

    for file_stem, class_name in resolved_clients:
        setup_blocks.append(
            "\n".join(
                [
                    f"        self.{file_stem} = {class_name}(",
                    "            tools=[],",
                    "        )",
                ]
            )
        )

    return _render_template(
        "src/graph.py",
        {
            "AGENT_CLASS_NAME": agent_class_name,
            "CLIENT_IMPORTS": client_imports,
            "CLIENT_SETUP": "\n\n".join(setup_blocks),
        },
    )


def _build_agent_json_content(agent_dir_name: str) -> str:
    return _render_template(
        "agent.json.template",
        {
            "AGENT_NAME": agent_dir_name,
        },
    )


def _build_env_template_content(
    *,
    port: int,
    model: str,
    model_provider: str,
) -> str:
    return _render_template(
        ".env.template",
        {
            "PORT": str(port),
            "MODEL": model,
            "MODEL_PROVIDER": model_provider,
        },
    )


def _build_llm_client_content(class_name: str) -> str:
    return _render_template(
        "llm_client.py.template",
        {
            "CLASS_NAME": class_name,
        },
    )


def _resolve_client_specs(clients: list[str] | None) -> list[tuple[str, str]]:
    if not clients:
        return [("example_client", "ExampleClient")]

    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()

    for raw_name in clients:
        snake_name = _normalize_name(raw_name, "snake")
        if not snake_name:
            continue

        file_stem = (
            snake_name
            if snake_name.endswith("_client")
            else f"{snake_name}_client"
        )
        if file_stem in seen:
            continue

        pascal_name = _normalize_name(raw_name, "pascal")
        if not pascal_name:
            continue
        class_name = (
            pascal_name
            if pascal_name.endswith("Client")
            else f"{pascal_name}Client"
        )

        seen.add(file_stem)
        resolved.append((file_stem, class_name))

    return resolved or [("example_client", "ExampleClient")]


def _client_specs_from_dir(llm_clients_dir: Path) -> list[tuple[str, str]]:
    """Derive (file_stem, class_name) for every client module on disk.

    The class name follows the same convention as ``_resolve_client_specs`` so a
    file ``check_request_client.py`` maps back to ``CheckRequestClient``.
    """
    specs: list[tuple[str, str]] = []
    for path in sorted(llm_clients_dir.glob("*.py")):
        if path.stem == "__init__":
            continue
        class_name = _normalize_name(path.stem, "pascal")
        if not class_name:
            continue
        if not class_name.endswith("Client"):
            class_name = f"{class_name}Client"
        specs.append((path.stem, class_name))
    return specs


def _build_llm_clients_init_content(client_specs: list[tuple[str, str]]) -> str:
    if not client_specs:
        return ""
    lines = [
        f"from .{file_stem} import {class_name}"
        for file_stem, class_name in client_specs
    ]
    return "\n".join(lines) + "\n"


def _write_llm_clients(
    llm_clients_dir: Path,
    *,
    clients: list[str] | None,
    force: bool,
) -> list[Path]:
    created: list[Path] = []
    for file_stem, class_name in _resolve_client_specs(clients):
        client_path = llm_clients_dir / f"{file_stem}.py"
        if client_path.exists() and not force:
            continue

        client_path.parent.mkdir(parents=True, exist_ok=True)
        client_path.write_text(
            _build_llm_client_content(class_name), encoding="utf-8"
        )
        created.append(client_path)

    # Regenerate the package __init__ so it re-exports every client present in
    # the directory (covers both `bat create` and `bat add client`).
    init_path = llm_clients_dir / "__init__.py"
    new_content = _build_llm_clients_init_content(
        _client_specs_from_dir(llm_clients_dir)
    )
    existing_content = (
        init_path.read_text(encoding="utf-8") if init_path.exists() else None
    )
    if new_content != existing_content:
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.write_text(new_content, encoding="utf-8")
        if init_path not in created:
            created.append(init_path)

    return created


def create_agent_scaffold(
    target_dir: Path,
    *,
    force: bool = False,
    clients: list[str] | None = None,
    port: int = 9900,
    model: str = "gpt-4o-mini",
    model_provider: str = "openai",
) -> list[Path]:
    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Target directory '{target_dir}' already exists and is not empty. "
            "Use --force to overwrite files."
        )

    target_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    for relative_path, content in _load_static_templates().items():
        file_path = target_dir / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists() and not force:
            continue

        file_path.write_text(content, encoding="utf-8")
        created.append(file_path)

    pyproject_path = target_dir / "pyproject.toml"

    if force or not pyproject_path.exists():
        pyproject_path.write_text(
            _build_pyproject_content(target_dir.name), encoding="utf-8"
        )
        created.append(pyproject_path)

    agent_json_path = target_dir / "agent.json"
    if force or not agent_json_path.exists():
        agent_json_path.write_text(
            _build_agent_json_content(target_dir.name), encoding="utf-8"
        )
        created.append(agent_json_path)

    env_path = target_dir / ".env"
    if force or not env_path.exists():
        env_path.write_text(
            _build_env_template_content(
                port=port,
                model=model,
                model_provider=model_provider,
            ),
            encoding="utf-8",
        )
        created.append(env_path)

    agent_spec_path = target_dir / "agent.spec"
    if force or not agent_spec_path.exists():
        agent_spec_path.write_text(
            _build_agent_spec_content(target_dir.name), encoding="utf-8"
        )
        created.append(agent_spec_path)

    dockerfile_path = target_dir / "Dockerfile"
    if force or not dockerfile_path.exists():
        dockerfile_path.write_text(
            _build_dockerfile_content(target_dir.name), encoding="utf-8"
        )
        created.append(dockerfile_path)

    makefile_path = target_dir / "Makefile"
    if force or not makefile_path.exists():
        makefile_path.write_text(
            _build_makefile_content(target_dir.name), encoding="utf-8"
        )
        created.append(makefile_path)

    graph_path = target_dir / "src" / "graph.py"
    if force or not graph_path.exists():
        graph_path.write_text(
            _build_graph_content(target_dir.name, clients), encoding="utf-8"
        )
        created.append(graph_path)

    src_init_path = target_dir / "src" / "__init__.py"
    if (
        force
        or not src_init_path.exists()
        or not src_init_path.read_text(encoding="utf-8").strip()
    ):
        src_init_path.parent.mkdir(parents=True, exist_ok=True)
        src_init_path.write_text(
            _build_src_init_content(target_dir.name), encoding="utf-8"
        )
        created.append(src_init_path)

    main_path = target_dir / "__main__.py"
    if force or not main_path.exists():
        main_path.write_text(
            _build_main_content(target_dir.name), encoding="utf-8"
        )
        created.append(main_path)

    created.extend(
        _write_llm_clients(
            target_dir / "src" / "llm_clients",
            clients=clients,
            force=force,
        )
    )

    return created
