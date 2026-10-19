"""The AIFabric of a blueprint, from its agents' config.yaml files (the
orama operator renders those from it). Hand edits and other blueprints'
agents are kept: one AIFabric often deploys several."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from create.agent import PROVIDER_API_KEY_VAR
from project import AgentTarget, blueprint_agents, unwired_agent_warning

API_VERSION = "orama.trirematics.io/v1"
KIND = "AIFabric"


@dataclass
class ManifestUpdate:
    document: dict[str, Any]
    added: list[str]  # entries new to the file
    changes: list[str]  # every other change, one line each
    warnings: list[str]


def check_kind(existing: dict[str, Any] | None, kind: str) -> None:
    if existing is not None and existing.get("kind") not in (None, kind):
        article = "an" if kind[0] in "AEIOU" else "a"
        raise ValueError(
            f"The file holds a {existing.get('kind')}, not {article} {kind}."
        )


def merged_metadata(
    metadata: dict[str, Any], changes: list[str], **values: str
) -> dict[str, Any]:
    """``metadata`` with ``values`` set, recording each change."""
    metadata = dict(metadata)
    for key, value in values.items():
        if metadata.get(key) != value:
            changes.append(f"metadata.{key} {metadata.get(key)} -> {value}")
        metadata[key] = value
    return metadata


def kubernetes_name(text: str) -> str:
    """``text`` as the operator's names must be: lowercase letters, digits
    and dashes, at most 63; CamelCase becomes words (cluster-view)."""
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", text)
    name = re.sub(r"[^a-z0-9]+", "-", words.lower()).strip("-")
    return name[:63].strip("-")


def _is_local(url: str | None) -> bool:
    hosts = ("localhost", "127.0.0.1", "0.0.0.0", "::1")
    return urlsplit(url or "").hostname in hosts


def _url_problem(url: str | None) -> str | None:
    """Why ``url`` cannot be used in the cluster, or None when it can."""
    unset = re.findall(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", url or "")
    if unset:
        return f"uses ${{{unset[0]}}}, which is not set"
    if not urlsplit(url or "").hostname:
        return f"'{url}' is not a URL"
    if _is_local(url):
        return f"{url} is localhost"
    return None


def _port_of(url: str | None) -> int | None:
    try:
        return urlsplit(url or "").port
    except ValueError:
        return None


@dataclass
class _Agent:
    directory: str
    name: str
    config: dict[str, Any]
    card_name: str | None
    port: int | None


def _listed(values: list[str] | None) -> str:
    return ", ".join(values) if values else "(none)"


def update_aifabric(
    blueprint_root: Path,
    existing: dict[str, Any] | None = None,
    *,
    name: str | None = None,
    namespace: str | None = None,
    image_pull_secrets: list[str],
    telemetry_endpoint: str | None = None,
) -> ManifestUpdate:
    """The AIFabric for ``blueprint_root``, built on ``existing`` if given.

    Raises:
        ValueError: If ``existing`` is not an AIFabric.
    """
    check_kind(existing, KIND)
    document = existing or {}
    spec = document.get("spec") or {}
    added: list[str] = []
    changes: list[str] = []
    warnings: list[str] = []

    model_prefix, cm_namespace, agent_modes, mcp_modes = _composition_model(
        blueprint_root
    )
    prefix = model_prefix or kubernetes_name(blueprint_root.name)
    ours = _our_agents(blueprint_root, warnings)

    agents = [dict(entry) for entry in spec.get("agents") or []]
    by_name = {entry.get("name"): entry for entry in agents}
    fabric_names = set(by_name) | {agent.name for agent in ours}
    llms = [dict(entry) for entry in spec.get("llms") or []]
    servers = [dict(entry) for entry in spec.get("mcp") or []]
    known_llms = {llm.get("name") for llm in llms}
    known_servers = {server.get("name") for server in servers}

    for agent in ours:
        model_config = agent.config.get("model") or {}
        if not model_config.get("provider") or not model_config.get("name"):
            warnings.append(
                f"{agent.directory}: config.yaml has no model; an internal "
                "agent of the fabric needs an LLM, so it is left out."
            )
            continue
        llm = _llm_for(agent, model_config, llms, warnings)
        dependencies = _dependencies(agent, ours, fabric_names, warnings)
        lists = {
            "mcpServers": _mcp_servers(agent, servers, mcp_modes, warnings),
            "dependencies": dependencies,
        }

        mode = agent_modes.get(agent.directory)
        if model_prefix and mode is None:
            warnings.append(
                f"{agent.directory}: no deployment mode of CompositionModel "
                f"'{model_prefix}' runs it; its model is set to "
                f"'{prefix}/{agent.name}'."
            )
        model = mode or f"{prefix}/{agent.name}"

        entry = by_name.get(agent.name)
        if entry is None:
            # Role "none": the dependencies are exactly the config's.
            internal = {"model": model, "llm": llm, "role": "none"}
            for key, value in lists.items():
                _set_or_drop(internal, key, value)
            agents.append(
                {"name": agent.name, "type": "internal", "internal": internal}
            )
            added.append(agent.name)
        elif entry.get("type", "internal") != "internal":
            warnings.append(
                f"{agent.directory}: '{agent.name}' is an external agent in "
                "the fabric; it is left as it is."
            )
        else:
            internal = dict(entry.get("internal") or {})
            internal.setdefault("model", model)
            if internal.get("llm") != llm:
                changes.append(
                    f"{agent.name}: llm {internal.get('llm')} -> {llm}"
                )
            internal["llm"] = llm
            for key, value in lists.items():
                if list(internal.get(key) or []) != value:
                    changes.append(
                        f"{agent.name}: {key} {_listed(internal.get(key))} "
                        f"-> {_listed(value)}"
                    )
                _set_or_drop(internal, key, value)
            entry["internal"] = internal

    telemetry = _telemetry(
        ours, spec.get("telemetry"), telemetry_endpoint, warnings
    )
    if telemetry != spec.get("telemetry"):
        changes.append(
            f"telemetry {spec.get('telemetry') or '(none)'} -> "
            f"{telemetry or '(none)'}"
        )
    changes += [
        f"new LLM {llm['name']} ({llm['provider']} {llm['model']})"
        for llm in llms
        if llm.get("name") not in known_llms
    ]
    changes += [
        f"new MCP server {server['name']}"
        for server in servers
        if server.get("name") not in known_servers
    ]
    secrets = [dict(entry) for entry in spec.get("imagePullSecrets") or []]
    for secret in image_pull_secrets:
        if all(entry.get("name") != secret for entry in secrets):
            secrets.append({"name": secret})
            changes.append(f"new image pull secret {secret}")

    new_spec = {
        "imagePullSecrets": secrets,
        "telemetry": telemetry,
        "llms": llms,
        "mcp": servers,
        "agents": agents,
    }
    ordered = {key: value for key, value in new_spec.items() if value}
    ordered.update(
        (key, value) for key, value in spec.items() if key not in new_spec
    )
    old_metadata = document.get("metadata") or {}
    metadata = merged_metadata(
        old_metadata,
        changes,
        name=name
        or old_metadata.get("name")
        or kubernetes_name(f"{blueprint_root.name}-fabric"),
        namespace=namespace
        or old_metadata.get("namespace")
        or cm_namespace
        or "default",
    )
    result = {
        "apiVersion": API_VERSION,
        "kind": KIND,
        "metadata": metadata,
        "spec": ordered,
    }
    result.update(
        (key, value) for key, value in document.items() if key not in result
    )
    return ManifestUpdate(result, added, changes, warnings)


def _set_or_drop(mapping: dict[str, Any], key: str, value: list) -> None:
    if value:
        mapping[key] = value
    else:
        mapping.pop(key, None)


def deployable_agents(root: Path, warnings: list[str]) -> list[str]:
    """The agent folders the blueprint's dispatcher can run, in order."""
    directories: list[str] = []
    for directory in blueprint_agents(root):
        target = AgentTarget(
            project_root=root, agent_dir=root / directory, agent_name=directory
        )
        if unwired_agent_warning(target) is not None:
            warnings.append(
                f"{directory}: __main__.py never names it, so no deployment "
                "can run it; it is left out. (Selected under another name? "
                "Add it by hand.)"
            )
            continue
        directories.append(directory)
    return directories


def agents_run_by(mode: dict[str, Any]) -> list[str]:
    """The agent folders a deployment mode runs: the one its first ``arg``
    selects, or the one whose card its ``AGENT_CARD_PATH`` points at --
    automation's ``logs_agent/`` is selected as ``logs``."""
    runs = []
    args = mode.get("args") or []
    if args:
        runs.append(str(args[0]))
    for env in mode.get("env") or []:
        if env.get("name") == "AGENT_CARD_PATH" and env.get("value"):
            runs.append(str(env["value"]).split("/")[0])
    return runs


def _our_agents(root: Path, warnings: list[str]) -> list[_Agent]:
    """The agents the blueprint's dispatcher can run, in folder order."""
    agents: list[_Agent] = []
    for directory in deployable_agents(root, warnings):
        # ${VAR} expanded first, as the ADK does when it loads the file.
        text = (root / directory / "config.yaml").read_text(encoding="utf-8")
        config = yaml.safe_load(os.path.expandvars(text)) or {}
        endpoint = config.get("endpoint") or {}
        port = endpoint.get("port")
        if not isinstance(port, int):
            port = _port_of(endpoint.get("url"))
        agents.append(
            _Agent(
                directory=directory,
                name=kubernetes_name(directory),
                config=config,
                card_name=_card_name(root / directory / "agent.json"),
                port=port,
            )
        )
    return agents


def _card_name(path: Path) -> str | None:
    try:
        card = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    return card.get("name") if isinstance(card, dict) else None


def _composition_model(
    root: Path,
) -> tuple[str | None, str | None, dict[str, str], dict[str, str]]:
    """``(name, namespace, agent folder -> model, MCP name -> model)`` from
    the blueprint's composition-model.yaml, when it has one."""
    path = root / "composition-model.yaml"
    if not path.is_file():
        return None, None, {}, {}
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    metadata = document.get("metadata") or {}
    model_name = kubernetes_name(str(metadata.get("name") or root.name))
    agent_modes: dict[str, str] = {}
    mcp_modes: dict[str, str] = {}
    for mode, desc in (document.get("deploymentModes") or {}).items():
        desc = desc or {}
        full = f"{model_name}/{mode}"
        if desc.get("kind") == "mcp":
            mcp_modes[mode] = full
            continue
        for directory in agents_run_by(desc):
            agent_modes.setdefault(directory, full)
    return model_name, metadata.get("namespace"), agent_modes, mcp_modes


def _llm_for(
    agent: _Agent,
    model_config: dict[str, Any],
    llms: list[dict[str, Any]],
    warnings: list[str],
) -> str:
    """The name of the fabric LLM serving ``model_config``, added if new."""
    provider = str(model_config["provider"])
    model = str(model_config["name"])
    base_url = model_config.get("base_url")
    problem = _url_problem(base_url) if base_url else None
    remote_base = base_url if base_url and not problem else None
    for llm in llms:
        endpoint = (llm.get("endpoint") or {}).get("baseUrl")
        if (
            llm.get("provider") == provider
            and llm.get("model") == model
            and (remote_base is None or endpoint == remote_base)
        ):
            return llm["name"]

    taken = {llm.get("name") for llm in llms}
    base_name = kubernetes_name(f"{provider}-{model}")
    llm_name = base_name
    suffix = 1
    while llm_name in taken:
        suffix += 1
        llm_name = f"{base_name}-{suffix}"
    entry: dict[str, Any] = {
        "name": llm_name,
        "provider": provider,
        "model": model,
    }
    key = PROVIDER_API_KEY_VAR.get(provider)
    if key:
        entry["apiKeySecretRef"] = {"name": f"{provider}-api-key", "key": key}
    if remote_base:
        entry["endpoint"] = {"baseUrl": remote_base}
    elif base_url:
        warnings.append(
            f"{agent.directory}: model.base_url {problem}; LLM "
            f"'{llm_name}' gets the operator's default {provider} endpoint "
            "-- set its endpoint.baseUrl if the cluster serves it elsewhere."
        )
    llms.append(entry)
    return llm_name


def _dependencies(
    agent: _Agent,
    ours: list[_Agent],
    fabric_names: set[str],
    warnings: list[str],
) -> list[str]:
    """The fabric agents ``agent`` calls, from its remote-agents.

    config.yaml names a remote agent by its card name and reaches it on a
    local port, so it is matched by port first, then by name.
    """
    by_port = {other.port: other.name for other in ours if other.port}
    by_card = {other.card_name: other.name for other in ours if other.card_name}
    dependencies: list[str] = []
    for remote in agent.config.get("remote-agents") or []:
        remote_name = str(remote.get("name") or "")
        url = remote.get("url")
        target = by_port.get(_port_of(url)) if _is_local(url) else None
        if target is None and kubernetes_name(remote_name) in fabric_names:
            target = kubernetes_name(remote_name)
        if target is None:
            target = by_card.get(remote_name)
        if target is None:
            warnings.append(
                f"{agent.directory}: remote agent '{remote_name}' ({url}) "
                "matches no agent of the fabric; add it (as an external "
                "agent?) and to this agent's dependencies by hand."
            )
            continue
        if target != agent.name and target not in dependencies:
            dependencies.append(target)
    return dependencies


def _mcp_servers(
    agent: _Agent,
    servers: list[dict[str, Any]],
    mcp_modes: dict[str, str],
    warnings: list[str],
) -> list[str]:
    """The fabric MCP servers ``agent`` uses, added to ``servers`` if new."""
    names: list[str] = []
    for server in agent.config.get("mcp-servers") or []:
        server_name = kubernetes_name(str(server.get("name") or ""))
        url = server.get("url")
        if not server_name:
            warnings.append(f"{agent.directory}: an MCP server has no name.")
            continue
        if server_name not in names:
            names.append(server_name)
        if any(entry.get("name") == server_name for entry in servers):
            continue
        if server_name in mcp_modes:
            entry = {
                "name": server_name,
                "type": "internal",
                "internal": {"model": mcp_modes[server_name]},
            }
        else:
            entry = {
                "name": server_name,
                "type": "external",
                "external": {"baseUrl": url},
            }
            problem = _url_problem(url)
            if problem:
                warnings.append(
                    f"{agent.directory}: MCP server '{server.get('name')}' "
                    f"{problem}; set spec.mcp '{server_name}' to its "
                    "in-cluster URL."
                )
        servers.append(entry)
    return names


def _telemetry(
    ours: list[_Agent],
    existing: dict[str, Any] | None,
    endpoint: str | None,
    warnings: list[str],
) -> dict[str, Any] | None:
    """spec.telemetry: the given endpoint, else the file's, else the
    agents' first one the cluster can reach."""
    endpoints: list[str] = []
    projects: list[str] = []
    for agent in ours:
        settings = agent.config.get("telemetry") or {}
        privacy = settings.get("privacy")
        if privacy not in (None, "none", 0):
            warnings.append(
                f"{agent.directory}: telemetry.privacy is '{privacy}' in "
                "config.yaml, but the operator does not render it: on the "
                "cluster only the agent's telemetry_privacy_floor applies."
            )
        for output in settings.get("output") or []:
            url = output.get("endpoint")
            if output.get("type") == "remote" and url and url not in endpoints:
                endpoints.append(url)
        project = settings.get("project_name")
        if project and project not in projects:
            projects.append(project)

    telemetry = dict(existing or {})
    if endpoint:
        telemetry["endpoint"] = endpoint
    elif not telemetry:
        remote = [url for url in endpoints if not _url_problem(url)]
        if remote:
            telemetry["endpoint"] = remote[0]
        elif endpoints:
            warnings.append(
                f"The agents export telemetry to {', '.join(endpoints)}, "
                "which the cluster cannot reach; pass --telemetry-endpoint "
                "<collector URL> to set spec.telemetry."
            )
    if telemetry and "projectName" not in telemetry and projects:
        telemetry["projectName"] = projects[0]
        if len(projects) > 1:
            warnings.append(
                "The agents use several telemetry projects "
                f"({', '.join(projects)}); a fabric has one, so a shared "
                f"trace would fragment. projectName is set to '{projects[0]}'."
            )
    return telemetry or None
