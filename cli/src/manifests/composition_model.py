"""The CompositionModel of a blueprint: one deployment mode per agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from image import registry_of

from .aifabric import (
    API_VERSION,
    ManifestUpdate,
    agents_run_by,
    check_kind,
    deployable_agents,
    kubernetes_name,
    merged_metadata,
)

KIND = "CompositionModel"


def _repository(image: str) -> str:
    """``image`` without its tag or digest."""
    name = image.split("@")[0]
    head, colon, tag = name.rpartition(":")
    return head if colon and "/" not in tag else name


def update_composition_model(
    blueprint_root: Path,
    existing: dict[str, Any] | None = None,
    *,
    image: str,
    name: str | None = None,
    namespace: str | None = None,
) -> ManifestUpdate:
    """The CompositionModel for ``blueprint_root``, built on ``existing``
    if given, its agents' modes running ``image``.

    Raises:
        ValueError: If ``existing`` is not a CompositionModel.
    """
    check_kind(existing, KIND)
    document = existing or {}
    added: list[str] = []
    changes: list[str] = []
    warnings: list[str] = []
    modes = {
        mode: dict(desc or {})
        for mode, desc in (document.get("deploymentModes") or {}).items()
    }

    ours = deployable_agents(blueprint_root, warnings)
    runs: dict[str, str] = {}
    for mode, desc in modes.items():
        if desc.get("kind") != "mcp":
            for directory in agents_run_by(desc):
                runs.setdefault(directory, mode)
    # The blueprint's image is the one its agents' modes ran, under whatever
    # repository it was pushed to before.
    repositories = {_repository(image)} | {
        _repository(str(modes[runs[directory]].get("imageTag") or ""))
        for directory in ours
        if directory in runs
    }
    for mode, desc in modes.items():
        tag = desc.get("imageTag")
        if tag and tag != image and _repository(str(tag)) in repositories:
            changes.append(f"{mode}: imageTag {tag} -> {image}")
            desc["imageTag"] = image

    for directory in ours:
        if directory in runs:
            continue
        mode = kubernetes_name(directory)
        if mode in modes:
            warnings.append(
                f"{directory}: mode '{mode}' already runs something else; "
                "add a mode for this agent by hand."
            )
            continue
        modes[mode] = {
            "kind": "agent",
            "name": mode,
            "imageTag": image,
            "args": [directory],
            # The config.yaml the operator mounts has no agent_card.
            "env": [
                {"name": "AGENT_CARD_PATH", "value": f"{directory}/agent.json"}
            ],
        }
        added.append(mode)

    if registry_of(image) is None:
        warnings.append(
            f"{image} names no registry, so the cluster can't pull it: pass "
            "--docker-registry, or set it with `bat set image "
            "--docker-registry`."
        )

    old_metadata = document.get("metadata") or {}
    metadata = merged_metadata(
        old_metadata,
        changes,
        name=name
        or old_metadata.get("name")
        or kubernetes_name(blueprint_root.name),
        namespace=namespace or old_metadata.get("namespace") or "default",
    )
    spec = dict(document.get("spec") or {})
    spec.setdefault("provider", "bubbleran")
    spec.setdefault("version", "v0.1.0")
    result = {
        "apiVersion": API_VERSION,
        "kind": KIND,
        "metadata": metadata,
        "spec": spec,
        "deploymentModes": modes,
    }
    result.update(
        (key, value) for key, value in document.items() if key not in result
    )
    return ManifestUpdate(result, added, changes, warnings)
