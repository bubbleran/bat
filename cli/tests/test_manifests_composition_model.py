"""`bat manifests composition-model`: one deployment mode per agent of a
blueprint, running the image `bat build` tags."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from cli import app

runner = CliRunner()

pytestmark = pytest.mark.skipif(
    shutil.which("make") is None, reason="make is not installed"
)

IMAGE_FLAGS = ("--docker-registry", "hub.example.com", "--version", "1.2.3")


def _blueprint(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "blueprint", "demo"])
    root = tmp_path / "demo"
    monkeypatch.chdir(root)
    runner.invoke(app, ["add", "agent", "netops", "--port", "9309"])
    runner.invoke(app, ["add", "agent", "hermes", "--port", "9310"])
    return root


def _generate(*args: str):
    result = runner.invoke(app, ["manifests", "composition-model", *args])
    assert result.exit_code == 0, result.output
    return result


def _model(root: Path) -> dict:
    return yaml.safe_load((root / "composition-model.yaml").read_text())


def test_each_agent_becomes_a_deployment_mode(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    result = _generate(*IMAGE_FLAGS)

    model = _model(root)
    assert model["apiVersion"] == "orama.trirematics.io/v1"
    assert model["kind"] == "CompositionModel"
    assert model["metadata"] == {"name": "demo", "namespace": "default"}
    assert model["spec"] == {"provider": "bubbleran", "version": "v0.1.0"}
    assert model["deploymentModes"]["netops"] == {
        "kind": "agent",
        "name": "netops",
        "imageTag": "hub.example.com/demo:1.2.3",
        "args": ["netops"],
        "env": [{"name": "AGENT_CARD_PATH", "value": "netops/agent.json"}],
        "rules": [],
    }
    assert set(model["deploymentModes"]) == {"hermes", "netops"}
    # PyYAML writes no comments: the example is added on every run.
    text = (root / "composition-model.yaml").read_text(encoding="utf-8")
    assert (
        "    rules: []\n"
        "    # Instead, e.g. to read the cluster's networks and terminals:\n"
        "    # rules:\n"
        "    #   - apiGroups: [athena.trirematics.io]\n"
        "    #     resources: [networks, terminals]\n"
        "    #     verbs: [get, watch]\n"
    ) in text
    assert "added: hermes, netops" in result.output
    assert "Warning" not in result.output


def test_the_aifabric_runs_its_agents_from_these_modes(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    _generate(*IMAGE_FLAGS, "--name", "network-ops", "--namespace", "orama")

    runner.invoke(app, ["manifests", "aifabric"])

    fabric = yaml.safe_load((root / "aifabric.yaml").read_text())
    models = {
        a["name"]: a["internal"]["model"] for a in fabric["spec"]["agents"]
    }
    assert models == {
        "hermes": "network-ops/hermes",
        "netops": "network-ops/netops",
    }
    assert fabric["metadata"]["namespace"] == "orama"


def test_without_a_registry_it_warns_the_cluster_cannot_pull(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    result = _generate()

    assert _model(root)["deploymentModes"]["netops"]["imageTag"] == "demo:dev"
    assert "can't pull" in result.output


def test_a_rerun_keeps_hand_edits_and_moves_the_image(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    _generate(*IMAGE_FLAGS)
    model = _model(root)
    rules = [{"apiGroups": [""], "resources": ["pods"], "verbs": ["get"]}]
    model["deploymentModes"]["netops"]["rules"] = rules
    model["deploymentModes"]["knowledge"] = {
        "kind": "mcp",
        "name": "knowledge",
        "imageTag": "hub.example.com/demo:1.2.3",
        "args": ["knowledge"],
    }
    model["deploymentModes"]["phoenix"] = {
        "kind": "mcp",
        "name": "phoenix",
        "imageTag": "arizephoenix/phoenix:latest",
    }
    model["spec"]["version"] = "v2.0.0"
    path = root / "composition-model.yaml"
    path.write_text("# Deployed by hand.\n" + yaml.safe_dump(model))

    result = _generate("--docker-registry", "hub.example.com", "--version", "2")

    modes = _model(root)["deploymentModes"]
    assert modes["netops"]["rules"] == rules
    assert modes["netops"]["imageTag"] == "hub.example.com/demo:2"
    # Same image, so a new build too; another image is left alone.
    assert modes["knowledge"]["imageTag"] == "hub.example.com/demo:2"
    assert modes["phoenix"]["imageTag"] == "arizephoenix/phoenix:latest"
    assert _model(root)["spec"]["version"] == "v2.0.0"
    assert path.read_text().startswith("# Deployed by hand.\n")
    assert (
        "netops: imageTag hub.example.com/demo:1.2.3 -> hub.example.com/demo:2"
        in result.output
    )


def test_an_agent_with_a_mode_under_another_name_gets_no_second_one(
    tmp_path, monkeypatch
) -> None:
    """automation's `logs` mode runs logs_agent/ through its card path."""
    root = _blueprint(tmp_path, monkeypatch)
    (root / "composition-model.yaml").write_text(
        yaml.safe_dump(
            {
                "kind": "CompositionModel",
                "deploymentModes": {
                    "ops": {
                        "kind": "agent",
                        "imageTag": "hub.example.com/demo:old",
                        "env": [
                            {
                                "name": "AGENT_CARD_PATH",
                                "value": "netops/agent.json",
                            }
                        ],
                    }
                },
            }
        )
    )

    _generate(*IMAGE_FLAGS)

    modes = _model(root)["deploymentModes"]
    assert set(modes) == {"ops", "hermes"}
    assert modes["ops"]["imageTag"] == "hub.example.com/demo:1.2.3"


def test_an_unchanged_rerun_changes_nothing(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    _generate(*IMAGE_FLAGS)
    before = (root / "composition-model.yaml").read_text()

    result = _generate(*IMAGE_FLAGS)

    assert "No changes." in result.output
    assert (root / "composition-model.yaml").read_text() == before


def test_a_file_holding_something_else_is_refused(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    (root / "composition-model.yaml").write_text("kind: AIFabric\n")

    result = runner.invoke(app, ["manifests", "composition-model"])

    assert result.exit_code == 1
    assert "not a CompositionModel" in result.output


def test_outside_a_blueprint_it_says_so(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["manifests", "composition-model"])

    assert result.exit_code == 1
    assert "Not inside a blueprint" in result.output
