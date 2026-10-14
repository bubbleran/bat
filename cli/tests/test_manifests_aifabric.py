"""`bat manifests aifabric`: a blueprint's AIFabric from its agents'
configs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def _config(root: Path, agent: str) -> dict:
    return yaml.safe_load((root / agent / "config.yaml").read_text())


def _write_config(root: Path, agent: str, config: dict) -> None:
    (root / agent / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


def _blueprint(tmp_path: Path, monkeypatch) -> Path:
    """demo: netops (9309) calls hermes (9310) and two MCP servers."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init", "blueprint", "demo"])
    root = tmp_path / "demo"
    monkeypatch.chdir(root)
    runner.invoke(app, ["add", "agent", "netops", "--port", "9309"])
    runner.invoke(app, ["add", "agent", "hermes", "--port", "9310"])

    netops = _config(root, "netops")
    netops["remote-agents"] = [
        {"name": "Hermes Agent", "url": "http://localhost:9310"}
    ]
    netops["mcp-servers"] = [
        {
            "name": "ClusterView",
            "url": "http://cluster-view.trirematics.svc:8080/mcp",
        },
        {"name": "Knowledge", "url": "http://localhost:9320/mcp"},
    ]
    netops["telemetry"] = {
        "project_name": "demo",
        "output": [{"type": "remote", "endpoint": "http://localhost:6006"}],
    }
    _write_config(root, "netops", netops)
    return root


def _generate(*args: str):
    result = runner.invoke(app, ["manifests", "aifabric", *args])
    assert result.exit_code == 0, result.output
    return result


def _fabric(root: Path) -> dict:
    return yaml.safe_load((root / "aifabric.yaml").read_text())


def _agent(fabric: dict, name: str) -> dict:
    return next(a for a in fabric["spec"]["agents"] if a["name"] == name)


def test_each_agent_becomes_an_internal_agent_of_the_fabric(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    _generate()

    fabric = _fabric(root)
    assert fabric["apiVersion"] == "orama.trirematics.io/v1"
    assert fabric["kind"] == "AIFabric"
    assert fabric["metadata"] == {"name": "demo-fabric", "namespace": "default"}
    hermes = _agent(fabric, "hermes")
    assert hermes["type"] == "internal"
    assert hermes["internal"]["model"] == "demo/hermes"
    # Role "none": the dependencies are exactly the config's.
    assert hermes["internal"]["role"] == "none"


def test_remote_agents_become_dependencies(tmp_path, monkeypatch) -> None:
    """Matched by port: config.yaml names remote agents by their card name
    and reaches them on localhost."""
    root = _blueprint(tmp_path, monkeypatch)

    _generate()

    netops = _agent(_fabric(root), "netops")
    assert netops["internal"]["dependencies"] == ["hermes"]
    assert "dependencies" not in _agent(_fabric(root), "hermes")["internal"]


def test_agents_on_the_same_model_share_one_llm(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    _generate()

    fabric = _fabric(root)
    assert fabric["spec"]["llms"] == [
        {
            "name": "openai-gpt-6-luna",
            "provider": "openai",
            "model": "gpt-6-luna",
            "apiKeySecretRef": {
                "name": "openai-api-key",
                "key": "OPENAI_API_KEY",
            },
        }
    ]
    assert {a["internal"]["llm"] for a in fabric["spec"]["agents"]} == {
        "openai-gpt-6-luna"
    }


def test_mcp_servers_become_fabric_mcp_servers(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    result = _generate()

    fabric = _fabric(root)
    # CamelCase names become words, as the orama manifests spell them.
    assert _agent(fabric, "netops")["internal"]["mcpServers"] == [
        "cluster-view",
        "knowledge",
    ]
    servers = {m["name"]: m for m in fabric["spec"]["mcp"]}
    assert servers["cluster-view"] == {
        "name": "cluster-view",
        "type": "external",
        "external": {"baseUrl": "http://cluster-view.trirematics.svc:8080/mcp"},
    }
    # A localhost URL means nothing inside the cluster.
    assert "knowledge" in result.output and "localhost" in result.output


def test_a_local_telemetry_endpoint_is_not_put_in_the_fabric(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    result = _generate()

    assert "telemetry" not in _fabric(root)["spec"]
    assert "--telemetry-endpoint" in result.output


def test_the_telemetry_endpoint_can_be_given(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)

    _generate(
        "--telemetry-endpoint",
        "http://phoenix-svc.phoenix.svc.cluster.local:6006",
    )

    assert _fabric(root)["spec"]["telemetry"] == {
        "endpoint": "http://phoenix-svc.phoenix.svc.cluster.local:6006",
        "projectName": "demo",
    }


def test_running_again_follows_the_configs_and_keeps_the_hand_edits(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    _generate()

    fabric = _fabric(root)
    fabric["spec"]["imagePullSecrets"] = [{"name": "bubbleran-hub"}]
    netops = _agent(fabric, "netops")
    netops["chattable"] = False
    netops["internal"]["model"] = "automation/netops"
    netops["internal"]["role"] = "worker"
    fabric["spec"]["llms"][0]["apiKeySecretRef"]["name"] = "my-openai-secret"
    fabric["spec"]["agents"].append(
        {
            "name": "supervisor",
            "type": "internal",
            "internal": {"model": "supervisor/supervisor-plan", "llm": "x"},
        }
    )
    (root / "aifabric.yaml").write_text(
        "# Deploys the demo blueprint.\n"
        + yaml.safe_dump(fabric, sort_keys=False)
    )
    config = _config(root, "netops")
    config["remote-agents"] = []
    _write_config(root, "netops", config)

    _generate()

    text = (root / "aifabric.yaml").read_text()
    assert text.startswith("# Deploys the demo blueprint.\n")
    fabric = yaml.safe_load(text)
    netops = _agent(fabric, "netops")
    # Kept: written by hand, or about other blueprints' agents.
    assert fabric["spec"]["imagePullSecrets"] == [{"name": "bubbleran-hub"}]
    assert netops["chattable"] is False
    assert netops["internal"]["model"] == "automation/netops"
    assert netops["internal"]["role"] == "worker"
    assert fabric["spec"]["llms"][0]["apiKeySecretRef"]["name"] == (
        "my-openai-secret"
    )
    assert _agent(fabric, "supervisor")["internal"]["llm"] == "x"
    # Followed: what config.yaml says.
    assert "dependencies" not in netops["internal"]


def test_agent_names_become_kubernetes_names(tmp_path, monkeypatch) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    runner.invoke(app, ["add", "agent", "supervisor_v4"])

    _generate()

    assert _agent(_fabric(root), "supervisor-v4")["internal"]["model"] == (
        "demo/supervisor-v4"
    )


def test_the_composition_model_names_the_deployment_modes(
    tmp_path, monkeypatch
) -> None:
    root = _blueprint(tmp_path, monkeypatch)
    (root / "composition-model.yaml").write_text(
        yaml.safe_dump(
            {
                "apiVersion": "orama.trirematics.io/v1",
                "kind": "CompositionModel",
                "metadata": {"name": "network-ops", "namespace": "trirematics"},
                "deploymentModes": {
                    "netops-agent": {"kind": "agent", "args": ["netops"]},
                    "cards-only": {
                        "kind": "agent",
                        "env": [
                            {
                                "name": "AGENT_CARD_PATH",
                                "value": "hermes/agent.json",
                            }
                        ],
                    },
                    "knowledge": {"kind": "mcp", "args": ["knowledge"]},
                },
            }
        )
    )

    _generate()

    fabric = _fabric(root)
    assert fabric["metadata"]["namespace"] == "trirematics"
    assert _agent(fabric, "netops")["internal"]["model"] == (
        "network-ops/netops-agent"
    )
    assert _agent(fabric, "hermes")["internal"]["model"] == (
        "network-ops/cards-only"
    )
    servers = {m["name"]: m for m in fabric["spec"]["mcp"]}
    assert servers["knowledge"] == {
        "name": "knowledge",
        "type": "internal",
        "internal": {"model": "network-ops/knowledge"},
    }


def _hand_made_agent(root: Path, name: str) -> None:
    folder = root / name
    folder.mkdir()
    (folder / "config.yaml").write_text(
        "model:\n  provider: openai\n  name: gpt-4o-mini\n", encoding="utf-8"
    )
    (folder / "agent.json").write_text(json.dumps({"name": name}))


def test_every_agent_folder_is_in_the_fabric(tmp_path, monkeypatch) -> None:
    """The scaffolded dispatcher runs any agent folder, one made by hand
    included."""
    root = _blueprint(tmp_path, monkeypatch)
    _hand_made_agent(root, "topology")

    _generate()

    names = [a["name"] for a in _fabric(root)["spec"]["agents"]]
    assert "topology" in names


def test_an_agent_a_hand_written_dispatcher_cannot_run_is_left_out(
    tmp_path, monkeypatch
) -> None:
    """automation's __main__.py names the agents it runs; a folder it never
    names cannot be deployed."""
    root = _blueprint(tmp_path, monkeypatch)
    (root / "__main__.py").write_text(
        'APPS = {"hermes", "netops"}\n', encoding="utf-8"
    )
    _hand_made_agent(root, "topology")

    result = _generate()

    names = [a["name"] for a in _fabric(root)["spec"]["agents"]]
    assert "topology" not in names
    assert "topology" in result.output


def test_a_privacy_level_in_config_is_flagged(tmp_path, monkeypatch) -> None:
    """The operator renders config.yaml without telemetry.privacy: on the
    cluster only the floor in the agent's code applies."""
    root = _blueprint(tmp_path, monkeypatch)
    config = _config(root, "hermes")
    config["telemetry"] = {"privacy": "content", "output": []}
    _write_config(root, "hermes", config)

    result = _generate()

    assert "privacy" in result.output and "hermes" in result.output


def test_outside_a_blueprint_it_says_so(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["manifests", "aifabric"])

    assert result.exit_code != 0
    assert "blueprint" in result.output


def test_urls_expand_environment_variables_like_the_adk(
    tmp_path, monkeypatch
) -> None:
    """The ADK expands ${VAR} in config.yaml when it loads it."""
    root = _blueprint(tmp_path, monkeypatch)
    config = _config(root, "hermes")
    config["mcp-servers"] = [
        {"name": "VictoriaMetrics", "url": "${VM_MCP_URL}"},
        {"name": "VictoriaLogs", "url": "${VL_MCP_URL}"},
    ]
    config["model"] = {
        "provider": "ollama",
        "name": "qwen3:4b",
        "base_url": "${OLLAMA_URL}",
    }
    _write_config(root, "hermes", config)
    monkeypatch.setenv("VM_MCP_URL", "http://vmm.monitoring.svc:8080/mcp")
    monkeypatch.delenv("VL_MCP_URL", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)

    result = _generate()

    spec = _fabric(root)["spec"]
    servers = {m["name"]: m for m in spec["mcp"]}
    assert servers["victoria-metrics"]["external"]["baseUrl"] == (
        "http://vmm.monitoring.svc:8080/mcp"
    )
    assert "VL_MCP_URL" in result.output
    # An unset variable is never written as an endpoint.
    ollama = next(llm for llm in spec["llms"] if llm["provider"] == "ollama")
    assert "endpoint" not in ollama
    assert "OLLAMA_URL" in result.output


def test_an_unchanged_fabric_is_rewritten_byte_for_byte(
    tmp_path, monkeypatch
) -> None:
    """Lists stay indented under their keys, as in the orama manifests, so
    a re-run only shows in a diff what actually changed."""
    root = _blueprint(tmp_path, monkeypatch)
    _generate()
    first = (root / "aifabric.yaml").read_text()

    _generate()

    assert (root / "aifabric.yaml").read_text() == first
    assert "\n  agents:\n    - name: " in first


def test_every_change_a_run_makes_is_reported(tmp_path, monkeypatch) -> None:
    """Following the configs can replace something chosen for the cluster
    (an LLM, a dependency): each run says what it changed."""
    root = _blueprint(tmp_path, monkeypatch)
    first = _generate()
    assert "added: hermes, netops" in first.output

    config = _config(root, "hermes")
    config["model"] = {"provider": "openai", "name": "gpt-5.6-luna"}
    _write_config(root, "hermes", config)
    config = _config(root, "netops")
    config["remote-agents"] = []
    _write_config(root, "netops", config)

    second = _generate()

    assert "hermes: llm openai-gpt-6-luna -> openai-gpt-5-6-luna" in (
        second.output
    )
    assert "netops: dependencies hermes -> (none)" in second.output
    assert "new LLM openai-gpt-5-6-luna" in second.output
    assert "No changes" in _generate().output
