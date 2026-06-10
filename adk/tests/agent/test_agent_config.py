"""Tests for the nested config.yaml schema on AgentConfig."""

from bat.agent.config import AgentConfig


def test_empty_config_has_no_optional_sections():
    cfg = AgentConfig()
    assert cfg.endpoint is None
    assert cfg.model is None
    assert cfg.telemetry is None
    assert cfg.checkpoints is False
    assert cfg.mcp_servers == []
    assert cfg.remote_agents == []


def test_full_nested_schema_parses():
    cfg = AgentConfig.model_validate(
        {
            "endpoint": {
                "url": "http://0.0.0.0/",
                "port": 9900,
                "mcp_port": 9800,
            },
            "model": {
                "provider": "openai",
                "name": "gpt-4.1-mini",
                "base_url": "http://localhost:11434",
            },
            "checkpoints": True,
            "telemetry": {
                "enabled": True,
                "exporter": "file",
                "endpoint": "http://localhost:6006",
                "service_name": "my-agent",
                "file_path": "spans.jsonl",
            },
            "remote-agents": [
                {
                    "name": "SMO Agent",
                    "url": "http://localhost:10001",
                    "protocol": "a2a",
                }
            ],
        }
    )

    assert cfg.endpoint.url == "http://0.0.0.0/"
    assert cfg.endpoint.port == 9900
    assert cfg.endpoint.mcp_port == 9800

    assert cfg.model.provider == "openai"
    assert cfg.model.name == "gpt-4.1-mini"
    assert cfg.model.base_url == "http://localhost:11434"

    assert cfg.checkpoints is True

    assert cfg.telemetry.enabled is True
    assert cfg.telemetry.exporter == "file"
    assert cfg.telemetry.endpoint == "http://localhost:6006"
    assert cfg.telemetry.service_name == "my-agent"
    assert cfg.telemetry.file_path == "spans.jsonl"

    # The protocol alias is still normalized to uppercase.
    assert cfg.remote_agents[0].name == "SMO Agent"
    assert cfg.remote_agents[0].protocol == "A2A"


def test_partial_sections_default_their_fields():
    cfg = AgentConfig.model_validate({"model": {"name": "gpt-4o"}})
    assert cfg.model.name == "gpt-4o"
    assert cfg.model.provider is None
    assert cfg.model.base_url is None
    # telemetry.enabled defaults to False when the section is given without it.
    cfg2 = AgentConfig.model_validate({"telemetry": {"exporter": "console"}})
    assert cfg2.telemetry.enabled is False
    assert cfg2.telemetry.exporter == "console"
