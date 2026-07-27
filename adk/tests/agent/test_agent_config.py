"""Tests for the nested config.yaml schema on AgentConfig."""

from bat.agent.config import AgentConfig


def test_empty_config_has_no_optional_sections():
    cfg = AgentConfig()
    assert cfg.endpoint is None
    assert cfg.model is None
    assert cfg.telemetry is None
    assert cfg.agent_card is None
    assert cfg.checkpoints is False
    assert cfg.mcp_servers == []
    assert cfg.remote_agents == []


def test_full_nested_schema_parses():
    cfg = AgentConfig.model_validate(
        {
            "endpoint": {
                "url": "http://0.0.0.0/",
                "port": 9900,
            },
            "model": {
                "provider": "openai",
                "name": "gpt-4.1-mini",
                "base_url": "http://localhost:11434",
            },
            "checkpoints": True,
            "agent_card": "./cards/agent.json",
            "telemetry": {
                "service_name": "my-agent",
                "output": [
                    {"type": "local", "file_path": "spans.jsonl"},
                    {"type": "remote", "endpoint": "http://localhost:6006"},
                ],
            },
            "remote-agents": [
                {
                    "name": "SMO Agent",
                    "url": "http://localhost:10001",
                }
            ],
        }
    )

    assert cfg.endpoint.url == "http://0.0.0.0/"
    assert cfg.endpoint.port == 9900

    assert cfg.model.provider == "openai"
    assert cfg.model.name == "gpt-4.1-mini"
    assert cfg.model.base_url == "http://localhost:11434"

    assert cfg.checkpoints is True
    assert cfg.agent_card == "./cards/agent.json"

    assert cfg.telemetry.service_name == "my-agent"
    assert cfg.telemetry.output[0].type == "local"
    assert cfg.telemetry.output[0].file_path == "spans.jsonl"
    assert cfg.telemetry.output[1].type == "remote"
    assert cfg.telemetry.output[1].endpoint == "http://localhost:6006"

    assert cfg.remote_agents[0].name == "SMO Agent"
    assert cfg.remote_agents[0].url == "http://localhost:10001"


def test_partial_sections_default_their_fields():
    cfg = AgentConfig.model_validate({"model": {"name": "gpt-4o"}})
    assert cfg.model.name == "gpt-4o"
    assert cfg.model.provider is None
    assert cfg.model.base_url is None
    # A telemetry section parses with only `output`; service_name defaults to
    # None. Enablement is no longer a config field -- it is derived (by
    # AgentApplication) from whether any output is configured.
    cfg2 = AgentConfig.model_validate(
        {"telemetry": {"output": [{"type": "console"}]}}
    )
    assert cfg2.telemetry.service_name is None
    assert cfg2.telemetry.output[0].type == "console"
