import asyncio

import pytest
from pydantic import ValidationError

from bat.config import load_agent_config


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in ("MODEL", "MODEL_PROVIDER", "BASE_URL", "CONFIG_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def _write(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return str(path)


def test_loads_inside_a_running_loop(tmp_path):
    path = _write(
        tmp_path,
        "mcp-servers:\n  - name: m\n    url: http://localhost:1/mcp\n"
        "remote-agents:\n  - name: a\n    url: http://localhost:2\n",
    )

    async def main():
        return load_agent_config(path)

    config = asyncio.run(main())

    assert [s.name for s in config.mcp_servers] == ["m"]
    assert [a.name for a in config.remote_agents] == ["a"]


def test_model_env_overrides_config(tmp_path, monkeypatch):
    path = _write(tmp_path, "model:\n  provider: openai\n  name: gpt-4.1\n")
    monkeypatch.setenv("MODEL", "ollama:llama3.1:8b")
    monkeypatch.setenv("BASE_URL", "http://localhost:11434")

    model = load_agent_config(path).model

    assert (model.provider, model.name) == ("ollama", "llama3.1:8b")
    assert model.base_url == "http://localhost:11434"


def test_model_provider_env_keeps_the_colon_in_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL", "llama3.1:8b")
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")

    model = load_agent_config(_write(tmp_path, "")).model

    assert (model.provider, model.name) == ("ollama", "llama3.1:8b")


def test_vars_are_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("DB", "postgresql://db/agent")
    path = _write(tmp_path, "persistence:\n  type: postgres\n  url: ${DB}\n")

    assert load_agent_config(path).persistence.url == "postgresql://db/agent"


def test_missing_file_gives_an_empty_config(tmp_path):
    config = load_agent_config(str(tmp_path / "nope.yaml"))

    assert config.model is None
    assert config.endpoint.port == 9900


def test_checkpoints_true_means_memory(tmp_path):
    config = load_agent_config(_write(tmp_path, "checkpoints: true\n"))

    assert config.persistence.type == "memory"


def test_postgres_needs_a_url(tmp_path):
    with pytest.raises(ValidationError, match="persistence.url"):
        load_agent_config(_write(tmp_path, "persistence:\n  type: postgres\n"))


def test_persistence_wins_over_checkpoints(tmp_path):
    text = "checkpoints: true\npersistence:\n  type: sqlite\n  url: /d/a.db\n"
    persistence = load_agent_config(_write(tmp_path, text)).persistence

    assert (persistence.type, persistence.url) == ("sqlite", "/d/a.db")


def test_a_database_without_its_extra_stops_the_load(tmp_path, monkeypatch):
    monkeypatch.setattr("bat.config.find_spec", lambda name: None)
    text = "persistence:\n  type: sqlite\n  url: a.db\n"
    hint = r"pip install 'bat-adk\[sqlite\]'"

    with pytest.raises(ValidationError, match=hint):
        load_agent_config(_write(tmp_path, text))


def test_port_in_url_moves_to_port(tmp_path):
    path = _write(tmp_path, "endpoint:\n  url: http://localhost:9300/\n")

    endpoint = load_agent_config(path).endpoint

    assert (endpoint.url, endpoint.port) == ("http://localhost", 9300)


def test_unknown_privacy_level_fails(tmp_path):
    with pytest.raises(ValidationError):
        load_agent_config(_write(tmp_path, "telemetry:\n  privacy: secret\n"))
