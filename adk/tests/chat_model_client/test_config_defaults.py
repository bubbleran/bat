"""ChatModelClientConfig.load precedence: env over config defaults."""

import pytest

from bat.chat_model_client import ChatModelClientConfig

_MODEL_ENV_VARS = ("MODEL", "MODEL_PROVIDER", "BASE_URL")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    for name in _MODEL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    ChatModelClientConfig._clear_defaults()
    yield
    ChatModelClientConfig._clear_defaults()


def test_falls_back_to_config_defaults_when_env_absent():
    ChatModelClientConfig._set_defaults(
        provider="openai", name="gpt-4.1-mini", base_url=None
    )
    cfg = ChatModelClientConfig.load(client_name="c")
    assert cfg.model == "gpt-4.1-mini"
    assert cfg.model_provider == "openai"
    assert cfg.base_url is None


def test_env_overrides_config_defaults(monkeypatch):
    ChatModelClientConfig._set_defaults(
        provider="openai", name="gpt-4.1-mini", base_url="http://from-config"
    )
    monkeypatch.setenv("MODEL", "llama3")
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.setenv("BASE_URL", "http://from-env")
    cfg = ChatModelClientConfig.load()
    assert cfg.model == "llama3"
    assert cfg.model_provider == "ollama"
    assert cfg.base_url == "http://from-env"


def test_env_provider_colon_form_still_works(monkeypatch):
    monkeypatch.setenv("MODEL", "openai:gpt-4o")
    cfg = ChatModelClientConfig.load()
    assert cfg.model_provider == "openai"
    assert cfg.model == "gpt-4o"


def test_env_model_with_config_provider(monkeypatch):
    # MODEL from env, provider from config (no colon in MODEL).
    ChatModelClientConfig._set_defaults(provider="anthropic", name="ignored")
    monkeypatch.setenv("MODEL", "claude-x")
    cfg = ChatModelClientConfig.load()
    assert cfg.model == "claude-x"
    assert cfg.model_provider == "anthropic"


def test_raises_without_model_anywhere():
    with pytest.raises(EnvironmentError, match="Model name not configured"):
        ChatModelClientConfig.load()


def test_raises_without_provider_anywhere(monkeypatch):
    monkeypatch.setenv("MODEL", "some-model")  # no provider, no colon
    with pytest.raises(EnvironmentError, match="provider not configured"):
        ChatModelClientConfig.load()
