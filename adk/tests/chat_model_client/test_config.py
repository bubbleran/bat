import pytest
from bat.chat_model_client import ChatModelClientConfig
from pydantic import ValidationError

def test_init_direct_instantiation():
    config = ChatModelClientConfig(
        model="llama3",
        model_provider="ollama",
        base_url="http://localhost:11434",
        client_name="TestClient",
    )

    assert config.model == "llama3"
    assert config.model_provider == "ollama"
    assert config.base_url == "http://localhost:11434"
    assert config.client_name == "TestClient"

def test_init_optional_fields_default_none():
    config = ChatModelClientConfig(
        model="llama3",
        model_provider="ollama",
    )

    assert config.model == "llama3"
    assert config.model_provider == "ollama"
    assert config.base_url is None
    assert config.client_name is None

def test_init_invalid_model_provider_raises_validation_error():
    with pytest.raises(ValidationError):
        ChatModelClientConfig(
            model="llama3",
            model_provider="invalid_provider",
        )

def test_from_env_requires_model(monkeypatch):
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)

    with pytest.raises(EnvironmentError, match="MODEL environment variable not set"):
        ChatModelClientConfig.from_env()

def test_from_env_requires_model_provider_if_model_not_prefixed(monkeypatch):
    monkeypatch.setenv("MODEL", "llama3")
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)

    with pytest.raises(EnvironmentError, match="MODEL_PROVIDER environment variable not set"):
        ChatModelClientConfig.from_env()

def test_from_env_parses_provider_and_model_from_model_env(monkeypatch):
    monkeypatch.setenv("MODEL", "ollama:llama3")
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)

    config = ChatModelClientConfig.from_env(client_name="TestClient")

    assert config.model_provider == "ollama"
    assert config.model == "llama3"
    assert config.base_url is None
    assert config.client_name == "TestClient"

def test_from_env_uses_model_provider_env_over_model_prefix(monkeypatch):
    monkeypatch.setenv("MODEL", "llama3")
    monkeypatch.setenv("MODEL_PROVIDER", "ollama")
    monkeypatch.delenv("BASE_URL", raising=False)

    config = ChatModelClientConfig.from_env()

    assert config.model_provider == "ollama"
    assert config.model == "llama3"

def test_from_env_reads_base_url(monkeypatch):
    monkeypatch.setenv("MODEL", "ollama:llama3")
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.setenv("BASE_URL", "http://localhost:11434")

    config = ChatModelClientConfig.from_env()

    assert config.model_provider == "ollama"
    assert config.model == "llama3"
    assert config.base_url == "http://localhost:11434"

def test_from_env_invalid_provider_in_model_prefix_raises_validation(monkeypatch):
    monkeypatch.setenv("MODEL", "badprovider:some-model")
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)

    with pytest.raises(ValidationError):
        ChatModelClientConfig.from_env()

def test_from_env_invalid_provider_env_raises_validation(monkeypatch):
    monkeypatch.setenv("MODEL", "llama3")
    monkeypatch.setenv("MODEL_PROVIDER", "badprovider")
    monkeypatch.delenv("BASE_URL", raising=False)

    with pytest.raises(ValidationError):
        ChatModelClientConfig.from_env()
