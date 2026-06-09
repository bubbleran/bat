"""Tests for ``TelemetryConfig.from_env`` — env-var resolution and defaults."""

import pytest

from bat.telemetry.config import (
    DEFAULT_COLLECTOR_ENDPOINT,
    DEFAULT_SERVICE_NAME,
    TelemetryConfig,
)

_TELEMETRY_ENV_VARS = (
    "TELEMETRY_ENABLED",
    "OTEL_SERVICE_NAME",
    "PHOENIX_COLLECTOR_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "PHOENIX_API_KEY",
    "OTEL_TRACES_EXPORTER",
    "OTEL_FILE_EXPORTER_PATH",
)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Isolate from the host environment so defaults are deterministic."""
    for name in _TELEMETRY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield


def test_disabled_by_default():
    cfg = TelemetryConfig.from_env()
    assert cfg.enabled is False
    assert cfg.service_name == DEFAULT_SERVICE_NAME
    assert cfg.traces_endpoint == DEFAULT_COLLECTOR_ENDPOINT + "/v1/traces"
    assert cfg.exporter == "otlp"
    assert cfg.headers == {}
    assert cfg.file_path is None


@pytest.mark.parametrize(
    "value,expected",
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("nope", False),
    ],
)
def test_enabled_flag_strict_allowlist(monkeypatch, value, expected):
    monkeypatch.setenv("TELEMETRY_ENABLED", value)
    assert TelemetryConfig.from_env().enabled is expected


def test_service_name_precedence(monkeypatch):
    # OTEL_SERVICE_NAME wins over the passed default.
    monkeypatch.setenv("OTEL_SERVICE_NAME", "from-env")
    assert (
        TelemetryConfig.from_env(default_service_name="from-arg").service_name
        == "from-env"
    )
    # Without the env var, the passed default is used.
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    assert (
        TelemetryConfig.from_env(default_service_name="from-arg").service_name
        == "from-arg"
    )


def test_endpoint_precedence_and_suffix(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4318")
    assert (
        TelemetryConfig.from_env().traces_endpoint
        == "http://otel:4318/v1/traces"
    )
    # PHOENIX_COLLECTOR_ENDPOINT takes precedence and a trailing slash is
    # stripped before the /v1/traces path is appended.
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:6006/")
    assert (
        TelemetryConfig.from_env().traces_endpoint
        == "http://phoenix:6006/v1/traces"
    )


def test_api_key_becomes_bearer_header(monkeypatch):
    monkeypatch.setenv("PHOENIX_API_KEY", "secret")
    assert TelemetryConfig.from_env().headers == {
        "Authorization": "Bearer secret"
    }


def test_file_exporter_config(monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "FILE")
    monkeypatch.setenv("OTEL_FILE_EXPORTER_PATH", "/tmp/spans.jsonl")
    cfg = TelemetryConfig.from_env()
    assert cfg.exporter == "file"  # normalized to lowercase
    assert cfg.file_path == "/tmp/spans.jsonl"
