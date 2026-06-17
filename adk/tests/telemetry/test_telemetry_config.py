"""Tests for ``TelemetryConfig.from_settings`` resolution and defaults.

Telemetry is configured exclusively from ``config.yaml`` (the ``telemetry``
section), so the only builder is ``from_settings``; there is no env-var path.
"""

from bat.telemetry.config import (
    DEFAULT_COLLECTOR_ENDPOINT,
    DEFAULT_FILE_PATH,
    DEFAULT_SERVICE_NAME,
    TelemetryConfig,
)

_OTLP_ENDPOINT = DEFAULT_COLLECTOR_ENDPOINT + "/v1/traces"


def test_disabled_with_no_outputs():
    cfg = TelemetryConfig.from_settings(default_service_name="Card")
    assert cfg.enabled is False
    assert cfg.service_name == "Card"
    # Disabled and no outputs -> nothing resolved.
    assert cfg.exporters == []


def test_service_name_precedence():
    # Explicit service_name wins over the passed default.
    cfg = TelemetryConfig.from_settings(
        service_name="explicit", default_service_name="Card"
    )
    assert cfg.service_name == "explicit"
    # Without it, the passed default is used.
    cfg = TelemetryConfig.from_settings(default_service_name="Card")
    assert cfg.service_name == "Card"
    # With neither, the module default.
    cfg = TelemetryConfig.from_settings()
    assert cfg.service_name == DEFAULT_SERVICE_NAME


def test_enabled_without_outputs_defaults_to_remote():
    cfg = TelemetryConfig.from_settings(enabled=True, default_service_name="C")
    assert len(cfg.exporters) == 1
    assert cfg.exporters[0].kind == "otlp"
    assert cfg.exporters[0].traces_endpoint == _OTLP_ENDPOINT


def test_multiple_outputs_fan_out():
    cfg = TelemetryConfig.from_settings(
        enabled=True,
        service_name="svc",
        outputs=[
            {"type": "local", "file_path": "/tmp/s.jsonl"},
            {"type": "remote", "endpoint": "http://phoenix:6006/"},
            {"type": "console"},
        ],
        default_service_name="Card",
    )
    assert cfg.enabled is True
    assert cfg.service_name == "svc"  # explicit wins over default
    kinds = [s.kind for s in cfg.exporters]
    assert kinds == ["file", "otlp", "console"]
    assert cfg.exporters[0].file_path == "/tmp/s.jsonl"
    # Trailing slash stripped before the /v1/traces path is appended.
    assert cfg.exporters[1].traces_endpoint == "http://phoenix:6006/v1/traces"


def test_local_and_remote_defaults():
    # `local` with no file_path -> default file; `remote` with no endpoint ->
    # default collector.
    cfg = TelemetryConfig.from_settings(
        enabled=True,
        outputs=[{"type": "local"}, {"type": "remote"}],
    )
    assert cfg.exporters[0].kind == "file"
    assert cfg.exporters[0].file_path == DEFAULT_FILE_PATH
    assert cfg.exporters[1].kind == "otlp"
    assert cfg.exporters[1].traces_endpoint == _OTLP_ENDPOINT


def test_unknown_type_is_skipped():
    # A typo'd type is dropped (with a warning), not silently turned into a
    # console exporter; valid entries alongside it still resolve.
    cfg = TelemetryConfig.from_settings(
        enabled=True,
        outputs=[{"type": "bogus"}, {"type": "local"}],
    )
    assert [s.kind for s in cfg.exporters] == ["file"]


def test_project_name_passthrough():
    # project_name flows through; absent -> None (Phoenix's "default" project).
    cfg = TelemetryConfig.from_settings(
        enabled=True, project_name="my-proj", outputs=[{"type": "console"}]
    )
    assert cfg.project_name == "my-proj"
    cfg = TelemetryConfig.from_settings(
        enabled=True, outputs=[{"type": "console"}]
    )
    assert cfg.project_name is None
