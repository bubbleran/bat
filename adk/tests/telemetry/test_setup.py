import pytest

from bat.telemetry import setup as setup_mod
from bat.telemetry.config import TelemetryConfig
from bat.telemetry.setup import setup_telemetry


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Reset the module-level init guard so each test starts uninitialized."""
    monkeypatch.setattr(setup_mod, "_initialized", False)
    yield
    # The guard is mutated via ``global`` inside setup_telemetry; force it back
    # so a successful run in one test cannot leak into the next.
    setup_mod._initialized = False


def test_setup_disables_without_crashing_when_enabled_but_otel_missing(
    monkeypatch,
):
    """Telemetry on + OpenTelemetry absent degrades gracefully, no crash.

    Rather than crashing the whole agent for a missing optional extra, setup
    logs a clear error (see ``setup_telemetry``) and returns ``False`` so the
    agent keeps serving requests instead of raising to the caller.
    """
    cfg = TelemetryConfig(enabled=True, service_name="test-agent")
    # Simulate the ``telemetry`` extra not being installed.
    monkeypatch.setattr(setup_mod, "trace", None)
    monkeypatch.setattr(setup_mod, "propagate", None)

    assert setup_telemetry(config=cfg) is False
    assert setup_mod.is_enabled() is False


def test_setup_disabled_does_not_raise_without_otel(monkeypatch):
    """Telemetry off stays import-safe even when OpenTelemetry is absent."""
    cfg = TelemetryConfig(enabled=False, service_name="test-agent")
    monkeypatch.setattr(setup_mod, "trace", None)
    monkeypatch.setattr(setup_mod, "propagate", None)

    assert setup_telemetry(config=cfg) is False
    assert setup_mod.is_enabled() is False
