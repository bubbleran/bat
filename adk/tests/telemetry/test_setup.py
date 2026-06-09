import pytest

from bat.telemetry import setup as setup_mod
from bat.telemetry.setup import setup_telemetry


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Reset the module-level init guard so each test starts uninitialized."""
    monkeypatch.setattr(setup_mod, "_initialized", False)
    monkeypatch.delenv("TELEMETRY_ENABLED", raising=False)
    yield
    # The guard is mutated via ``global`` inside setup_telemetry; force it back
    # so a successful run in one test cannot leak into the next.
    setup_mod._initialized = False


def test_setup_raises_when_enabled_but_otel_missing(monkeypatch):
    """Telemetry on + OpenTelemetry absent must hard-fail, not no-op."""
    monkeypatch.setenv("TELEMETRY_ENABLED", "1")
    # Simulate the ``telemetry`` extra not being installed.
    monkeypatch.setattr(setup_mod, "trace", None)
    monkeypatch.setattr(setup_mod, "propagate", None)

    with pytest.raises(RuntimeError, match="OpenTelemetry is not installed"):
        setup_telemetry(service_name="test-agent")

    assert setup_mod.is_enabled() is False


def test_setup_disabled_does_not_raise_without_otel(monkeypatch):
    """Telemetry off stays import-safe even when OpenTelemetry is absent."""
    # TELEMETRY_ENABLED unset by the fixture -> disabled.
    monkeypatch.setattr(setup_mod, "trace", None)
    monkeypatch.setattr(setup_mod, "propagate", None)

    assert setup_telemetry(service_name="test-agent") is False
    assert setup_mod.is_enabled() is False
