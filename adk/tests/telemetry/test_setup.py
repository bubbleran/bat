import pytest

from bat.telemetry import setup as setup_mod
from bat.telemetry.attributes import OPENINFERENCE_PROJECT_NAME
from bat.telemetry.config import ExporterSpec, TelemetryConfig
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


def test_project_name_sets_resource_attribute():
    """A configured project_name lands on the Resource as
    ``openinference.project.name`` — the attribute Phoenix routes traces on."""
    cfg = TelemetryConfig(
        enabled=True,
        service_name="svc",
        project_name="my-proj",
        exporters=[ExporterSpec(kind="console")],
    )
    try:
        assert setup_telemetry(config=cfg) is True
        attrs = setup_mod._provider.resource.attributes
        assert attrs["service.name"] == "svc"
        assert attrs[OPENINFERENCE_PROJECT_NAME] == "my-proj"
    finally:
        setup_mod.shutdown_telemetry()


def test_no_project_name_omits_resource_attribute():
    """Without project_name the attribute is absent (Phoenix's "default")."""
    cfg = TelemetryConfig(
        enabled=True,
        service_name="svc",
        exporters=[ExporterSpec(kind="console")],
    )
    try:
        assert setup_telemetry(config=cfg) is True
        assert OPENINFERENCE_PROJECT_NAME not in setup_mod._provider.resource.attributes
    finally:
        setup_mod.shutdown_telemetry()
