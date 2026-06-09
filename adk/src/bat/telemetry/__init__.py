"""OpenTelemetry integration for the ADK.

Public surface kept intentionally small. All helpers are safe to call whether
or not the optional ``telemetry`` extra is installed and whether or not
telemetry has been enabled via ``TELEMETRY_ENABLED``.
"""

from . import attributes
from .config import TelemetryConfig
from .setup import (
    SpanKind,
    extract_context,
    get_tracer,
    inject_context,
    is_enabled,
    setup_telemetry,
    shutdown_telemetry,
)

__all__ = [
    "attributes",
    "TelemetryConfig",
    "SpanKind",
    "extract_context",
    "get_tracer",
    "inject_context",
    "is_enabled",
    "setup_telemetry",
    "shutdown_telemetry",
]
