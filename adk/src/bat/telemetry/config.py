"""Telemetry configuration, read from environment variables.

All telemetry is opt-in: unless ``TELEMETRY_ENABLED`` is truthy the ADK behaves
exactly as before (no spans, no exporters, no extra dependencies required).
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..logging import create_logger

logger = create_logger(__name__, "debug")

DEFAULT_SERVICE_NAME = "bat-agent"
# Arize Phoenix listens on :6006 by default and ingests OTLP/HTTP at /v1/traces.
DEFAULT_COLLECTOR_ENDPOINT = "http://localhost:6006"
_TRACES_PATH = "/v1/traces"

_TRUTHY = {"1", "true", "yes", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY


@dataclass
class TelemetryConfig:
    """Resolved telemetry settings.

    Attributes
    -------
        enabled (bool): Master switch (``TELEMETRY_ENABLED``).
        service_name (str): Value of the ``service.name`` resource attribute.
        traces_endpoint (str): Full OTLP/HTTP traces endpoint URL.
        headers (Dict[str, str]): Extra headers for the exporter (e.g. auth).
        exporter (str): ``"otlp"`` or ``"console"``.
    """

    enabled: bool
    service_name: str
    traces_endpoint: str
    headers: Dict[str, str] = field(default_factory=dict)
    exporter: str = "otlp"
    file_path: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        default_service_name: Optional[str] = None,
    ) -> "TelemetryConfig":
        """Build a :class:`TelemetryConfig` from environment variables.

        Recognized variables:
            - ``TELEMETRY_ENABLED``: master switch (default off).
            - ``OTEL_SERVICE_NAME``: overrides ``default_service_name``.
            - ``PHOENIX_COLLECTOR_ENDPOINT`` / ``OTEL_EXPORTER_OTLP_ENDPOINT``:
              base collector URL (default ``http://localhost:6006``).
            - ``PHOENIX_API_KEY``: if set, sent as ``Authorization: Bearer``.
            - ``OTEL_TRACES_EXPORTER``: ``otlp`` (default), ``console`` or
              ``file``.
            - ``OTEL_FILE_EXPORTER_PATH``: target file for the ``file`` exporter
              (JSON Lines, one span per line).

        Args:
            default_service_name (Optional[str]): Fallback service name when
                ``OTEL_SERVICE_NAME`` is not set (e.g. the agent card name).
        """
        service_name = (
            os.getenv("OTEL_SERVICE_NAME")
            or default_service_name
            or DEFAULT_SERVICE_NAME
        )

        base_endpoint = (
            os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            or DEFAULT_COLLECTOR_ENDPOINT
        ).rstrip("/")
        traces_endpoint = base_endpoint + _TRACES_PATH

        headers: Dict[str, str] = {}
        api_key = os.getenv("PHOENIX_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        exporter = os.getenv("OTEL_TRACES_EXPORTER", "otlp").strip().lower()
        file_path = os.getenv("OTEL_FILE_EXPORTER_PATH")

        return cls(
            enabled=_env_bool("TELEMETRY_ENABLED", False),
            service_name=service_name,
            traces_endpoint=traces_endpoint,
            headers=headers,
            exporter=exporter,
            file_path=file_path,
        )

    @classmethod
    def from_settings(
        cls,
        *,
        enabled: bool = False,
        service_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        exporter: Optional[str] = None,
        file_path: Optional[str] = None,
        default_service_name: Optional[str] = None,
    ) -> "TelemetryConfig":
        """Build a :class:`TelemetryConfig` from explicit settings.

        Used when telemetry is configured from ``config.yaml`` (the
        ``telemetry`` section) rather than the environment. Only the API key is
        still sourced from the environment (``PHOENIX_API_KEY``), since secrets
        never live in config.yaml.

        Args:
            enabled (bool): Master switch.
            service_name (Optional[str]): ``service.name``; falls back to
                ``default_service_name`` then ``DEFAULT_SERVICE_NAME``.
            endpoint (Optional[str]): OTLP collector base URL; defaults to
                ``DEFAULT_COLLECTOR_ENDPOINT``.
            exporter (Optional[str]): ``otlp`` (default), ``console`` or
                ``file``.
            file_path (Optional[str]): Target file for the ``file`` exporter.
            default_service_name (Optional[str]): Fallback service name (e.g.
                the agent card name).
        """
        resolved_service_name = (
            service_name or default_service_name or DEFAULT_SERVICE_NAME
        )
        base_endpoint = (endpoint or DEFAULT_COLLECTOR_ENDPOINT).rstrip("/")
        traces_endpoint = base_endpoint + _TRACES_PATH

        headers: Dict[str, str] = {}
        api_key = os.getenv("PHOENIX_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        return cls(
            enabled=enabled,
            service_name=resolved_service_name,
            traces_endpoint=traces_endpoint,
            headers=headers,
            exporter=(exporter or "otlp").strip().lower(),
            file_path=file_path,
        )
