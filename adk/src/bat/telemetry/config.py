from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..logging import create_logger

logger = create_logger(__name__, "debug")

DEFAULT_SERVICE_NAME = "bat-agent"
# Arize Phoenix listens on :6006 by default and ingests OTLP/HTTP at /v1/traces.
DEFAULT_COLLECTOR_ENDPOINT = "http://localhost:6006"
DEFAULT_FILE_PATH = "spans.jsonl"
_TRACES_PATH = "/v1/traces"



@dataclass
class ExporterSpec:
    """A single resolved telemetry destination.

    Attributes:
        kind (str): Canonical exporter kind: ``"file"``, ``"otlp"`` or
            ``"console"``.
        file_path (Optional[str]): Target JSONL file (``kind == "file"``).
        traces_endpoint (Optional[str]): Full OTLP/HTTP traces URL
            (``kind == "otlp"``).
    """

    kind: str
    file_path: Optional[str] = None
    traces_endpoint: Optional[str] = None


def _spec_from_type(
    type_value: Optional[str],
    *,
    file_path: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> Optional[ExporterSpec]:
    """Resolve a single output entry into an :class:`ExporterSpec`.

    Unknown ``type`` values are skipped (with a warning) rather than raising,
    so one bad entry never disables the whole telemetry pipeline.
    """
    kind= type_value.strip().lower() if type_value else None
    if kind is None:
        logger.warning(
            "Unknown telemetry output type %r; skipping (expected one of "
            "local, remote, console).",
            type_value,
        )
        return None
    if kind == "file" or kind == "local":
        return ExporterSpec(
            kind="file", file_path=file_path or DEFAULT_FILE_PATH
        )
    if kind == "otlp" or kind == "remote":
        base = (endpoint or DEFAULT_COLLECTOR_ENDPOINT).rstrip("/")
        return ExporterSpec(
            kind="otlp",
            traces_endpoint=base + _TRACES_PATH,
        )
    if kind == "console":
        return ExporterSpec(kind="console")
    
    logger.warning("Unknown telemetry output type %r; skipping (expected one of local, remote, console).",
        type_value,
    )
    return None


@dataclass
class TelemetryConfig:
    """Resolved telemetry settings.

    Attributes:
        enabled (bool): Master switch.
        service_name (str): Value of the ``service.name`` resource attribute.
        project_name (Optional[str]): OpenInference/Phoenix project name, set as
            the ``openinference.project.name`` resource attribute. ``None``
            leaves Phoenix's ``default`` project.
        exporters (List[ExporterSpec]): One entry per active destination; the
            spans are fanned out to all of them.
    """

    enabled: bool
    service_name: str
    project_name: Optional[str] = None
    exporters: List[ExporterSpec] = field(default_factory=list)
    
    @classmethod
    def from_settings(
        cls,
        *,
        enabled: bool = False,
        service_name: Optional[str] = None,
        project_name: Optional[str] = None,
        outputs: Optional[List[Any]] = None,
        default_service_name: Optional[str] = None,
    ) -> "TelemetryConfig":
        """Build a :class:`TelemetryConfig` from explicit settings.

        Used when telemetry is configured from ``config.yaml`` (the
        ``telemetry`` section).

        Args:
            enabled (bool): Master switch.
            service_name (Optional[str]): ``service.name``; falls back to
                ``default_service_name`` then ``DEFAULT_SERVICE_NAME``.
            project_name (Optional[str]): OpenInference/Phoenix project name
                (the ``openinference.project.name`` resource attribute); ``None``
                leaves Phoenix's ``default`` project.
            outputs (Optional[List[Any]]): One entry per destination, each a
                dict (or object) with ``type`` (``local``/``remote``/
                ``console``) plus ``file_path`` / ``endpoint`` as relevant.
            default_service_name (Optional[str]): Fallback service name (e.g.
                the agent card name).
        """
        resolved_service_name = (
            service_name or default_service_name or DEFAULT_SERVICE_NAME
        )
        specs: List[ExporterSpec] = []
        for out in outputs or []:
            spec = _spec_from_type(
                out.get("type"),
                file_path=out.get("file_path"),
                endpoint=out.get("endpoint"),
            )
            if spec is not None:
                specs.append(spec)

        # Enabled with no usable output -> fall back to a remote (OTLP)
        # exporter at the default collector, matching the historical default.
        if enabled and not specs:
            logger.debug(
                "Telemetry enabled with no outputs; defaulting to remote "
                "(OTLP) at %s.",
                DEFAULT_COLLECTOR_ENDPOINT,
            )
            specs.append(_spec_from_type("remote"))

        return cls(
            enabled=enabled,
            service_name=resolved_service_name,
            project_name=project_name,
            exporters=specs,
        )
