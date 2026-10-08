from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..logging import create_logger
from .privacy import TelemetryPrivacy, parse_privacy

logger = create_logger(__name__, "debug")

DEFAULT_SERVICE_NAME = "bat-agent"
# Arize Phoenix listens on :6006 by default and ingests OTLP/HTTP at /v1/traces.
DEFAULT_COLLECTOR_ENDPOINT = "http://localhost:6006"
DEFAULT_FILE_PATH = "spans.jsonl"


@dataclass
class ExporterSpec:
    """One span destination.

    Attributes:
        kind (str): ``"file"``, ``"otlp"`` or ``"console"``.
        file_path (Optional[str]): JSONL file, for ``file``.
        traces_endpoint (Optional[str]): OTLP/HTTP traces URL, for ``otlp``.
    """

    kind: str
    file_path: Optional[str] = None
    traces_endpoint: Optional[str] = None


def _exporter_spec(output: Dict[str, Any]) -> Optional[ExporterSpec]:
    """Resolve one ``telemetry.output`` entry; ``None`` for an unknown type."""
    kind = output.get("type")
    if kind == "local":
        file_path = output.get("file_path") or DEFAULT_FILE_PATH
        return ExporterSpec(kind="file", file_path=file_path)
    if kind == "remote":
        base = output.get("endpoint") or DEFAULT_COLLECTOR_ENDPOINT
        endpoint = base.rstrip("/") + "/v1/traces"
        return ExporterSpec(kind="otlp", traces_endpoint=endpoint)
    if kind == "console":
        return ExporterSpec(kind="console")
    logger.warning(
        f"Unknown telemetry output type {kind!r}; skipping "
        f"(expected one of local, remote, console)."
    )
    return None


@dataclass
class TelemetryConfig:
    """Resolved telemetry settings.

    Attributes:
        enabled (bool): Master switch.
        service_name (str): The ``service.name`` resource attribute.
        project_name (Optional[str]): Phoenix project (the
            ``openinference.project.name`` resource attribute); ``None``
            leaves Phoenix's ``default`` project.
        privacy (TelemetryPrivacy): See :mod:`bat.telemetry.privacy`.
        exporters (List[ExporterSpec]): Every span goes to all of them.
    """

    enabled: bool
    service_name: str
    project_name: Optional[str] = None
    privacy: TelemetryPrivacy = TelemetryPrivacy.NONE
    exporters: List[ExporterSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.privacy = parse_privacy(self.privacy)

    @classmethod
    def from_settings(
        cls,
        *,
        enabled: bool = False,
        service_name: Optional[str] = None,
        project_name: Optional[str] = None,
        privacy: object = None,
        outputs: Optional[List[Dict[str, Any]]] = None,
        default_service_name: Optional[str] = None,
    ) -> "TelemetryConfig":
        """Build a :class:`TelemetryConfig` from the ``config.yaml``
        ``telemetry`` section.

        ``service_name`` falls back to ``default_service_name`` (the agent
        card name), then ``DEFAULT_SERVICE_NAME``. Each ``outputs`` entry is
        a dict with ``type`` (``local``/``remote``/``console``) and
        ``file_path`` or ``endpoint``; unknown types are skipped.
        """
        exporters = []
        for output in outputs or []:
            spec = _exporter_spec(output)
            if spec is not None:
                exporters.append(spec)

        return cls(
            enabled=enabled,
            service_name=(
                service_name or default_service_name or DEFAULT_SERVICE_NAME
            ),
            project_name=project_name,
            privacy=privacy,
            exporters=exporters,
        )
