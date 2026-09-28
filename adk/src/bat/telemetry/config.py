from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..logging import create_logger
from .privacy import TelemetryPrivacy, parse_privacy

logger = create_logger(__name__, "debug")

DEFAULT_SERVICE_NAME = "bat-agent"
# Arize Phoenix listens on :6006 by default and ingests OTLP/HTTP at /v1/traces.
DEFAULT_COLLECTOR_ENDPOINT = "http://localhost:6006"
DEFAULT_FILE_PATH = "spans.jsonl"
_TRACES_PATH = "/v1/traces"


@dataclass
class ExporterSpec:
    """The destination of telemetry spans. One per active exporter.
    
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
    kind = type_value.strip().lower() if type_value else None
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

    logger.warning(
        f"Unknown telemetry output type {type_value!r}; skipping "
        f"(expected one of local, remote, console)."
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
        privacy (TelemetryPrivacy): How much of the agent's internals may
            leave the process -- ``none``/``content``/``names``/``full``, each
            level a superset of the one below. Token counts, span kinds,
            hierarchy and timing survive at every level, so cost accounting
            keeps working. See :mod:`bat.telemetry.privacy`.
        exporters (List[ExporterSpec]): One entry per active destination; the
            spans are fanned out to all of them.
    """

    enabled: bool
    service_name: str
    project_name: Optional[str] = None
    privacy: TelemetryPrivacy = TelemetryPrivacy.NONE
    exporters: List[ExporterSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Coerce ``privacy`` however this dataclass was constructed.

        ``from_settings`` already parses it, but this is a public dataclass:
        code that builds one directly (tests, embedders) would otherwise hold
        a plain string and fail later, inside ``setup_telemetry``, with an
        ``AttributeError`` far from the mistake.
        """
        self.privacy = parse_privacy(self.privacy)

    @classmethod
    def from_settings(
        cls,
        *,
        enabled: bool = False,
        service_name: Optional[str] = None,
        project_name: Optional[str] = None,
        privacy: object = None,
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
                (the ``openinference.project.name`` resource attribute);
                ``None`` leaves Phoenix's ``default`` project.
            privacy (object): Privacy level as written in ``config.yaml``
                (level name or ordinal); coerced via
                :func:`bat.telemetry.privacy.parse_privacy`.
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

        return cls(
            enabled=enabled,
            service_name=resolved_service_name,
            project_name=project_name,
            privacy=parse_privacy(privacy),
            exporters=specs,
        )
