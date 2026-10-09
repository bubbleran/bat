import atexit
import contextlib
from typing import Any, Dict, Optional

from ..logging import create_logger
from .attributes import OPENINFERENCE_PROJECT_NAME
from .config import TelemetryConfig

logger = create_logger(__name__, "debug")

try:
    from opentelemetry import propagate, trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
except ImportError:  # pragma: no cover - exercised only without the extra
    propagate = None  # type: ignore[assignment]
    trace = None  # type: ignore[assignment]
    Status = StatusCode = None  # type: ignore[assignment,misc]

    class SpanKind:  # type: ignore[no-redef]
        """Minimal stand-in so callers can reference SpanKind anywhere."""

        INTERNAL = 0
        SERVER = 1
        CLIENT = 2
        PRODUCER = 3
        CONSUMER = 4


_provider = None


# --- No-op fallbacks (used when OTel is not installed) ---------------------
class _NoopSpan:
    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def record_exception(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_event(self, *args: Any, **kwargs: Any) -> None:
        pass

    def end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False


class _NoopTracer:
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> _NoopSpan:
        return _NoopSpan()

    def start_span(self, *args: Any, **kwargs: Any) -> _NoopSpan:
        return _NoopSpan()


def setup_telemetry(config: Optional[TelemetryConfig] = None) -> bool:
    """Install the global tracer provider, exporters and LangChain
    instrumentation. Idempotent.

    Returns:
        bool: ``False`` when telemetry is disabled or the ``telemetry`` extra
            is missing (logged; the agent keeps running without it).
    """
    global _provider

    if _provider is not None:
        return True

    if config is None or not config.enabled:
        logger.debug(
            "Telemetry disabled (add a `telemetry.output` entry in "
            "config.yaml to enable)."
        )
        return False

    try:
        if trace is None:
            raise ImportError
        from openinference.instrumentation import TraceConfig
        from openinference.instrumentation.langchain import (
            LangChainInstrumentor,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        from .file_exporter import JsonFileSpanExporter
        from .redaction import RedactingSpanExporter
    except ImportError:
        logger.error(
            "Telemetry is enabled in config.yaml but no telemetry is "
            "installed; continuing with telemetry disabled. Install the "
            "extra to enable it: pip install 'bat-adk[telemetry]'."
        )
        return False

    resource_attributes = {"service.name": config.service_name}
    if config.project_name:
        resource_attributes[OPENINFERENCE_PROJECT_NAME] = config.project_name
    provider = TracerProvider(resource=Resource.create(resource_attributes))

    def _add(processor, exporter):
        if config.privacy.hides_content:
            exporter = RedactingSpanExporter(exporter, privacy=config.privacy)
        provider.add_span_processor(processor(exporter))

    for spec in config.exporters:
        if spec.kind == "console":
            _add(BatchSpanProcessor, ConsoleSpanExporter())
            logger.info("Telemetry: console exporter active.")
        elif spec.kind == "file":
            try:
                _add(SimpleSpanProcessor, JsonFileSpanExporter(spec.file_path))
            except OSError as e:
                logger.error(
                    "Telemetry: cannot open file exporter at %s (%s); "
                    "skipping this destination.",
                    spec.file_path,
                    e,
                )
                continue
            logger.info("Telemetry: file exporter -> %s.", spec.file_path)
        elif spec.kind == "otlp":
            otlp = OTLPSpanExporter(endpoint=spec.traces_endpoint)
            _add(BatchSpanProcessor, otlp)
            logger.info(
                "Telemetry: OTLP exporter -> %s.", spec.traces_endpoint
            )

    logger.info(
        "Telemetry enabled (%d exporter(s), service=%s, privacy=%s).",
        len(config.exporters),
        config.service_name,
        config.privacy.name.lower(),
    )

    trace.set_tracer_provider(provider)

    instrument_kwargs: Dict[str, Any] = {"tracer_provider": provider}
    if config.privacy.hides_content:
        instrument_kwargs["config"] = TraceConfig(
            hide_inputs=True,
            hide_outputs=True,
            hide_prompts=True,
            hide_llm_invocation_parameters=True,
            hide_llm_tools=True,
        )
    LangChainInstrumentor().instrument(**instrument_kwargs)
    logger.debug("OpenInference LangChain instrumentation active.")

    _provider = provider
    atexit.register(shutdown_telemetry)
    return True


def shutdown_telemetry() -> None:
    """Flush buffered spans and close the exporters. Idempotent; registered
    with ``atexit`` by :func:`setup_telemetry`."""
    global _provider
    if _provider is None:
        return
    with contextlib.suppress(Exception):
        _provider.shutdown()
    _provider = None


def mark_span_error(span: Any, description: str) -> None:
    """Set ``span``'s status to ERROR, saying why.

    Trace readers (the eval engine, Phoenix) key off the status, not the
    recorded exception. A no-op without the telemetry extra.
    """
    if Status is None:
        return
    span.set_status(Status(StatusCode.ERROR, description))


def get_tracer(name: str) -> Any:
    """Return a tracer for ``name``; a no-op one without the telemetry extra.

    OTel's tracer is a proxy that starts recording once
    :func:`setup_telemetry` installs the provider, so a module-level
    ``tracer = get_tracer(__name__)`` captured at import time still works.
    """
    if trace is None:
        return _NoopTracer()
    return trace.get_tracer(name)


def inject_context(
    carrier: Dict[str, str],
    span: Optional[Any] = None,
) -> Dict[str, str]:
    """Inject the current trace context, or ``span``'s, into ``carrier``.

    Pass ``span`` from async generators that yield across tasks, where
    attaching it to the ambient context would break the ``contextvars``
    detach.
    """
    if _provider is not None:
        ctx = trace.set_span_in_context(span) if span is not None else None
        propagate.inject(carrier, context=ctx)
    return carrier


def extract_context(carrier: Dict[str, str]) -> Optional[Any]:
    """Extract a trace context from ``carrier``; ``None`` when unavailable."""
    if _provider is None or not carrier:
        return None
    return propagate.extract(carrier)


def is_enabled() -> bool:
    """Whether telemetry has been successfully initialized."""
    return _provider is not None
