import atexit
import contextlib
from typing import Any, Dict, Optional
from .file_exporter import JsonFileSpanExporter
from ..logging import create_logger
from .config import TelemetryConfig

logger = create_logger(__name__, "debug")

# --- Optional OpenTelemetry import -----------------------------------------
# Imported defensively so this module stays import-safe when the ``telemetry``
# extra is absent. ``trace``/``propagate`` are only ever dereferenced after a
# successful ``setup_telemetry`` (which requires the extra), and ``SpanKind``
# falls back to a stand-in so call sites can reference it unconditionally.
try:
    from opentelemetry import propagate, trace
    from opentelemetry.trace import SpanKind
except ImportError:  # pragma: no cover - exercised only without the extra
    propagate = None  # type: ignore[assignment]
    trace = None  # type: ignore[assignment]

    class SpanKind:  # type: ignore[no-redef]
        """Minimal stand-in so callers can reference SpanKind anywhere."""

        INTERNAL = 0
        SERVER = 1
        CLIENT = 2
        PRODUCER = 3
        CONSUMER = 4


_initialized = False
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


# LangGraph 1.x callback hooks that the OpenInference LangChain tracer
# (<=0.1.66) does not implement. Calling them raises a noisy AttributeError.
_LANGGRAPH_CALLBACK_SHIMS = ("on_interrupt", "on_resume")


def _patch_openinference_langgraph_callbacks() -> None:
    """Add no-op LangGraph callbacks to the OpenInference LangChain tracer.

    LangGraph 1.x dispatches ``on_interrupt`` / ``on_resume`` callbacks (fired
    by human-in-the-loop ``interrupt()`` and its resume), but the OpenInference
    LangChain instrumentation (<=0.1.66), being a callback handler, does not
    implement them, which logs a noisy ``AttributeError`` on every interrupt or
    resume. This shim silences that without affecting behavior; it is a no-op
    for hooks that already exist or if the internal module layout changes.
    Remove once OpenInference ships these methods.
    """
    try:
        from openinference.instrumentation.langchain._tracer import (
            OpenInferenceTracer,
        )

        for hook in _LANGGRAPH_CALLBACK_SHIMS:
            if not hasattr(OpenInferenceTracer, hook):
                setattr(
                    OpenInferenceTracer,
                    hook,
                    lambda self, *args, **kwargs: None,
                )
    except Exception:  # pragma: no cover - defensive, version-dependent
        pass


def setup_telemetry(
    service_name: Optional[str] = None,
    *,
    config: Optional[TelemetryConfig] = None,
) -> bool:
    """Configure the global tracer provider and auto-instrumentation.

    Idempotent: safe to call multiple times; only the first call has effect.

    Args:
        service_name (Optional[str]): Unused; telemetry is configured solely
            via ``config``. Kept for signature compatibility.
        config (TelemetryConfig): Resolved telemetry settings, built from the
            ``config.yaml`` ``telemetry`` section by ``AgentApplication``.

    Returns:
        bool: ``True`` if telemetry was activated, ``False`` if it is disabled
            (no outputs configured, or enabled but the ``telemetry`` extra is
            not installed — in which case a clear error is logged and the agent
            keeps running without telemetry rather than crashing).
    """
    global _initialized, _provider

    # Idempotent check
    if _initialized:
        return True

    cfg = config
    if cfg is None or not cfg.enabled:
        logger.debug(
            "Telemetry disabled (add a `telemetry.output` entry in "
            "config.yaml to enable)."
        )
        return False

    if trace is None:
        logger.error(
            "Telemetry is enabled in config.yaml but no telemetry is "
            "installed; continuing with telemetry disabled. Install the "
            "extra to enable it: pip install 'bat-adk[telemetry]'."
        )
        return False

    # Necessary imports (OTel SDK and exporters) are deferred until this point
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    resource = Resource.create({"service.name": cfg.service_name})
    provider = TracerProvider(resource=resource)

    # Fan out to every configured destination: one span processor per exporter,
    # so the same spans reach a local file and a remote collector together.
    for exporter in cfg.exporters:
        if exporter.kind == "console":
            provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )
            logger.info("Telemetry: console exporter active.")
        elif exporter.kind == "file":
            path = exporter.file_path or "spans.jsonl"
            # SimpleSpanProcessor: synchronous write on span end, so a reader
            # (e.g. the eval engine) sees spans without a flush/batch delay.
            provider.add_span_processor( SimpleSpanProcessor(JsonFileSpanExporter(path)))
            logger.info("Telemetry: file exporter -> %s.", path)
            
        elif exporter.kind == "otlp":
            otlp_exp = OTLPSpanExporter(endpoint=exporter.traces_endpoint)
            provider.add_span_processor(BatchSpanProcessor(otlp_exp))
            logger.info(
                "Telemetry: OTLP exporter -> %s.", exporter.traces_endpoint
            )

    logger.info(
        "Telemetry enabled (%d exporter(s), service=%s).",
        len(cfg.exporters),
        cfg.service_name,
    )

    trace.set_tracer_provider(provider)
    _provider = provider
    # Flush and close exporters on interpreter exit. The OTLP/console paths use
    # a BatchSpanProcessor whose buffered spans would otherwise be dropped when
    # the process ends; shutdown() forces a final flush. (The file exporter is a
    # synchronous SimpleSpanProcessor, but shutdown() still closes its handle.)
    atexit.register(shutdown_telemetry)

    # Auto-instrument LangChain / LangGraph via OpenInference. This captures
    # LLM calls, graph nodes and tool executions without touching call sites.
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor

        LangChainInstrumentor().instrument(tracer_provider=provider)
        _patch_openinference_langgraph_callbacks()
        logger.debug("OpenInference LangChain instrumentation active.")
        
    except ImportError:
        logger.warning(
            "openinference-instrumentation-langchain not installed: "
            "LLM/tool spans will not be auto-captured."
        )

    _initialized = True
    return True


def shutdown_telemetry() -> None:
    """Flush and shut down the tracer provider (final span export).

    Forces buffered spans out of the ``BatchSpanProcessor`` and closes the
    exporters. Registered with ``atexit`` by :func:`setup_telemetry`, but also
    safe to call explicitly (e.g. from an application shutdown hook or a test).
    Idempotent and a no-op when telemetry was never initialized.
    """
    global _initialized, _provider
    provider = _provider
    if provider is None:
        return
    _provider = None
    _initialized = False
    with contextlib.suppress(Exception):
        provider.shutdown()


def get_tracer(name: str) -> Any:
    """Return a tracer for ``name``.

    When OpenTelemetry is not installed, returns a no-op tracer. Otherwise it
    returns OTel's tracer, which is a *proxy*: before :func:`setup_telemetry`
    installs a provider it produces non-recording spans, and it automatically
    starts recording once the provider is set. This is what makes a
    module-level ``tracer = get_tracer(__name__)`` (captured at import time,
    before ``setup_telemetry`` runs) work -- returning a ``_NoopTracer`` here
    would cache a dead no-op and silently drop every manual span.
    """
    if trace is None:
        return _NoopTracer()
    return trace.get_tracer(name)


def inject_context(
    carrier: Dict[str, str],
    span: Optional[Any] = None,
) -> Dict[str, str]:
    """Inject a trace context into ``carrier`` (W3C traceparent).

    If ``span`` is given, the context built around that span is injected
    instead of the current context. This matters for async generators that
    yield across tasks, where the span must not be attached to the ambient
    context manager (which would break ``contextvars`` detach).
    """
    if _initialized:
        ctx = trace.set_span_in_context(span) if span is not None else None
        propagate.inject(carrier, context=ctx)
    return carrier


def extract_context(carrier: Dict[str, str]) -> Optional[Any]:
    """Extract a trace context from ``carrier``; ``None`` when unavailable."""
    if not _initialized or not carrier:
        return None
    return propagate.extract(carrier)


def is_enabled() -> bool:
    """Whether telemetry has been successfully initialized."""
    return _initialized
