"""Exporter-side redaction for content OpenInference's ``TraceConfig`` misses.

``TraceConfig`` masks *inside* the instrumentation, which covers prompts,
messages, completions and the LLM's declared tool list (``llm.tools.*``, via
``hide_llm_tools``). It has no option for the attributes a **tool span**
carries: ``tool.description``, ``tool.parameters`` and the arguments of each
tool call all export in the clear even with every ``hide_*`` flag set.

Tool descriptions are the agent's capability surface -- they read as
documentation of what the agent can do and how to ask for it -- so for an
agent sold as a packaged artifact they leak as much design as the prompts do.
This module closes that gap by wrapping each exporter: spans are rewritten
on the way out, so every destination (file, OTLP, console) sees the same
redacted span.

What is kept at the ``content`` level:

- ``tool.name`` / ``tool_call.function.name``: the eval engine reconstructs
  per-episode tool calls from spans, and its tool-call metrics key off the
  name. Redacting names silently empties those metrics, so it is deferred to
  the ``full`` level where that trade is made explicitly.
- token counts, span kind, hierarchy and timing: cost accounting and the
  eval engine depend on them. These survive at *every* level.

Two more kinds of content are masked here because no ``hide_*`` flag reaches
them: ``input.value`` / ``output.value`` on the ADK's own spans (what one
agent asked another, and the answer), and error text -- a span's status
description and its exception events' message and stack trace, which can quote
the very values ``content`` hides. The exception *type* and the ERROR status
survive: they say that and how a step failed, not what it was handling.

Span (graph node) names and tool names sit at higher privacy levels because
they are identity rather than content, and blanking them costs trace
readability (and, for tool names, the eval engine's tool-call metrics). See
:mod:`bat.telemetry.privacy` for the ladder.
"""

from typing import Dict, Optional, Sequence

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Status

from .privacy import TelemetryPrivacy

# Same sentinel OpenInference's TraceConfig writes.
REDACTED = "__BLOCKED__"


def _hides(key: str, hide_tool_names: bool) -> bool:
    """True when the value under ``key`` must not leave the process."""
    if key in (
        "input.value",
        "output.value",
        "exception.message",
        "exception.stacktrace",
    ):
        return True

    tool_content = (
        "tool.description",
        "tool.parameters",
        "tool.json_schema",
        "tool_call.function.arguments",
    )
    if any(part in key for part in tool_content):
        return True
    tool_names = ("tool.name", "tool_call.function.name")
    return hide_tool_names and any(part in key for part in tool_names)


def redact_attributes(
    attributes: Optional[Dict[str, object]],
    *,
    hide_tool_names: bool = False,
) -> Dict[str, object]:
    """Replace content values with :data:`REDACTED`, keeping the keys.

    Args:
        attributes (Optional[Dict[str, object]]): Span or event attributes.
        hide_tool_names (bool): Also redact tool identity (the ``full``
            privacy level).
    """
    return {
        key: REDACTED if _hides(key, hide_tool_names) else value
        for key, value in (attributes or {}).items()
    }


class RedactingSpanExporter(SpanExporter):
    """Wrap an exporter, rewriting each span before it is handed over.

    Args:
        delegate (SpanExporter): The exporter that performs the real export.
        privacy (TelemetryPrivacy): The effective privacy level; decides
            which of tool content, span names and tool names are redacted.
    """

    def __init__(
        self,
        delegate: SpanExporter,
        *,
        privacy: TelemetryPrivacy = TelemetryPrivacy.CONTENT,
    ) -> None:
        self._delegate = delegate
        self._privacy = privacy

    def _redact(self, span: ReadableSpan) -> ReadableSpan:
        attributes = dict(span.attributes or {})
        events = span.events
        status = span.status
        if self._privacy.hides_content:
            hide = self._privacy.hides_tool_names
            attributes = redact_attributes(attributes, hide_tool_names=hide)
            events = [
                Event(
                    event.name,
                    redact_attributes(event.attributes, hide_tool_names=hide),
                    timestamp=event.timestamp,
                )
                for event in span.events
            ]
            if status is not None and status.description:
                status = Status(status.status_code, REDACTED)

        name = span.name
        if self._privacy.hides_span_names:
            kind = attributes.get("openinference.span.kind")
            name = str(kind) if kind else REDACTED

        # Rebuild rather than mutate: ReadableSpan is the exporter-facing
        # read-only view, and the live Span it came from is already ended.
        return ReadableSpan(
            name=name,
            context=span.get_span_context(),
            parent=span.parent,
            resource=span.resource,
            attributes=attributes,
            events=events,
            links=span.links,
            kind=span.kind,
            status=status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._delegate.export([self._redact(s) for s in spans])

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self._delegate.shutdown()
