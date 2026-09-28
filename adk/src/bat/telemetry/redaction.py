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

Span (graph node) names and tool names sit at higher privacy levels because
they are identity rather than content, and blanking them costs trace
readability (and, for tool names, the eval engine's tool-call metrics). See
:mod:`bat.telemetry.privacy` for the ladder.
"""

from typing import Dict, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .privacy import TelemetryPrivacy

# Same sentinel OpenInference's TraceConfig writes, so a consumer sees one
# consistent marker regardless of which layer did the redaction.
REDACTED = "__REDACTED__"

# Matched as substrings because OpenInference nests these under indexed
# paths, e.g.
#   llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments
_CONTENT_MARKERS = (
    "tool.description",
    "tool.parameters",
    "tool.json_schema",
    "tool_call.function.arguments",
)

# Attribute holding the OpenInference span kind (LLM / CHAIN / TOOL / ...).
# Reused as the replacement span name so redacted traces keep their shape.
_SPAN_KIND_KEY = "openinference.span.kind"

# Tool identity, redacted only at the `full` level.
_NAME_MARKERS = (
    "tool.name",
    "tool_call.function.name",
    "gen_ai.tool.name",
)


def _is_tool_content(key: str) -> bool:
    """True when ``key`` names tool content rather than tool identity."""
    return any(marker in key for marker in _CONTENT_MARKERS)


def _is_tool_name(key: str) -> bool:
    """True when ``key`` names a tool's identity."""
    return any(marker in key for marker in _NAME_MARKERS)


def redact_attributes(
    attributes: Optional[Dict[str, object]],
    *,
    hide_tool_names: bool = False,
) -> Dict[str, object]:
    """Replace tool attribute values with :data:`REDACTED`.

    Keys are preserved so consumers can still tell *that* a tool span carried
    a description or arguments, only not what they said.

    Args:
        attributes (Optional[Dict[str, object]]): The span's attributes.
        hide_tool_names (bool): Also redact tool identity (the ``full``
            privacy level).
    """
    if not attributes:
        return {}

    def _redact(key: str, value: object) -> object:
        if _is_tool_content(key):
            return REDACTED
        if hide_tool_names and _is_tool_name(key):
            return REDACTED
        return value

    return {key: _redact(key, value) for key, value in attributes.items()}


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
        if self._privacy.hides_content:
            attributes = redact_attributes(
                attributes,
                hide_tool_names=self._privacy.hides_tool_names,
            )

        name = span.name
        if self._privacy.hides_span_names:
            kind = attributes.get(_SPAN_KIND_KEY)
            name = str(kind) if kind else REDACTED

        # Rebuild rather than mutate: ReadableSpan is the exporter-facing
        # read-only view, and the live Span it came from is already ended.
        return ReadableSpan(
            name=name,
            context=span.get_span_context(),
            parent=span.parent,
            resource=span.resource,
            attributes=attributes,
            events=span.events,
            links=span.links,
            kind=span.kind,
            status=span.status,
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
