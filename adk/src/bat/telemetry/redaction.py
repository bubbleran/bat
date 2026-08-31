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

What is deliberately **kept**:

- ``tool.name`` / ``tool_call.function.name``: the eval engine reconstructs
  per-episode tool calls from spans, and its tool-call metrics key off the
  name. Redacting names would silently empty those metrics.
- token counts, span kind, hierarchy and timing: cost accounting and the
  eval engine depend on them.

Span (graph node) names are handled separately behind ``hide_span_names``:
they are structural rather than content, and blanking them costs a lot of
trace readability, so it is its own opt-in.
"""

from typing import Dict, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

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


def _is_tool_content(key: str) -> bool:
    """True when ``key`` names tool content rather than tool identity."""
    return any(marker in key for marker in _CONTENT_MARKERS)


def redact_attributes(
    attributes: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Replace tool-content attribute values with :data:`REDACTED`.

    Keys are preserved so consumers can still tell *that* a tool span carried
    a description or arguments, only not what they said.
    """
    if not attributes:
        return {}
    return {
        key: (REDACTED if _is_tool_content(key) else value)
        for key, value in attributes.items()
    }


class RedactingSpanExporter(SpanExporter):
    """Wrap an exporter, rewriting each span before it is handed over.

    Args:
        delegate (SpanExporter): The exporter that performs the real export.
        hide_tool_content (bool): Redact tool descriptions, parameter schemas
            and tool-call arguments (tool *names* are kept -- see the module
            docstring).
        hide_span_names (bool): Replace span names with the span's
            OpenInference kind (``LLM``/``CHAIN``/``TOOL``/...), falling back
            to :data:`REDACTED`, so graph node names stop leaking while the
            trace keeps its shape.
    """

    def __init__(
        self,
        delegate: SpanExporter,
        *,
        hide_tool_content: bool = True,
        hide_span_names: bool = False,
    ) -> None:
        self._delegate = delegate
        self._hide_tool_content = hide_tool_content
        self._hide_span_names = hide_span_names

    def _redact(self, span: ReadableSpan) -> ReadableSpan:
        attributes = dict(span.attributes or {})
        if self._hide_tool_content:
            attributes = redact_attributes(attributes)

        name = span.name
        if self._hide_span_names:
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
