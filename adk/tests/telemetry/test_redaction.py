"""Tests for exporter-side redaction (tool content and span names).

These cover the gap OpenInference's TraceConfig leaves open: a tool span's
description, parameter schema and call arguments are not reachable by any
``hide_*`` flag, so they are redacted on the way out instead.
"""

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from opentelemetry.trace import SpanContext, SpanKind, TraceFlags

from bat.telemetry.privacy import TelemetryPrivacy
from bat.telemetry.redaction import (
    REDACTED,
    RedactingSpanExporter,
    redact_attributes,
)


class _CapturingExporter:
    """Stand-in delegate that records what it was handed."""

    def __init__(self):
        self.spans = []
        self.flushed = False
        self.shut_down = False

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis=30000):
        self.flushed = True
        return True

    def shutdown(self):
        self.shut_down = True


def _span(name="tools", attributes=None):
    ctx = SpanContext(
        trace_id=0x1,
        span_id=0x2,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return ReadableSpan(
        name=name,
        context=ctx,
        attributes=attributes or {},
        kind=SpanKind.INTERNAL,
        start_time=1,
        end_time=2,
    )


def test_tool_description_and_schema_are_redacted():
    out = redact_attributes(
        {
            "tool.name": "reboot_ran",
            "tool.description": "Reboots the RAN node; disruptive",
            "tool.parameters": '{"cell_id": "string"}',
        }
    )
    assert out["tool.description"] == REDACTED
    assert out["tool.parameters"] == REDACTED
    # Identity is preserved: the eval engine keys tool-call metrics off it.
    assert out["tool.name"] == "reboot_ran"


def test_nested_tool_call_arguments_are_redacted():
    key = (
        "llm.output_messages.0.message.tool_calls.0"
        ".tool_call.function.arguments"
    )
    name_key = (
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"
    )
    out = redact_attributes({key: '{"cell_id": "42"}', name_key: "reboot_ran"})
    assert out[key] == REDACTED
    assert out[name_key] == "reboot_ran"


def test_usage_attributes_survive_redaction():
    out = redact_attributes(
        {"llm.token_count.total": 42, "llm.token_count.prompt": 10}
    )
    assert out["llm.token_count.total"] == 42
    assert out["llm.token_count.prompt"] == 10


def test_exporter_redacts_tool_content_but_keeps_span_name():
    cap = _CapturingExporter()
    exp = RedactingSpanExporter(cap, privacy=TelemetryPrivacy.CONTENT)
    exp.export([_span("call_tool", {"tool.description": "secret"})])

    (got,) = cap.spans
    assert got.attributes["tool.description"] == REDACTED
    assert got.name == "call_tool"


def test_span_names_replaced_by_openinference_kind():
    cap = _CapturingExporter()
    exp = RedactingSpanExporter(cap, privacy=TelemetryPrivacy.NAMES)
    exp.export(
        [_span("classify_intent", {"openinference.span.kind": "CHAIN"})]
    )

    (got,) = cap.spans
    assert got.name == "CHAIN"
    # Structure and timing must survive so traces stay navigable.
    assert got.start_time == 1
    assert got.end_time == 2
    assert got.get_span_context().trace_id == 0x1


def test_span_name_falls_back_to_redacted_without_kind():
    cap = _CapturingExporter()
    exp = RedactingSpanExporter(cap, privacy=TelemetryPrivacy.NAMES)
    exp.export([_span("my_private_node", {})])
    assert cap.spans[0].name == REDACTED


def test_flush_and_shutdown_delegate():
    cap = _CapturingExporter()
    exp = RedactingSpanExporter(cap)
    assert exp.force_flush() is True
    exp.shutdown()
    assert cap.flushed and cap.shut_down


def test_full_level_also_redacts_tool_names():
    """`full` trades the eval engine's tool-call metrics for a hidden
    tool inventory."""
    out = redact_attributes(
        {"tool.name": "reboot_ran", "llm.token_count.total": 7},
        hide_tool_names=True,
    )
    assert out["tool.name"] == REDACTED
    assert out["llm.token_count.total"] == 7


def test_levels_are_cumulative():
    assert TelemetryPrivacy.NONE.hides_content is False
    assert TelemetryPrivacy.CONTENT.hides_content is True
    assert TelemetryPrivacy.CONTENT.hides_span_names is False
    assert TelemetryPrivacy.NAMES.hides_span_names is True
    assert TelemetryPrivacy.NAMES.hides_tool_names is False
    assert TelemetryPrivacy.FULL.hides_tool_names is True


def test_inputs_and_outputs_are_content():
    """What one agent asked another, and what came back, are content."""
    out = redact_attributes(
        {
            "input.value": "Create network lab-1",
            "output.value": "Draft ready",
            "gen_ai.agent.name": "Network Operations Agent",
        }
    )

    assert out["input.value"] == REDACTED
    assert out["output.value"] == REDACTED
    assert out["gen_ai.agent.name"] == "Network Operations Agent"


def test_error_messages_are_redacted_as_content():
    """An exception message can quote the very values `content` hides; its
    type and the ERROR status are identity and survive."""
    from opentelemetry.sdk.trace import Event
    from opentelemetry.trace import Status, StatusCode

    ctx = SpanContext(
        trace_id=0x1,
        span_id=0x2,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    span = ReadableSpan(
        name="tools",
        context=ctx,
        attributes={},
        kind=SpanKind.INTERNAL,
        start_time=1,
        end_time=2,
        status=Status(StatusCode.ERROR, "invalid band 'n78-secret'"),
        events=[
            Event(
                "exception",
                {
                    "exception.type": "ValueError",
                    "exception.message": "invalid band 'n78-secret'",
                    "exception.stacktrace": "Traceback ... n78-secret",
                },
                timestamp=1,
            )
        ],
    )
    delegate = _CapturingExporter()

    RedactingSpanExporter(
        delegate, privacy=TelemetryPrivacy.CONTENT
    ).export([span])

    out = delegate.spans[0]
    assert out.status.status_code is StatusCode.ERROR
    assert out.status.description == REDACTED
    event = out.events[0]
    assert event.attributes["exception.type"] == "ValueError"
    assert event.attributes["exception.message"] == REDACTED
    assert event.attributes["exception.stacktrace"] == REDACTED
