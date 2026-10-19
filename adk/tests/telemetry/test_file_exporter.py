"""Tests for the JSON-Lines span file exporter.

The exporter only duck-types ``ReadableSpan``, so the tests feed lightweight
fakes. They still require the OpenTelemetry SDK (the module imports it at
import time), hence the ``importorskip``.
"""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from bat.telemetry.file_exporter import (  # noqa: E402
    JsonFileSpanExporter,
    _span_to_dict,
)


def _fake_span(
    *,
    name="invoke_agent test",
    trace_id=0x1234,
    span_id=0xABCD,
    parent_span_id=None,
    status_name="OK",
    attributes=None,
    start_time=1_000,
    end_time=2_000,
):
    return SimpleNamespace(
        name=name,
        kind=SimpleNamespace(name="INTERNAL"),
        status=SimpleNamespace(
            status_code=SimpleNamespace(name=status_name), description=None
        ),
        events=(),
        start_time=start_time,
        end_time=end_time,
        attributes=attributes or {},
        parent=(
            SimpleNamespace(span_id=parent_span_id)
            if parent_span_id is not None
            else None
        ),
        get_span_context=lambda: SimpleNamespace(
            trace_id=trace_id, span_id=span_id
        ),
    )


def test_span_to_dict_hex_widths_and_fields():
    span = _fake_span(
        trace_id=0x1,
        span_id=0x2,
        parent_span_id=0x3,
        attributes={"gen_ai.operation.name": "invoke_agent"},
    )
    d = _span_to_dict(span)

    assert d["name"] == "invoke_agent test"
    assert d["kind"] == "INTERNAL"
    assert d["status"] == "OK"
    assert d["trace_id"] == "0" * 31 + "1"  # 32 hex chars
    assert d["span_id"] == "0" * 15 + "2"  # 16 hex chars
    assert d["parent_span_id"] == "0" * 15 + "3"
    assert d["start_time"] == 1_000
    assert d["end_time"] == 2_000
    assert d["attributes"] == {"gen_ai.operation.name": "invoke_agent"}


def test_span_to_dict_handles_missing_parent():
    d = _span_to_dict(_fake_span(parent_span_id=None))
    assert d["parent_span_id"] is None


def test_exporter_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "spans.jsonl"
    exporter = JsonFileSpanExporter(str(path))
    try:
        exporter.export([_fake_span(span_id=0x1), _fake_span(span_id=0x2)])
        exporter.export([_fake_span(span_id=0x3)])
    finally:
        exporter.shutdown()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    span_ids = [json.loads(line)["span_id"] for line in lines]
    assert span_ids == ["0" * 15 + "1", "0" * 15 + "2", "0" * 15 + "3"]


def test_exporter_returns_success():
    from opentelemetry.sdk.trace.export import SpanExportResult

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        exporter = JsonFileSpanExporter(f"{d}/spans.jsonl")
        try:
            result = exporter.export([_fake_span()])
            assert result is SpanExportResult.SUCCESS
        finally:
            exporter.shutdown()


def test_span_to_dict_keeps_the_error_message_and_events():
    """The status says a step failed; only its description and the exception
    event say why -- which is what a reader of the file needs."""
    span = _fake_span(status_name="ERROR")
    span.status.description = "TimeoutError: operator did not answer"
    span.events = [
        SimpleNamespace(
            name="exception",
            timestamp=1_500,
            attributes={
                "exception.type": "TimeoutError",
                "exception.message": "operator did not answer",
            },
        )
    ]

    d = _span_to_dict(span)

    assert d["status_description"] == "TimeoutError: operator did not answer"
    assert d["events"] == [
        {
            "name": "exception",
            "time": 1_500,
            "attributes": {
                "exception.type": "TimeoutError",
                "exception.message": "operator did not answer",
            },
        }
    ]


def test_span_to_dict_without_an_error_has_no_description_or_events():
    d = _span_to_dict(_fake_span())

    assert d["status_description"] is None
    assert d["events"] == []
