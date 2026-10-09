import contextlib
import json
import os
import threading
from typing import Any, Dict, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult


def _span_to_dict(span: ReadableSpan) -> Dict[str, Any]:
    ctx = span.get_span_context()
    return {
        "name": span.name,
        "kind": span.kind.name,
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_span_id": (
            format(span.parent.span_id, "016x") if span.parent else None
        ),
        "start_time": span.start_time,  # unix nanoseconds
        "end_time": span.end_time,  # unix nanoseconds
        "attributes": dict(span.attributes),
        "status": span.status.status_code.name,
        "status_description": span.status.description,
        "events": [
            {
                "name": event.name,
                "time": event.timestamp,  # unix nanoseconds
                "attributes": dict(event.attributes or {}),
            }
            for event in span.events
        ],
    }


class JsonFileSpanExporter(SpanExporter):
    """Append finished spans to a file, one JSON object per line."""

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._file = open(path, "a", encoding="utf-8")

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with self._lock:
                for span in spans:
                    self._file.write(json.dumps(_span_to_dict(span)) + "\n")
                self._file.flush()
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        with self._lock:
            self._file.flush()
        return True

    def shutdown(self) -> None:
        with self._lock, contextlib.suppress(Exception):
            self._file.close()
