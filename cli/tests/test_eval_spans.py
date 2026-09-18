"""Tests for reconstructing per-episode usage/tool-calls from OTel spans.

These cover ``eval.engine.adapter`` reading a directory of JSON-Lines span
files and aggregating by ``trace_id`` — including the multi-agent case where a
remote sub-agent writes its own file but shares the trace via ``traceparent``.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval.engine.adapter import _aggregate_from_spans, _read_spans_dir


def _write_spans(path: Path, spans: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(span) for span in spans) + "\n",
        encoding="utf-8",
    )


def _llm_span(trace_id: str, prompt: int, completion: int, **extra) -> dict:
    span = {
        "trace_id": trace_id,
        "span_id": f"s{prompt}{completion}",
        "start_time": 1_000,
        "end_time": 1_000 + 5_000_000_000,  # +5s in nanoseconds
        "attributes": {
            "openinference.span.kind": "LLM",
            "llm.token_count.prompt": prompt,
            "llm.token_count.completion": completion,
        },
    }
    span["attributes"].update(extra)
    return span


def _root_span(trace_id: str, conversation_id: str) -> dict:
    return {
        "trace_id": trace_id,
        "span_id": "root",
        "start_time": 0,
        "end_time": 1,
        "attributes": {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.conversation.id": conversation_id,
        },
    }


def test_read_spans_dir_missing_directory(tmp_path: Path) -> None:
    assert _read_spans_dir(str(tmp_path / "does-not-exist")) == []


def test_read_spans_dir_merges_multiple_files_and_skips_bad_lines(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"trace_id": "t", "span_id": "1"}) + "\n"
        "not json\n"  # malformed line is skipped, not fatal
        "\n",  # blank line ignored
        encoding="utf-8",
    )
    (tmp_path / "b.jsonl").write_text(
        json.dumps({"trace_id": "t", "span_id": "2"}) + "\n",
        encoding="utf-8",
    )
    # A non-jsonl file is ignored.
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")

    spans = _read_spans_dir(str(tmp_path))
    assert {s["span_id"] for s in spans} == {"1", "2"}


def test_aggregate_single_agent(tmp_path: Path) -> None:
    trace = "trace-aaa"
    _write_spans(
        tmp_path / "agent.jsonl",
        [
            _root_span(trace, "conv-1"),
            _llm_span(trace, prompt=10, completion=5),
            _llm_span(trace, prompt=3, completion=2),
            {
                "trace_id": trace,
                "span_id": "tool-1",
                "attributes": {
                    "openinference.span.kind": "TOOL",
                    "tool.name": "search",
                    "input.value": json.dumps({"q": "hi"}),
                },
            },
        ],
    )

    usage, tool_calls, found = _aggregate_from_spans(str(tmp_path), "conv-1")

    assert found is True
    assert usage["input_tokens"] == 13
    assert usage["output_tokens"] == 7
    assert usage["total_tokens"] == 20
    assert usage["inference_time"] == 10.0  # two 5s LLM spans
    assert tool_calls == [
        {"name": "search", "args": {"q": "hi"}, "id": "tool-1"}
    ]


def test_aggregate_not_found_when_conversation_absent(tmp_path: Path) -> None:
    _write_spans(
        tmp_path / "agent.jsonl",
        [_llm_span("trace-x", prompt=10, completion=5)],
    )

    usage, tool_calls, found = _aggregate_from_spans(str(tmp_path), "conv-1")

    assert found is False
    assert usage["input_tokens"] == 0
    assert usage["total_tokens"] == 0
    assert tool_calls == []


def test_aggregate_multi_agent_recomposes_by_trace_id(tmp_path: Path) -> None:
    """A remote sub-agent's spans live in a separate file but share the
    trace_id (propagated via traceparent); their tokens must be included."""
    trace = "trace-shared"
    # Entry agent: root (carries conversation id) + its own LLM call.
    _write_spans(
        tmp_path / "agent.jsonl",
        [
            _root_span(trace, "conv-1"),
            _llm_span(trace, prompt=10, completion=5),
        ],
    )
    # Remote sub-agent: same trace_id, NO conversation id attribute, own file.
    _write_spans(
        tmp_path / "subagent.jsonl",
        [_llm_span(trace, prompt=100, completion=50)],
    )

    usage, _tool_calls, found = _aggregate_from_spans(str(tmp_path), "conv-1")

    assert found is True
    assert usage["input_tokens"] == 110
    assert usage["output_tokens"] == 55
    assert usage["total_tokens"] == 165


def test_aggregate_ignores_other_conversations(tmp_path: Path) -> None:
    """Spans from a different conversation's trace must not leak in."""
    _write_spans(
        tmp_path / "agent.jsonl",
        [
            _root_span("trace-1", "conv-1"),
            _llm_span("trace-1", prompt=10, completion=5),
            _root_span("trace-2", "conv-2"),
            _llm_span("trace-2", prompt=999, completion=999),
        ],
    )

    usage, _tool_calls, found = _aggregate_from_spans(str(tmp_path), "conv-1")

    assert found is True
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 5
