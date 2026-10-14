"""Reading span files, and the tokens of a conversation spread over them."""

from __future__ import annotations

import json
from pathlib import Path

from eval.engine.adapter import read_spans
from eval.engine.trajectory import build_trajectory


def _write_spans(path: Path, spans: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(span) for span in spans) + "\n",
        encoding="utf-8",
    )


def _llm_span(trace_id: str, prompt: int, completion: int, cached: int = 0):
    return {
        "trace_id": trace_id,
        "span_id": f"s{prompt}{completion}",
        "start_time": 1_000,
        "end_time": 2_000,
        "attributes": {
            "openinference.span.kind": "LLM",
            "llm.token_count.prompt": prompt,
            "llm.token_count.completion": completion,
            "llm.token_count.prompt_details.cache_read": cached,
        },
    }


def _root_span(trace_id: str, conversation_id: str) -> dict:
    return {
        "trace_id": trace_id,
        "span_id": "root",
        "start_time": 0,
        "end_time": 3_000,
        "attributes": {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.conversation.id": conversation_id,
        },
    }


def test_read_spans_of_a_missing_directory(tmp_path: Path) -> None:
    assert read_spans([str(tmp_path / "does-not-exist")]) == []


def test_read_spans_merges_files_and_skips_bad_lines(tmp_path: Path) -> None:
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"trace_id": "t", "span_id": "1"}) + "\nnot json\n\n",
        encoding="utf-8",
    )
    (tmp_path / "b.jsonl").write_text(
        json.dumps({"trace_id": "t", "span_id": "2"}) + "\n", encoding="utf-8"
    )
    (tmp_path / "ignore.txt").write_text("nope", encoding="utf-8")

    spans = read_spans([str(tmp_path)])

    assert {span["span_id"] for span in spans} == {"1", "2"}


def test_tokens_count_every_agent_of_the_conversation(tmp_path: Path) -> None:
    """A called agent writes its own file, but shares the caller's trace;
    another conversation's trace stays out."""
    _write_spans(
        tmp_path / "agent.jsonl",
        [
            _root_span("trace-1", "conv-1"),
            _llm_span("trace-1", prompt=10, completion=5, cached=4),
            _root_span("trace-2", "conv-2"),
            _llm_span("trace-2", prompt=999, completion=999),
        ],
    )
    _write_spans(
        tmp_path / "called.jsonl",
        [_llm_span("trace-1", prompt=100, completion=50, cached=60)],
    )

    totals = build_trajectory(
        read_spans([str(tmp_path)]), "conv-1", ["hi"]
    ).totals

    assert (totals.tokens_in, totals.tokens_out, totals.tokens_cached) == (
        110,
        55,
        64,
    )
