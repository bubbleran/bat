"""The LLM judge: what it is shown, and reading its answer back.

On-prem judges answer less tidily than hosted ones: reasoning models print
their thinking, small models wrap the JSON in prose. The verdict must survive
that, and a score off the 0-1 scale must be retried rather than recorded.
"""

from __future__ import annotations

import http.server
import json
import threading
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from eval.engine.adapter import read_spans
from eval.engine.contracts import (
    EpisodeResult,
    EpisodeTrace,
    JudgeSpec,
    TaskSpec,
    TraceEvent,
    Trajectory,
    TrajectoryTurn,
)
from eval.engine.eval_config import load_eval_config
from eval.engine.judge import (
    RUBRICS,
    Judge,
    JudgeVerdict,
    _parse_judge_json,
    build_judge_context,
    score,
)
from eval.engine.trajectory import build_trajectory, tool_calls_from

FIXTURE = Path(__file__).parent / "fixtures" / "spans" / "probe_two_turns"

# What qwen3:4b on Ollama actually returned with thinking turned off: its
# reasoning, a closing tag with no opening one, then the JSON.
QWEN3_ANSWER = (
    "Okay, let's see. The user wants me to reply with a specific JSON: "
    '{"score": 1, "reasoning": "ok"} and nothing else. Hmm, I need to make '
    "sure I follow exactly what they're asking for.\n</think>\n\n"
    '{"score": 1, "reasoning": "ok"}'
)


@pytest.mark.parametrize(
    "answer",
    [
        '{"reasoning": "ok", "score": 0.5}',
        '```json\n{"reasoning": "ok", "score": 0.5}\n```',
        '<think>scores could be 0.2 or 0.5</think>{"reasoning": "ok", '
        '"score": 0.5}',
        'Here is my evaluation: {"reasoning": "ok", "score": 0.5} '
        "Hope it helps.",
    ],
)
def test_the_verdict_is_found_in_untidy_answers(answer) -> None:
    assert _parse_judge_json(answer) == {"reasoning": "ok", "score": 0.5}


def test_the_last_verdict_wins_over_one_quoted_in_the_thinking() -> None:
    assert _parse_judge_json(QWEN3_ANSWER) == {"score": 1, "reasoning": "ok"}


def test_braces_inside_the_reasoning_do_not_confuse_it() -> None:
    answer = '{"reasoning": "the {draft} was saved, not applied", "score": 0}'

    assert _parse_judge_json(answer)["reasoning"] == (
        "the {draft} was saved, not applied"
    )


def test_an_answer_without_a_verdict_is_an_error() -> None:
    with pytest.raises(ValueError):
        _parse_judge_json("I think the agent did fine.")


class _StructuredJudge:
    """A schema-bound judge client: ``invoke`` returns the verdict or raises."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.prompts: list[str] = []

    def invoke(self, message):
        self.prompts.append(message)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _TextJudge:
    """A judge client without a schema: ``invoke`` returns a message."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def invoke(self, message):
        self.prompts.append(message)
        return AIMessage(content=self.answers.pop(0))


def _judge(structured, text=None) -> Judge:
    judge = Judge(JudgeSpec(provider="openai", model="judge"))
    judge.client = lambda rubric, schema: structured if schema else text
    return judge


def test_the_verdict_schema_bounds_the_score_and_reasons_first() -> None:
    """The schema is what the server constrains the answer to: the score
    must be bounded there, and reasoning comes before it so the model
    reasons before it scores."""
    schema = JudgeVerdict.model_json_schema()

    assert list(schema["properties"]) == ["reasoning", "score"]
    assert schema["properties"]["score"]["minimum"] == 0
    assert schema["properties"]["score"]["maximum"] == 1


def test_the_verdict_comes_back_structured() -> None:
    judge = _StructuredJudge([JudgeVerdict(reasoning="good", score=0.8)])

    result = _judge(judge).ask("task_completion", "prompt")

    assert result == {"reasoning": "good", "score": 0.8}


def test_an_answer_that_does_not_fit_the_schema_is_retried() -> None:
    judge = _StructuredJudge(
        [
            ValueError("score 8 is greater than the maximum of 1"),
            JudgeVerdict(reasoning="good", score=0.8),
        ]
    )

    result = _judge(judge).ask("task_completion", "prompt")

    assert result["score"] == 0.8
    assert len(judge.prompts) == 2


def test_without_structured_output_the_answer_is_read_as_text() -> None:
    """A server or model that cannot do structured output still gets a
    verdict read, and is not asked for structured output again."""
    structured = _StructuredJudge(
        [NotImplementedError("json_schema"), NotImplementedError("json_schema")]
    )
    text = _TextJudge([QWEN3_ANSWER, '{"reasoning": "fine", "score": 0.5}'])
    judge = _judge(structured, text)

    first = judge.ask("task_completion", "prompt")
    second = judge.ask("task_completion", "prompt")

    assert first == {"score": 1.0, "reasoning": "ok"}
    assert second["score"] == 0.5
    assert len(structured.prompts) == 2
    assert len(text.prompts) == 2


def test_a_judge_that_never_scores_properly_leaves_no_score() -> None:
    structured = _StructuredJudge([ValueError("bad"), ValueError("bad")])
    judge = _judge(structured, _TextJudge(['{"score": 9}']))

    result = judge.ask("task_completion", "prompt")

    assert result["score"] is None
    assert result["reasoning"].startswith("Error: ")


@pytest.fixture(scope="module")
def trajectory() -> Trajectory:
    return build_trajectory(
        read_spans([str(FIXTURE)]),
        "probe-conv-1",
        ["Create lab-1 with two cells on n78", "Make it three cells"],
    )


def test_the_judge_reads_the_trajectory_when_there_is_one(trajectory) -> None:
    context = build_judge_context(trajectory, events=[], max_chars=24000)

    assert "[tool check_operator]" in context
    assert "FAILED: TimeoutError" in context
    # The legend explains the line format to the judge.
    assert "[agent <name>]" in context


def test_the_judge_falls_back_to_the_a2a_transcript() -> None:
    events = [
        TraceEvent(
            t_ms=1.0,
            task_status="working",
            content_preview="",
            user_input="Create lab-1",
        ),
        TraceEvent(t_ms=9.0, task_status="completed", content_preview="Done."),
    ]
    no_spans = Trajectory(found=False, turns=[TrajectoryTurn(user="x")])

    context = build_judge_context(no_spans, events=events, max_chars=24000)

    assert "[1ms | USER] Create lab-1" in context


def test_the_judge_context_respects_its_budget(trajectory) -> None:
    context = build_judge_context(trajectory, events=[], max_chars=900)

    assert len(context) <= 900


@pytest.mark.parametrize("rubric", RUBRICS)
def test_every_judge_is_shown_what_the_agent_did(rubric) -> None:
    assert "{context}" in RUBRICS[rubric][0]


def test_the_tool_judge_gets_the_calls_in_order_not_their_json(
    trajectory, monkeypatch
) -> None:
    """The [tool] lines carry the arguments already; the judge needs only
    the complete list, which a rendering over budget may shorten."""
    prompts: dict[str, str] = {}

    def ask(self, rubric: str, prompt: str) -> dict:
        prompts[rubric] = prompt
        return {"reasoning": "", "score": 1.0}

    monkeypatch.setattr(Judge, "ask", ask)
    episode = EpisodeResult(
        task_id="t",
        final_status="completed",
        final_output="done",
        trace=EpisodeTrace(
            trajectory=trajectory, tool_calls=tool_calls_from(trajectory)
        ),
    )
    task = TaskSpec(
        id="t",
        turns=["x"],
        expected={"tool_calls": [{"name": "list_networks"}]},
    )

    score(JudgeSpec(provider="openai", model="m"), [episode], {"t": task})

    assert (
        "list_networks, check_operator (FAILED), list_networks, "
        "check_operator (FAILED)\n"
    ) in prompts["tool_call"]
    assert '"args"' not in prompts["tool_call"]


def test_the_outcome_judge_sees_only_the_request_and_the_result(
    trajectory, monkeypatch
) -> None:
    """One lenient call per episode: no steps, so nothing it could check
    a step-level expectation against."""
    prompts: dict[str, str] = {}

    def ask(self, rubric: str, prompt: str) -> dict:
        prompts[rubric] = prompt
        return {"reasoning": "done", "score": 0.9}

    monkeypatch.setattr(Judge, "ask", ask)
    episode = EpisodeResult(
        task_id="t",
        final_status="completed",
        final_output="lab-1 now has three cells.",
        trace=EpisodeTrace(
            trajectory=trajectory, tool_calls=tool_calls_from(trajectory)
        ),
    )
    task = TaskSpec(
        id="t",
        turns=["Create lab-1", "Make it three cells"],
        expected={
            "expected_outcome": "lab-1 has three cells",
            "tool_calls": [{"name": "list_networks"}],
        },
    )
    spec = JudgeSpec(provider="openai", model="m", mode="outcome")

    score(spec, [episode], {"t": task})

    assert list(prompts) == ["task_completion"]
    prompt = prompts["task_completion"]
    assert "Create lab-1 -> Make it three cells" in prompt
    assert "lab-1 has three cells" in prompt
    assert "lab-1 now has three cells." in prompt
    assert "[tool" not in prompt and "list_networks" not in prompt
    scores = episode.qualitative_scores
    assert scores.task_completion_quality == 0.9
    assert scores.response_relevance is None
    assert scores.hallucination_score is None
    assert scores.judge_reasoning == {"task_completion": "done"}


def _eval_yaml(tmp_path: Path, extra: str) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(
        "evaluation:\n  dataset: eval/input/tasks.json\n"
        f"{extra}"
        "models:\n  - provider: openai\n    model: gpt-4.1-mini\n",
        encoding="utf-8",
    )
    return path


def test_the_judges_trajectory_budget_is_configurable(tmp_path) -> None:
    path = _eval_yaml(
        tmp_path,
        "judge:\n  provider: ollama\n  model: qwen3:4b\n"
        "  max_trajectory_chars: 8000\n",
    )

    assert load_eval_config(tmp_path, path).judge.max_trajectory_chars == 8000


def test_the_judge_mode_is_configurable(tmp_path) -> None:
    path = _eval_yaml(
        tmp_path,
        "judge:\n  provider: ollama\n  model: qwen3:4b\n  mode: outcome\n",
    )

    assert load_eval_config(tmp_path, path).judge.mode == "outcome"


def test_an_unknown_judge_mode_is_rejected(tmp_path) -> None:
    path = _eval_yaml(
        tmp_path,
        "judge:\n  provider: ollama\n  model: qwen3:4b\n  mode: quick\n",
    )

    with pytest.raises(ValueError, match="mode"):
        load_eval_config(tmp_path, path)


def test_a_nonsense_trajectory_budget_is_rejected(tmp_path) -> None:
    path = _eval_yaml(
        tmp_path,
        "judge:\n  provider: ollama\n  model: qwen3:4b\n"
        "  max_trajectory_chars: 0\n",
    )

    with pytest.raises(ValueError, match="max_trajectory_chars"):
        load_eval_config(tmp_path, path)


def test_extra_span_files_resolve_against_the_agent(tmp_path) -> None:
    path = _eval_yaml(
        tmp_path,
        "  extra_spans:\n    - ../netops/spans.jsonl\n",
    )

    cfg = load_eval_config(tmp_path, path)

    assert cfg.extra_spans == [(tmp_path / "../netops/spans.jsonl").resolve()]


class _FakeModelServer:
    """A local HTTP server answering like Ollama (``/api/chat``) or an
    OpenAI-compatible server such as vLLM (``/v1/chat/completions``), and
    recording what it was sent. No model, no container: it checks what the
    real LangChain clients put on the wire."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.requests: list[tuple[str, dict]] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append((self.path, body))
                content = outer.replies.pop(0)
                if self.path.endswith("/api/chat"):
                    payload = {
                        "model": body.get("model"),
                        "created_at": "2026-09-30T00:00:00Z",
                        "message": {"role": "assistant", "content": content},
                        "done": True,
                        "done_reason": "stop",
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    }
                    data = (json.dumps(payload) + "\n").encode()
                    kind = "application/x-ndjson"
                else:
                    payload = {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body.get("model"),
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": content,
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                    data = json.dumps(payload).encode()
                    kind = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args) -> None:
                pass

        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), Handler
        )
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def model_server(monkeypatch):
    servers: list[_FakeModelServer] = []

    def start(provider: str, model: str, replies: list[str], path: str = ""):
        server = _FakeModelServer(replies)
        servers.append(server)
        monkeypatch.setenv("OPENAI_API_KEY", "not-needed-by-a-local-server")
        spec = JudgeSpec(
            provider=provider, model=model, base_url=server.url + path
        )
        return server, Judge(spec)

    yield start
    for server in servers:
        server.close()


def test_an_ollama_judge_is_constrained_to_the_verdict_schema(
    model_server,
) -> None:
    server, judge = model_server(
        "ollama", "qwen3:4b", ['{"reasoning": "grounded", "score": 0.9}']
    )

    result = judge.ask("hallucination", "prompt")

    assert result == {"reasoning": "grounded", "score": 0.9}
    path, body = server.requests[0]
    assert path == "/api/chat"
    # Ollama decodes against this schema, so the answer cannot be thinking
    # or prose around the JSON.
    assert body["format"]["properties"]["score"]["maximum"] == 1


def test_an_openai_compatible_judge_is_asked_for_the_verdict_schema(
    model_server,
) -> None:
    server, judge = model_server(
        "openai",
        "openai/gpt-oss-120b",
        ['{"reasoning": "complete", "score": 1.0}'],
        path="/v1",
    )

    result = judge.ask("task_completion", "prompt")

    assert result == {"reasoning": "complete", "score": 1.0}
    path, body = server.requests[0]
    assert path == "/v1/chat/completions"
    assert body["response_format"]["type"] == "json_schema"
    schema = body["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {"reasoning", "score"}


def test_a_server_that_ignores_the_schema_is_read_as_text(
    model_server,
) -> None:
    """An older server that drops the schema answers the way qwen3 did:
    thinking, then the JSON. Structured parsing fails; the text path reads
    it, and later calls go straight to text."""
    server, judge = model_server(
        "ollama", "qwen3:4b", [QWEN3_ANSWER, QWEN3_ANSWER, QWEN3_ANSWER]
    )

    first = judge.ask("relevance", "prompt")

    assert first == {"score": 1.0, "reasoning": "ok"}
    asked_for_schema = ["format" in body for _, body in server.requests]
    assert asked_for_schema == [True, True, False]
