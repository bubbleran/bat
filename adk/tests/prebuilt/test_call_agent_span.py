"""What CallAgentNode records about a call to another agent.

The called agent's own spans hold what it did; only the caller's CLIENT span
can hold what was asked and what came back, so an evaluation (or anyone
reading the trace) can follow the exchange without the other agent's spans.
"""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from a2a.types import (  # noqa: E402
    AgentCard,
    Artifact,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402

from bat.prebuilt import call_agent_node  # noqa: E402
from bat.prebuilt.call_agent_node import CallAgentNode  # noqa: E402


@pytest.fixture
def exported(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(call_agent_node, "tracer", provider.get_tracer("t"))
    return exporter


def _status(state, text=None):
    status = TaskStatus(state=state)
    if text is not None:
        status.message.CopyFrom(
            Message(
                message_id="s", role=Role.ROLE_AGENT, parts=[Part(text=text)]
            )
        )
    return StreamResponse(
        status_update=TaskStatusUpdateEvent(
            task_id="t1", context_id="c1", status=status
        )
    )


def _artifact(text):
    return StreamResponse(
        artifact_update=TaskArtifactUpdateEvent(
            task_id="t1",
            context_id="c1",
            artifact=Artifact(artifact_id="a1", parts=[Part(text=text)]),
        )
    )


_SUBMITTED = StreamResponse(
    task=Task(
        id="t1",
        context_id="c1",
        status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
    )
)


def _call(monkeypatch, chunks, *, error=None, stop_after=None):
    """Consume one call the way CallAgentNode's worker does."""

    class _Client:
        def send_message(self, request):
            async def stream():
                for chunk in chunks:
                    yield chunk
                if error is not None:
                    raise error

            return stream()

    async def create_client(agent, client_config):
        return _Client()

    monkeypatch.setattr(call_agent_node, "create_client", create_client)
    card = AgentCard(name="Network Operations Agent")
    message = Message(
        message_id="m1",
        role=Role.ROLE_USER,
        parts=[Part(text="Create network lab-1: 2 cells, band n78")],
    )

    async def run():
        stream = CallAgentNode.consume_agent_stream(
            SimpleNamespace(), agent_card=card, message=message
        )
        try:
            seen = 0
            async for _ in stream:
                seen += 1
                if stop_after is not None and seen == stop_after:
                    break
        finally:
            await stream.aclose()

    asyncio.run(run())


def _only_span(exported):
    spans = exported.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


def test_call_span_records_what_was_asked_and_answered(
    monkeypatch, exported
) -> None:
    _call(
        monkeypatch,
        [
            _SUBMITTED,
            _status(TaskState.TASK_STATE_WORKING, "Generating response..."),
            _artifact("Draft ready: lab-1 (2 cells, n78)."),
            _status(TaskState.TASK_STATE_COMPLETED),
        ],
    )

    span = _only_span(exported)
    assert span.attributes["input.value"] == (
        "Create network lab-1: 2 cells, band n78"
    )
    # The closing status carries no text: it must not erase the answer.
    assert span.attributes["output.value"] == (
        "Draft ready: lab-1 (2 cells, n78)."
    )
    assert span.attributes["bat.a2a.task_state"] == "TASK_STATE_COMPLETED"
    assert span.status.status_code is not StatusCode.ERROR


def test_call_span_records_a_question_back_as_input_required(
    monkeypatch, exported
) -> None:
    _call(
        monkeypatch,
        [_status(TaskState.TASK_STATE_INPUT_REQUIRED, "Confirm to apply?")],
    )

    span = _only_span(exported)
    assert span.attributes["output.value"] == "Confirm to apply?"
    assert span.attributes["bat.a2a.task_state"] == (
        "TASK_STATE_INPUT_REQUIRED"
    )


def test_a_failed_remote_task_marks_the_call_as_failed(
    monkeypatch, exported
) -> None:
    _call(
        monkeypatch,
        [_status(TaskState.TASK_STATE_FAILED, "operator unreachable")],
    )

    span = _only_span(exported)
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "operator unreachable"


def test_a_broken_stream_marks_the_call_as_failed(
    monkeypatch, exported
) -> None:
    with pytest.raises(ConnectionError):
        _call(
            monkeypatch,
            [_status(TaskState.TASK_STATE_WORKING, "Generating response...")],
            error=ConnectionError("connection reset by peer"),
        )

    span = _only_span(exported)
    assert span.status.status_code is StatusCode.ERROR
    assert "connection reset by peer" in span.status.description


def test_the_answer_is_kept_when_the_caller_stops_at_it(
    monkeypatch, exported
) -> None:
    """CallAgentNode's worker stops reading at the first terminal chunk."""
    _call(
        monkeypatch,
        [
            _artifact("Draft ready."),
            _status(TaskState.TASK_STATE_COMPLETED),
        ],
        stop_after=1,
    )

    span = _only_span(exported)
    assert span.attributes["output.value"] == "Draft ready."
