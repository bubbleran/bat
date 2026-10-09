"""The trajectory digest: what an agent did, read from its spans.

The fixture is a real two-turn conversation: a supervisor (a ReAct loop with
one tool, a call to a second agent, a summarizing model call) and the called
``netops`` agent (a ReAct loop whose first tool always fails), both on a
local model, both writing spans into the same directory. Only the prompts
and raw generation dumps were stripped from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.engine.adapter import read_spans
from eval.engine.trajectory import build_trajectory, render_trajectory

FIXTURE = Path(__file__).parent / "fixtures" / "spans" / "probe_two_turns"
CONVERSATION = "probe-conv-1"
TURNS = [
    "Create a 5G network named lab-1 with two cells on band n78.",
    "Actually make it three cells.",
]


@pytest.fixture(scope="module")
def spans() -> list[dict]:
    return read_spans([str(FIXTURE)])


@pytest.fixture(scope="module")
def trajectory(spans):
    return build_trajectory(spans, CONVERSATION, TURNS)


def _kinds(steps) -> list[str]:
    return [step.kind for step in steps]


def test_one_turn_per_request_under_its_user_message(trajectory) -> None:
    assert trajectory.found is True
    assert [turn.user for turn in trajectory.turns] == TURNS


def test_steps_follow_the_order_they_happened_in(trajectory) -> None:
    first = trajectory.turns[0].steps

    # The call to netops is recorded under the supervisor's root span, not
    # under its graph node: only the start times put it before summarize.
    assert _kinds(first) == ["model", "tool", "model", "agent", "model"]


def test_model_steps_are_labelled_with_the_agents_graph_node(
    trajectory,
) -> None:
    first = trajectory.turns[0].steps

    assert [s.node for s in first if s.kind == "model"] == [
        "plan",
        "plan",
        "summarize",
    ]


def test_a_model_step_keeps_what_the_model_decided_and_said(
    trajectory,
) -> None:
    decided, _, silent, _, summary = trajectory.turns[0].steps

    assert decided.model == "qwen3:1.7b"
    assert (decided.tokens_in, decided.tokens_out) == (195, 803)
    assert decided.called == ["list_networks"]
    assert decided.said is None
    # The planner's second call produced nothing -- which is why the
    # supervisor went on to send "None" to netops.
    assert silent.said is None and silent.called == []
    assert summary.said == (
        "The network operations agent reported a stream error due to the "
        "operator not responding within 30 seconds."
    )


def test_a_tool_step_keeps_its_result(trajectory) -> None:
    tool = trajectory.turns[0].steps[1]

    assert tool.name == "list_networks"
    assert tool.args == {}
    assert tool.result == '[{"name": "lab-0", "status": "running"}]'
    assert tool.error is None


def test_an_agent_step_keeps_the_exchange_and_its_failure(trajectory) -> None:
    call = trajectory.turns[1].steps[3]

    assert call.agent == "netops"
    assert call.asked == "lab-0, 3 cells, 5G"
    assert call.answered == "Stream error: operator did not answer in 30s"
    assert call.status == "error"
    assert call.error == "Stream error: operator did not answer in 30s"


def test_the_called_agents_steps_nest_under_the_call(trajectory) -> None:
    call = trajectory.turns[1].steps[3]

    assert _kinds(call.steps) == ["model", "tool"]
    decided, failed = call.steps
    assert decided.node == "draft_network"
    assert decided.called == [
        "check_operator",
        "validate_network_spec",
        "save_draft",
    ]
    assert failed.name == "check_operator"
    assert failed.error == "TimeoutError: operator did not answer in 30s"


def test_a_failure_is_reported_once(trajectory) -> None:
    """The failing tool's error also fails every enclosing node span; those
    must not show up as extra error steps."""
    call = trajectory.turns[0].steps[3]

    assert _kinds(call.steps) == ["model", "tool"]
    assert not any(
        step.kind == "error" for turn in trajectory.turns for step in turn.steps
    )


def test_totals_cover_every_agent_in_the_trace(trajectory) -> None:
    totals = trajectory.totals

    assert totals.model_calls == 8
    assert totals.tokens_in == 1629
    assert totals.tokens_out == 2997
    assert totals.tool_calls == 4
    assert totals.agent_calls == 2
    # Two failed tool calls, two failed agent calls.
    assert totals.errors == 4


TRANSFER = {
    "type": "tool",
    "data": {
        "content": "Transferred to netops",
        "additional_kwargs": {},
        "tool_call_id": "call-1",
    },
}


@pytest.mark.parametrize(
    "update",
    [
        {"messages": [TRANSFER], "network": {"name": "lab-0", "cells": None}},
        # Command(update=[(key, value), ...]) serializes as pairs.
        [["messages", [TRANSFER]], ["network", {"name": "lab-0"}]],
    ],
)
def test_a_tool_returning_a_command_shows_only_its_message(
    spans, update
) -> None:
    """A LangGraph Command records the whole state update; the tool's
    output is the message in it, wherever the update put it."""
    command = {"graph": None, "update": update, "resume": None, "goto": "x"}
    edited = [dict(span, attributes=dict(span["attributes"])) for span in spans]
    for span in edited:
        if span["attributes"].get("tool.name") == "list_networks":
            span["attributes"]["output.value"] = json.dumps(command)

    tool = build_trajectory(edited, CONVERSATION, TURNS).turns[0].steps[1]

    assert tool.result == "Transferred to netops"


def test_tool_arguments_come_from_the_models_call(spans) -> None:
    """Tool spans of a ReAct loop carry the result but not the arguments;
    the model's tool-call decision does, matched by tool-call id."""
    edited = [dict(span, attributes=dict(span["attributes"])) for span in spans]
    for span in edited:
        attributes = span["attributes"]
        key = "llm.output_messages.0.message.tool_calls.0.tool_call"
        if attributes.get(f"{key}.function.name") == "list_networks":
            attributes[f"{key}.function.arguments"] = '{"region": "eu"}'

    trajectory = build_trajectory(edited, CONVERSATION, TURNS)

    assert trajectory.turns[0].steps[1].args == {"region": "eu"}


def test_long_text_is_capped_and_marked(spans) -> None:
    trajectory = build_trajectory(
        spans, CONVERSATION, TURNS, max_field_chars=20
    )

    summary = trajectory.turns[0].steps[4]
    assert summary.said.startswith("The network operati")
    assert "[truncated: 106 chars]" in summary.said
    assert trajectory.truncated is True


def test_other_conversations_are_ignored(spans) -> None:
    assert build_trajectory(spans, "someone-else", TURNS).found is False


def test_no_spans_means_not_found(spans) -> None:
    trajectory = build_trajectory([], CONVERSATION, TURNS)

    assert trajectory.found is False
    assert [turn.user for turn in trajectory.turns] == TURNS
    assert all(turn.steps == [] for turn in trajectory.turns)


def test_a_call_recorded_by_an_older_adk_still_shows_up(spans) -> None:
    """bat-adk 2026.9.29a0 records only the called agent's name."""
    edited = []
    for span in spans:
        span = dict(span, attributes=dict(span["attributes"]))
        if span.get("kind") == "CLIENT":
            for key in ("input.value", "output.value", "bat.a2a.task_state"):
                span["attributes"].pop(key, None)
            span["status"] = "UNSET"
            span["status_description"] = None
        edited.append(span)

    call = build_trajectory(edited, CONVERSATION, TURNS).turns[0].steps[3]

    assert call.agent == "netops"
    assert call.asked is None and call.answered is None
    assert call.error is None
    assert _kinds(call.steps) == ["model", "tool"]


def test_the_digest_is_json_serialisable(trajectory) -> None:
    dumped = json.loads(trajectory.model_dump_json())

    assert dumped["turns"][0]["steps"][3]["kind"] == "agent"


def test_rendering_reads_as_numbered_steps(trajectory) -> None:
    text = render_trajectory(trajectory, max_chars=24000)

    assert text.startswith(f"Turn 1 - user: {TURNS[0]}")
    assert "[tool list_networks]" in text
    assert "4.2 [tool check_operator]" in text
    assert "FAILED: TimeoutError: operator did not answer in 30s" in text
    assert '[agent netops] asked: "lab-0, 3 cells, 5G"' in text


def test_rendering_leaves_out_models_and_tokens(trajectory) -> None:
    """No rubric scores cost; the counts stay in the digest for the checks."""
    text = render_trajectory(trajectory, max_chars=24000)

    assert "qwen3" not in text and " in / " not in text
    assert "[model plan] called: list_networks" in text
    assert text.endswith(
        "Totals: 8 model calls, 4 tool calls, 2 agent calls, 4 failed steps"
    )


def test_rendering_stays_within_its_budget(trajectory) -> None:
    text = render_trajectory(trajectory, max_chars=700)

    assert len(text) <= 700
    assert text.startswith("Turn 1 - user:")
    assert "omitted" in text
