import asyncio

import pytest
from a2a.helpers import new_message, new_text_part
from a2a.types import (
    StreamResponse,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils.errors import InvalidParamsError

from bat.a2a_stream import (
    RemoteTaskContext,
    _parse_resume,
    _read_agent_response,
)


@pytest.mark.parametrize(
    "value", [False, 0, "", None, [], {}, {"approved": True}]
)
@pytest.mark.parametrize("position", [0, 1, 2])
def test_resume_envelope_preserves_value_and_application_parts(value, position):
    application = [{"network": "net-a"}, {"options": []}]
    parts = application.copy()
    parts.insert(position, {"resume": value})
    original = parts.copy()

    answer, data = _parse_resume("context", parts, waiting=True)

    assert answer == value
    assert type(answer) is type(value)
    assert data == application
    assert parts == original


@pytest.mark.parametrize(
    "text, parts, waiting",
    [
        ("yes", [{"approved": True}], True),
        ("", [None], True),
        ("", [{"resume": True}, {"resume": False}], True),
        ("", [{"resume": True, "network": "net-a"}], True),
        ("", [{"resume": None}], False),
        ("", [], True),
        ("   ", [], True),
    ],
)
def test_invalid_resume_requests_are_rejected(text, parts, waiting):
    with pytest.raises(InvalidParamsError):
        _parse_resume(text, parts, waiting=waiting)


def test_text_replies_preserve_original_text():
    assert _parse_resume(" yes \n", [], waiting=True) == (" yes \n", [])


def test_ordinary_application_data_is_not_interpreted_as_resume():
    data = [{"request": {"resume": "an application field"}}, False]

    assert _parse_resume("hello", data, waiting=False) == (None, data)


def test_message_only_remote_reply_uses_the_shared_reply_converter():
    async def stream():
        yield StreamResponse(message=new_message([new_text_part("Done")]))

    reply = asyncio.run(
        _read_agent_response(
            stream(), RemoteTaskContext(context_id="context"), lambda _: None
        )
    )

    assert reply.text == "Done"


def test_status_only_remote_question_preserves_task_identity():
    async def stream():
        yield StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="remote-task",
                context_id="remote-context",
                status=TaskStatus(
                    state=TaskState.TASK_STATE_INPUT_REQUIRED,
                    message=new_message([new_text_part("Continue?")]),
                ),
            )
        )

    reply = asyncio.run(
        _read_agent_response(
            stream(), RemoteTaskContext(context_id="context"), lambda _: None
        )
    )

    assert reply.text == "Continue?"
    assert reply.session.context_id == "remote-context"
    assert reply.session.task_id == "remote-task"
