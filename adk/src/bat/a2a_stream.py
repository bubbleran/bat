import json
from typing import AsyncIterator, Callable, List, Optional

from a2a.helpers import (
    get_data_parts,
    get_text_parts,
    new_data_part,
    new_text_part,
)
from a2a.server.tasks.task_manager import append_artifact_to_task
from a2a.types import Part, StreamResponse, Task, TaskState
from a2a.utils.errors import InvalidParamsError
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel, JsonValue
from pydantic_core import to_jsonable_python

_COMPLETED = TaskState.TASK_STATE_COMPLETED
_FAILED = TaskState.TASK_STATE_FAILED
_INPUT_REQUIRED = TaskState.TASK_STATE_INPUT_REQUIRED
_RUNNING = (TaskState.TASK_STATE_SUBMITTED, TaskState.TASK_STATE_WORKING)


class RemoteTaskContext(BaseModel):
    """A conversation with a remote agent: its A2A context, and its task
    while the remote waits for input. Keep it in the graph's state to
    survive restarts."""

    context_id: str
    task_id: Optional[str] = None


class RemoteAgentResponse(BaseModel):
    """How a call ended: `status` is a2a's `TaskState` COMPLETED,
    INPUT_REQUIRED (`text` is the question) or FAILED. Pending replies may
    include a response schema describing the expected answer."""

    status: int
    text: str
    data: List[JsonValue] = []
    session: RemoteTaskContext
    response_schema: Optional[dict[str, JsonValue]] = None


async def _read_agent_response(
    stream: AsyncIterator[StreamResponse],
    session: RemoteTaskContext,
    on_progress: Callable[[List[Part]], None],
) -> RemoteAgentResponse:
    """The remote's first final state. Its events update its task as on the
    A2A server; the progress on the way goes to `on_progress`."""
    task = Task(id=session.task_id or "", context_id=session.context_id)
    async for response in stream:
        if response.HasField("message"):
            return _to_agent_response(
                _COMPLETED, list(response.message.parts), session
            )
        if response.HasField("task"):
            task.CopyFrom(response.task)
        if response.HasField("artifact_update"):
            task.id = response.artifact_update.task_id
            task.context_id = response.artifact_update.context_id
            append_artifact_to_task(task, response.artifact_update)
            continue
        if response.HasField("status_update"):
            task.id = response.status_update.task_id
            task.context_id = response.status_update.context_id
            task.status.CopyFrom(response.status_update.status)
        if task.status.state not in _RUNNING:
            return _final_agent_response(task, session)
        on_progress(list(task.status.message.parts))
    raise RuntimeError("The remote's stream ended without a final state.")


def _final_agent_response(
    task: Task, session: RemoteTaskContext
) -> RemoteAgentResponse:
    status = task.status
    parts = list(status.message.parts)
    session.context_id = task.context_id
    session.task_id = None
    if status.state == _INPUT_REQUIRED:
        session.task_id = task.id
        state = _INPUT_REQUIRED
    elif status.state == _COMPLETED:
        answer: List[Part] = []
        for artifact in task.artifacts:
            answer.extend(artifact.parts)
        state = _COMPLETED
        parts = answer or parts
    else:
        state = _FAILED
    reply = _to_agent_response(state, parts, session)
    if state == _INPUT_REQUIRED:
        metadata = MessageToDict(status.message.metadata)
        details = metadata.get("interrupt")
        schema = (
            details.get("response_schema")
            if isinstance(details, dict)
            else None
        )
        if isinstance(schema, dict):
            reply.response_schema = schema
    return reply


def _to_agent_response(
    state: int, parts: List[Part], session: RemoteTaskContext
) -> RemoteAgentResponse:
    return RemoteAgentResponse(
        status=state,
        text="\n".join(get_text_parts(parts)),
        data=get_data_parts(parts),
        session=session.model_copy(),
    )

def _to_part(value: object) -> Part:
    """A str as a text part, any other value as a data part."""
    if isinstance(value, str):
        part = new_text_part(value)
    else:
        part = new_data_part(to_jsonable_python(value))
    return part


def _serialize_text_and_data(text: str, data: List[JsonValue]) -> str:
    """Combine text and every data part, serialized as JSON, in order."""
    lines = [text] if text else []
    lines.extend(json.dumps(value) for value in data)
    return "\n".join(lines)


def _parse_resume(
    text: str, data: List[JsonValue], *, waiting: bool
) -> tuple[JsonValue, List[JsonValue]]:
    """Separate the resume control envelope from ordinary application data.

    A structured reply must have exactly one `{"resume": value}` part.
    Text-only replies remain supported. Validate before applying updates.
    """
    indexes = [
        index
        for index, value in enumerate(data)
        if isinstance(value, dict) and "resume" in value
    ]
    if len(indexes) > 1:
        raise InvalidParamsError("Send at most one resume data part.")
    if indexes:
        index = indexes[0]
        envelope = data[index]
        if set(envelope) != {"resume"}:
            raise InvalidParamsError(
                "A resume envelope must contain only the 'resume' key; "
                "send application data in separate parts."
            )
        if not waiting:
            raise InvalidParamsError("No interrupt is waiting for a reply.")
        return envelope["resume"], data[:index] + data[index + 1 :]
    if waiting:
        if data:
            raise InvalidParamsError(
                'Structured interrupt replies require {"resume": value}.'
            )
        if not text.strip():
            raise InvalidParamsError("An interrupt reply cannot be empty.")
    return text if waiting else None, data
