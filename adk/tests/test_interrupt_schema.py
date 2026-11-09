"""Response schemas across local interrupts and remote A2A tool calls."""

import asyncio
import json
from itertools import cycle

import httpx
import pytest
from a2a.helpers import new_message, new_text_part
from a2a.types import StreamResponse, Task
from conftest import agent_card, free_port, send_message, serving
from google.protobuf.json_format import MessageToDict
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt
from pydantic import BaseModel, TypeAdapter
from test_bat_executor import _events, _graph, _states

from bat.a2a_stream import (
    _COMPLETED,
    _FAILED,
    _INPUT_REQUIRED,
    RemoteTaskContext,
    _read_agent_response,
)
from bat.config import BATAgentConfig, EndpointConfig, RemoteAgentConfig
from bat.executor import BATAgentExecutor
from bat.remote_agents import RemoteAgentClient
from bat.server import BATAgentServer


class Approval(BaseModel):
    approved: bool


def _metadata(schema):
    return {"interrupt": {"response_schema": schema}}


@pytest.mark.parametrize(
    "schema, question, answer",
    [
        (None, "Continue?", "yes"),
        (str, "Describe the change", "Some details"),
        (Approval, {"question": "Approve?"}, {"approved": True}),
    ],
)
def test_schema_is_metadata_and_text_replies_still_work(
    schema, question, answer
):
    seen = []

    def confirm(state):
        reply = interrupt(question, response_schema=schema)
        seen.append(
            reply.model_dump() if isinstance(reply, BaseModel) else reply
        )
        return {"messages": [AIMessage("Done")]}

    graph = _graph(confirm, checkpointer=InMemorySaver())
    asked = _events(graph, "start")
    message = asked[0].status.message
    expected_part = (
        {"text": question} if isinstance(question, str) else {"data": question}
    )
    assert [MessageToDict(part) for part in message.parts] == [expected_part]
    expected_metadata = (
        _metadata(TypeAdapter(schema).json_schema()) if schema else {}
    )
    assert MessageToDict(message.metadata) == expected_metadata

    done = (
        _events(graph, answer)
        if isinstance(answer, str)
        else _events(graph, "", {"resume": answer})
    )
    assert _states(done) == ["TASK_STATE_COMPLETED"]
    assert seen == [answer]


def _read(metadata, state=_INPUT_REQUIRED):
    task = Task(id="remote-task", context_id="remote-context")
    task.status.state = state
    task.status.message.CopyFrom(new_message([new_text_part("Question")]))
    task.status.message.metadata.update(metadata)

    async def stream():
        yield StreamResponse(task=task)

    return asyncio.run(
        _read_agent_response(
            stream(),
            RemoteTaskContext(context_id="remote-context"),
            lambda _: None,
        )
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"interrupt": "unknown format"},
        {"interrupt": []},
        _metadata("unknown format"),
    ],
)
def test_missing_or_unrecognized_schema_metadata_is_optional(metadata):
    reply = _read(metadata)

    assert reply.response_schema is None
    assert reply.text == "Question"
    assert reply.data == []


@pytest.mark.parametrize("state", [_INPUT_REQUIRED, _COMPLETED, _FAILED])
def test_only_pending_replies_expose_a_response_schema(state):
    schema = Approval.model_json_schema()
    reply = _read(_metadata(schema), state)

    assert reply.response_schema == (
        schema if state == _INPUT_REQUIRED else None
    )


class _FakeModel(GenericFakeChatModel):
    def bind_tools(self, *args, **kwargs):
        return self


@pytest.mark.parametrize(
    "schema, remote_question, answer",
    [
        (str, "What is your answer?", "Some details"),
        (
            Approval,
            "What is your answer?",
            {"approved": True},
        ),
        (
            Approval,
            {"question": "Approve?", "network": "net-b"},
            {"approved": True},
        ),
    ],
)
@pytest.mark.parametrize("mode", ["send_request", "as_tool"])
def test_schema_survives_remote_call_and_supervisor_a2a_stream(
    schema, remote_question, answer, mode
):
    received = []
    confirmed = []

    def receive(state):
        received.append(state["messages"][-1].text)
        return {}

    def confirm(state):
        result = interrupt(remote_question, response_schema=schema)
        confirmed.append(
            result.model_dump() if isinstance(result, BaseModel) else result
        )
        return {"messages": [AIMessage("Confirmed")]}

    def server(graph, name, port):
        config = BATAgentConfig(
            endpoint=EndpointConfig(url="http://127.0.0.1", port=port)
        )
        return BATAgentServer(
            agent_card(name), config, BATAgentExecutor(graph, name)
        )

    async def main():
        remote_graph = _graph(receive, confirm, checkpointer=InMemorySaver())
        remote_port = free_port()
        remote_server = server(remote_graph, "Remote", remote_port)
        async with serving(remote_server.app, remote_port) as remote_url:
            remote = RemoteAgentClient(
                RemoteAgentConfig(name="Remote", url=remote_url)
            )
            if mode == "as_tool":
                tool = await remote.as_tool()
                call = {
                    "name": tool.name,
                    "args": {"message": "deploy"},
                    "id": "call",
                }
                model = _FakeModel(
                    messages=cycle(
                        [AIMessage("", tool_calls=[call]), AIMessage("Done")]
                    )
                )
                caller = create_agent(
                    model, [tool], checkpointer=InMemorySaver()
                )
            else:

                async def call_remote(state):
                    response = await remote.send_request("deploy")
                    return {"messages": [AIMessage(response.text)]}

                caller = _graph(call_remote, checkpointer=InMemorySaver())
            port = free_port()
            supervisor = server(caller, "Supervisor", port)
            async with (
                serving(supervisor.app, port) as url,
                httpx.AsyncClient(timeout=10) as http,
            ):
                headers = {"A2A-Version": "1.0"}
                first = await http.post(
                    url, json=send_message("start"), headers=headers
                )
                waiting = first.json()["result"]["task"]
                assert waiting["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
                question = waiting["status"]["message"]
                assert len(question["parts"]) == 1
                assert set(question["parts"][0]) == {"text"}
                text = question["parts"][0]["text"]
                if isinstance(remote_question, str):
                    assert text == remote_question
                else:
                    assert json.loads(text) == remote_question
                assert question["metadata"] == _metadata(
                    TypeAdapter(schema).json_schema()
                )

                body = send_message("reply")
                message = body["params"]["message"]
                message.update(
                    taskId=waiting["id"], contextId=waiting["contextId"]
                )
                message["parts"] = (
                    [{"text": answer}]
                    if isinstance(answer, str)
                    else [{"data": {"resume": answer}}]
                )
                response = await http.post(url, json=body, headers=headers)
                assert (
                    response.json()["result"]["task"]["status"]["state"]
                    == "TASK_STATE_COMPLETED"
                )

    asyncio.run(main())
    assert confirmed == [answer]
    assert received == ["deploy"]
