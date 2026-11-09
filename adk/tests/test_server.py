import asyncio
import json

import httpx
import pytest
from conftest import CARD, agent_card, free_port, send_message, serving
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import interrupt

from bat.config import BATAgentConfig, EndpointConfig
from bat.executor import BATAgentExecutor
from bat.server import BATAgentServer, load_agent_card, ping_route
from bat.state import BATAgentState


def _card_file(tmp_path, card=CARD):
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(card))
    return str(path)


def _echo():
    def echo(state):
        return {"messages": [AIMessage("echo " + state["messages"][-1].text)]}

    builder = StateGraph(MessagesState)
    builder.add_node(echo)
    builder.add_edge(START, "echo")
    return builder.compile()


def _server(tmp_path, port):
    card = load_agent_card(_card_file(tmp_path))
    config = BATAgentConfig(
        endpoint=EndpointConfig(url="http://localhost", port=port)
    )
    server = BATAgentServer(card, config, BATAgentExecutor(_echo(), card.name))
    server.app.add_route("/ping", ping_route, methods=["GET"])
    return server


def test_card_interface_comes_from_the_endpoint(tmp_path):
    server = _server(tmp_path, 9300)

    (interface,) = server.agent_card.supported_interfaces
    assert interface.url == "http://localhost:9300"
    assert interface.protocol_binding == "JSONRPC"


def test_card_with_interfaces_is_rejected(tmp_path):
    card = {**CARD, "supportedInterfaces": [{"url": "http://x"}]}

    with pytest.raises(ValueError, match="supportedInterfaces"):
        load_agent_card(_card_file(tmp_path, card))


def test_agent_answers_over_http_in_a_running_loop(tmp_path):
    port = free_port()
    server = _server(tmp_path, port)

    async def main():
        async with (
            serving(server.app, port) as url,
            httpx.AsyncClient() as http,
        ):
            pong = await http.get(f"{url}/ping")
            card = await http.get(f"{url}/.well-known/agent-card.json")
            reply = await http.post(
                url, json=send_message("hi"), headers={"A2A-Version": "1.0"}
            )
        return pong.json(), card.json(), reply.json()["result"]["task"]

    pong, card, task = asyncio.run(main())

    assert pong == "pong"
    assert card["name"] == "Echo Agent"
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["artifacts"][0]["parts"][0]["text"] == "echo hi"


@pytest.mark.parametrize("method", ["SendMessage", "SendStreamingMessage"])
def test_invalid_resume_requests_leave_task_and_checkpoint_unchanged(method):
    seen = {}

    def prepare(state: BATAgentState) -> BATAgentState:
        state.data_out = {"stale": True}
        return state

    def confirm(state: BATAgentState):
        answer = interrupt("Approve?")
        seen.update(
            answer=answer,
            data_in=state.data_in,
            data_out=state.data_out,
            messages=[message.text for message in state.messages],
        )
        return {"messages": [AIMessage("Done")]}

    builder = StateGraph(BATAgentState)
    builder.add_node(prepare)
    builder.add_node(confirm)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "confirm")
    graph = builder.compile(checkpointer=InMemorySaver())
    port = free_port()
    config = BATAgentConfig(
        endpoint=EndpointConfig(url="http://127.0.0.1", port=port)
    )
    server = BATAgentServer(
        agent_card("Approval"), config, BATAgentExecutor(graph, "Approval")
    )

    async def main():
        async with (
            serving(server.app, port) as url,
            httpx.AsyncClient(timeout=10) as http,
        ):

            async def request(parts, task=None, rpc_method="SendMessage"):
                body = send_message("deploy")
                body["method"] = rpc_method
                message = body["params"]["message"]
                message["parts"] = parts
                if task:
                    message.update(
                        taskId=task["id"], contextId=task["contextId"]
                    )
                response = await http.post(
                    url, json=body, headers={"A2A-Version": "1.0"}
                )
                if "text/event-stream" in response.headers.get(
                    "content-type", ""
                ):
                    events = [
                        json.loads(line.removeprefix("data:"))
                        for line in response.text.splitlines()
                        if line.startswith("data:")
                    ]
                    return next(event for event in events if "error" in event)
                return response.json()

            unexpected = await request(
                [{"data": {"resume": True}}], rpc_method=method
            )
            assert unexpected["error"]["code"] == -32602
            assert seen == {}
            first = await request([{"text": "deploy"}])
            waiting = first["result"]["task"]
            assert waiting["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
            graph_config = {"configurable": {"thread_id": waiting["contextId"]}}
            before = await graph.aget_state(graph_config)

            for parts in (
                [{"text": "yes"}, {"data": {"unrelated": True}}],
                [{"data": {"resume": True}}, {"data": {"resume": False}}],
                [{"data": {"resume": True, "network": "net-a"}}],
                [{"text": "   "}],
            ):
                rejected = await request(parts, waiting, method)
                assert rejected["error"]["code"] == -32602
                current = await http.post(
                    url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "GetTask",
                        "params": {"id": waiting["id"]},
                    },
                    headers={"A2A-Version": "1.0"},
                )
                assert (
                    current.json()["result"]["status"]["state"]
                    == "TASK_STATE_INPUT_REQUIRED"
                )
                after = await graph.aget_state(graph_config)
                assert after.config == before.config
                assert after.values == before.values
                assert after.interrupts == before.interrupts

            done = await request(
                [
                    {"text": "context"},
                    {"data": {"network": "net-a"}},
                    {"data": {"resume": False}},
                ],
                waiting,
            )
            assert (
                done["result"]["task"]["status"]["state"]
                == "TASK_STATE_COMPLETED"
            )

    asyncio.run(main())
    assert seen == {
        "answer": False,
        "data_in": {"network": "net-a"},
        "data_out": None,
        "messages": ["deploy", "context"],
    }
