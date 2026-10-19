import asyncio

from a2a.types import Message

from bat.agent import AgentTaskResult, AgentTaskStatus
from bat.prebuilt import CallAgentNode


async def _unreachable(agent_card, message):
    raise RuntimeError("http://user:pw@remote-agent:9900 refused")
    yield  # pragma: no cover - makes this an async generator


def test_remote_agent_error_is_reported_by_type_only():
    node = object.__new__(CallAgentNode)
    node._agent_name = "remote"
    node._agent_card = None
    node._streams = {}
    node.consume_agent_stream = _unreachable

    async def drive():
        await node._start_stream("call", Message())
        return await node._streams["call"]["queue"].get()

    assert asyncio.run(drive()) == AgentTaskResult(
        task_status=AgentTaskStatus.AGENT_TASK_STATUS_FAILED,
        content="Call to agent 'remote' failed (RuntimeError).",
    )
