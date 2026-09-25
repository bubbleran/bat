import asyncio
from typing import Optional
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage
from typing_extensions import Self, override

from bat.agent import AgentState, AgentTaskResult, AgentTaskStatus
from bat.prebuilt import ReActLoop


class _State(AgentState):
    input: str = ""
    output: Optional[str] = None

    @classmethod
    @override
    def from_query(cls, query: str) -> Self:
        return cls(input=query)

    @override
    def to_task_result(self) -> AgentTaskResult:
        return AgentTaskResult(
            task_status=AgentTaskStatus.AGENT_TASK_STATUS_COMPLETED,
            content=self.output or "",
        )


def _loop(response: AIMessage) -> ReActLoop:
    """A ReActLoop with only the attributes `_llm` reads, so the node can be
    exercised without compiling a graph."""
    loop = object.__new__(ReActLoop)
    loop.loop_name = "loop"
    loop.input_key = "input"
    loop.output_key = "output"
    loop.status_key = None
    loop._internal_messages_key = "loop.messages"
    loop._internal_final_response_key = "loop.final_response"
    loop._internal_trace_key = "loop.trace.tool_calls"
    loop.chat_model_client = MagicMock()
    loop.chat_model_client.invoke = MagicMock(return_value=response)
    return loop


def _run_llm(loop: ReActLoop, state: _State) -> _State:
    async def drive():
        last = None
        async for item in loop._llm(state):
            last = item
        return last

    return asyncio.run(drive())


def _state() -> _State:
    state = _State(input="question")
    state.bat_extra["loop.messages"] = []
    state.bat_extra["loop.trace.tool_calls"] = []
    return state


def test_final_response_from_block_list_content_is_a_string():
    response = AIMessage(
        content=[
            {"type": "text", "text": "the "},
            {"type": "text", "text": "answer"},
        ]
    )
    out = _run_llm(_loop(response), _state())
    assert out.bat_extra["loop.final_response"] == "the answer"


def test_final_response_from_plain_content_is_unchanged():
    out = _run_llm(_loop(AIMessage(content="the answer")), _state())
    assert out.bat_extra["loop.final_response"] == "the answer"


def test_reasoning_blocks_are_kept_out_of_the_final_response():
    response = AIMessage(
        content=[
            {"type": "reasoning", "summary": [], "id": "rs_1"},
            {"type": "text", "text": "the answer"},
        ]
    )
    out = _run_llm(_loop(response), _state())
    assert out.bat_extra["loop.final_response"] == "the answer"


def test_tool_calls_bypass_the_final_response():
    response = AIMessage(
        content=[],
        tool_calls=[
            {"name": "get_weather", "args": {}, "id": "call_1"},
        ],
    )
    state = _state()
    out = _run_llm(_loop(response), state)
    assert "loop.final_response" not in out.bat_extra
    assert out.bat_buffer == [response]
