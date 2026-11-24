from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, JsonValue
from typing import Annotated


def _latest_reducer(_: object, new: object) -> object:
    """The newest write wins: interrupt answers in one step each write."""
    return new


class BATAgentState(BaseModel):
    """Base graph state with message history and the latest input/output data.

    `add_messages` assigns missing IDs and merges messages by ID, including
    when nodes return the full state. Data fields keep the newest write,
    including `None`, so a node can take several interrupt answers in one
    step; parallel branches writing the same field in one step get no error,
    the last write wins. The executor resets them on each request.
    Extend this class with fields specific to your agent.
    """
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list
    )
    data_in: Annotated[JsonValue, _latest_reducer] = None
    data_out: Annotated[JsonValue | BaseModel, _latest_reducer] = None
