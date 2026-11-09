from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, JsonValue
from typing import Annotated

class BATAgentState(BaseModel):
    """Base graph state with message history and the latest input/output data.

    `add_messages` assigns missing IDs and merges messages by ID, including
    when nodes return the full state. Data fields use LangGraph's default
    LastValue channel: values replace previous ones, including with `None`,
    with at most one update per graph step. The executor resets them on each
    request. Put successive interrupts in separate sequential nodes.
    Extend this class with fields specific to your agent.
    """
    messages: Annotated[list[AnyMessage], add_messages] = Field(
        default_factory=list
    )
    data_in: JsonValue = None
    data_out: JsonValue | BaseModel = None
