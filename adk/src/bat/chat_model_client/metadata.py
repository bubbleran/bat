import bisect
import time

from langchain_core.messages import ToolCall
from pydantic import BaseModel, model_validator
from functools import reduce
from typing import Any, Dict, List, Optional, Self

USAGE_METADATA_KEY = "usage"
TRACE_METADATA_KEY = "trace"
TOOL_CALLS_METADATA_KEY = "tool_calls"

class UsageMetadata(BaseModel):
    """Metadata about the usage of the chat model.

    Note: Defining a ChatModelClient as a property of an object deriving the `AgentGraph` class
    allows to automatically collect and aggregate usage metadata from the chat model
    and return it as part of the streaming response metadata.

    Attributes
    -------
        input_tokens (int): Number of input tokens used in the request.
        output_tokens (int): Number of output tokens generated in the response.
        total_tokens (int): Total number of tokens used (input + output).
        inference_time (float): Time taken for the inference in seconds.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    inference_time: float = 0.0

    def __add__(
        self,
        other: Self | Dict[str, int]
    ) -> Self:
        """Add two UsageMetadata instances."""
        if isinstance(other, Dict):
            return UsageMetadata(
                input_tokens=self.input_tokens + other.get("input_tokens", 0),
                output_tokens=self.output_tokens + other.get("output_tokens", 0),
                total_tokens=self.total_tokens + other.get("total_tokens", 0),
                inference_time=self.inference_time + other.get("inference_time", 0.0),
            )
        return UsageMetadata(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            inference_time=self.inference_time + other.inference_time,
        )
    def __sub__(
        self,
        other: Self | Dict
    ) -> Self:
        """Subtract two UsageMetadata instances."""
        if isinstance(other, Dict):
            return UsageMetadata(
                input_tokens=self.input_tokens - other.get("input_tokens", 0),
                output_tokens=self.output_tokens - other.get("output_tokens", 0),
                total_tokens=self.total_tokens - other.get("total_tokens", 0),
                inference_time=self.inference_time - other.get("inference_time", 0.0),
            )
        return UsageMetadata(
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            total_tokens=self.total_tokens - other.total_tokens,
            inference_time=self.inference_time - other.inference_time,
        )
    @model_validator(mode="after")
    def check_non_negative(
        self
    ) -> Self:
        if self.input_tokens < 0:
            raise ValueError("input_tokens count should be non-negative.")
        if self.output_tokens < 0:
            raise ValueError("output_tokens count should be non-negative.")
        if self.total_tokens < 0:
            raise ValueError("total_tokens count should be non-negative.")
        if self.inference_time < 0:
            raise ValueError("inference_time should be non-negative.")
        return self

class TraceMetadata(BaseModel):
    """Aggregated trace metadata emitted alongside agent responses.

    Attributes
    -------
        tool_calls (List[ToolCall]): Tool calls observed during execution, in order.
    """
    tool_calls: List[ToolCall] = []


class MetadataCollector:
    """Collect and aggregate timestamped usage and trace metadata.

    This helper centralizes metadata buffering logic used by prebuilt components,
    so all nodes expose consistent `get_usage_metadata` / `get_trace_metadata`
    behavior.
    """

    def __init__(self) -> None:
        self._usage_metadatas: List[tuple[float, UsageMetadata]] = []
        self._trace_metadatas: List[tuple[float, List[ToolCall]]] = []

    def add_usage(
        self,
        usage: UsageMetadata | Dict[str, Any],
        *,
        timestamp: Optional[float] = None,
    ) -> None:
        t = time.time() if timestamp is None else timestamp
        usage_metadata = usage if isinstance(usage, UsageMetadata) else UsageMetadata.model_validate(usage)
        self._usage_metadatas.append((t, usage_metadata))

    def add_tool_calls(
        self,
        tool_calls: List[ToolCall],
        *,
        timestamp: Optional[float] = None,
    ) -> None:
        if not tool_calls:
            return
        t = time.time() if timestamp is None else timestamp
        self._trace_metadatas.append((t, list(tool_calls)))

    def observe_metadata(
        self,
        metadata: Any,
        *,
        timestamp: Optional[float] = None,
    ) -> None:
        if not metadata or not hasattr(metadata, "get"):
            return

        usage = metadata.get(USAGE_METADATA_KEY)
        if usage is not None:
            self.add_usage(usage, timestamp=timestamp)

        trace = metadata.get(TRACE_METADATA_KEY, {})
        if isinstance(trace, dict):
            tool_calls = trace.get(TOOL_CALLS_METADATA_KEY, [])
            if isinstance(tool_calls, list):
                self.add_tool_calls(tool_calls, timestamp=timestamp)

    def get_usage_metadata(
        self,
        from_timestamp: Optional[float] = None,
    ) -> UsageMetadata:
        i = bisect.bisect_left(
            self._usage_metadatas,
            0 if from_timestamp is None else from_timestamp,
            key=lambda x: x[0],
        )
        return reduce(
            lambda acc, metadata: acc + metadata[1],
            self._usage_metadatas[i:],
            UsageMetadata(),
        )

    def get_trace_metadata(
        self,
        from_timestamp: Optional[float] = None,
    ) -> TraceMetadata:
        i = bisect.bisect_left(
            self._trace_metadatas,
            0 if from_timestamp is None else from_timestamp,
            key=lambda x: x[0],
        )

        tool_calls: List[ToolCall] = []
        for _, calls in self._trace_metadatas[i:]:
            tool_calls.extend(calls)

        return TraceMetadata(tool_calls=tool_calls)
