import bisect
import time
from ..logging import create_logger
from .config import ChatModelClientConfig
from functools import reduce
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.chat_models.base import BaseChatModel
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, model_validator
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
)
from typing_extensions import Self

_logger = create_logger(__name__, "debug")

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

class ChatModelClient:
    """Client that facilitates interaction with a chat model.

    This client can be used to send user instructions to the chat model and receive responses.
    It supports both single and batch invocations, and can handle tool calls if tools are provided.

    If stored as a property of an object deriving the `AgentGraph` class, `UsageMetadata` will be
    automatically collected and returned as metadata of the streaming response.

    Args:
        chat_model_config (ChatModelClientConfig, optional):
            Configuration for the chat model client.
        system_instructions (str):
            System instructions to be used in the chat model.
        tools (Sequence[Dict[str, Any] | type | Callable | BaseTool | None], optional):
            LangChain-defined tools to be used by the chat model.

    Examples:
    ```python
    config = ChatModelClientConfig.from_env(
        client_name="SampleClient",
    )
    client = ChatModelClient(
        chat_model_config=config,
        system_instructions="You always reply in pirate language.",
    )
    response = client.invoke(HumanMessage("What is the weather like today?"))
    ```
    """

    def __init__(
        self,
        chat_model_config: ChatModelClientConfig | None = None,
        system_instructions: str = "You are a helpful assistant.",
        tools: Sequence[Dict[str, Any] | type | Callable | BaseTool | None] = None,
    ):
        """Initialize the ChatModelClient with the given configuration, system instructions, and tools.

        Args:
            chat_model_config (ChatModelClientConfig, optional):
                Configuration for the chat model client. If None, it will be loaded from environment variables.
            system_instructions (str):
                System instructions to be used by the chat model.
            tools (Sequence[Dict[str, Any] | type | Callable | BaseTool | None], optional):
                LangChain-defined tools to be used by the chat model.
        Raises:
            EnvironmentError: If the chat model configuration is not provided and cannot be loaded from environment variables.
        """
        self.config = ChatModelClientConfig.from_env() if chat_model_config is None else chat_model_config
        self.system_instructions = SystemMessage(system_instructions)
        self.tools = tools

        self._chat_model = init_chat_model(
            model=self.config.model,
            model_provider=self.config.model_provider,
            base_url=self.config.base_url,
        )
        _logger.info(f"model {self.config.model_provider}:{self.config.model} initialized{' with tools' if self.tools else ''}")
        if self.tools:
            self._chat_model = self._chat_model.bind_tools(self.tools)
        
        self.usage_metadatas = []
        

    def get_chat_model(self) -> BaseChatModel:
        """Get the chat model instance.
        
        Returns:
            BaseChatModel: The chat model instance configured with the provided model and tools.
        """
        return self._chat_model
    
    @classmethod
    def _validate_input_type(
        cls,
        input: HumanMessage | List[ToolMessage],
    ):
        """Validate the input for the invoke and stream method."""
        if isinstance(input, HumanMessage):
            return True
        if isinstance(input, list) and all(isinstance(msg, ToolMessage) for msg in input):
            return True
        return False
    
    def _build_messages_list(
        self,
        input: HumanMessage | List[ToolMessage],
        history: Optional[List[BaseMessage]] = None,
    ) -> List[BaseMessage]:
        """Build the messages list for the chat model.
        
        The system instructions are always included as the first message.
        If `history` is provided, it is prepended to the messages list.
        If the `input` is a `HumanMessage`, it is appended to the messages list.
        If the `input` is a list of `ToolMessage`, they are appended to the messages list.

        Returns:
            List[BaseMessage]: The list of messages to be sent to the chat model.
        """
        messages = [self.system_instructions]
        if history:
            messages += history
        if isinstance(input, HumanMessage):
            messages.append(input)
        else:
            messages += input
        return messages
    
    def _update_history(
        self,
        history: List[BaseMessage],
        input: HumanMessage | List[ToolMessage],
        response: AIMessage,
    ) -> None:
        """Update the history **in-place** with the input and response.
        
        If the input is a HumanMessage, it is appended directly.
        If the input is a list of ToolMessages, they are appended to the history.
        The response is always appended to the history.
        """
        if isinstance(input, HumanMessage):
            history.append(input)
        else:
            history += input
        history.append(response)

    def invoke(
        self,
        input: HumanMessage | List[ToolMessage],
        history: Optional[List[BaseMessage]] = None,
    ) -> AIMessage:
        """Invoke the chat model with user instructions or tool call results.

        If the `history` is provided, it will be prepended to the input message.
        This method modifies the `history` in-place to include the input and output messages.
        
        Parameters:
            input (HumanMessage | List[ToolMessage]): The user input or tool call results to process.
            history (Optional[List[BaseMessage]]): Optional history of messages.
        
        Returns:
            AIMessage: The response from the chat model.
        Raises:
            ValueError: If the input type is invalid or if the response from the chat model is not an `AIMessage`.
        """
        assert self._validate_input_type(input), f"Invalid input type: {type(input)}. Expected HumanMessage or List[ToolMessage]."

        # Build the messages for the chat model
        messages = self._build_messages_list(input, history)

        # Invoke the chat model and extract the response
        t_start = time.time()
        try:
            response = self._chat_model.invoke(messages)
        except Exception as e:
            raise RuntimeError(message="Error invoking chat model.") from e
        t_end = time.time()
        if not isinstance(response, AIMessage):
            raise ValueError(f"Expected AIMessage after invocation of chat model, got {type(response)}")

        if not response.usage_metadata:
            _logger.warning("Chat model did not return usage metadata.")
        usage_metadata = {**(response.usage_metadata or {}) | {'inference_time': t_end - t_start}}
        self.usage_metadatas.append((t_start, UsageMetadata.model_validate(usage_metadata)))

        # Update the history
        if history is not None:
            self._update_history(history, input, response)
        
        # Return the response
        return response

    def batch(
        self,
        inputs: List[HumanMessage],
        history: Optional[List[BaseMessage]] = None,
    ) -> List[AIMessage]:
        """Batch process multiple human messages in batch.

        If the `history` is provided, it will be prepended to each input message.
        This method does NOT modify the `history` in-place.
        
        Parameters:
            inputs (List[HumanMessage]): List of user inputs to process.
            history (Optional[List[BaseMessage]]): Optional history of messages.

        Returns:
            List[AIMessage]: List of responses from the chat model for each input.
        Raises:
            ValueError: If the input type is invalid or if the response from the chat model is not an AIMessage.
        """
        if not all([self._validate_input_type(input) for input in inputs]):
            raise ValueError(f"Invalid input type in batch: {[type(input) for input in inputs]}. Expected HumanMessage.")
        full_history = [self.system_instructions] + history if history else [self.system_instructions]
        messages = [
            full_history + [input]
            for input in inputs
        ]
        t_start = time.time()
        responses = self._chat_model.batch(messages)
        t_end = time.time()
        if not all(isinstance(response, AIMessage) for response in responses):
            raise ValueError("Expected all responses to be AIMessage instances after batch invocation of chat model.")

        usage_metadatas = [response.usage_metadata or {} for response in responses]
        if None in usage_metadatas:
            _logger.warning("Some responses from chat model did not return usage metadata.")
        aggregated_metadata = reduce(
            lambda acc, metadata: acc + metadata,
            usage_metadatas,
            UsageMetadata(inference_time=t_end - t_start),
        )
        self.usage_metadatas.append((t_start, aggregated_metadata))
        return responses
    
    def get_usage_metadata(
        self,
        from_timestamp: Optional[float] = None,
    ) -> UsageMetadata:
        """
        Get the aggregated usage metadata from the chat model client.

        Args:
            from_timestamp (Optional[float]): If provided, only usage metadata after this timestamp will be considered.
                If None, all usage metadata will be considered.
        
        Returns:
            UsageMetadata: The aggregated usage metadata.
        """
        # lower bound binary search to find the first usage metadata after the timestamp
        i = bisect.bisect_left(
            self.usage_metadatas,
            0 if from_timestamp is None else from_timestamp,
            key=lambda x: x[0]
        )
        # call reduce to aggregate usage metadata
        return reduce(
            lambda acc, metadata: acc + metadata[1],
            self.usage_metadatas[i:],
            UsageMetadata(),
        )
