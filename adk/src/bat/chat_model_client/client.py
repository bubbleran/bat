import time
from functools import reduce
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from ..logging import create_logger
from .config import ChatModelClientConfig
from .metadata import MetadataCollector, UsageMetadata

logger = create_logger(__name__, "debug")

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
        output_schema: Optional[type[BaseModel]] = None,
    ):
        """Initialize the ChatModelClient with the given configuration, system instructions, and tools.

        Args:
            chat_model_config (ChatModelClientConfig, optional):
                Configuration for the chat model client. If None, it will be loaded from environment variables.
            system_instructions (str):
                System instructions to be used by the chat model.
            tools (Sequence[Dict[str, Any] | type | Callable | BaseTool | None], optional):
                LangChain-defined tools to be used by the chat model.
            output_schema (type[BaseModel], optional):
                If provided, the chat model response will be parsed according to this Pydantic schema. The raw response
                from the chat model will be included in the output as well.
        Raises:
            EnvironmentError: If the chat model configuration is not provided and cannot be loaded from environment variables.
        """
        if not isinstance(system_instructions, str):
            raise TypeError(f"Expected system_instructions to be of type 'str', got {type(system_instructions)}")

        self.config = ChatModelClientConfig.from_env() if chat_model_config is None else chat_model_config
        self.system_instructions = SystemMessage(system_instructions)
        self.tools = tools

        self._chat_model = init_chat_model(
            model=self.config.model,
            model_provider=self.config.model_provider,
            base_url=self.config.base_url,
            default_headers=self.config.build_default_headers(),
        )
        _full_model_name = self.config.model_provider + ":" + self.config.model
        logger.info(f"ChatModelClient {self.config.client_name or ''} initialized with: model={_full_model_name}, #tools={len(self.tools or [])}")
        if self.tools:
            self._chat_model = self._chat_model.bind_tools(self.tools)

        if output_schema:
            if not isinstance(output_schema, type) or not issubclass(output_schema, BaseModel):
                raise TypeError(f"Expected output_schema {output_schema} to be a subclass of pydantic BaseModel")
            self.output_schema = output_schema
            self._chat_model = self._chat_model.with_structured_output(
                schema=output_schema,
                include_raw=True,
            )
        else:
            self.output_schema = AIMessage

        self._metadata_collector = MetadataCollector()

    @property
    def chat_model(self) -> BaseChatModel:
        """The chat model instance configured with the provided model and tools."""
        return self._chat_model

    @classmethod
    def _validate_input_type(
        cls,
        input: str | HumanMessage | List[ToolMessage],
    ):
        """Validate the input for the invoke and stream method."""
        if isinstance(input, str):
            return True
        if isinstance(input, HumanMessage):
            return True
        if isinstance(input, list) and all(isinstance(msg, ToolMessage) for msg in input):
            return True
        return False

    def _build_messages_list(
        self,
        input: str | HumanMessage | List[ToolMessage],
        history: Optional[List[BaseMessage]] = None,
    ) -> List[BaseMessage]:
        """Build the messages list for the chat model.

        The system instructions are always included as the first message.
        If `history` is provided, it is prepended to the messages list.
        If the `input` is a `str`, it is converted to a `HumanMessage` and appended to the messages list.
        If the `input` is a `HumanMessage`, it is appended to the messages list.
        If the `input` is a list of `ToolMessage`, they are appended to the messages list.

        Returns:
            List[BaseMessage]: The list of messages to be sent to the chat model.
        """
        messages = [self.system_instructions]
        if history:
            messages += history
        if isinstance(input, str):
            messages.append(HumanMessage(input))
        elif isinstance(input, HumanMessage):
            messages.append(input)
        else:
            messages += input
        return messages

    def _update_history(
        self,
        history: List[BaseMessage],
        input: str | HumanMessage | List[ToolMessage],
        response: AIMessage,
    ) -> None:
        """Update the history **in-place** with the input and response.

        If the input is a HumanMessage, it is appended directly.
        If the input is a list of ToolMessages, they are appended to the history.
        The response is always appended to the history.
        """
        if isinstance(input, str):
            history.append(HumanMessage(input))
        elif isinstance(input, HumanMessage):
            history.append(input)
        else:
            history += input
        history.append(response)

    def _process_response(
        self,
        response: AIMessage | Dict[str, Any],
    ) -> Tuple[AIMessage, Any]:
        """
        Process the response from the chat model, checking for its type and parsing it according to the
        output schema if provided.

        Args:
            response (AIMessage | Dict[str, Any]): The response from the chat model.

        Returns:
            Tuple[AIMessage, Any]: A tuple containing the processed AIMessage and the parsed output.

        Raises:
            ValueError: If the response is not of the expected type or if there is an
                error parsing the response according to the output schema.
            KeyError: If the expected keys are not found in the response when the output schema is used.
            ValidationError: If the parsed response does not conform to the output schema.
        """
        if not isinstance(response, AIMessage) and not isinstance(response, Dict):
            raise ValueError(
                "Expected AIMessage or dict after invocation of chat model, "
                f"got {type(response)}, value={response}"
            )
        if isinstance(response, AIMessage):
            return response, response.model_copy()
        if 'raw' not in response:
            raise KeyError("Key 'raw' not found in response after chat model invocation")
        if 'parsed' not in response:
            raise KeyError("Key 'parsed' not found in response after chat model invocation")
        if 'parsing_error' not in response:
            raise KeyError("Key 'parsing_error' not found in response after chat model invocation")
        raw: AIMessage = response['raw']
        parsed = response['parsed']
        parsing_error = response['parsing_error']
        if parsing_error is not None:
            raise ValueError(
                f"Error parsing chat model response into the output schema {self.output_schema}: "
                f"{parsing_error}. Raw response: {raw}"
            )
        try:
            parsed_obj = self.output_schema.model_validate(parsed)
        except ValidationError as e:
            raise e
        return raw, parsed_obj

    def invoke(
        self,
        input: str | HumanMessage | List[ToolMessage],
        history: Optional[List[BaseMessage]] = None,
    ) -> Union[AIMessage, Any]:
        """Invoke the chat model with user instructions or tool call results.

        If the `history` is provided, it will be prepended to the input message.
        This method modifies the `history` in-place to include the input and output messages.

        Parameters:
            input (str | HumanMessage | List[ToolMessage]): The user input or tool call results to process.
            history (Optional[List[BaseMessage]]): Optional history of messages.

        Returns:
            AIMessage | Any: The response from the chat model.
        Raises:
            ValueError: If the input/output type is invalid or if there is an error parsing the
                response according to the output schema.
            KeyError: If the expected keys are not found in the response when the output schema is used.
            ValidationError: If the parsed response does not conform to the output schema.
        """
        assert self._validate_input_type(input), f"Invalid input type: {type(input)}. Expected str or HumanMessage or List[ToolMessage]."

        # Build the messages for the chat model
        messages = self._build_messages_list(input, history)

        # Invoke the chat model and extract the response
        t_start = time.time()
        try:
            response = self._chat_model.invoke(messages)
        except Exception as e:
            raise e
        t_end = time.time()
        r_for_history, r_to_return = self._process_response(response)

        if not r_for_history.usage_metadata:
            logger.warning("Chat model did not return usage metadata.")
        usage_metadata = {**(r_for_history.usage_metadata or {}) | {'inference_time': t_end - t_start}}
        self._metadata_collector.add_usage(usage_metadata, timestamp=t_start)

        # Update the history
        if history is not None:
            self._update_history(history, input, r_for_history)

        # Return the response
        return r_to_return

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
            logger.warning("Some responses from chat model did not return usage metadata.")
        aggregated_metadata = reduce(
            lambda acc, metadata: acc + metadata,
            usage_metadatas,
            UsageMetadata(inference_time=t_end - t_start),
        )
        self._metadata_collector.add_usage(aggregated_metadata, timestamp=t_start)
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
        return self._metadata_collector.get_usage_metadata(from_timestamp=from_timestamp)
