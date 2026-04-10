from bat.chat_model_client import ChatModelClient, ChatModelClientConfig
from langchain_core.messages import HumanMessage


class ExampleClient(ChatModelClient):

    SYSTEM_INSTRUCTIONS = (
        "INSERT INSTRUCTIONS HERE"
    )
    USER_INSTRUCTIONS = (
        "USER QUERY: {message}"
    )

    def __init__(
        self,
        tools,
    ):
        super().__init__(
            system_instructions=self.SYSTEM_INSTRUCTIONS,
            chat_model_config=ChatModelClientConfig.from_env(client_name="ExampleClient"),
            tools=tools,
        )

    def invoke(
        self,
        query: str,
    ) -> str:
        """Format the query based on what the client needs to do."""
        input_message = HumanMessage(
            content=self.USER_INSTRUCTIONS.format(
                message=query,
            )
        )
        response = super().invoke(input_message)
        response_content = response.content.strip()
        return response_content
