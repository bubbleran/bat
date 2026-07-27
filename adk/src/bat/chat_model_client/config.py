import os
from typing import ClassVar, Dict, Optional

from pydantic import BaseModel, Field
from typing_extensions import Literal

from ..logging import create_logger

logger = create_logger(__name__, "debug")

ModelProvider = Literal[
    "anthropic",
    "deepseek",
    "nvidia",
    "ollama",
    "openai",
]
"""`ModelProvider` is a type alias for the supported model providers.
The currently supported providers are:
- `anthropic`
- `deepseek`
- `nvidia`
- `ollama`
- `openai`
"""


class ChatModelClientConfig(BaseModel):
    """Configuration for the chat model client.

    This class is used to configure the chat model client with the necessary
    parameters.
    Some model providers may require specific environment variables to be set,
    like OPENAI_API_KEY for OpenAI.

    Attributes
    -------
        model (str): The name of the model to use.
        model_provider (ModelProvider): The provider of the model
            (e.g., OpenAI, Meta, etc.).
        base_url (str, optional): The base URL for the model provider, required
            for non-OpenAI providers.
        client_name (str, optional): Name for the client.

    The class can be instantiated directly or created from environment variables
    using the `load` class method (usually preferred).

    Examples
    -------
    Direct instantiation:
    ```python
    config = ChatModelClientConfig(
        model="gpt-4o-mini",
        model_provider="openai",
        base_url="https://api.openai.com/v1",
        client_name="SampleClient",
    )
    ```

    From environment variables:
    ```python
    config = ChatModelClientConfig.load(
        client_name="SampleClient",
    )
    ```
    """

    class ConfigDict:
        arbitrary_types_allowed = True
    _config_defaults: ClassVar[Dict[str, Optional[str]]] = {}

    model: str
    model_provider: ModelProvider
    base_url: Optional[str] = Field(
        default=None,
        description=(
            "Base URL for the model provider. "
            "Required for non-OpenAI providers."
        ),
    )
    client_name: Optional[str] = Field(
        default=None,
        description="Name for the client.",
    )

    @classmethod
    def _set_defaults(
        cls,
        *,
        provider: Optional[str] = None,
        name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Install fallback model settings (typically from config.yaml).

        ``load`` uses these only when the corresponding environment variable
        is absent, implementing the precedence ``env > config.yaml`` for the
        model, provider and base URL.
        """
        cls._config_defaults = {
            "provider": provider,
            "name": name,
            "base_url": base_url,
        }

    @classmethod
    def _clear_defaults(cls) -> None:
        """Drop any installed fallback model settings."""
        cls._config_defaults = {}

    def __init__(
        self,
        model: str,
        model_provider: ModelProvider,
        base_url: Optional[str] = None,
        client_name: Optional[str] = None,
    ):
        """Initialize the ChatModelClientConfig with the provided parameters.

        Args:
            model (str): The name of the model to use.
            model_provider (ModelProvider): The provider of the model
                (e.g., openai, nvidia, etc.).
            base_url (Optional[str]): The base URL for the model provider,
                required for non-OpenAI providers.
            client_name (Optional[str]): Name for the client.
        """
        super().__init__(
            model=model,
            model_provider=model_provider,
            base_url=base_url,
            client_name=client_name,
        )

    @classmethod
    def load(
        cls,
        client_name: Optional[str] = None,
    ) -> "ChatModelClientConfig":
        """Create a `ChatModelClientConfig` from environment variables, falling
        back to the config.yaml.

        Resolution follows the precedence ``env > config.yaml``:
        - `MODEL`: model name (or `<provider>:<model>`); falls back to
            `model.name` from config.yaml.
        - `MODEL_PROVIDER`: provider (e.g. openai, ollama); falls back to the
            provider in `MODEL` or `model.provider` from config.yaml.
        - `BASE_URL`: optional base URL; falls back to `model.base_url`.

        Args:
            client_name (Optional[str]): Name for the client.

        Returns:
            An instance of `ChatModelClientConfig`.

        Raises:
            EnvironmentError: If neither the environment nor config.yaml provide
                the model name or the provider.
        """
        defaults = cls._config_defaults

        raw_model = os.getenv("MODEL")
        model_provider = os.getenv("MODEL_PROVIDER")
        base_url = os.getenv("BASE_URL")

        model_name: Optional[str]
        if raw_model:
            if not model_provider and ":" in raw_model:
                model_provider, model_name = raw_model.split(":", 1)
            else:
                model_name = raw_model
        else:
            model_name = defaults.get("name")

        if not model_provider:
            model_provider = defaults.get("provider")
        if base_url is None:
            base_url = defaults.get("base_url")

        if not model_name:
            raise EnvironmentError(
                "Model name not configured: set the MODEL environment variable "
                "or the 'model.name' field in config.yaml."
            )
        if not model_provider:
            raise EnvironmentError(
                "Model provider not configured: set MODEL_PROVIDER (or use the "
                "'<provider>:<model>' format in MODEL), or 'model.provider' in "
                "config.yaml."
            )

        return cls(
            model=model_name,
            model_provider=model_provider,
            base_url=base_url,
            client_name=client_name,
        )

    @classmethod
    def from_env(
        cls,
        client_name: Optional[str] = None,
    ) -> "ChatModelClientConfig":
        """Backward-compatible alias for :meth:`load`.

        ``from_env`` was the original name of this constructor; it is retained
        so existing callers keep working. New code should call :meth:`load`,
        which this delegates to verbatim (same environment/config.yaml
        precedence and the same raised errors).

        Args:
            client_name (Optional[str]): Name for the client.

        Returns:
            An instance of `ChatModelClientConfig`.
        """
        return cls.load(client_name=client_name)

    def build_default_headers(
        self,
    ) -> Dict[str, str]:
        if self.model_provider == "nvidia":
            api_key = os.getenv("API_KEY")
            if api_key is None:
                logger.warning("API_KEY environment variable not set")
                api_key = "<not-used>"
            result = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        else:
            result = {}
        return result
