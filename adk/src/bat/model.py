from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .config import ModelConfig
from .logging import create_logger

logger = create_logger(__name__, "debug")


class _ProviderKwargs(TypedDict, total=False):
    use_responses_api: bool
    reasoning_effort: str
    reasoning: str
    service_tier: str


def init_model(model: ModelConfig) -> BaseChatModel:
    """The LangChain chat model of config.yaml's `model`."""
    if not model.name or not model.provider:
        raise ValueError(
            "Set the model in config.yaml (`model.provider`, `model.name`) "
            "or with MODEL / MODEL_PROVIDER."
        )
    return init_chat_model(
        model=model.name,
        model_provider=model.provider,
        base_url=model.base_url,
        **_provider_kwargs(model),
    )


def _provider_kwargs(model: ModelConfig) -> _ProviderKwargs:
    kwargs: _ProviderKwargs = {}
    # OpenAI's own endpoint (no base_url) is used through the Responses API.
    if model.provider == "openai" and model.base_url is None:
        kwargs["use_responses_api"] = True
    if model.reasoning_effort is not None:
        if model.provider in ("anthropic", "openai"):
            kwargs["reasoning_effort"] = model.reasoning_effort
        elif model.provider == "ollama":
            kwargs["reasoning"] = model.reasoning_effort
        else:
            logger.warning(f"reasoning_effort is ignored for {model.provider}.")
    if model.service_tier is not None:
        if model.provider == "openai":
            kwargs["service_tier"] = model.service_tier
        else:
            logger.warning(f"service_tier is ignored for {model.provider}.")
    return kwargs
