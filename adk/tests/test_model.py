import pytest

from bat import model as model_module
from bat.config import ModelConfig
from bat.model import init_model


@pytest.fixture
def calls(monkeypatch):
    seen = []

    def fake_init_chat_model(**kwargs):
        seen.append(kwargs)
        return "chat model"

    monkeypatch.setattr(model_module, "init_chat_model", fake_init_chat_model)
    return seen


def test_openai_without_base_url_uses_the_responses_api(calls):
    init_model(ModelConfig(provider="openai", name="gpt-4.1"))

    assert calls == [
        {
            "model": "gpt-4.1",
            "model_provider": "openai",
            "base_url": None,
            "use_responses_api": True,
        }
    ]


def test_openai_compatible_endpoint_does_not(calls):
    init_model(ModelConfig(provider="openai", name="m", base_url="http://x"))

    assert "use_responses_api" not in calls[0]


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("anthropic", {"reasoning_effort": "low"}),
        ("openai", {"reasoning_effort": "low"}),
        ("ollama", {"reasoning": "low"}),
        ("deepseek", {}),
    ],
)
def test_reasoning_effort_per_provider(calls, provider, expected):
    init_model(
        ModelConfig(
            provider=provider, name="m", base_url="http://x",
            reasoning_effort="low",
        )
    )

    for key in ("reasoning_effort", "reasoning"):
        assert calls[0].get(key) == expected.get(key)


def test_service_tier_only_for_openai(calls):
    init_model(ModelConfig(provider="openai", name="m", service_tier="flex"))
    init_model(ModelConfig(provider="ollama", name="m", service_tier="flex"))

    assert calls[0]["service_tier"] == "flex"
    assert "service_tier" not in calls[1]


def test_missing_name_is_an_error(calls):
    with pytest.raises(ValueError, match="MODEL"):
        init_model(ModelConfig(provider="openai"))
