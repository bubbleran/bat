from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from .contracts import EvalConfig, JudgeSpec, ModelSpec

_ENV_VAR_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _to_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_path(base_dir: Path, raw_path: str | None, fallback: str) -> Path:
    path_value = raw_path or fallback
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _to_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_positive_int(value: Any, *, field_name: str, default: int) -> int:
    raw = default if value is None else value
    try:
        parsed = int(raw)
    except Exception as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{field_name} must be >= 1")
    return parsed


def _split_provider_model(value: str, *, field_name: str) -> tuple[str, str]:
    raw = value.strip()
    if not raw or ":" not in raw:
        raise ValueError(
            f"{field_name} must use '<provider>:<model>' format when provider is omitted"
        )

    provider, model = raw.split(":", 1)
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        raise ValueError(
            f"{field_name} must use '<provider>:<model>' format when provider is omitted"
        )
    return provider, model


_JUDGE_PROMPT_KEYS = ("relevance", "task_completion", "hallucination", "tool_call")
_JUDGE_PROMPT_MAX_LEN = 1000


def _parse_judge_prompts(raw: Any) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("judge.prompts must be a mapping")

    unknown = set(raw) - set(_JUDGE_PROMPT_KEYS)
    if unknown:
        raise ValueError(
            f"judge.prompts has unknown key(s) {sorted(unknown)}; "
            f"allowed: {list(_JUDGE_PROMPT_KEYS)}"
        )

    out: dict[str, str] = {}
    for key in _JUDGE_PROMPT_KEYS:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"judge.prompts.{key} must be a string")
        text = value.strip()
        if not text:
            continue
        if len(text) > _JUDGE_PROMPT_MAX_LEN:
            raise ValueError(
                f"judge.prompts.{key} exceeds the {_JUDGE_PROMPT_MAX_LEN}-character limit "
                f"(got {len(text)})"
            )
        out[key] = text
    return out


def _parse_env_map(raw: Any, *, section_name: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{section_name}.env must be a mapping of environment variables")

    parsed: dict[str, str] = {}
    for key, value in raw.items():
        env_key = str(key).strip()
        if not env_key or value is None:
            continue
        parsed[env_key] = str(value)
    return parsed


def _parse_model_spec(item: Any, *, section_name: str) -> ModelSpec:
    if isinstance(item, str):
        provider, model = _split_provider_model(item, field_name=section_name)
        return ModelSpec(provider=provider, model=model)

    if not isinstance(item, dict):
        raise ValueError(f"{section_name} must be either a mapping or '<provider>:<model>' string")

    provider = _to_optional_str(item.get("provider"))
    model = _to_optional_str(item.get("model"))
    base_url = _to_optional_str(item.get("base_url"))
    env = _parse_env_map(item.get("env"), section_name=section_name)

    if model and not provider and ":" in model:
        provider, model = _split_provider_model(model, field_name=f"{section_name}.model")

    if not provider or not model:
        raise ValueError(
            f"{section_name} must define at least one valid provider and model (or model as '<provider>:<model>')"
        )

    return ModelSpec(
        provider=provider,
        model=model,
        base_url=base_url,
        env=env,
    )


def _parse_judge_spec(item: Any) -> JudgeSpec | None:
    if item is None:
        return None

    if isinstance(item, str):
        provider, model = _split_provider_model(item, field_name="judge")
        return JudgeSpec(provider=provider, model=model)

    if not isinstance(item, dict):
        raise ValueError("judge must be either a mapping or '<provider>:<model>' string")

    provider = _to_optional_str(item.get("provider"))
    model = _to_optional_str(item.get("model"))
    base_url = _to_optional_str(item.get("base_url"))
    api_key_env = _to_optional_str(item.get("api_key_env"))
    env = _parse_env_map(item.get("env"), section_name="judge")
    prompts = _parse_judge_prompts(item.get("prompts"))

    if not any([provider, model, base_url, api_key_env, env, prompts]):
        return None

    if api_key_env and not _ENV_VAR_NAME.fullmatch(api_key_env):
        raise ValueError(f"judge.api_key_env is not a valid environment variable name: {api_key_env}")

    if model and not provider and ":" in model:
        provider, model = _split_provider_model(model, field_name="judge.model")

    if not provider or not model:
        raise ValueError(
            "judge must define provider and model (or model as '<provider>:<model>')"
        )

    return JudgeSpec(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        env=env,
        prompts=prompts,
    )


def load_eval_config(agent_root: Path, config_path: Path) -> EvalConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("eval.yaml must define a mapping at top level")

    evaluation_section = raw.get("evaluation") or {}
    if not isinstance(evaluation_section, dict):
        raise ValueError("evaluation section must be a mapping")

    models_raw = raw.get("models") or []
    if not isinstance(models_raw, list):
        raise ValueError("models section must be a list")

    models: list[ModelSpec] = []
    for idx, item in enumerate(models_raw):
        models.append(_parse_model_spec(item, section_name=f"models[{idx}]"))

    if not models:
        raise ValueError("No valid models configured in eval/eval.yaml")

    dataset = _resolve_path(agent_root, evaluation_section.get("dataset"), "eval/input/tasks.json")
    output_dir = _resolve_path(agent_root, evaluation_section.get("output_dir"), "eval/output")
    agent_url = _to_optional_str(evaluation_section.get("agent_url")) or "http://127.0.0.1:9900"

    agent_startup_timeout_s = _to_positive_int(
        evaluation_section.get("agent_startup_timeout_s"),
        field_name="evaluation.agent_startup_timeout_s",
        default=45,
    )
    agent_shutdown_timeout_s = _to_positive_int(
        evaluation_section.get("agent_shutdown_timeout_s"),
        field_name="evaluation.agent_shutdown_timeout_s",
        default=10,
    )

    k = int(evaluation_section.get("k", 1))
    if k < 1:
        raise ValueError("evaluation.k must be >= 1")

    qualitative = _to_bool(evaluation_section.get("qualitative"), default=False)
    run_name = _to_optional_str(evaluation_section.get("run_name")) or "benchmark"

    judge = _parse_judge_spec(raw.get("judge"))
    if qualitative and judge is None:
        raise ValueError(
            "When evaluation.qualitative is true, set judge.provider and judge.model in eval/eval.yaml"
        )

    return EvalConfig(
        dataset=dataset,
        output_dir=output_dir,
        agent_url=agent_url,
        agent_startup_timeout_s=agent_startup_timeout_s,
        agent_shutdown_timeout_s=agent_shutdown_timeout_s,
        k=k,
        qualitative=qualitative,
        run_name=run_name,
        models=models,
        judge=judge,
    )


def default_eval_yaml() -> str:
    return (
        "evaluation:\n"
        "  dataset: eval/input/tasks.json\n"
        "  output_dir: eval/output\n"
        "  agent_url: http://127.0.0.1:9900\n"
        "  agent_startup_timeout_s: 45\n"
        "  agent_shutdown_timeout_s: 10\n"
        "  k: 1\n"
        "  qualitative: false\n"
        "\n"
        "judge:\n"
        "  provider: ollama\n"
        "  model: local-judge-model\n"
        "  base_url: http://localhost:11434\n"
        "  # api_key_env: BAT_JUDGE_API_KEY   # name of the env var holding the judge's API key\n"
        "\n"
        "models:\n"
        "  - provider: openai\n"
        "    model: your-model-name\n"
        "  - provider: ollama\n"
        "    model: your-local-model\n"
        "    base_url: http://localhost:11434\n"
    )


def default_tasks_json() -> str:
    return (
        "[\n"
        "  {\n"
        "    \"id\": \"smoke_test\",\n"
        "    \"turns\": [\n"
        "      \"Describe what you can do in one short paragraph.\"\n"
        "    ],\n"
        "    \"expected\": {\n"
        "      \"status\": \"completed\",\n"
        "      \"expected_outcome\": \"The agent describes its capabilities clearly in one short paragraph.\"\n"
        "    },\n"
        "    \"meta\": {\n"
        "      \"category\": \"smoke\"\n"
        "    }\n"
        "  }\n"
        "]\n"
    )
