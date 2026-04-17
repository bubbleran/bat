from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import JudgeSpec, ModelSpec, EvalConfig
import yaml



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
    env=_parse_env_map(item.get("env"), section_name=section_name)
    
    if model and not provider and ":" in model:
        provider, model = _split_provider_model(model, field_name=f"{section_name}.model")

    if not provider or not model:
        raise ValueError(
            f"{section_name} must define at least one valid provider and model (or model as '<provider>:<model>')"
        )
    
    if section_name == "judge":
        is_empty = not any([provider, model, base_url, env])
        if is_empty:
            return None
        return JudgeSpec(
                provider=provider, 
                model=model, 
                base_url=base_url,
                env=env
                )
    else:
        return ModelSpec(
            provider=provider,
            model=model,
            base_url=base_url,
            env=env
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

    k = int(evaluation_section.get("k", 1))
    if k < 1:
        raise ValueError("evaluation.k must be >= 1")

    qualitative = _to_bool(evaluation_section.get("qualitative"), default=True)
    save_attempts = _to_bool(evaluation_section.get("save_attempts"), default=False)
    run_name = _to_optional_str(evaluation_section.get("run_name")) or "benchmark"

    judge = _parse_model_spec(raw.get("judge"), section_name="judge")
    if qualitative and judge is None:
        raise ValueError(
            "When evaluation.qualitative is true, set judge.provider and judge.model in eval/eval.yaml"
        )

    return EvalConfig(
        dataset=dataset,
        output_dir=output_dir,
        k=k,
        qualitative=qualitative,
        save_attempts=save_attempts,
        run_name=run_name,
        models=models,
        judge=judge,
    )


def default_eval_yaml() -> str:
    return (
        "evaluation:\n"
        "  dataset: eval/input/tasks.json\n"
        "  output_dir: eval/output\n"
        "  k: 1\n"
        "  qualitative: true\n"
        "  save_attempts: false\n"
        "\n"
        "judge:\n"
        "  provider: ollama\n"
        "  model: local-judge-model\n"
        "  base_url: http://localhost:11434\n"
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
        "      \"must_succeed\": true\n"
        "    },\n"
        "    \"meta\": {\n"
        "      \"category\": \"smoke\"\n"
        "    }\n"
        "  }\n"
        "]\n"
    )
