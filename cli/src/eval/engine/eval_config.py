from __future__ import annotations

from pathlib import Path
from typing import Any, get_args

import yaml
from bat.chat_model_client.config import ModelProvider

from .contracts import EvalConfig, JudgeSpec, ModelSpec

PROVIDERS = sorted(get_args(ModelProvider))
JUDGE_PROMPTS = ("relevance", "task_completion", "hallucination", "tool_call")

DEFAULT_EVAL_YAML = """\
evaluation:
  dataset: eval/input/tasks.json
  output_dir: eval/output
  agent_startup_timeout_s: 45
  agent_shutdown_timeout_s: 10
  k: 1
  qualitative: false
  # extra_spans:   # span files of the agents this one calls
  #   - ../other-agent/spans.jsonl

judge:
  provider: ollama
  model: local-judge-model
  base_url: http://localhost:11434
  # api_key_env: BAT_JUDGE_API_KEY   # name of the env var holding the judge's API key
  # mode: full   # outcome: one lenient score from request + final response
  # max_trajectory_chars: 24000   # of steps shown (~4 chars/token)
  # prompts:   # agent-specific context added to a judge's rubric
  #   task_completion: "A draft is not a deployed network."

models:
  - provider: openai
    model: your-model-name
  - provider: ollama
    model: your-local-model
    base_url: http://localhost:11434
"""

DEFAULT_TASKS_JSON = """\
[
  {
    "id": "smoke_test",
    "turns": [
      "Describe what you can do in one short paragraph."
    ],
    "expected": {
      "status": "completed",
      "expected_outcome": "The agent describes its capabilities clearly in one short paragraph."
    },
    "meta": {
      "category": "smoke"
    }
  }
]
"""


def _spec(raw: Any, where: str) -> dict[str, Any]:
    """A model or judge entry: a mapping, or a '<provider>:<model>' string."""
    spec = {"model": raw} if isinstance(raw, str) else dict(raw)
    provider = spec.get("provider")
    model = str(spec.get("model") or "")
    if not provider and ":" in model:
        provider, model = model.split(":", 1)
    if not provider or not model:
        raise ValueError(
            f"{where} needs a provider and a model, or '<provider>:<model>'"
        )
    if provider not in PROVIDERS:
        raise ValueError(
            f"{where}.provider '{provider}' is not supported. "
            f"Valid providers: {', '.join(PROVIDERS)}."
        )
    env = spec.get("env") or {}
    spec.update(
        provider=provider,
        model=model,
        env={
            key: str(value) for key, value in env.items() if value is not None
        },
    )
    return spec


def _judge(raw: Any) -> JudgeSpec | None:
    if not raw:
        return None
    spec = _spec(raw, "judge")
    prompts = spec.get("prompts") or {}
    unknown = sorted(set(prompts) - set(JUDGE_PROMPTS))
    if unknown:
        raise ValueError(
            f"judge.prompts has unknown key(s) {unknown}; "
            f"allowed: {list(JUDGE_PROMPTS)}"
        )
    for key, text in prompts.items():
        if len(str(text)) > 1000:
            raise ValueError(
                f"judge.prompts.{key} exceeds the 1000-character limit "
                f"(got {len(str(text))})"
            )
    spec["prompts"] = prompts
    return JudgeSpec(**spec)


def load_eval_config(agent_root: Path, config_path: Path) -> EvalConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    evaluation = raw.get("evaluation") or {}
    models = [
        ModelSpec(**_spec(item, f"models[{index}]"))
        for index, item in enumerate(raw.get("models") or [])
    ]
    if not models:
        raise ValueError("No valid models configured in eval/eval.yaml")
    judge = _judge(raw.get("judge"))
    qualitative = bool(evaluation.get("qualitative"))
    if qualitative and judge is None:
        raise ValueError(
            "When evaluation.qualitative is true, set judge.provider and "
            "judge.model in eval/eval.yaml"
        )
    extra_spans = evaluation.get("extra_spans") or []
    if isinstance(extra_spans, str):
        extra_spans = [extra_spans]
    dataset = evaluation.get("dataset") or "eval/input/tasks.json"
    output_dir = evaluation.get("output_dir") or "eval/output"
    return EvalConfig(
        dataset=(agent_root / dataset).resolve(),
        output_dir=(agent_root / output_dir).resolve(),
        agent_startup_timeout_s=evaluation.get("agent_startup_timeout_s", 45),
        agent_shutdown_timeout_s=evaluation.get("agent_shutdown_timeout_s", 10),
        k=evaluation.get("k", 1),
        qualitative=qualitative,
        run_name=evaluation.get("run_name") or "benchmark",
        models=models,
        judge=judge,
        extra_spans=[(agent_root / path).resolve() for path in extra_spans],
    )
