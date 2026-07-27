from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AgentTaskStatus = Literal["working", "input-required", "completed", "error"]


class ExpectedToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    args_subset: dict[str, Any] = Field(default_factory=dict)
    times: int = 1


class TaskExpected(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Expected final status from the agent. None = skip status check entirely.
    status: AgentTaskStatus | None = "completed"
    # Free-text description of the desired outcome, evaluated semantically by the LLM judges.
    expected_outcome: str | None = None
    # All phrases must appear in the final output text. None/empty = skip substring check.
    output_must_contain: list[str] | None = None
    tool_calls: list[ExpectedToolCall] = Field(default_factory=list)


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    turns: list[str]
    expected: TaskExpected = Field(default_factory=TaskExpected)
    meta: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    t_ms: float
    task_status: AgentTaskStatus
    content_preview: str
    user_input: str | None = None


class EpisodeTrace(BaseModel):
    model_config = ConfigDict(extra="ignore")

    events: list[TraceEvent] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class QualitativeScores(BaseModel):
    model_config = ConfigDict(extra="ignore")

    response_relevance: float | None = None
    task_completion_quality: float | None = None
    hallucination_score: float | None = None
    tool_call_appropriateness: float | None = None
    judge_reasoning: dict[str, str] = Field(default_factory=dict)


class EpisodeVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    passed: bool
    reason: str = ""


class EpisodeResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: str | None = None
    task_id: str
    expected_outcome: str | None = None
    final_status: AgentTaskStatus
    final_output: str
    verdict: EpisodeVerdict | None = None
    qualitative_scores: QualitativeScores | None = None
    aux: dict[str, Any] = Field(default_factory=dict)
    trace: EpisodeTrace = Field(default_factory=EpisodeTrace)


class ModelSpec(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class JudgeSpec(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    prompts: dict[str, str] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    dataset: Path
    output_dir: Path
    agent_url: str
    agent_startup_timeout_s: int = 45
    agent_shutdown_timeout_s: int = 10
    k: int
    qualitative: bool
    run_name: str
    models: list[ModelSpec]
    judge: JudgeSpec | None
