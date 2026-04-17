from __future__ import annotations
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, List, Literal, Optional


AgentTaskStatus = Literal["working", "input-required", "completed", "error"]


class ExpectedToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    args_subset: Dict[str, Any] = Field(default_factory=dict)
    times: int = 1


class TaskExpected(BaseModel):
    model_config = ConfigDict(extra="ignore")

    must_succeed: bool = True
    final_contains: Optional[str] = None
    tool_calls: List[ExpectedToolCall] = Field(default_factory=list)


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    turns: List[str]
    expected: TaskExpected = Field(default_factory=TaskExpected)
    meta: Dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    t_ms: float
    task_status: AgentTaskStatus
    content_preview: str
    user_input: Optional[str] = None


class EpisodeTrace(BaseModel):
    model_config = ConfigDict(extra="ignore")

    events: List[TraceEvent] = Field(default_factory=list)
    usage: Dict[str, Any] = Field(default_factory=dict)
    timings: Dict[str, float] = Field(default_factory=dict)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)


class QualitativeScores(BaseModel):
    model_config = ConfigDict(extra="ignore")

    response_relevance: Optional[float] = None
    task_completion_quality: Optional[float] = None
    hallucination_score: Optional[float] = None
    judge_reasoning: Dict[str, str] = Field(default_factory=dict)


class EpisodeResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_id: str
    status: AgentTaskStatus
    output_text: str
    success: bool
    trace: EpisodeTrace = Field(default_factory=EpisodeTrace)
    aux: Dict[str, Any] = Field(default_factory=dict)
    model_name: Optional[str] = None
    qualitative_scores: Optional[QualitativeScores] = None


class ModelSpec(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class JudgeSpec(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


class EvalConfig(BaseModel):
    dataset: Path
    output_dir: Path
    k: int
    qualitative: bool
    save_attempts: bool
    run_name: str
    models: list[ModelSpec]
    judge: JudgeSpec | None