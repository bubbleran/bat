from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

AgentTaskStatus = Literal["working", "input-required", "completed", "error"]


class ExpectedToolCall(BaseModel):
    name: str
    args_subset: dict[str, Any] = Field(default_factory=dict)
    times: int = 1


class ExpectedAgentCall(BaseModel):
    agent: str  # the called agent's card name
    times: int = 1


class TaskExpected(BaseModel):
    status: AgentTaskStatus | None = "completed"  # None skips the check
    expected_outcome: str | None = None  # scored by the judge
    output_must_contain: list[str] | None = None
    # Read off the spans: without any, they fail as one "no spans" check.
    tool_calls: list[ExpectedToolCall] = Field(default_factory=list)
    agent_calls: list[ExpectedAgentCall] = Field(default_factory=list)
    max_model_calls: int | None = None
    max_tokens: int | None = None
    no_errors: bool = False

    @property
    def needs_spans(self) -> bool:
        return bool(
            self.tool_calls
            or self.agent_calls
            or self.no_errors
            or self.max_model_calls is not None
            or self.max_tokens is not None
        )


class TaskSpec(BaseModel):
    id: str
    turns: list[str]
    expected: TaskExpected = Field(default_factory=TaskExpected)
    meta: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    t_ms: float
    task_status: AgentTaskStatus
    content_preview: str
    user_input: str | None = None


class ModelStep(BaseModel):
    kind: Literal["model"] = "model"
    node: str | None = None
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    reasoning_tokens: int = 0
    said: str | None = None
    called: list[str] = Field(default_factory=list)
    error: str | None = None


class ToolStep(BaseModel):
    kind: Literal["tool"] = "tool"
    node: str | None = None
    name: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    steps: list[Step] = Field(default_factory=list)


class AgentStep(BaseModel):
    kind: Literal["agent"] = "agent"
    node: str | None = None
    agent: str | None = None
    asked: str | None = None
    answered: str | None = None
    status: AgentTaskStatus | None = None
    error: str | None = None
    steps: list[Step] = Field(default_factory=list)


class ErrorStep(BaseModel):
    """A failure outside any model, tool or agent call."""

    kind: Literal["error"] = "error"
    node: str | None = None
    span: str | None = None
    error: str


Step = Annotated[
    Union[ModelStep, ToolStep, AgentStep, ErrorStep],
    Field(discriminator="kind"),
]
ToolStep.model_rebuild()
AgentStep.model_rebuild()


class TrajectoryTurn(BaseModel):
    user: str | None = None
    steps: list[Step] = Field(default_factory=list)


class TrajectoryTotals(BaseModel):
    model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0
    tool_calls: int = 0
    agent_calls: int = 0
    errors: int = 0


class Trajectory(BaseModel):
    found: bool = False  # False: no span of the conversation was found
    turns: list[TrajectoryTurn] = Field(default_factory=list)
    totals: TrajectoryTotals = Field(default_factory=TrajectoryTotals)
    truncated: bool = False


class EpisodeTrace(BaseModel):
    events: list[TraceEvent] = Field(default_factory=list)
    wall_ms: float = 0.0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    trajectory: Trajectory = Field(default_factory=Trajectory)


class QualitativeScores(BaseModel):
    response_relevance: float | None = None
    task_completion_quality: float | None = None
    hallucination_score: float | None = None
    tool_call_appropriateness: float | None = None
    judge_reasoning: dict[str, str] = Field(default_factory=dict)


class EpisodeVerdict(BaseModel):
    passed: bool
    reason: str = ""


class EpisodeResult(BaseModel):
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
    # ~4 characters per token: fits a 16k-token context with the rubric.
    max_trajectory_chars: int = Field(default=24000, ge=1)
    # full: every rubric, on the trajectory. outcome: one lenient
    # task-success score from the request and the final response alone.
    mode: Literal["full", "outcome"] = "full"


class EvalConfig(BaseModel):
    dataset: Path
    output_dir: Path
    agent_startup_timeout_s: int = Field(default=45, ge=1)
    agent_shutdown_timeout_s: int = Field(default=10, ge=1)
    k: int = Field(default=1, ge=1)
    qualitative: bool = False
    run_name: str = "benchmark"
    models: list[ModelSpec]
    judge: JudgeSpec | None = None
    extra_spans: list[Path] = Field(default_factory=list)
