from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from bat.chat_model_client import ChatModelClient, ChatModelClientConfig
from bat.logging import create_logger
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .contracts import (
    EpisodeResult,
    JudgeSpec,
    QualitativeScores,
    TaskExpected,
    TaskSpec,
    TraceEvent,
    Trajectory,
)
from .trajectory import render_trajectory

logger = create_logger(__name__, level="info")

TRAJECTORY_LEGEND = "\n".join(
    [
        "Legend -- the trajectory below was recorded from the agent's "
        "telemetry, in the order things happened:",
        '- "Turn N - user: ..." is what the user said in that turn (the '
        "only ground truth for what the user asked).",
        '- "[model <node>]" is one call to the agent\'s language model in '
        'graph node <node>: the tools it decided to call ("called:") and '
        'what it said ("said:"). Its prompt is left out.',
        '- "[tool <name>] args: ... -> ..." is a tool call with its '
        'arguments and its result, or "FAILED: <error>".',
        '- "[agent <name>] asked: ..." is a call to another agent: what '
        "was asked, that agent's own steps (numbered under it) and its "
        'answer ("-> ...").',
        '- "[error <node>]" is a failure outside any of these. A FAILED '
        "step produced nothing.",
        "",
    ]
)

A2A_LEGEND = "\n".join(
    [
        "Legend -- no telemetry was recorded, so below are only the "
        "messages the agent streamed:",
        "- [USER] explicit user input (the only ground truth for what the "
        "user stated).",
        "- [AGENT OUTPUT] content the agent sent while asking for input; "
        "its values are the agent's proposals.",
        "- [SYSTEM] status updates from the runtime.",
        "",
    ]
)


RESPONSE_RELEVANCE_PROMPT = """You are an evaluator of CONVERSATIONAL RELEVANCE.

**User Queries:**
{query}

**What happened (read the legend at its top):**
{context}

**Final Response:**
{response}

Your job is to judge whether the agent stays on the topic the user raised and avoids detours. There are two axes, scored together on one scale:

1. **On-topic / no detours (primary axis, 0.0–0.8).** Does the agent keep the conversation on the subject the user actually asked about? Penalize:
   - drifting to unrelated subjects mid-conversation
   - addressing something the user never asked about
   - going off on tangents and not coming back
   - sending a tool or another agent a request about something the user did not ask for (visible in the trajectory as the arguments of a [tool] line or the "asked:" of an [agent] line)
   Reward staying consistently on the user's subject across all turns, even if intermediate steps don't immediately resolve the question.

2. **Response craft (refinement axis, 0.8–1.0).** Of the responses that are on-topic, refine the score based on shape:
   - heavy padding, restating system errors verbatim as new analysis, hedging instead of answering → stay at 0.8
   - clean, direct, proportionate response → 0.9–1.0
   This axis only matters once the on-topic floor of 0.8 is reached. Do NOT lower an on-topic response below 0.8 for padding or verbosity alone — mild verbosity is acceptable.

Score bands:
  1.0 — On-topic throughout, no detours, AND a clean direct response.
  0.9 — On-topic throughout, no detours, with light padding or one small hedge.
  0.8 — On-topic throughout, no detours, but noticeable padding / restating / verbosity. This is the floor for "the agent did not go off-topic".
  0.6 — Mostly on-topic with one meaningful detour that the agent recovered from, OR briefly drifted before returning to the subject.
  0.4 — Significant off-topic content — a real portion of the conversation is about something the user didn't ask.
  0.2 — Mostly off-topic, only a small thread relates to the user's actual subject.
  0.0 — Wrong topic entirely, non-sequitur, raw error dump with no engagement.

Do NOT score based on whether the agent's answer is factually correct, whether the action succeeded, or whether the expected outcome was reached. Those are scored by other evaluators. An on-topic wrong answer scores at least 0.8 here.

Return JSON only:
{{
    "reasoning": "1-2 sentences: first whether the agent stayed on topic / had any detours, then briefly note the response shape if it affected the 0.8–1.0 band.",
    "score": float
}}
"""

TASK_COMPLETION_PROMPT = """You are an evaluator of TASK COMPLETION. The score is driven first by whether the expected outcome was actually reached, then refined by how well the agent executed along the way.

**User Queries:**
{query}

**Expected Behavior (the reference):**
{expected_desc}

**What happened (read the legend at its top):**
{context}

**Final Response:**
{response}

**Actual Final Status:** {status}

Start by establishing what was expected and what actually happened. The expected behavior tells you what the final status should be and what the outcome should look like — compare that to the actual final status and to what the agent actually did. This match or mismatch is the dominant factor in your score.

**Verify, do not trust the response.** When a trajectory is shown, every action the final response claims (created, saved, applied, sent, configured, fixed) must be backed by a step that did it and did not fail — a [tool] or [agent] line whose result shows it. A claimed outcome that no successful step supports was NOT reached, however confident the response sounds: treat it as a mismatch. A FAILED step that produced nothing cannot support a claim.

If expected status is "completed", the task was meant to finish with a real, concrete result — anything else is a failure. If expected is "input-required", stopping to ask for missing info IS the success condition. If expected is "error", a clean refusal or failure IS the success.

Use this as your base score:

  1.0  — actual matches expected, with a complete concrete result fully satisfying the stated expectations.
  0.8  — actual matches expected, minor gaps in the result (small omission, slightly incomplete).
  0.6  — actual matches expected in status, but the deliverable is shallow or barely meets the bar.
  0.4  — actual does NOT match expected, but the agent did substantial relevant work and came close.
  0.2  — actual does NOT match expected, the work was shallow or went off-track early.
  0.0  — actual does NOT match expected, no meaningful work, refusal, or total failure.

When the expected outcome was not reached, the score must be at most 0.4 regardless of effort. Reaching the wrong terminal state is a failure — do not reward process over outcome.

After establishing the base score, look at the intermediate steps. Even when the agent reached the right terminal status, check whether it made significant errors, unnecessary detours, or wrong turns along the way. If the path had clear missteps (e.g. tried an invalid value multiple times, looped on the same error, passed the wrong value to a tool or another agent, went in circles), adjust down by up to 0.2. A FAILED step the agent recovered from is a misstep; a FAILED step the response hides or misreports means the outcome was not reached. If the execution was clean, direct, and correct, nudge up by 0.1. For cases where the expected outcome was not reached, intermediate steps can still lift the score from 0.2 to 0.4 if the agent made genuine meaningful progress before diverging.

Return JSON only:
{{
    "reasoning": "1-2 sentences: how the actual outcome compared to expected (naming the step that supports or contradicts it), then the execution flaws or merits and the adjustment from the base score.",
    "score": float
}}
"""

HALLUCINATION_DETECTION_PROMPT = """You are an evaluator of GROUNDEDNESS. Your job is to score how closely the agent stays anchored to what it was told and what it actually observed. Hallucination happens when the agent introduces specifics nobody provided, alters something the user provided, or reports something the recorded steps contradict.

**User Queries:**
{query}

**All facts the user explicitly stated (ground truth for the user's domain — use this as your checklist):**
{user_facts}

**Expected Behavior:**
{expected_desc}

**What happened (read the legend at its top):**
{context}

**Final Response:**
{response}

Sources the agent may legitimately rely on:
- the user's own statements (the only ground truth for what the user wants: names, values, quantities);
- results of tool calls and answers of other agents shown in the trajectory, and runtime status messages — these are observations, not user facts, and using them is not hallucination;
- widely-known public facts.

Walk through every specific claim in the final response (concrete value, name, number, identifier, status, outcome), and also every value the agent itself passed on to a tool or another agent, and classify each significant one:

- GROUNDED — traces back to a user statement or to a tool/agent result, reproduced faithfully.
- FABRICATED — a specific user-domain fact or value that nobody provided.
- ALTERED — the agent changed something the user did specify (user said X, the agent used or reported Y), including when it passed the changed value on to a tool or another agent.
- CONTRADICTED — the response says the opposite of what a recorded step shows: claims success where the step FAILED, reports a value the tool did not return, or states an outcome no step produced.

**Policy: echoed user values.** If the user explicitly stated a value — valid or invalid for any underlying schema — and the agent reproduces it faithfully, it is GROUNDED, never hallucination.

**Policy: legitimate non-user sources.** Values the agent took from a tool result or another agent's answer are GROUNDED even though the user never said them. Flag a claim as FABRICATED only when it is user-domain content nobody provided.

Score bands:
  1.0  — Every specific claim is GROUNDED.
  0.8  — One minor FABRICATED detail, harmless, no effect on the outcome.
  0.6  — One or two non-trivial FABRICATED or ALTERED claims the user would notice.
  0.4  — Several FABRICATED or ALTERED claims, or a single one that caused a wrong outcome, or one CONTRADICTED claim about the outcome.
  0.2  — The agent substantially invented specifics or misreported what happened; most claims are FABRICATED, ALTERED or CONTRADICTED.
  0.0  — Almost nothing in the response corresponds to what the user said or what the steps show.

If the response makes no specific claims (only clarifying questions or acknowledged uncertainty), score 1.0. A claim that misleads the user about what actually happened weighs more than a harmless invented detail.

Return JSON only:
{{
    "reasoning": "List the FABRICATED, ALTERED or CONTRADICTED claims, each with the user statement or step it conflicts with. If everything is grounded, write 'Fully grounded.'",
    "score": float
}}
"""

TOOL_CALL_APPROPRIATENESS_PROMPT = """You are an evaluator of TOOL USAGE.

**User Queries:**
{query}

**Expected Behavior (the reference):**
{expected_desc}

**What happened (read the legend at its top):**
{context}

**Tool Calls Made, in order (source of truth, called agents' included):**
{tool_calls}

**Final Response:**
{response}

TASK: Evaluate whether the tool usage was appropriate for what was expected: the right tools, with the right arguments, in a sensible order, and their results actually used.

CRITICAL RULE: Score using ONLY the recorded tool calls (the list above, and the [tool] lines of the trajectory with their arguments) as evidence. NEVER infer tool usage from prose in the response — if a tool isn't recorded, it wasn't called. Tool lines numbered under an [agent] line were made by that called agent; they count toward the expected tools.

Check the arguments against what the user said: a tool called with a value the user did not give (a different name, a wrong quantity) is a wrong argument. A tool call that FAILED and was neither retried nor worked around, when its result was needed, counts as missing.

SCORE BANDS — use ALL of them:

  1.0  — All expected tools called, with correct arguments, in a sensible order. No redundant calls. Results clearly drive the response.
  0.8  — Right tools called with minor flaws: one redundant call, slightly off arguments that still work, or mild inefficiency in ordering.
  0.6  — Right idea, flawed execution: most expected tools called but one missing or extra, OR arguments partially wrong but partially recoverable.
  0.4  — Significant gaps: roughly half the expected tool usage is correct; the other half is missing, wrong, failed, or used with broken arguments.
  0.2  — Largely wrong: wrong tools selected, OR correct tools called with mostly broken arguments.
  0.0  — No tool calls when tools were required, OR every tool call is wrong/fabricated, OR the agent claims tool usage in prose with no recorded calls.

ANTI-CLUSTERING: A single missing expected tool call is 0.6, not 0.8. Wrong arguments on the right tool is 0.4–0.6 depending on severity.

REASON FIRST, SCORE SECOND. Name the expected tools, mark each as called/missing/wrong/failed, then score.

Return JSON only:
{{
    "reasoning": "Map expected tools to the recorded calls — what's present, missing, wrong or failed",
    "score": float
}}
"""

TASK_OUTCOME_PROMPT = """You are an evaluator of TASK SUCCESS. You see only what the user asked and how the conversation ended, not the steps in between. Judge whether the user got what they asked for.

**User Queries:**
{query}

**Expected Behavior (the reference):**
{expected_desc}

**Final Response:**
{response}

**Actual Final Status:** {status}

Be generous. Judge the outcome, not the style:
- Take the response's claims at face value unless they contradict themselves or the final status. You cannot see the steps, so never lower the score for missing evidence.
- Do not lower the score for verbosity, formatting, tone, or extra helpful detail.
- If expected status is "input-required", asking for the missing information IS success. If expected is "error", a clean refusal or failure IS success.
- When torn between two bands, pick the higher one.

Score bands:
  1.0 — Done: the response delivers what was asked.
  0.8 — Done, with small gaps a user would not mind.
  0.6 — Mostly done: the core of the request is met, a secondary part is missing.
  0.4 — Partly done: real progress, but the user would have to ask again.
  0.2 — Barely: on the right subject, little of what was asked.
  0.0 — Not done: wrong task, unexpected refusal or error, or nothing usable.

Return JSON only:
{{
    "reasoning": "1-2 sentences: what was asked, and whether the final response delivers it.",
    "score": float
}}
"""

# Rubric name (as in judge.prompts) -> its prompt and the score it sets.
RUBRICS = {
    "relevance": (RESPONSE_RELEVANCE_PROMPT, "response_relevance"),
    "task_completion": (TASK_COMPLETION_PROMPT, "task_completion_quality"),
    "hallucination": (HALLUCINATION_DETECTION_PROMPT, "hallucination_score"),
    "tool_call": (
        TOOL_CALL_APPROPRIATENESS_PROMPT,
        "tool_call_appropriateness",
    ),
}
# judge.mode outcome: task completion alone, from the request and result.
OUTCOME_RUBRICS = {
    "task_completion": (TASK_OUTCOME_PROMPT, "task_completion_quality"),
}
_SYSTEM = "You are a precise evaluator. Always respond with valid JSON only."


class JudgeVerdict(BaseModel):
    """What every judge answers, sent to its server as the output schema.
    Reasoning comes first so the model reasons before it scores."""

    reasoning: str = Field(
        description="The evidence and how it maps to the rubric, 1-3 sentences."
    )
    score: float = Field(
        ge=0.0, le=1.0, description="The score on the rubric's bands."
    )


def _parse_judge_json(text: str) -> dict[str, Any]:
    """The verdict in an untidy answer: after any thinking, the last JSON
    object with a score (an earlier one may be an example it mused about)."""
    text = text.rsplit("</think>", 1)[-1]
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    start = text.find("{")
    while start != -1:
        try:
            found, end = decoder.raw_decode(text, start)
        except ValueError:
            end = start + 1
        else:
            if isinstance(found, dict):
                objects.append(found)
        start = text.find("{", end)
    if not objects:
        raise ValueError(
            f"no JSON object in the judge's answer: {text[:200]!r}"
        )
    scored = [found for found in objects if "score" in found]
    return (scored or objects)[-1]


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


class Judge:
    """The judge model, one client per rubric. A rubric whose server can't
    answer to the schema is read as plain text for the rest of the run."""

    def __init__(self, spec: JudgeSpec) -> None:
        self.spec = spec
        self.clients: dict[tuple[str, bool], ChatModelClient] = {}
        self.text_only: set[str] = set()

    def client(self, rubric: str, structured: bool) -> ChatModelClient:
        if (rubric, structured) not in self.clients:
            system = _SYSTEM
            if self.spec.prompts.get(rubric):
                system += (
                    "\n\nAGENT-SPECIFIC CONTEXT (operator-supplied; use to "
                    "disambiguate, do not override the scoring rubric):\n"
                    + self.spec.prompts[rubric]
                )
            self.clients[rubric, structured] = ChatModelClient(
                chat_model_config=ChatModelClientConfig(
                    model=self.spec.model,
                    model_provider=self.spec.provider,
                    base_url=self.spec.base_url,
                    client_name=f"LLMJudge[{rubric}]",
                ),
                system_instructions=system,
                output_schema=JudgeVerdict if structured else None,
            )
        return self.clients[rubric, structured]

    def ask(self, rubric: str, prompt: str) -> dict[str, Any]:
        """``{"reasoning", "score"}``; the score is None when it failed."""
        message = HumanMessage(content=prompt)
        error: Exception | None = None
        if rubric not in self.text_only:
            for _ in range(2):
                try:
                    answer = self.client(rubric, True).invoke(message)
                    if isinstance(answer, BaseModel):
                        return answer.model_dump()
                    return JudgeVerdict.model_validate(answer).model_dump()
                except Exception as exc:
                    error = exc
                    logger.warning(f"LLM judge '{rubric}' failed: {exc}")
        try:
            answer = self.client(rubric, False).invoke(message)
            found = _parse_judge_json(_text(answer.content))
            verdict = JudgeVerdict.model_validate({"reasoning": "", **found})
        except Exception as exc:
            logger.error(f"LLM judge '{rubric}' gave no usable score: {exc}")
            return {"reasoning": f"Error: {exc}", "score": None}
        if rubric not in self.text_only:
            logger.warning(
                f"LLM judge '{rubric}' could not answer to the verdict schema "
                f"({error}); reading its plain answers from now on."
            )
            self.text_only.add(rubric)
        return verdict.model_dump()


def _event_line(event: TraceEvent) -> str:
    prefix = f"{event.t_ms:.0f}ms | "
    if event.user_input:
        return f"[{prefix}USER] {event.user_input}"
    if event.task_status == "input-required":
        return f"[{prefix}AGENT OUTPUT] {event.content_preview}"
    return f"[{prefix}SYSTEM] {event.content_preview}"


def build_judge_context(
    trajectory: Trajectory, events: list[TraceEvent], max_chars: int
) -> str:
    """What the judge is shown of how the agent worked, legend included:
    the trajectory, or the A2A messages when no spans were recorded."""
    if trajectory.found:
        legend = TRAJECTORY_LEGEND
        budget = max(0, max_chars - len(legend) - 1)
        body = render_trajectory(trajectory, max_chars=budget)
    else:
        legend = A2A_LEGEND
        body = "\n".join(_event_line(event) for event in events) or "No events"
    return f"{legend}\n{body}"[:max_chars]


def _times(name: str, times: int) -> str:
    return f"'{name}' (at least {times}×)" if times > 1 else f"'{name}'"


def _expected(expected: TaskExpected, *, steps: bool = True) -> str:
    """What the task expects; without the expectations about its steps
    when the judge is not shown them."""
    parts: list[str] = []
    if expected.expected_outcome:
        parts.append(f"Expected outcome: {expected.expected_outcome.strip()}")
    if expected.status is not None:
        parts.append(f"The task should reach final status '{expected.status}'.")
    if expected.output_must_contain:
        quoted = ", ".join(f'"{s}"' for s in expected.output_must_contain)
        parts.append(f"Output must contain: {quoted}.")
    if steps and expected.tool_calls:
        calls = ", ".join(_times(c.name, c.times) for c in expected.tool_calls)
        parts.append(f"Expected tool calls: {calls}.")
    if steps and expected.agent_calls:
        calls = ", ".join(
            _times(c.agent, c.times) for c in expected.agent_calls
        )
        parts.append(f"Expected calls to other agents: {calls}.")
    if steps and expected.no_errors:
        parts.append("No step should fail along the way.")
    return " ".join(parts) or "No specific expectations defined."


def score(
    spec: JudgeSpec, episodes: list[EpisodeResult], tasks: dict[str, TaskSpec]
) -> None:
    """Every rubric on every episode, at most 16 judge calls at a time.
    Tool usage is scored only for tasks that expect tool calls."""
    outcome = spec.mode == "outcome"
    rubrics = OUTCOME_RUBRICS if outcome else RUBRICS
    judge = Judge(spec)
    asked = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for episode in episodes:
            task = tasks[episode.task_id]
            fields = {
                "query": " -> ".join(task.turns),
                "response": episode.final_output,
                "status": episode.final_status,
                "context": build_judge_context(
                    episode.trace.trajectory,
                    episode.trace.events,
                    spec.max_trajectory_chars,
                ),
                "expected_desc": _expected(task.expected, steps=not outcome),
                "user_facts": "\n".join(f"- {turn}" for turn in task.turns),
                # Names only: the [tool] lines carry the arguments, but a
                # rendering over budget may leave some of them out.
                "tool_calls": ", ".join(
                    call["name"] + (" (FAILED)" if call.get("error") else "")
                    for call in episode.trace.tool_calls
                )
                or "none",
            }
            episode.qualitative_scores = QualitativeScores()
            for rubric, (prompt, _) in rubrics.items():
                if rubric == "tool_call" and not task.expected.tool_calls:
                    episode.qualitative_scores.judge_reasoning[rubric] = (
                        "skipped: no tool calls expected for this task"
                    )
                    continue
                job = pool.submit(judge.ask, rubric, prompt.format(**fields))
                asked.append((episode.qualitative_scores, rubric, job))
        for scores, rubric, job in asked:
            result = job.result()
            setattr(scores, rubrics[rubric][1], result["score"])
            scores.judge_reasoning[rubric] = result["reasoning"]
