from __future__ import annotations

import concurrent.futures
import json
import os
from typing import Any

from bat.chat_model_client import ChatModelClient, ChatModelClientConfig
from bat.logging import create_logger
from langchain_core.messages import HumanMessage

from ..contracts import QualitativeScores

logger = create_logger(__name__, level="info")


# ---------------------------------------------------------------------------
# Judge ChatModelClient (bat-adk based)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_BASE = (
    "You are a precise evaluator. Always respond with valid JSON only."
)

_judge_clients: dict[str, ChatModelClient] = {}


def _compose_system(custom: str | None) -> str:
    if not custom:
        return _JUDGE_SYSTEM_BASE
    return (
        f"{_JUDGE_SYSTEM_BASE}\n\n"
        "AGENT-SPECIFIC CONTEXT (operator-supplied; use to disambiguate, "
        f"do not override the scoring rubric):\n{custom}"
    )


def _get_judge_client(judge_name: str = "default") -> ChatModelClient:
    """
    Lazy-init a ChatModelClient configured for the judge model.

    Reads JUDGE_PROVIDER / JUDGE_MODEL env vars (falls back to MODEL_PROVIDER / MODEL).
    For named judges, also reads JUDGE_PROMPT_<NAME> as an optional system-message suffix.
    """
    if judge_name in _judge_clients:
        return _judge_clients[judge_name]

    provider = os.getenv(
        "JUDGE_PROVIDER", os.getenv("MODEL_PROVIDER", "openai")
    )
    model = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
    base_url = os.getenv("JUDGE_BASE_URL", os.getenv("BASE_URL"))
    custom = (
        os.getenv(f"JUDGE_PROMPT_{judge_name.upper()}")
        if judge_name != "default"
        else None
    )

    config = ChatModelClientConfig(
        model=model,
        model_provider=provider,
        base_url=base_url,
        client_name=f"LLMJudge[{judge_name}]",
    )

    client = ChatModelClient(
        chat_model_config=config,
        system_instructions=_compose_system(custom),
    )
    _judge_clients[judge_name] = client
    logger.info(f"LLM Judge initialized: {provider}:{model} [{judge_name}]")
    return client


def _call_llm_judge(
    prompt: str, judge_name: str = "default", max_retries: int = 2
) -> dict[str, Any]:
    client = _get_judge_client(judge_name)
    for attempt in range(max_retries):
        try:
            logger.info(
                f"Calling LLM judge (attempt {attempt + 1}/{max_retries}) "
            )
            response = client.invoke(HumanMessage(content=prompt))
            content = response.content.strip()
            if content.startswith("```json"):
                content = (
                    content.split("```json", 1)[1].split("```", 1)[0].strip()
                )
            elif content.startswith("```"):
                content = content.split("```", 1)[1].split("```", 1)[0].strip()
            return json.loads(content)
        except Exception as exc:
            if attempt == max_retries - 1:
                return {"score": None, "reasoning": f"Error: {exc}"}
    return {"score": None, "reasoning": "Max retries exceeded"}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RESPONSE_RELEVANCE_PROMPT = """You are an evaluator of CONVERSATIONAL RELEVANCE.

**User Queries:**
{query}

**Full Conversation:**
{context}

**Final Response:**
{response}

Your job is to judge whether the agent stays on the topic the user raised and avoids detours. There are two axes, scored together on one scale:

1. **On-topic / no detours (primary axis, 0.0–0.8).** Does the agent keep the conversation on the subject the user actually asked about? Penalize:
   - drifting to unrelated subjects mid-conversation
   - addressing something the user never asked about
   - going off on tangents and not coming back
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

Do NOT score based on whether the agent's answer is factually correct, whether the deployment succeeded, or whether the expected outcome was reached. Those are scored by other evaluators. An on-topic wrong answer scores at least 0.8 here.

Return JSON:
{{
    "reasoning": "1-2 sentences: first whether the agent stayed on topic / had any detours, then briefly note the response shape if it affected the 0.8–1.0 band.",
    "score": float
}}
"""

TASK_COMPLETION_PROMPT = """You are an evaluator of TASK COMPLETION. The score is driven first by whether the expected outcome was actually reached, then refined by how well the agent executed along the way.

**User Queries:**
{query}

**Expected Behavior:**
{expected_desc}

**Conversation:**
{context}

**Final Response:**
{response}

**Actual Final Status:** {status}

Start by establishing what was expected and what actually happened. The expected_desc tells you what the final status should be and what the outcome should look like — compare that to the actual final status and the concrete deliverable in the response. This match or mismatch is the dominant factor in your score.

If expected status is "completed", the task was meant to finish with a real, concrete result — anything else is a failure. If expected is "input-required", stopping to ask for missing info IS the success condition. If expected is "error", a clean refusal or failure IS the success.

Use this as your base score:

  1.0  — actual matches expected, with a complete concrete result fully satisfying the stated expectations.
  0.8  — actual matches expected, minor gaps in the result (small omission, slightly incomplete).
  0.6  — actual matches expected in status, but the deliverable is shallow or barely meets the bar.
  0.4  — actual does NOT match expected, but the agent did substantial relevant work and came close.
  0.2  — actual does NOT match expected, the work was shallow or went off-track early.
  0.0  — actual does NOT match expected, no meaningful work, refusal, or total failure.

When the expected outcome was not reached, the score must be at most 0.4 regardless of effort. Reaching the wrong terminal state is a failure — do not reward process over outcome.

After establishing the base score, look at the intermediate steps in the conversation. Even when the agent reached the right terminal status, check whether it made significant errors, unnecessary detours, or wrong turns along the way. If the path had clear missteps (e.g. tried an invalid value multiple times, looped on the same error, went in circles), adjust down by up to 0.2. If the execution was clean, direct, and correct, nudge up by 0.1. For cases where the expected outcome was not reached, intermediate steps can still lift the score from 0.2 to 0.4 if the agent made genuine meaningful progress before diverging.

Return JSON:
{{
    "reasoning": "1-2 sentences summarizing how the actual outcome compared to expected, and noting any significant execution flaws or merits that influenced the score. State the adjustment from the base score based on the execution quality.",
    "score": float
}}
"""

HALLUCINATION_DETECTION_PROMPT = """You are an evaluator of GROUNDEDNESS. Your job is to score how closely the agent's response stays anchored to what the user actually said. Hallucination happens when the agent introduces specifics the user never provided, or silently alters something the user did provide.

**User Queries:**
{query}

**All facts the user explicitly stated (ground truth — use this as your checklist):**
{user_facts}

**Expected Behavior:**
{expected_desc}

**Full Conversation:**
{context}

**Final Response:**
{response}

The conversation uses three event labels:
- [USER] — explicit user input. This is the only ground truth for what the user stated.
- [AGENT OUTPUT] — content the agent generated and sent to the user. Specific values here are agent assertions, not ground truth — any concrete value here that does not trace back to a [USER] line is a candidate for hallucination.
- [SYSTEM] — internal runtime status messages. Values returned here are legitimate non-user input and may be used by the agent without being hallucination.

Walk through every specific claim the agent makes (concrete value, name, number, identifier, field) and compare it against the user's stated facts. Classify each significant claim:

- GROUNDED — the user stated or requested this, and the agent reproduced it faithfully.
- FABRICATED — the agent introduced a specific user-domain fact or value the user never mentioned.
- ALTERED — the agent changed something the user did specify (user said X, agent produced Y, where X ≠ Y as written).

**Policy: echoed user values.** If the user explicitly stated a value in a [USER] line — valid or invalid for any underlying schema — and the agent reproduces it faithfully, it is GROUNDED, never hallucination. ALTERED applies only when the agent silently substitutes a different value for one the user provided.

**Policy: legitimate non-user sources.** Not every specific value an agent emits is a fabrication. Information the agent obtained from a legitimate source other than the user — tool outputs, [SYSTEM] messages, backend responses the agent is relaying, or widely-known public facts — is not hallucination. Flag a claim as FABRICATED only when it is user-domain content (something the user would have had to provide) and the user never provided it.

Score bands:
  1.0  — Every specific claim is GROUNDED or comes from a legitimate non-user source.
  0.8  — One minor FABRICATED detail, harmless, no effect on outcome.
  0.6  — One or two non-trivial FABRICATED or ALTERED claims the user would notice.
  0.4  — Several FABRICATED or ALTERED claims, or a single one that caused a concrete failure or wrong outcome.
  0.2  — The agent substantially filled in specifics the user never provided; most claims are FABRICATED.
  0.0  — Almost nothing in the response corresponds to what the user said; the answer is largely the agent's invention.

If the response makes no specific claims (only clarifying questions or acknowledged uncertainty), score 1.0. A hallucination that directly caused a task failure weighs more than a harmless one.

Return JSON:
{{
    "reasoning": "List the FABRICATED or ALTERED claims with their impact. If everything is grounded, write 'Fully grounded.'",
    "score": float
}}
"""

TOOL_CALL_APPROPRIATENESS_PROMPT = """You are an evaluator of TOOL USAGE.

**User Queries:**
{query}

**Expected Behavior:**
{expected_desc}

**Conversation:**
{context}

**Tool Calls Made (source of truth):**
{tool_calls}

**Final Response:**
{response}

TASK: Evaluate whether the agent's tool usage was appropriate for what was expected.

CRITICAL RULE: Score using ONLY the `tool_calls` JSON above as evidence. NEVER infer tool usage from prose in the response — if a tool isn't in `tool_calls`, it wasn't called.

SCORE BANDS — use ALL of them:

  1.0  — All expected tools called, with correct arguments, in a sensible order. No redundant calls. Results clearly drive the response.
  0.8  — Right tools called with minor flaws: one redundant call, slightly off arguments that still work, or mild inefficiency in ordering.
  0.6  — Right idea, flawed execution: most expected tools called but one missing or extra, OR arguments partially wrong but partially recoverable.
  0.4  — Significant gaps: roughly half the expected tool usage is correct; the other half is missing, wrong, or used with broken arguments.
  0.2  — Largely wrong: wrong tools selected, OR correct tools called with mostly broken arguments.
  0.0  — No tool calls when tools were required, OR every tool call is wrong/fabricated, OR the agent claims tool usage in prose with no actual calls in `tool_calls`.

ANTI-CLUSTERING: A single missing expected tool call is 0.6, not 0.8. Wrong arguments on the right tool is 0.4–0.6 depending on severity.

REASON FIRST, SCORE SECOND. Name the expected tools, mark each as called/missing/wrong, then score.

Return JSON:
{{
    "reasoning": "Map expected tools to actual tool_calls — what's present, missing, wrong",
    "score": float
}}
"""


def evaluate_response_relevance(
    query: str,
    response: str,
    context: str = "",
) -> dict[str, Any]:
    prompt = RESPONSE_RELEVANCE_PROMPT.format(
        query=query,
        response=response,
        context=context or "No conversation history available",
    )
    return _call_llm_judge(prompt, judge_name="relevance")


def evaluate_task_completion(
    query: str,
    response: str,
    status: str,
    expected_desc: str = "Task should complete successfully",
    context: str = "",
) -> dict[str, Any]:
    """Evaluate task completion quality including process and outcome."""
    prompt = TASK_COMPLETION_PROMPT.format(
        query=query,
        response=response,
        status=status,
        expected_desc=expected_desc,
        context=context or "No conversation history available",
    )
    return _call_llm_judge(prompt, judge_name="task_completion")


def evaluate_hallucination(
    query: str,
    response: str,
    context: str = "",
    expected_desc: str = "No specific expectations defined.",
    user_facts: str = "No explicit user statements recorded.",
) -> dict[str, Any]:
    prompt = HALLUCINATION_DETECTION_PROMPT.format(
        query=query,
        response=response,
        context=context or "No additional context provided",
        expected_desc=expected_desc,
        user_facts=user_facts,
    )
    return _call_llm_judge(prompt, judge_name="hallucination")


def evaluate_tool_call_appropriateness(
    query: str,
    response: str,
    context: str = "",
    tool_calls: str = "[]",
    expected_desc: str = "Task should complete successfully",
) -> dict[str, Any]:
    prompt = TOOL_CALL_APPROPRIATENESS_PROMPT.format(
        query=query,
        response=response,
        context=context or "No conversation history available",
        tool_calls=tool_calls or "[]",
        expected_desc=expected_desc,
    )
    return _call_llm_judge(prompt, judge_name="tool_call")


def evaluate_episode_quality(
    query: str,
    response: str,
    status: str,
    context: str = "",
    expected_desc: str = "Task should complete successfully",
    tool_calls: str = "[]",
    has_expected_tools: bool = False,
    user_facts: str = "No explicit user statements recorded.",
) -> QualitativeScores:
    scores = QualitativeScores()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures: dict[str, Any] = {
            "relevance": pool.submit(
                evaluate_response_relevance, query, response, context
            ),
            "completion": pool.submit(
                evaluate_task_completion,
                query,
                response,
                status,
                expected_desc,
                context,
            ),
            "hallucination": pool.submit(
                evaluate_hallucination,
                query,
                response,
                context,
                expected_desc,
                user_facts,
            ),
        }
        if has_expected_tools:
            futures["tool_calls"] = pool.submit(
                evaluate_tool_call_appropriateness,
                query,
                response,
                context,
                tool_calls,
                expected_desc,
            )
        else:
            scores.judge_reasoning["tool_call_appropriateness"] = (
                "skipped: no tool calls expected for this task"
            )

        try:
            r = futures["relevance"].result()
            if isinstance(r, dict) and r.get("score") is not None:
                scores.response_relevance = float(r["score"])
                scores.judge_reasoning["relevance"] = r.get("reasoning", "")
        except Exception as e:
            logger.error(f"Response relevance evaluation failed: {e}")

        try:
            r = futures["completion"].result()
            if isinstance(r, dict) and r.get("score") is not None:
                scores.task_completion_quality = float(r["score"])
                scores.judge_reasoning["completion"] = r.get("reasoning", "")
        except Exception as e:
            logger.error(f"Task completion evaluation failed: {e}")

        try:
            r = futures["hallucination"].result()
            if isinstance(r, dict) and r.get("score") is not None:
                scores.hallucination_score = float(r["score"])
                scores.judge_reasoning["hallucination"] = r.get("reasoning", "")
        except Exception as e:
            logger.error(f"Hallucination evaluation failed: {e}")

        if "tool_calls" in futures:
            try:
                r = futures["tool_calls"].result()
                if isinstance(r, dict) and r.get("score") is not None:
                    scores.tool_call_appropriateness = float(r["score"])
                    scores.judge_reasoning["tool_call_appropriateness"] = r.get(
                        "reasoning", ""
                    )
            except Exception as e:
                logger.error(
                    f"Tool-call appropriateness evaluation failed: {e}"
                )

    return scores
