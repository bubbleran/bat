# llm_evaluators.py
"""
LLM-as-Judge evaluators for qualitative metrics:
- Response Relevance
- Task Completion Quality
- Hallucination Detection

Uses bat-adk ChatModelClient for LLM calls.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage

from bat.chat_model_client import ChatModelClient, ChatModelClientConfig
from bat.logging import create_logger
logger = create_logger(__name__, level="info")
from ..contracts import QualitativeScores


# ---------------------------------------------------------------------------
# Judge ChatModelClient (bat-adk based)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = "You are a precise evaluator. Always respond with valid JSON only."

_judge_client: Optional[ChatModelClient] = None


def _get_judge_client() -> ChatModelClient:
    """
    Lazy-init a ChatModelClient configured for the judge model.
    
    Reads JUDGE_PROVIDER / JUDGE_MODEL env vars (falls back to MODEL_PROVIDER / MODEL).
    """
    global _judge_client
    if _judge_client is not None:
        return _judge_client

    provider = os.getenv("JUDGE_PROVIDER", os.getenv("MODEL_PROVIDER", "openai"))
    model = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
    base_url = os.getenv("JUDGE_BASE_URL", os.getenv("BASE_URL"))

    config = ChatModelClientConfig(
        model=model,
        model_provider=provider,
        base_url=base_url,
        client_name="LLMJudge",
    )

    _judge_client = ChatModelClient(
        chat_model_config=config,
        system_instructions=_JUDGE_SYSTEM,
    )
    logger.info(f"LLM Judge initialized: {provider}:{model}")
    return _judge_client

def _call_llm_judge(prompt: str, max_retries: int = 2) -> Dict[str, Any]:
    client = _get_judge_client()
    for attempt in range(max_retries):
        try:
            logger.info (f"Calling LLM judge (attempt {attempt + 1}/{max_retries}) ")
            response = client.invoke(HumanMessage(content=prompt))
            content = response.content.strip()
            if content.startswith("```json"):
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
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

RESPONSE_RELEVANCE_PROMPT = """You are a strict evaluator of response relevance.

**User Queries:**
{query}

**Full Conversation:**
{context}

**Final Response:**
{response}

TASK: Score how directly the agent answered the user’s request.

HARD RULES (override all other reasoning):
- If the final response does NOT contain a concrete answer/result → score ≤ 0.3
- If the response is mostly procedural ("how to do it") → score ≤ 0.3
- If the response includes fabricated example outputs (e.g., "item1", "...") → score ≤ 0.5
- If the response directly answers the query with concrete content → score ≥ 0.7

EVALUATION:
- Final response weight: 70%
- Intermediate steps: 30%

CHECKS:
1. Does the final response directly answer the question?
2. Is it a real answer or just instructions?
3. Are there fake placeholders instead of real data?
4. Any off-topic drift?

SCORING:
- 1.0: Direct, concrete, complete answer
- 0.7–0.9: Mostly correct, minor issues
- 0.4–0.6: Partial answer or weak relevance
- 0.2–0.3: Mostly procedural or indirect
- 0.0–0.1: No real answer

Return JSON:
{{
    "score": float,
    "reasoning": "short explanation"
}}
"""

TASK_COMPLETION_PROMPT = """You are a strict evaluator of task completion.

**User Queries:**
{query}

**Expected Behavior:**
{expected_desc}

**Conversation:**
{context}

**Final Response:**
{response}

**Status:** {status}

TASK: Did the agent ACTUALLY complete the task?

HARD RULES:
- If no real result/output is provided → score ≤ 0.3
- If only instructions are given without the REAL results→ score ≤ 0.2
- If output is fabricated or placeholder, check other steps to understand the context → score ≤ 0.4
- If required data should come from tools but none is shown → score ≤ 0.3
- Only give ≥0.8 if the result is concrete AND matches expectations. NOT ONLY THE DESCRIPTION OF WHAT SHOULD HAVE BEEN DONE.

IMPORTANT:
The FIRST user request is mandatory. If not fulfilled → score ≤ 0.3

EVALUATION:
- Final result: 80%
- Process: 20%

CHECKS:
1. Was the task actually completed?
2. Is there a real, usable result?
3. Is the result verifiable (not fake)?
4. Does it match expected behavior?

SCORING:
- 1.0: Fully completed with real output
- 0.7–0.9: Completed with minor issues
- 0.4–0.6: Partial completion
- 0.2–0.3: Barely progressed / no real output
- 0.0–0.1: Failed / only instructions

Return JSON:
{{
    "score": float,
    "reasoning": "short explanation"
}}
"""

HALLUCINATION_DETECTION_PROMPT = """You are a strict hallucination detector.

**User Queries:**
{query}

**Conversation (with tools):**
{context}

**Final Response:**
{response}

TASK: Detect fabricated or unsupported information.

HARD RULES:
- If the response invents data not present in context/tools → score ≤ 0.3
- If placeholders are used (e.g., "...", "item1") → score ≤ 0.4
- If the agent claims results without evidence → score ≤ 0.2
- If tool output is expected but missing → penalize heavily
- Only give ≥0.8 if ALL claims are grounded

CHECK:
1. Are outputs backed by tool_calls or context?
2. Any invented values?
3. Any fake examples pretending to be real?
4. Any claim of completion without proof?

SCORING:
- 1.0: Fully grounded
- 0.7–0.9: Minor assumptions
- 0.4–0.6: Some unsupported claims
- 0.2–0.3: Significant fabrication
- 0.0–0.1: Mostly fabricated or misleading

Return JSON:
{{
    "score": float,
    "reasoning": "short explanation"
}}
"""

TOOL_CALL_APPROPRIATENESS_PROMPT = """You are a strict evaluator of tool usage.

**User Queries:**
{query}

**Expected Behavior:**
{expected_desc}

**Conversation:**
{context}

**Tool Calls (source of truth):**
{tool_calls}

**Final Response:**
{response}

TASK: Evaluate tool usage ONLY based on tool_calls JSON.

HARD RULES:
- If tool_calls is EMPTY and task requires tools → score = 0.0
- If the agent CLAIMS tool usage but tool_calls is empty → score = 0.0
- NEVER infer tool usage from text
- If fake outputs appear without tool calls → score ≤ 0.2

CHECK:
1. Were tools REQUIRED?
2. Are they present in tool_calls?
3. Are results tied to tool outputs?
4. Any mismatch between claims and evidence?

SCORING:
- 1.0: Correct and efficient usage
- 0.7–0.9: Mostly correct
- 0.4–0.6: Partial usage
- 0.2–0.3: Weak or inconsistent
- 0.0–0.1: No tool calls or fake usage

Return JSON:
{{
    "score": float,
    "reasoning": "short explanation"
}}
"""


def evaluate_response_relevance(query: str, response: str, context: str = "") -> Dict[str, Any]:
    """Evaluate how relevant the response is to the query throughout the conversation."""
    prompt = RESPONSE_RELEVANCE_PROMPT.format(
        query=query,
        response=response,
        context=context or "No conversation history available"
    )
    return _call_llm_judge(prompt)


def evaluate_task_completion(
    query: str,
    response: str,
    status: str,
    expected_desc: str = "Task should complete successfully",
    context: str = ""
) -> Dict[str, Any]:
    """Evaluate task completion quality including process and outcome."""
    prompt = TASK_COMPLETION_PROMPT.format(
        query=query,
        response=response,
        status=status,
        expected_desc=expected_desc,
        context=context or "No conversation history available"
    )
    return _call_llm_judge(prompt)


def evaluate_hallucination(
    query: str,
    response: str,
    context: str = ""
) -> Dict[str, Any]:
    """Detect hallucinations in the response."""
    prompt = HALLUCINATION_DETECTION_PROMPT.format(
        query=query,
        response=response,
        context=context or "No additional context provided"
    )
    return _call_llm_judge(prompt)


def evaluate_tool_call_appropriateness(
    query: str,
    response: str,
    status: str,
    context: str = "",
    tool_calls: str = "[]",
    expected_desc: str = "Task should complete successfully",
) -> Dict[str, Any]:
    """Evaluate whether tool calls were appropriate for the task."""
    prompt = TOOL_CALL_APPROPRIATENESS_PROMPT.format(
        query=query,
        response=response,
        status=status,
        context=context or "No conversation history available",
        tool_calls=tool_calls or "[]",
        expected_desc=expected_desc,
    )
    return _call_llm_judge(prompt)


def evaluate_episode_quality(
    query: str,
    response: str,
    status: str,
    context: str = "",
    expected_desc: str = "Task should complete successfully",
    tool_calls: str = "[]",
) -> QualitativeScores:
    """
    Run all qualitative evaluations for an episode.
    Returns QualitativeScores with all metrics.
    """
    scores = QualitativeScores()

    # Response Relevance
    try:
        relevance_result = evaluate_response_relevance(query, response, context)
        if isinstance(relevance_result, dict) and relevance_result.get("score") is not None:
            scores.response_relevance = float(relevance_result["score"])
            logger.debug(f"Relevance evaluation result: {relevance_result}")
            scores.judge_reasoning["relevance"] = relevance_result.get("reasoning", "")
    except Exception as e:
        logger.error(f"Response relevance evaluation failed: {e}")

    # Task Completion Quality
    try:
        completion_result = evaluate_task_completion(query, response, status, expected_desc, context)
        if isinstance(completion_result, dict) and completion_result.get("score") is not None:
            scores.task_completion_quality = float(completion_result["score"])
            logger.debug(f"Task completion evaluation result: {completion_result}")
            scores.judge_reasoning["completion"] = completion_result.get("reasoning", "")
    except Exception as e:
        logger.error(f"Task completion evaluation failed: {e}")

    # Hallucination Detection
    try:
        hallucination_result = evaluate_hallucination(query, response, context)
        if isinstance(hallucination_result, dict) and hallucination_result.get("score") is not None:
            scores.hallucination_score = float(hallucination_result["score"])
            logger.debug(f"Hallucination evaluation result: {hallucination_result}")
            scores.judge_reasoning["hallucination"] = hallucination_result.get("reasoning", "")
    except Exception as e:
        logger.error(f"Hallucination evaluation failed: {e}")

    # Tool-call appropriateness
    try:
        tool_call_result = evaluate_tool_call_appropriateness(
            query=query,
            response=response,
            status=status,
            context=context,
            tool_calls=tool_calls,
            expected_desc=expected_desc,
        )
        if isinstance(tool_call_result, dict) and tool_call_result.get("score") is not None:
            scores.tool_call_appropriateness = float(tool_call_result["score"])
            logger.debug(f"Tool-call appropriateness evaluation result: {tool_call_result}")
            scores.judge_reasoning["tool_call_appropriateness"] = tool_call_result.get("reasoning", "")
    except Exception as e:
        logger.error(f"Tool-call appropriateness evaluation failed: {e}")

    return scores
