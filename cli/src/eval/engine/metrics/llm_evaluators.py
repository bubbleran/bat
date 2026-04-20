"""Lightweight LLM-as-judge qualitative evaluators."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage

from bat.chat_model_client import ChatModelClient, ChatModelClientConfig
from bat.logging import create_logger

from ..contracts import QualitativeScores


logger = create_logger(__name__, level="info")

_JUDGE_SYSTEM = "You are a precise evaluator. Respond with valid JSON only."
_judge_client: Optional[ChatModelClient] = None


def _get_judge_client() -> ChatModelClient:
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
            logger.info(f"LLM judge response: {response.content}")
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


def _prompt_common_context(
    *,
    query: str,
    response: str,
    status: str,
    context: str,
    expected_desc: str,
) -> str:
    expected_outcome = _infer_expected_outcome(expected_desc)
    return (
        f"User queries (chronological):\n{query}\n\n"
        f"Expected behavior:\n{expected_desc}\n\n"
        f"Inferred expected outcome: {expected_outcome}\n\n"
        f"Final status: {status}\n\n"
        f"Trace context:\n{context or 'No trace context provided'}\n\n"
        f"Final response:\n{response}\n\n"
    )


def _infer_expected_outcome(expected_desc: str) -> str:
    text = expected_desc.lower()
    failure_markers = (
        "fail",
        "failure",
        "refuse",
        "reject",
        "blocked",
        "error expected",
        "should not",
        "must not",
        "unsafe",
        "denied",
    )
    success_markers = (
        "success",
        "successfully",
        "complete",
        "completed",
        "solve",
        "resolved",
    )
    if any(marker in text for marker in failure_markers):
        return "expected_failure_or_refusal"
    if any(marker in text for marker in success_markers):
        return "expected_success"
    return "unclear"


def _response_relevance_prompt(
    *,
    query: str,
    response: str,
    status: str,
    context: str,
    expected_desc: str,
) -> str:
    common = _prompt_common_context(
        query=query,
        response=response,
        status=status,
        context=context,
        expected_desc=expected_desc,
    )
    return (
        "You are scoring only: response_relevance.\n"
        "Evaluate how directly the final response addresses the user's request and constraints.\n"
        "Use weighted evidence: final response 80%, intermediate trace context 20%.\n"
        "Ignore factual correctness unless it affects relevance.\n\n"
        "Scoring anchors:\n"
        "- 1.0: Fully on-topic, directly answers the request, no significant off-topic content.\n"
        "- 0.7: Mostly relevant, minor digressions or small missed constraints.\n"
        "- 0.4: Partially relevant, addresses only part of the request or includes notable unrelated content.\n"
        "- 0.1: Mostly irrelevant or mismatched to the request.\n"
        "- 0.0: Completely irrelevant.\n\n"
        f"{common}"
        "Return strict JSON only (no markdown, no extra keys):\n"
        '{"score": <float 0.0-1.0>, "reasoning": "<max 35 words>"}'
    )


def _task_completion_quality_prompt(
    *,
    query: str,
    response: str,
    status: str,
    context: str,
    expected_desc: str,
) -> str:
    common = _prompt_common_context(
        query=query,
        response=response,
        status=status,
        context=context,
        expected_desc=expected_desc,
    )
    return (
        "You are scoring only: task_completion_quality.\n"
        "Evaluate whether the task outcome matches the expected behavior, not whether it merely ended in success.\n"
        "If expected behavior implies refusal/failure, a correct refusal/failure handling should score high.\n"
        "If expected behavior implies success, unresolved errors or refusal should score low.\n"
        "Use weighted evidence: final outcome and final response 70%, intermediate steps/trace quality 30%.\n"
        "Penalize missing key steps, unresolved failures, or outcomes opposite to expectation.\n\n"
        "Scoring anchors:\n"
        "- 1.0: Outcome fully matches expected behavior (including expected refusal/failure when applicable), no material gaps.\n"
        "- 0.7: Mostly correct outcome, minor omissions in process or explanation.\n"
        "- 0.4: Partial alignment; key requirements or checks missing.\n"
        "- 0.1: Outcome mostly opposite to expected behavior.\n"
        "- 0.0: Completely wrong outcome or not completed.\n\n"
        f"{common}"
        "Return strict JSON only (no markdown, no extra keys):\n"
        '{"score": <float 0.0-1.0>, "reasoning": "<max 35 words>"}'
    )


def _hallucination_score_prompt(
    *,
    query: str,
    response: str,
    status: str,
    context: str,
    expected_desc: str,
) -> str:
    common = _prompt_common_context(
        query=query,
        response=response,
        status=status,
        context=context,
        expected_desc=expected_desc,
    )
    return (
        "You are scoring only: hallucination_score.\n"
        "Measure factual grounding and consistency with the provided trace/context.\n"
        "Use weighted evidence: final response 80%, intermediate trace context 20%.\n"
        "Score high when claims are supported, cautious, and non-fabricated.\n\n"
        "Scoring anchors:\n"
        "- 1.0: No unsupported claims; statements align with available evidence/context.\n"
        "- 0.7: Minor speculative or weakly supported claims, low risk.\n"
        "- 0.4: Multiple unsupported claims or inconsistencies.\n"
        "- 0.1: Major fabricated details likely to mislead.\n"
        "- 0.0: Predominantly fabricated or contradictory response.\n\n"
        f"{common}"
        "Return strict JSON only (no markdown, no extra keys):\n"
        '{"score": <float 0.0-1.0>, "reasoning": "<max 35 words>"}'
    )


def _score_dimension(prompt: str) -> Dict[str, Any]:
    return _call_llm_judge(prompt)


def evaluate_episode_quality(
    query: str,
    response: str,
    status: str,
    context: str = "",
    expected_desc: str = "Task should complete successfully",
) -> QualitativeScores:
    scores = QualitativeScores()
    logger.info(f"Evaluating episode quality for query: {query} | status: {status}")
    try:
        relevance = _score_dimension(
            _response_relevance_prompt(
            query=query,
            response=response,
            status=status,
            context=context,
            expected_desc=expected_desc,
            )
        )
        if relevance.get("score") is not None:
            scores.response_relevance = float(relevance["score"])
            scores.judge_reasoning["relevance"] = str(relevance.get("reasoning", ""))
    except Exception as exc:
        logger.error(f"Response relevance evaluation failed: {exc}")

    try:
        completion = _score_dimension(
            _task_completion_quality_prompt(
            query=query,
            response=response,
            status=status,
            context=context,
            expected_desc=expected_desc,
            )
        )
        if completion.get("score") is not None:
            scores.task_completion_quality = float(completion["score"])
            scores.judge_reasoning["completion"] = str(completion.get("reasoning", ""))
    except Exception as exc:
        logger.error(f"Task completion evaluation failed: {exc}")

    try:
        hallucination = _score_dimension(
            _hallucination_score_prompt(
            query=query,
            response=response,
            status=status,
            context=context,
            expected_desc=expected_desc,
            )
        )
        if hallucination.get("score") is not None:
            scores.hallucination_score = float(hallucination["score"])
            scores.judge_reasoning["hallucination"] = str(hallucination.get("reasoning", ""))
    except Exception as exc:
        logger.error(f"Hallucination evaluation failed: {exc}")

    return scores
