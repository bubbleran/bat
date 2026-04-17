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


def _dimension_prompt(
    *,
    dimension: str,
    guidance: str,
    query: str,
    response: str,
    status: str,
    context: str,
    expected_desc: str,
) -> str:
    return (
        "Score the agent behavior from 0.0 to 1.0 and return JSON only.\n"
        f"Dimension: {dimension}\n"
        f"Guidance: {guidance}\n"
        f"User queries (chronological):\n{query}\n\n"
        f"Expected behavior:\n{expected_desc}\n\n"
        f"Final status: {status}\n\n"
        f"Trace context:\n{context or 'No trace context provided'}\n\n"
        f"Final response:\n{response}\n\n"
        "Output format:\n"
        '{"score": <float 0.0-1.0>, "reasoning": "<short explanation>"}'
    )


def _score_dimension(**kwargs: str) -> Dict[str, Any]:
    prompt = _dimension_prompt(**kwargs)
    return _call_llm_judge(prompt)


def evaluate_episode_quality(
    query: str,
    response: str,
    status: str,
    context: str = "",
    expected_desc: str = "Task should complete successfully",
) -> QualitativeScores:
    scores = QualitativeScores()

    try:
        relevance = _score_dimension(
            dimension="response_relevance",
            guidance="Measure topical relevance of intermediate and final responses.",
            query=query,
            response=response,
            status=status,
            context=context,
            expected_desc=expected_desc,
        )
        if relevance.get("score") is not None:
            scores.response_relevance = float(relevance["score"])
            scores.judge_reasoning["relevance"] = str(relevance.get("reasoning", ""))
    except Exception as exc:
        logger.error(f"Response relevance evaluation failed: {exc}")

    try:
        completion = _score_dimension(
            dimension="task_completion_quality",
            guidance="Measure how completely and correctly the task was solved.",
            query=query,
            response=response,
            status=status,
            context=context,
            expected_desc=expected_desc,
        )
        if completion.get("score") is not None:
            scores.task_completion_quality = float(completion["score"])
            scores.judge_reasoning["completion"] = str(completion.get("reasoning", ""))
    except Exception as exc:
        logger.error(f"Task completion evaluation failed: {exc}")

    try:
        hallucination = _score_dimension(
            dimension="hallucination_score",
            guidance="Measure grounding and factual consistency. 1.0 means no hallucination.",
            query=query,
            response=response,
            status=status,
            context=context,
            expected_desc=expected_desc,
        )
        if hallucination.get("score") is not None:
            scores.hallucination_score = float(hallucination["score"])
            scores.judge_reasoning["hallucination"] = str(hallucination.get("reasoning", ""))
    except Exception as exc:
        logger.error(f"Hallucination evaluation failed: {exc}")

    return scores
