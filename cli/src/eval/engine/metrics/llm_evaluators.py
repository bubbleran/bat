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

RESPONSE_RELEVANCE_PROMPT = """You are an expert evaluator assessing the relevance of an AI agent's response throughout its entire interaction.

**User Queries (chronological):**
{query}

**Full Agent Conversation:**
{context}

**Final Agent Response:**
{response}

**Task:** Evaluate how relevant and on-topic the agent's responses are to the user's queries throughout the entire conversation. You can be harsh, use the whole scale from 0.0 to 1.0, and consider the following:

**Evaluation Instructions:**
1. **Consider the full conversation flow**: Review all intermediate steps and responses
2. **Weight the final response more heavily** (60% weight) but don't ignore intermediate steps (40% weight)
3. **Check for topic drift**: Did the agent stay focused throughout or introduce irrelevant information?
4. **Assess contextual awareness**: Did intermediate responses show understanding of the evolving task and the user's needs?
5. **Evaluate final response relevance**: Does the final response directly address the user's queries without tangents?

**Scoring Criteria:**
- **1.0**: All responses perfectly relevant; final response directly addresses all queries; no tangents
- **0.8-0.9**: Mostly relevant throughout; final response addresses main points; minor tangents in intermediate steps
- **0.6-0.7**: Generally relevant but some intermediate drift; final response mostly on-topic
- **0.4-0.5**: Partial relevance; significant tangents or off-topic intermediate steps; final response somewhat addresses queries
- **0.2-0.3**: Barely relevant; mostly off-topic throughout; final response weakly connected to queries
- **0.0-0.1**: Completely irrelevant conversation; final response unrelated to user queries

Respond ONLY with valid JSON in this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<explanation covering intermediate steps and final response>"
}}"""


TASK_COMPLETION_PROMPT = """You are an expert evaluator assessing task completion quality across the full agent workflow.

**User Queries (chronological):**
{query}

**Expected Behavior:**
{expected_desc}

**Full Agent Conversation:**
{context}

**Final Agent Response:**
{response}

**Final Agent Status:** {status}

**Task:** Evaluate how well the agent completed the requested task, considering both the process and final outcome.

**Evaluation Instructions:**
1. **Analyze the problem-solving process**: Review how the agent progressed through intermediate steps
2. **Weight the final outcome heavily** (70% weight) but consider process quality (30% weight)
3. **Check for corrections and refinements**: Did the agent properly handle feedback and iterate toward the goal?
4. **Validate against expected behavior**: Does the final result meet all stated requirements?
5. **Consider error recovery**: If errors occurred, did the agent recover appropriately?

**Scoring Criteria:**
- **1.0**: Task perfectly completed; efficient process; all requirements met; handled feedback well
- **0.8-0.9**: Task completed successfully; minor inefficiencies or extra iterations; all core requirements met
- **0.6-0.7**: Task mostly completed; some stumbling in process but recovered; minor requirements missing
- **0.4-0.5**: Task partially completed; inefficient process with multiple issues; key requirements missing
- **0.2-0.3**: Task barely progressed; poor process; most requirements unmet
- **0.0-0.1**: Task failed; incorrect approach throughout; requirements completely unmet

Respond ONLY with valid JSON in this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<explanation covering process quality, iterations, and final completeness>"
}}"""


HALLUCINATION_DETECTION_PROMPT = """You are an expert evaluator detecting hallucinations and fabricated information throughout an agent conversation.

**User Queries (chronological):**
{query}

**Full Agent Conversation with Tool Outputs:**
{context}

**Final Agent Response:**
{response}

**Task:** Assess whether the agent introduced hallucinated or fabricated information at any point, with emphasis on the final response.

**Evaluation Instructions:**
1. **Trace information flow**: Check if each claim in the conversation is grounded in user input or tool outputs
2. **Weight final response heavily** (60% weight) but scan all intermediate steps (40% weight)
3. **Identify fabrications**: Look for invented values, false assumptions, or unsupported assertions
4. **Check consistency**: Verify intermediate claims align with available context
5. **Assess appropriate uncertainty**: Does the agent admit when it lacks information?

**Common Hallucination Patterns to Check:**
- Inventing parameter values not provided by user or tools
- Making assumptions about data not in context
- Contradicting earlier statements without justification
- Claiming actions were successful without evidence
- Fabricating tool outputs or system responses
- Intermidiate steps that diverge from user intent without explanation

**Scoring Criteria (inverted - higher is better):**
- **1.0**: Zero hallucinations; all claims grounded in context; appropriate uncertainty when needed
- **0.8-0.9**: Negligible hallucinations; minor assumptions but clearly stated as such; core facts correct
- **0.6-0.7**: Minor hallucinations in intermediate steps but corrected; final response grounded
- **0.4-0.5**: Some fabricated details throughout; mix of correct and unsupported claims in final response
- **0.2-0.3**: Significant hallucinations; multiple false claims; final response includes fabrications
- **0.0-0.1**: Pervasive hallucinations; most claims fabricated; contradicts known facts

Respond ONLY with valid JSON in this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<explanation identifying any hallucinations across intermediate steps and final response>"
}}"""


TOOL_CALL_APPROPRIATENESS_PROMPT = """You are an expert evaluator assessing whether an agent used tools appropriately.

**User Queries (chronological):**
{query}

**Expected Behavior:**
{expected_desc}

**Full Agent Conversation:**
{context}

**Observed Tool Calls (JSON):**
{tool_calls}

**Final Agent Response:**
{response}

**Final Agent Status:** {status}

**Task:** Evaluate whether the selected tools, their frequency, and timing were appropriate for solving the task.

**Evaluation Instructions:**
1. Check tool necessity: were tools used only when needed?
2. Check tool choice: were the chosen tools relevant to the task?
3. Check execution strategy: was the number/order of tool calls reasonable?
4. Check overuse/underuse: did the agent spam tools or avoid needed tool usage?
5. Consider final outcome: if task failed, assess whether tool strategy contributed to failure.

**Scoring Criteria:**
- **1.0**: Tool strategy was optimal; calls were necessary, relevant, and efficient.
- **0.8-0.9**: Mostly good tool usage; minor inefficiencies.
- **0.6-0.7**: Acceptable but with noticeable inefficiency or suboptimal choice.
- **0.4-0.5**: Several inappropriate or redundant calls; strategy only partially adequate.
- **0.2-0.3**: Poor tool strategy; many irrelevant or missing tool calls.
- **0.0-0.1**: Tool usage was fundamentally inappropriate and harmful to task completion.

Respond ONLY with valid JSON in this exact format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<explanation focused on tool choice, necessity, and timing>"
}}"""



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
