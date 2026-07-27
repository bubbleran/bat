"""Centralized OpenTelemetry attribute names and operation values.
"""

# --- GenAI semantic conventions (write side: our manual spans) -------------
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_AGENT_ID = "gen_ai.agent.id"
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

# --- Operation name values -------------------------------------------------
OP_INVOKE_AGENT = "invoke_agent"
OP_INVOKE_WORKFLOW = "invoke_workflow"
OP_EXECUTE_TOOL = "execute_tool"

# --- ADK-specific (custom) attributes --------------------------------------
# Per-turn correlation key: the A2A task id. Paired with gen_ai.conversation.id
# (the A2A context id), it lets consumers (eval engine, UIs) fetch the spans of
# a single turn from the telemetry backend now that usage/tool-call metadata is
# no longer carried inside the A2A message.
BAT_TASK_ID = "bat.a2a.task_id"

# OpenInference/Phoenix session grouping. Each turn (and each interrupt/resume)
# is a separate request -> a separate trace; setting session.id to the A2A
# context id lets Phoenix group all of a conversation's traces under one
# Session.
SESSION_ID = "session.id"

# OpenInference/Phoenix project routing. Set as a *Resource* attribute, it tells
# Phoenix which project to file a trace under (default project: "default").
# Distinct from service.name: service.name only labels spans *within* a project,
# whereas this selects the project itself. NOTE: Phoenix reconstructs a trace
# within a single project, so agents that share a distributed trace (via the
# propagated traceparent) must use the same project or the trace fragments.
OPENINFERENCE_PROJECT_NAME = "openinference.project.name"

# --- OpenInference attributes (read side: Fase C aggregation) --------------
OPENINFERENCE_LLM_TOKEN_PROMPT = "llm.token_count.prompt"
OPENINFERENCE_LLM_TOKEN_COMPLETION = "llm.token_count.completion"
OPENINFERENCE_LLM_TOKEN_TOTAL = "llm.token_count.total"
OPENINFERENCE_LLM_MODEL_NAME = "llm.model_name"
OPENINFERENCE_TOOL_NAME = "tool.name"
