"""OpenTelemetry attribute names and values set on the ADK's own spans."""

# GenAI semantic conventions

#: The operation a span performs, e.g. ``OP_INVOKE_AGENT``.
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
#: The name of the agent the span belongs to.
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
#: The conversation (A2A context) id.
GEN_AI_CONVERSATION_ID = "gen_ai.conversation.id"
#: ``GEN_AI_OPERATION_NAME`` value of a span invoking an agent.
OP_INVOKE_AGENT = "invoke_agent"

#: The A2A task id: with the conversation id, it selects one turn's spans.
BAT_TASK_ID = "bat.a2a.task_id"
#: Final A2A TaskState of a call to another agent, on CallAgentNode's span.
BAT_A2A_TASK_STATE = "bat.a2a.task_state"

#: Groups a conversation's traces into one Phoenix session (the context id).
SESSION_ID = "session.id"
#: Resource attribute selecting the Phoenix project. Agents that share a
#: distributed trace must share a project, or the trace fragments.
OPENINFERENCE_PROJECT_NAME = "openinference.project.name"
#: Span input, shown by Phoenix as a span's Input.
INPUT_VALUE = "input.value"
#: Span output, shown by Phoenix as a span's Output.
OUTPUT_VALUE = "output.value"
