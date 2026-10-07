# Eval Trajectory Digest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `bat eval` judges *how* an agent reached its answer — model outputs, tool calls, calls to other agents, failures — read from the agent's OpenTelemetry spans and condensed into a digest small enough for an on-prem judge.

**Architecture:** A pure `trajectory.py` module turns the spans of one conversation into a per-turn, time-ordered digest (model outputs kept, model inputs dropped). The adapter stores it on each episode; the evaluator adds span-based deterministic checks; the judge reads a budgeted text rendering of it. A small ADK change makes agent-to-agent calls and errors visible in spans.

**Tech Stack:** Python 3.12, pydantic v2, OpenInference/OTel span JSON (bat-adk `JsonFileSpanExporter`), LangChain `ChatModelClient` for the judge, pytest.

**Spec:** the "Design" section below (agreed in conversation on 2026-09-30; also in memory `eval-trajectory-digest`).

## Design

- **A2A stays the source for** final status, final answer and input-required prompts. With the current ADK an agent streams only what its `to_task_result()` returns, so intermediate A2A events carry almost nothing.
- **The trajectory comes from spans**, condensed by code, not by an LLM summariser. Per turn (one trace per request, ordered by the root span's start), in time order:
  - `model`: node (`metadata.langgraph_node`), model name, tokens (in, out, reasoning), `said` (output message text), `called` (tool-call names the model decided on). Inputs are dropped — they are the history the digest already holds.
  - `tool`: name, args, result, error.
  - `agent`: agent name, `asked`, `answered`, remote status, error, and the called agent's own steps nested one level deep.
  - `error`: a failing span with no failing descendant (so one failure is reported once).
  - Totals across the whole conversation, sub-agents included.
- Every text field is capped (`…[truncated: N chars]`); the judge rendering has an overall budget and elides middle steps when over it.
- **Missing spans fail loudly** when a task asks for a span-based check, instead of reporting "called 0×".
- **ADK**: `CallAgentNode`'s CLIENT span records the request (`input.value`), the answer (`output.value`), the remote state (`bat.a2a.task_state`) and ERROR status; the exporter-side redaction masks `input.value`/`output.value` at `content`; the file exporter writes the status description and exception events.
- **Judge**: trajectory-aware prompts; tool results and other agents' answers are legitimate sources for groundedness; robust JSON extraction (thinking text, `</think>`, fences); scores outside [0, 1] rejected.

## Global Constraints

- CLI floor stays `bat-adk>=2026.9.29a0`; the digest must degrade gracefully on spans from that release (no `asked`/`answered`, no error messages).
- Eval-time privacy is `none` (the eval runs from source and replaces the telemetry block); the digest must still not crash on `__REDACTED__` values.
- `EpisodeTrace.events` and `EpisodeTrace.tool_calls` stay (plots, metrics and existing episode readers use them).
- Line length 80, ruff rules of each package.

## Review Focus

- A conversation with several turns: each turn's steps land under the right user message, even when a turn produced no spans.
- A tool that fails and is retried: two tool steps, one error, no duplicate error from the enclosing node spans.
- A sub-agent that is itself traced into the same file: its steps nest under the agent step, not at top level.
- Huge tool results (catalogs, logs): capped per field; the judge rendering stays under budget.
- A judge that "thinks" before answering (qwen3, deepseek-r1): the score is still parsed.

## File Structure

- `adk/src/bat/prebuilt/call_agent_node.py` — record request/answer/state/error on the CLIENT span.
- `adk/src/bat/telemetry/attributes.py` — `BAT_A2A_TASK_STATE`, `INPUT_VALUE`, `OUTPUT_VALUE` constants.
- `adk/src/bat/telemetry/redaction.py` — `input.value`/`output.value` are content.
- `adk/src/bat/telemetry/file_exporter.py` — `status_description`, `events`.
- `cli/src/eval/engine/trajectory.py` (new) — spans → `Trajectory`; `render_trajectory()`.
- `cli/src/eval/engine/contracts.py` — `Trajectory*` models, `EpisodeTrace.trajectory`, new `TaskExpected` fields.
- `cli/src/eval/engine/adapter.py` — read spans once, build usage, tool calls and trajectory.
- `cli/src/eval/engine/evaluator.py` + `bench_runner.py` — span-based checks.
- `cli/src/eval/engine/metrics/llm_evaluators.py`, `qualitative_helpers.py`, `orchestrator.py` — judge prompts, parsing, context from the trajectory.
- `cli/src/eval/engine/eval_config.py`, `cli/src/eval/commands.py` — `judge.max_trajectory_chars`.
- Tests: `adk/tests/...`, `cli/tests/test_eval_trajectory.py`, `cli/tests/test_eval_evaluator.py`, `cli/tests/test_eval_judge.py`, fixtures under `cli/tests/fixtures/spans/`.

---

### Task 1: ADK — agent calls and errors visible in spans

**Files:** `call_agent_node.py`, `attributes.py`, `redaction.py`, `file_exporter.py`; tests `adk/tests/prebuilt/test_call_agent_span.py`, `adk/tests/telemetry/test_redaction.py`, `adk/tests/telemetry/test_file_exporter.py`.

- [x] Failing tests: the CLIENT span of a streamed call carries `input.value` = request text, `output.value` = last non-empty answer text, `bat.a2a.task_state` = final state name; a stream that raises ends the span with status ERROR and the message as description; `redact_attributes` masks `input.value`/`output.value`; the file exporter writes `status_description` and exception events.
- [x] Implement; run the ADK suite.

### Task 2: Real span fixtures

- [x] Probe blueprint (two agents: a ReActLoop supervisor with a tool and a `CallAgentNode` to a second agent with its own tool) on a local Ollama model; run `bat eval`, keep the episode's `spans-*/` files.
- [x] Trim to one conversation, store as `cli/tests/fixtures/spans/*.jsonl`; note the exact attribute names used.

### Task 3: `trajectory.py`

**Interfaces — Produces:**
- `build_trajectory(spans: list[dict], conversation_id: str, turns: list[str], *, max_field_chars: int = 1500) -> Trajectory`
- `render_trajectory(trajectory: Trajectory, *, max_chars: int = 24000) -> str`
- models in `contracts.py`: `ModelStep`, `ToolStep`, `AgentStep`, `ErrorStep` (discriminated by `kind`), `TrajectoryTurn(user, steps)`, `TrajectoryTotals`, `Trajectory(found, turns, totals, truncated)`.

- [x] Failing tests against the fixtures: step order and kinds; node labels; `said`/`called` from output messages (string and content-block forms); tool args/result; agent step with nested steps; error reported once; totals; truncation marker; turns mapped by order; spans of other conversations ignored; `found=False` on no spans.
- [x] Rendering tests: labels, budget respected with an elision marker.
- [x] Implement.

### Task 4: Adapter + evaluator

- [x] Failing tests: `EpisodeTrace.trajectory` filled from the spans dir; `no_errors`, `max_model_calls`, `max_tokens`, `agent_calls` checks; a span-based check with `found=False` fails with a "no spans" reason and suppresses the misleading per-check failures.
- [x] Implement (`BatA2AAdapter._collect_from_spans` returns the trajectory too; `EpisodeEvaluator.evaluate(..., trajectory=None)`).

### Task 5: Judge

- [x] Failing tests: `_parse_judge_json` handles fenced JSON, prose around JSON, `<think>…</think>`, a bare `</think>` preamble; out-of-range scores rejected; the qualitative context is the rendered trajectory when spans were found, the A2A transcript otherwise; `judge.max_trajectory_chars` parsed from eval.yaml.
- [x] Rewrite the four prompts to read the trajectory; implement.

### Task 6: Docs, template, end-to-end

- [x] `bat eval init` template and `cli/docs/bat-cli.md`: new expected fields, trajectory, judge settings, on-prem judge notes; ADK docs: what `CallAgentNode` records.
- [x] Re-run the probe with the local judge; read the episode JSON and judge reasoning.
