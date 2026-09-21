# Telemetry

**BAT-ADK** exports its internals as **OpenTelemetry** spans. Token usage, tool calls, LLM timing and the shape of the graph are no longer collected by hand into the A2A stream — they are read back from the spans.

Two things follow from that:

- Anything that consumes an agent's traces (a backend like Arize Phoenix, or `bat eval`) reads spans, not message metadata.
- An agent that calls another agent propagates the W3C `traceparent` through the A2A message, so **both agents' spans land in the same trace** even though they are separate processes.

## Installation

Telemetry lives behind an extra, so an agent that does not want it does not pay for it:

```toml
dependencies = ["bat-adk[telemetry]"]   # or bat-adk[all]
```

Without the extra the SDK still runs: the tracers degrade to no-ops and nothing is exported.

## Turning it on

Everything is configured in the agent's `config.yaml`, under `telemetry`:

```yaml
telemetry:
  # service_name: my-agent      # optional; defaults to the agent card name
  # project_name: my-agent      # optional; Phoenix project (default: "default")
  privacy: none                 # none | content | names | full
  output:
    - type: remote
      endpoint: http://localhost:6006
    - type: local
      file_path: spans.jsonl
```

**Telemetry is on when `output` has at least one entry.** Spans fan out to *every* entry, so a run can go to a collector and a file at once.

| `type` | destination |
|---|---|
| `remote` | OTLP/HTTP collector, e.g. Arize Phoenix. `endpoint` defaults to `http://localhost:6006` |
| `local` | JSON Lines file, one span per line. `file_path` defaults to `spans.jsonl` |
| `console` | stdout, for debugging |

An unknown `type` is skipped with a warning rather than disabling the whole pipeline.

`project_name` is distinct from `service_name`: the first is the Phoenix project the trace is filed under, the second labels the spans within it. **Agents that share a distributed trace must use the same `project_name`**, or the trace fragments across projects.

## Privacy

By default a span carries prompts, completions and tool definitions in full. `telemetry.privacy` decides how much of that may leave the process. It is a single ordered dial, and each level redacts everything the level below it does:

| level | redacts |
|---|---|
| `none` | nothing — the default |
| `content` | prompts, messages, completions, invocation parameters, and every tool's description, parameter schema and call arguments |
| `names` | also span names, i.e. the LangGraph node names. Span kinds (`LLM`/`CHAIN`/`TOOL`) replace them, so the trace keeps its shape |
| `full` | also tool names |

Redacted values are replaced with `__REDACTED__` before any exporter sees them; they never leave the process.

**Token counts, span kinds, hierarchy and timing survive at every level**, so cost accounting keeps working even at `full`. The one thing `full` costs you is the eval engine's tool-call metrics, which key off the tool name — use it only when the tool inventory itself is considered proprietary.

The level is written as its name (`content`) or its ordinal (`1`). An unknown value is a **validation error**, not a silent fallback to `none`: a typo must not export in the clear precisely when someone was trying to lock the agent down.

### Build-time floor

`config.yaml` is editable at runtime — a mounted ConfigMap, a replaced file or `CONFIG_PATH` pointing elsewhere can all turn privacy back off. For an agent shipped as a packaged artifact that makes it a default, not a guarantee.

```bash
bat build --telemetry-privacy content
```

bakes a **minimum** level into the frozen binary. The effective level is `max(floor, config.yaml)`: a replaced `config.yaml` can raise privacy but never lower it. Undoing it means decompiling the binary, not editing a mounted file. Without the flag there is no floor and `config.yaml` is the sole authority.

## What gets instrumented

- **LangChain / LangGraph** — automatically, through OpenInference. This is where token counts, prompts, completions and tool calls come from.
- **`AgentExecutor`** — one root `invoke_agent` span per request, carrying the conversation and task ids; it continues an incoming trace when the caller propagated one.
- **`CallAgentNode`** — a `CLIENT` span around the remote call, injecting `traceparent` into the outgoing A2A message.

## How `bat eval` uses it

`bat eval run` enables the `local` exporter for the duration of the run, pointing each episode at its own spans directory, and restores `config.yaml` afterwards. It then groups spans by the trace whose root carries the episode's conversation id and aggregates them — **including spans written by remote sub-agents in their own processes**, which is how multi-agent token usage is recomposed.

This is also why `privacy: full` empties the tool-call metrics: the eval keys them off the tool name.

## Environment variables

Telemetry itself is configured only from `config.yaml`. The surrounding knobs are:

| variable | effect |
|---|---|
| `CONFIG_PATH` | path to the agent's `config.yaml` (default `./config.yaml`) |
| `LOG_LEVEL` | SDK log verbosity |
