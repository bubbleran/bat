# OpenTelemetry Telemetry in the ADK (`bat-adk`)

Reference document for the OpenTelemetry (OTel) integration in the ADK: why it was
done, what changed file by file (with the reasons), what the various components are,
how to use it, and finally what can be improved, the alternatives, and the next
steps.

---

## Table of contents

1. [Context and goal](#1-context-and-goal)
2. [Architecture at a glance](#2-architecture-at-a-glance)
3. [What changed (file by file)](#3-what-changed-file-by-file)
4. [Key components explained](#4-key-components-explained)
5. [How to use it](#5-how-to-use-it)
6. [Instrumentation alternatives](#6-instrumentation-alternatives)
7. [Export and backend alternatives](#7-export-and-backend-alternatives)
8. [Next steps](#8-next-steps)

---

## 1. Context and goal

Before this integration the ADK collected metadata (tokens, tool calls, timing)
**by hand**: each component (`ChatModelClient`, `ReActLoop`, `CallAgentNode`)
instantiated a `MetadataCollector`, timed the calls with `time.time()`, summed
tokens and de-duplicated tool calls. Those data were then injected into the A2A
protocol `metadata` (`usage` + `trace`) and read back by the eval engine.

Limits of the manual approach:

- timing/aggregation/dedup logic duplicated in several places, fragile;
- no distributed tracing: an agent→agent (A2A) call was not correlated into a
  single trace;
- no export to an observability backend;
- custom naming (`input_tokens`, `inference_time`) not interoperable with the GenAI
  ecosystem.

**Goal:** make OpenTelemetry the **single source of truth** for telemetry,
replacing the manual collection and exporting to **Arize Phoenix**.

**Decisions taken:**

- **Scope:** *full* replacement. The legacy `MetadataCollector` (and the
  `usage`/`trace` A2A message metadata it fed) has been **removed**: token usage
  and tool calls are now derived exclusively from OpenTelemetry spans. The eval
  engine reads them back from the spans the agent writes (see the `file`
  exporter in [§5](#5-how-to-use-it) and the read-back in
  [§3](#3-what-changed-file-by-file)).
- **Instrumentation:** *hybrid* — **OpenInference auto-instrumentation** for
  LangChain/LangGraph **+ manual spans** at the ADK-specific boundaries (executor,
  remote A2A calls).
- **Backend:** **Arize Phoenix** over OTLP/HTTP, runnable locally; but the setup is
  vendor-neutral (see [§7](#7-export-and-backend-alternatives)).
- **Fully opt-in:** without `TELEMETRY_ENABLED=1` everything is a no-op and the ADK
  behaves exactly as before (no spans, no exporters). The OTel dependencies live
  in an optional extra.

---

## 2. Architecture at a glance

The integration is split into two implemented "phases":

- **Phase A — Infrastructure + auto-instrumentation.** A new `bat.telemetry` module
  configures the `TracerProvider`, the OTLP exporter to Phoenix and enables the
  **OpenInference** auto-instrumentation of LangChain/LangGraph. This *automatically*
  captures LLM calls, graph nodes and tool calls.
- **Phase B — Manual spans + context propagation.** At the only points the
  auto-instrumentation cannot see (the ADK itself) we create manual spans following
  the GenAI semantic conventions: the root `invoke_agent` span in the executor and
  the `invoke_agent` (CLIENT) span on agent→agent calls, with W3C `traceparent`
  propagation through the A2A message metadata.

```
invoke_agent <agent>             (manual, Phase B — SpanKind.INTERNAL)
  ├─ ChatOpenAI / chat ...       (auto, Phase A — OpenInference)
  ├─ <LangGraph nodes> ...       (auto, Phase A — OpenInference)
  └─ invoke_agent <remote>       (manual, Phase B — SpanKind.CLIENT)
        └─ [remote process]       (same trace_id via traceparent)
```

Everything is exported in **OTLP** to Phoenix (or any OTLP-compatible backend).

---

## 3. What changed (file by file)

### Created files — the `bat/telemetry/` module

#### `src/bat/telemetry/__init__.py`
Public facade of the module, deliberately minimal. Re-exports:
`setup_telemetry`, `get_tracer`, `inject_context`, `extract_context`, `is_enabled`,
`SpanKind`, `TelemetryConfig` and the `attributes` submodule.
**Why:** a single entry point; all exports are safe both with and without the
`telemetry` extra installed, and both with telemetry enabled and disabled.

#### `src/bat/telemetry/attributes.py`
Centralized registry of span attribute names and operation values. Two families:

- **Write side (our manual spans) — GenAI semantic conventions:**
  `GEN_AI_OPERATION_NAME` (`gen_ai.operation.name`), `GEN_AI_PROVIDER_NAME`,
  `GEN_AI_AGENT_NAME`, `GEN_AI_AGENT_ID`, `GEN_AI_CONVERSATION_ID`,
  `GEN_AI_REQUEST_MODEL`, `GEN_AI_RESPONSE_FINISH_REASONS`,
  `GEN_AI_USAGE_INPUT_TOKENS`, `GEN_AI_USAGE_OUTPUT_TOKENS`, `GEN_AI_TOOL_NAME`.
- **Operation values:** `OP_INVOKE_AGENT` (`invoke_agent`), `OP_INVOKE_WORKFLOW`,
  `OP_EXECUTE_TOOL`.
- **ADK-specific (custom) attributes:** `BAT_TASK_ID` (`bat.a2a.task_id`, the
  per-turn A2A task id) and `SESSION_ID` (`session.id`, set to the A2A context
  id so Phoenix groups all of a conversation's traces under one Session).
- **Read side (to read OpenInference spans when reconstructing usage):**
  `OPENINFERENCE_LLM_TOKEN_PROMPT` (`llm.token_count.prompt`),
  `OPENINFERENCE_LLM_TOKEN_COMPLETION`, `OPENINFERENCE_LLM_TOKEN_TOTAL`,
  `OPENINFERENCE_LLM_MODEL_NAME`, `OPENINFERENCE_TOOL_NAME`.

**Why:** if the (still partly experimental) semantic conventions change, only *one
file* needs updating. The write/read split lets the two specs evolve independently.

#### `src/bat/telemetry/config.py`
Resolves configuration from environment variables. Exposes the `TelemetryConfig`
dataclass with `from_env()`. Variables read:

| Variable | Effect | Default |
|---|---|---|
| `TELEMETRY_ENABLED` | master switch (opt-in) | `false` |
| `OTEL_SERVICE_NAME` | `service.name` | agent card name |
| `PHOENIX_COLLECTOR_ENDPOINT` | base endpoint (checked first) | `http://localhost:6006` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | base endpoint (fallback) | — |
| `PHOENIX_API_KEY` | sent as `Authorization: Bearer` header | — |
| `OTEL_TRACES_EXPORTER` | `otlp` or `console` | `otlp` |

The final traces endpoint is `<base>/v1/traces` (OTLP/HTTP). Boolean parsing uses a
strict allowlist (`1/true/yes/on`) to avoid accidental enablement.
**Why:** a single source of truth for endpoint, headers and flags; it reads both the
Phoenix variables and the standard OTel ones, so it's already ready for other
backends.

#### `src/bat/telemetry/setup.py`
The OpenTelemetry bootstrap and the helpers. Main components:

- **`setup_telemetry(service_name=None) -> bool`**: idempotent. If telemetry is
  disabled or the dependencies are missing, it is a no-op and returns `False` (logs
  a warning guiding to install the extra). Otherwise it creates
  `Resource(service.name=...)`, `TracerProvider`, a `BatchSpanProcessor` with an
  `OTLPSpanExporter` (HTTP) — or a `ConsoleSpanExporter` if
  `OTEL_TRACES_EXPORTER=console` — sets the global tracer provider and enables
  `LangChainInstrumentor().instrument(...)` from OpenInference. Finally it applies
  the `_patch_openinference_langgraph_callbacks()` shim. It also registers
  `shutdown_telemetry()` with `atexit` to flush the exporters on exit.
- **`get_tracer(name)`**: returns a real tracer, or a `_NoopTracer` if OTel is not
  installed. Even with OTel installed but not initialized, it yields non-recording
  spans → always safe at the call site.
- **`inject_context(carrier, span=None)`**: injects the W3C `traceparent` into the
  carrier; with an explicit `span` it uses *that* span's context instead of the
  ambient one (crucial for async generators — see [§4](#4-key-components-explained)).
- **`extract_context(carrier)`**: extracts a context from a carrier; returns `None`
  if unavailable (never raises).
- **`SpanKind`**: the real OTel enum, or a stub with the same constants if OTel is
  not installed (so call sites can reference it unconditionally).
- **`_NoopSpan` / `_NoopTracer`**: fallbacks implementing the minimal interface
  (`start_span`, `start_as_current_span`, `set_attribute`, `record_exception`,
  `end`, context manager). They guarantee the instrumentation code never raises
  `AttributeError`.
- **`_patch_openinference_langgraph_callbacks()`**: defensive shim (see [§4](#4-key-components-explained)).
- **`shutdown_telemetry()`**: flushes and shuts the provider down; registered with
  `atexit` so the `BatchSpanProcessor` does not drop buffered spans at exit.

**Why:** import-safe and no-op by design; zero runtime impact when telemetry is off
or the extra is not installed.

`setup_telemetry` registers `shutdown_telemetry()` with `atexit` (also exported
for explicit calls). On interpreter exit it flushes and shuts the provider down,
so the `BatchSpanProcessor` used by the OTLP/console exporters does not drop its
buffered spans; it is idempotent and a no-op when telemetry was never set up.

#### `src/bat/telemetry/file_exporter.py`
A minimal `SpanExporter` that appends each finished span to a file as one JSON
object per line (JSON Lines). Selected with `OTEL_TRACES_EXPORTER=file` and
`OTEL_FILE_EXPORTER_PATH=<path>`; paired with a `SimpleSpanProcessor` so spans
hit disk **synchronously** the moment they end (no batch/flush delay).
**Why:** the eval engine runs the agent as a subprocess, so an in-memory
exporter cannot reach its spans — a file is the cross-process equivalent. The
eval reconstructs per-episode token usage and tool calls from these files (this
is what replaced the A2A `usage`/`trace` metadata).

#### Removed: the legacy `MetadataCollector`
`src/bat/chat_model_client/metadata.py` (the `MetadataCollector` plus the
`UsageMetadata`/`TraceMetadata` data classes) has been **deleted**, along with
all its call sites: the per-call usage collection in `ChatModelClient`, the
tool-call collection in `ReActLoop`, the metadata observation in
`CallAgentNode.consume_agent_stream`, and the (already unused) aggregation
helpers in `AgentGraph`. The executor no longer injects `usage`/`trace` into the
A2A messages. Spans are now the only source of this data.

### Modified files

#### `pyproject.toml`
Added the optional **`telemetry`** extra:
`opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`,
`openinference-instrumentation-langchain`. Also added to the `all` meta-extra.
**Why:** keep the base install lightweight; telemetry is pulled only with
`pip install 'bat-adk[telemetry]'`. Installation is decoupled from enablement
(`TELEMETRY_ENABLED`).

#### `src/bat/agent/application.py`
Imports `setup_telemetry` and calls it in `AgentApplication.__init__`, after loading
the agent card, passing `service_name=self._agent_card.name`.
**Why:** a single initialization at startup; the agent card name becomes the OTel
`service.name`. If telemetry is off, it's a no-op (backward compatible).

#### `src/bat/agent/_executor.py`
- Module-level tracer `tracer = get_tracer(__name__)`.
- `_carrier_from_message(message)`: defensively extracts a `{str: str}` dict from the
  incoming A2A message metadata (it may contain the `traceparent`); returns `{}` on
  any error (telemetry must never break the request).
- The agent name is derived from the graph class name (the `Graph` suffix dropped),
  so it travels with the graph rather than being passed in.
- In `execute()` the streaming loop is wrapped in
  `tracer.start_as_current_span("invoke_agent <agent>", context=parent_ctx,
  kind=SpanKind.INTERNAL)`, with attributes `gen_ai.operation.name`,
  `gen_ai.agent.name`, `gen_ai.conversation.id` (= `task.context_id`),
  `session.id` (= `task.context_id`) and `bat.a2a.task_id` (= `task.id`).
  `parent_ctx` is obtained from `extract_context(_carrier_from_message(...))` to
  continue a trace propagated by a calling agent. `session.id` + `bat.a2a.task_id`
  are the keys consumers use to fetch a turn's spans now that usage/tool-call
  metadata is no longer carried in the A2A message.

**Why the context manager is safe here:** `execute()` is a **regular coroutine**
(not an async generator). The `with` opens and closes in the same task, around the
`async for` that awaits the entire stream before returning — no `yield` to an
external driver, hence no context attach/detach issue. `SpanKind.INTERNAL` is correct
because `execute()` is invoked internally by the framework, it's not the entry point
of an external RPC.

#### `src/bat/prebuilt/call_agent_node.py`
- Module-level tracer.
- `_inject_traceparent(message, span=None)`: serializes `span`'s context into the
  `traceparent` and updates `message.metadata`; logs and suppresses any error.
- In `consume_agent_stream` the `invoke_agent <remote>` span is created with
  **`tracer.start_span(...)` (NOT `start_as_current_span`)**, `kind=SpanKind.CLIENT`,
  sets the `gen_ai.*` attributes, injects the `traceparent` into the outgoing message
  by passing the span **explicitly**, iterates the stream, on error calls
  `span.record_exception(e)`, and closes with `span.end()` in a `finally`.

**Why `start_span` and not `start_as_current_span`:** `consume_agent_stream` is an
**async generator** driven by a separate task (a background worker + queue). Using
`start_as_current_span` as a context manager would attach a context token
(`contextvars`) around the `yield`s; when the generator is resumed/closed in a
*different task*, OTel raises `ValueError: Token was created in a different Context`
(logged as "Failed to detach context"). By creating the span manually and injecting
the context *from the span* (a plain Python object, not `contextvars`-dependent) the
problem disappears. This is the fix for a bug actually hit while running two agents.

---

## 4. Key components explained

**No-op safety.** The entire `bat.telemetry` module is designed to have no effect
when: (a) the `telemetry` extra is not installed (`_OTEL_AVAILABLE=False` →
`_NoopTracer`), or (b) `TELEMETRY_ENABLED` is not set (`setup_telemetry` returns
`False` and does not initialize the provider). In both cases the call sites
(`tracer.start_as_current_span(...)`, `inject_context(...)`, etc.) work without
raising and at negligible cost.

**Context propagation (W3C `traceparent`).** The W3C Trace Context standard travels
in a `traceparent` header/field. In multi-agent the channel is the **A2A message
metadata**: the caller (`CallAgentNode`) injects it on the way out, the callee
(`MinimalAgentExecutor`) extracts it on the way in. Result: a **single `trace_id`**
crossing multiple processes → one end-to-end trace in Phoenix, with the remote's
execution nested under the caller's CLIENT span. Each agent keeps its own
`service.name`, so it remains a distinct entity (not "squashed").

**The async-generator context bug.** This is why `consume_agent_stream` uses
`start_span`. General rule: **never use `start_as_current_span` around a `yield`** in
an async generator that may be resumed/closed in a different task. Either create the
span manually (as now), or wrap only the synchronous section with no yield.

**The `_patch_openinference_langgraph_callbacks` shim.** LangGraph 1.x dispatches
`on_interrupt` / `on_resume` callbacks (fired by human-in-the-loop `interrupt()`
and its resume), but the OpenInference instrumentation (≤ 0.1.66) — being a
callback handler — does not implement them and logs a noisy `AttributeError` on
every interrupt or resume. The shim adds no-op `on_interrupt` / `on_resume`
methods to the tracer class *only if missing*. It is defensive (try/except,
`hasattr`) and should be removed when OpenInference ships these methods
(see [§8](#8-next-steps)).

---

## 5. How to use it

**Installation** (the `telemetry` extra, plus Phoenix for the local server):
```bash
pip install 'bat-adk[telemetry]'
pip install arize-phoenix     # only to run Phoenix locally
```

**Start Phoenix** (UI at http://localhost:6006):
```bash
phoenix serve
```

**Environment variables** before launching the agent:
```bash
export TELEMETRY_ENABLED=1                              # opt-in (mind the trailing D!)
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 # where to export
export OTEL_INSTRUMENTATION_A2A_SDK_ENABLED=false       # silence the a2a.server.* spans (optional)
```
Optional: `OTEL_SERVICE_NAME=<name>` (default = agent card name),
`OTEL_TRACES_EXPORTER=console` (print spans to the log instead of exporting them).

**File exporter (used by the eval engine).** With
`OTEL_TRACES_EXPORTER=file` and `OTEL_FILE_EXPORTER_PATH=<path>` the agent writes
each finished span as one JSON line to `<path>` (synchronously). The `bat eval`
runner sets this automatically: it points the agent at
`<output>/<task_id>/spans-<i>/agent.jsonl` and then reads **every `*.jsonl` in
that directory**, grouping spans by `trace_id`, to reconstruct each episode's
token usage and tool calls. For a **multi-agent** eval, point each remote
sub-agent's `OTEL_FILE_EXPORTER_PATH` at a distinct file in that same directory:
the shared `trace_id` (carried by the propagated `traceparent`) ties their spans
to the entry agent's trace, so their tokens are recomposed into the totals.

> Note: `OTEL_INSTRUMENTATION_A2A_SDK_ENABLED` is an **a2a-sdk** variable, not an ADK
> one: the a2a-sdk's own OTel instrumentation hooks into our tracer provider and
> produces many `a2a.server.*` spans; setting it to `false` leaves only the ADK +
> OpenInference spans.

**Launch** and verify: at startup the log must show
```
Telemetry enabled (OTLP exporter -> http://localhost:6006/v1/traces, service=<agent>)
```
Spans appear in Phoenix **only after** the agent handles a request (nothing is
produced at startup).


---

## 6. Instrumentation alternatives

This section compares the options available (2026) for instrumenting a Python
**LangChain + LangGraph** application with OpenTelemetry. The adopted choice is the
**hybrid OpenInference (auto) + manual spans**, with a hand-rolled `TracerProvider`
setup in `setup.py`.

A cross-cutting point: **LangGraph `interrupt()` (human-in-the-loop) is not covered by
LangChain's callback system** — see the final subsection.

### 1. OpenInference — `openinference-instrumentation-langchain` (Arize) — *adopted choice*

A tracing spec for LLM apps maintained by Arize, built on top of OTel. The
instrumentor hooks into the LangChain/LangGraph callback system and produces
OTel-compatible spans.

It uses *its own* namespaces (`llm.*`, `openinference.*`): `openinference.span.kind`
(`LLM`/`CHAIN`/`TOOL`/`RETRIEVER`/`AGENT`), tokens
`llm.token_count.prompt/completion/total` (with cache and reasoning granularity),
messages `llm.input_messages.*`/`llm.output_messages.*`, model
`llm.model_name`/`llm.provider`.

**Pros:** rich data model for LLMs/agents (span.kind, messages, tools, retrievers);
token granularity beyond the current OTel conventions; first-class Phoenix
integration; still OTLP-standard, hence exportable anywhere; zero-code activation.
**Cons:** **non-standard OTel** namespaces (`llm.token_count.total` ≠ `gen_ai.usage.*`),
so generic backends show the spans but don't recognize them as GenAI metrics;
ecosystem oriented to Arize/Phoenix; does not emit the OTel GenAI metrics.

Sources: [OpenInference Semantic Conventions](https://arize-ai.github.io/openinference/spec/semantic_conventions.html) · [GitHub](https://github.com/Arize-ai/openinference) · [Arize AX docs](https://arize.com/docs/ax/observe/tracing-concepts/openinference-semantic-conventions)

### 2. OpenLLMetry — `opentelemetry-instrumentation-langchain` (Traceloop)

A suite of OTel instrumentation from Traceloop (Apache 2.0), 20+ providers/frameworks.
It primarily uses `gen_ai.*`, in a historical variant: tokens
`gen_ai.usage.prompt_tokens`/`completion_tokens` (+`llm.usage.*`), prompt/completion
`gen_ai.prompt`/`gen_ai.completion`, context `traceloop.*`.

**Pros:** closer to the OTel `gen_ai.*` conventions → more portability (Datadog,
Honeycomb, New Relic) and less vendor lock-in; "two-line" SDK; broad coverage; the
Traceloop conventions partly merged into OTel.
**Cons:** **naming mismatch** between the historical `prompt_tokens/completion_tokens`
and the now-canonical OTel `input_tokens/output_tokens`; the agent/chain model is less
expressive than OpenInference; Phoenix integration is not native.

Sources: [GitHub](https://github.com/traceloop/openllmetry) · [GenAI Semantic Conventions](https://www.traceloop.com/docs/openllmetry/contributing/semantic-conventions) · [Dynatrace](https://www.dynatrace.com/knowledge-base/openllmetry/)

### 3. Manual spans following the OTel GenAI semantic conventions (no library)

You create spans by hand with the OTel SDK: `gen_ai.operation.name`,
`gen_ai.request.model`, `gen_ai.usage.input_tokens`/`output_tokens`,
`gen_ai.response.finish_reasons`, plus the `gen_ai.client.token.usage` metric.
Maturity status: still **Development/experimental** in 2026; opt-in via
`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`.

**Pros:** total control (cardinality, noise); maximum portability ("pure" standard);
no third-party dependency; the **only way to explicitly cover** points the
auto-instrumentation can't see (e.g. a span around `interrupt()`/resume).
**Cons:** high development/maintenance cost (hooking into the callbacks by hand);
still-experimental conventions (risk of breaking changes); easy to diverge and to
lose the out-of-the-box richness.

> It's precisely because option 3 is needed for the ADK-specific boundaries that we
> chose the **hybrid**: OpenInference (option 1) for base coverage + targeted manual
> spans (executor, A2A).

Sources: [GenAI client spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) · [GenAI overview](https://opentelemetry.io/docs/specs/semconv/gen-ai/) · [GenAI metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/)

### 4. `phoenix.otel.register(auto_instrument=True)` vs manual `TracerProvider` setup

This is not a conventions alternative but a **bootstrap mode** (orthogonal to 1–3).
`register()` reads the Phoenix env vars and configures provider/processor/exporter
with Phoenix-aware defaults; with `auto_instrument=True` it activates *all* installed
OpenInference instrumentors.

**Pros (`register`):** almost zero boilerplate, automatic reading of the Phoenix env
vars. **Cons (`register`):** implicit behavior (activates everything),
Phoenix-oriented defaults, explicit `shutdown()` handling with `batch=True`.
**Pros (manual setup, our choice in `setup.py`):** explicit control over exporters,
sampler, resource attributes and **custom processors** (needed for Phase C);
independent of the Phoenix SDK → any OTLP backend.
**Cons (manual setup):** more boilerplate; each instrumentation must be activated by
hand (`LangChainInstrumentor().instrument(...)`).

> **We chose the manual setup** precisely so we can later add a custom `SpanProcessor`
> (Phase C) and stay backend-agnostic.

Sources: [Phoenix — OTEL setup](https://arize.com/docs/phoenix/tracing/how-to-tracing/setup-tracing/setup-using-phoenix-otel) · [arize-phoenix-otel](https://pypi.org/project/arize-phoenix-otel/) · [register() reference](https://arize-phoenix.readthedocs.io/projects/otel/en/latest/api/register.html)

### The `on_interrupt` / LangGraph callback gap

All auto-instrumentation hooks into LangChain's callbacks (`BaseCallbackHandler`),
which **does not provide `on_interrupt`**. LangGraph's HITL interruptions live on a
different plane: `interrupt()` saves state and raises `GraphInterrupt`, outside the
callback chain. Observed consequences: tools that call `interrupt()` marked as
**ERROR**, and interrupt→resume cycles generating **separate traces** instead of a
single trace. Implication: to trace HITL pauses you need **manual instrumentation**
around the `interrupt()`/`Command(resume=...)` boundaries — another argument for the
hybrid approach.

Sources: [BaseCallbackHandler](https://python.langchain.com/api_reference/core/callbacks/langchain_core.callbacks.base.BaseCallbackHandler.html) · [LangGraph Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts) · [Langfuse issue #10962](https://github.com/langfuse/langfuse/issues/10962)

### Comparison summary

| Aspect | OpenInference (1) | OpenLLMetry (2) | Manual OTel spans (3) |
|---|---|---|---|
| Token namespace | `llm.token_count.total` | `gen_ai.usage.prompt_tokens/completion_tokens` | `gen_ai.usage.input_tokens/output_tokens` (canonical) |
| OTel-standard adherence | Low (own namespace) | Medium-high (historical naming) | Maximum |
| Agent/chain model richness | High (`span.kind`, messages) | Medium | Depends on the implementation |
| Generic-backend compatibility | Limited without mapping | Good | Excellent |
| Control vs simplicity | Simplicity | Simplicity | Total control |
| Conventions maturity | Stable (de facto Arize) | Aligning to OTel | Experimental |
| Covers LangGraph `interrupt()` | No | No | Yes (manual) |

---

## 7. Export and backend alternatives

The ADK exports in **OTLP**, today (2026) the de-facto substrate for LLM and agent
observability. This decouples instrumentation (what) from destination (where): any
OTLP-compatible backend receives the traces without code changes. Our baseline is
OTLP/**HTTP** to Phoenix.

### 1. Direct OTLP export to a backend

Simplest configuration (it's the default): the agent sends OTLP straight to the
backend's endpoint.
**Pros:** zero additional infrastructure, immediate setup, minimal latency.
**Cons:** backend coupling (changing/duplicating the destination needs changes), no
centralized processing (sampling/PII), dependence on backend availability (limited
buffer).

OTLP backend overview:

| Backend | Type | Notes |
|---|---|---|
| **Arize Phoenix** | GenAI-native, OSS | OTel-native, OpenInference conventions, prompt/eval viewer. ADK default, great for dev. |
| **Langfuse** | GenAI-native | `/api/public/otel` endpoint; **OTLP/HTTP only** (gRPC unsupported); maps GenAI + OpenInference. |
| **SigNoz** | Classic full-stack tracing | OSS on ClickHouse, OTLP-native; not GenAI-specialized. |
| **Jaeger v2** | Classic tracing | Rebuilt on the OTel Collector; shows agent/model spans but without LLM-native views. |
| **Grafana Tempo (LGTM)** | Classic tracing | OTLP direct or via Collector; Grafana visualization; not GenAI-aware. |
| **Cloud (Datadog…)** | Managed | Datadog LLM Obs supports the GenAI sem-conv natively; paid, lock-in. |

### 2. Transport: OTLP/HTTP vs OTLP/gRPC

**gRPC** (port 4317, HTTP/2+protobuf, persistent connection): higher throughput,
minimal latency, rich errors for retry. *Cons:* issues with proxies/firewalls/LBs
(needs an L7 HTTP/2-aware LB), heavier client.
**HTTP** (port 4318, protobuf or JSON — our choice): maximum network compatibility
(passes proxies/CDNs/firewalls), simple to debug, consistent with HTTP-only backends
(Langfuse). *Cons:* lower throughput; HTTP/JSON has larger payloads (use protobuf in
production).

**Recommendation (2026):** start with **HTTP/protobuf** (compatibility + simplicity),
migrate to gRPC only when you hit throughput bottlenecks and control the
infrastructure. The ADK's current choice is aligned.

Sources: [OneUptime — gRPC vs HTTP](https://oneuptime.com/blog/post/2026-02-06-otlp-grpc-vs-http-comparison/view) · [SigNoz](https://signoz.io/comparisons/opentelemetry-grpc-vs-http/) · [OTLP spec](https://opentelemetry.io/docs/specs/otlp/)

### 3. Routing through an OpenTelemetry Collector

As an alternative to direct export, the agent sends to a **Collector** (a separate
process), the single point for sampling, PII redaction and fan-out. Two patterns,
often combined:

- **agent/sidecar** (close to the workload): fast offload, buffer/retry out of the
  process; *but* it lacks the cluster-wide view for tail-sampling.
- **gateway** (centralized): ideal for tail-sampling, PII redaction and consistent
  fan-out; *but* stateful, must be made HA, one extra hop.

What it enables: **sampling** (head in the SDK, tail in the gateway — e.g. 100% of
errors/slow, 5% of the rest), **PII redaction** (the `gen_ai.prompt`/`completion`
carry the full text; redaction/transform processor + OTTL), **fan-out** to multiple
backends.
**Pros:** total backend decoupling, centralized policies, resilience, migration
without touching the app. **Cons:** extra infrastructure/maintenance, an additional
hop, configuration complexity.

Sources: [Collector deploy](https://opentelemetry.io/docs/collector/deploy/agent/) · [Sidecar vs agent](https://last9.io/blog/opentelemetry-sidecar-vs-agent/) · [Handling sensitive data](https://opentelemetry.io/docs/security/handling-sensitive-data/)

### 4. GenAI-native vs classic tracing backends

**GenAI-native (Phoenix, Langfuse, Datadog LLM Obs)** — *Pros:* LLM-aware views
(prompt/completion, tokens and costs, multi-turn, eval), native understanding of the
conventions. *Cons:* partial infrastructure coverage, conventions still evolving
(OTel GenAI and OpenInference coexist → mapping), possible lock-in.
**Classic tracing (Jaeger/Tempo/SigNoz)** — *Pros:* reuse of the existing stack (LLM
next to microservices), mature distributed tracing, unified stack. *Cons:* no GenAI
semantics (attributes as generic tags), LLM analysis "done by hand".

**Summary:** OTLP/HTTP-protobuf with a GenAI-native default (Phoenix) for the
LLM-aware experience, keeping the option to route via a Collector to classic or cloud
backends for full-stack correlation or centralized redaction. Portability is
guaranteed by the OTLP standard.

Sources: [opentelemetry.io — GenAI observability (2026)](https://opentelemetry.io/blog/2026/genai-observability/) · [Jaeger v2 + GenAI](https://dev.to/thegatewayguy/ai-agents-are-opaque-jaeger-v2-otel-genai-conventions-are-the-fix-48b8) · [Datadog](https://www.datadoghq.com/blog/llm-otel-semantic-convention/)

---

## 8. Next steps

Ordered roughly by priority/effort ratio.

### 1. Upstream the `on_interrupt` / `on_resume` fix to OpenInference (low effort, high priority)
Open a PR on `Arize-ai/openinference` adding `on_interrupt` / `on_resume` to
`OpenInferenceTracer`, aligning it with LangGraph 1.x. Once released: **remove the
shim** `_patch_openinference_langgraph_callbacks()` and bump the minimum version
constraint in `pyproject.toml`. The shim's docstring already documents the removal
(tracked debt).
Ref.: [Arize-ai/phoenix#3120](https://github.com/Arize-ai/phoenix/issues/3120), [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts).

### 2. Resource attributes: `service.version` and `deployment.environment.name` (low effort)
Today the `Resource` only has `service.name`. Add `service.version` (semver/git hash,
from the package version) and `deployment.environment.name` (`staging`/`production`).
They are **stable** and enable segmentation/comparison at zero runtime cost. Extend
`TelemetryConfig.from_env()` respecting `OTEL_RESOURCE_ATTRIBUTES`.
Ref.: [Resource semconv](https://opentelemetry.io/docs/specs/semconv/resource/), [Deployment environment](https://opentelemetry.io/docs/specs/semconv/resource/deployment-environment/).

### 3. Add the METRICS signal (medium effort)
Metrics answer the aggregate trend (token burn rate, p99 latency, error rate) at low
storage cost. GenAI metrics: `gen_ai.client.token.usage` (Histogram, by
`gen_ai.token.type`), `gen_ai.client.operation.duration`, `...time_to_first_chunk`,
`...time_per_output_chunk`. Configure a `MeterProvider` +
`PeriodicExportingMetricReader` + `OTLPMetricExporter` (to a Collector/Prometheus;
Phoenix is primarily tracing), with `View` and the recommended buckets, reusing the
same attributes as the spans (for exemplar-linking) and the same no-op pattern.
Ref.: [GenAI metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/).

### 4. "Phase C" — spans as the single source of truth ✅ done (for the eval path)
The legacy `MetadataCollector` has been **removed**: usage/tool-calls are derived
exclusively from spans. The chosen mechanism is **out-of-process read-back**: the
agent exports spans to a file (`OTEL_TRACES_EXPORTER=file`) and the eval engine
reconstructs per-episode usage/tool-calls by reading the run's span directory and
grouping by `trace_id` / `gen_ai.conversation.id` (multi-agent included). This was
preferred over an in-process `SpanProcessor` because the eval runs the agent as a
**subprocess**, so an in-memory processor cannot reach its spans.
**Still open:** if a *live* (non-eval) caller ever needs usage back **synchronously
inside the A2A response**, that would require an in-process `SpanProcessor` (read
the `OPENINFERENCE_*` attributes in `on_end`, aggregate by `trace_id`), with the
sampling caveat — derive counts *before* any drop, or accept sampled-only totals
([spec#3145](https://github.com/open-telemetry/opentelemetry-specification/issues/3145)).
Ref.: [Custom SpanProcessors](https://oneuptime.com/blog/post/2026-01-30-opentelemetry-span-processors/view), [Span Metrics connector](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/connector/spanmetricsconnector/README.md).

### 5. Introduce an OpenTelemetry Collector (medium effort)
Insert a Collector between agents and backends: local agent (buffering) + gateway
(global policies). Key processors: `batch`, `transform`/redaction, `tail_sampling`,
multi-backend routing. The `traceparent` propagation already in place lets the
Collector recompose the distributed traces across agents.
Ref.: [Tail sampling](https://grafana.com/docs/opentelemetry/collector/sampling/tail/).

### 6. Sampling strategies (medium effort; requires step 5)
Looping agents generate many spans. Avoid random head-sampling (it would drop exactly
the errors/high-latency traces); prefer **tail-based** on the Collector (100% errors,
100% slow/high-consumption, 100% agent runs, ~10% normal successes). Constraint: all
spans of a trace must reach the same Collector instance (`loadbalancing` exporter).
Consistency with Phase C: the metadata derivation must be robust to dropping.
Ref.: [Tail-based sampling (2026)](https://oneuptime.com/blog/post/2026-01-25-tail-based-sampling-opentelemetry/view).

### 7. Add the LOGS signal / GenAI events (medium-high effort)
Correlate application logs to spans via `trace_id`/`span_id`. The GenAI conventions
move content (prompts, completions, tool args) toward **structured events/logs**
(`gen_ai.client.inference.operation.details`, `gen_ai.evaluation.result`). Connect
`bat.logging` to the OTel `LoggingHandler`. **Privacy by default:** no prompt/tool
content captured by default; introduce an opt-in flag in `TelemetryConfig` and leave
redaction to the Collector.
Ref.: [GenAI events](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/).

### 8. Track the maturity of the GenAI conventions (ongoing, low effort)
The GenAI conventions are not yet fully stable: in early 2026 the **client spans**
exited experimental, but **metrics, events and agent/framework spans remain in
Development**. The `gen_ai.*` attributes we use may change: the mitigation already
adopted — **centralizing the keys in `attributes.py`** — is the correct one. Adopt
`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` to consciously choose the
conventions version and periodically check releases.
Ref.: [GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/), [Inside the LLM Call (2026)](https://opentelemetry.io/blog/2026/genai-observability/).

### Fix the `.env` gotcha (low effort)
Independent of OTel but surfaced during development: change `load_dotenv()` in
`application.py` to look for the `.env` in the **cwd** (e.g.
`load_dotenv(find_dotenv(usecwd=True))`), so it works both with `bat-adk` installed
and with a local/editable source. See [§5](#5-how-to-use-it).

---

*Document produced alongside the OpenTelemetry integration of the ADK. For
implementation details see `src/bat/telemetry/` and the hook points in
`src/bat/agent/_executor.py` and `src/bat/prebuilt/call_agent_node.py`.*
