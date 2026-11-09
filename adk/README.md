# BubbleRAN Agentic Toolkit - Agent Development Kit (ADK)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](../LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/bat-adk)](https://pypi.org/project/bat-adk/)

The **BAT-ADK** is a Python-based Software Development Kit designed to simplify the development, deployment, and integration of AI Agents within the BubbleRAN architecture.
This repository includes the ADK framework ([BubbleRAN Software License](https://bubbleran.com/resources/files/BubbleRAN_Licence-Agreement-1.3.pdf)).

It is a toolkit: an agent's `main` loads its config and card,
opens what it needs with small `bat.*` functions, compiles its own LangGraph
graph and serves it over A2A.

The previous API (`bat.agent`, `bat.prebuilt`, `bat.chat_model_client`,
`bat.interop`) still works unchanged and is documented in
[docs/bat-adk.md](docs/bat-adk.md). It is deprecated: its classes warn with a
`FutureWarning`, and it will be removed in the next release.

## Installation

```bash
uv add bat-adk        # or: pip install bat-adk
```

## An agent

```python
import asyncio
import bat
from bat.server import ping_route
from graph import build_graph          # yours: returns a compiled graph


async def main():
    config = bat.load_agent_config()
    card = bat.load_agent_card(config.agent_card)
    bat.init_telemetry(config, service_name=card.name)
    checkpointer = await bat.init_checkpointer(config.persistence)
    graph = await build_graph(config, checkpointer)
    executor = bat.BATAgentExecutor(graph, card.name)
    server = bat.BATAgentServer(card, config, executor)
    server.app.add_route("/bat/ping", ping_route, methods=["GET"])
    await server.serve()


asyncio.run(main())
```

`main` is async because MCP sessions, remote agent cards and the Postgres pool
are opened in the event loop that then serves the agent.

## The building blocks

| `bat.` | Reads from config.yaml | Gives |
|---|---|---|
| `load_agent_config(path=None)` | the whole file (`CONFIG_PATH`, `${VAR}`, `.env`; `MODEL` / `MODEL_PROVIDER` / `BASE_URL` override `model`) | `BATAgentConfig`, data only, no network |
| `load_agent_card(path=None)` | `agent_card` (default `AGENT_CARD_PATH` or `./agent.json`) | `AgentCard` |
| `init_telemetry(config, service_name, privacy_floor="none")` | `telemetry` | OpenTelemetry exporters |
| `init_model(config.model)` | `model` | LangChain `BaseChatModel` |
| `await init_mcp_clients(config.mcp_servers)` | `mcp-servers` | `MultiServerMCPClient`; a server that does not answer stops the agent |
| `await list_mcp_tools(client, *servers)` | | `list[BaseTool]` of the named servers, or all |
| `await init_a2a_clients(config.remote_agents)` | `remote-agents` | `dict[str, RemoteAgentClient]`: `send_request(message, session)` (a str as text, any other value as data; automatically forwards interrupts inside a graph), `as_tool()` (the model gets the reply's text and data as JSON), `card()` |
| `await init_checkpointer(config.persistence)` | `persistence` (`memory`, `postgres`) | checkpointer (Postgres: on a connection pool), or `None`; `BATAgentServer` keeps A2A tasks in the same Postgres |
| `BATAgentExecutor(graph, agent_name)` | | runs a compiled graph per A2A request |
| `BATAgentState` | | Pydantic base state with `messages`, `data_in`, and `data_out` |
| `BATAgentServer(card, config, executor)` | `endpoint` | `serve()` / `run()`: card and JSON-RPC; add routes to `server.app`, e.g. `/bat/ping` with `bat.server.ping_route` |

Every block returns a native LangChain/LangGraph object, so any of them can
be swapped for one you build yourself.

## How a graph run maps to A2A

Use `bat.BATAgentState` as the base state for your graph:

```python
class State(bat.BATAgentState):
    network: str | None = None
```

It provides `messages` with the `add_messages` reducer and an independent
empty list by default, plus `data_in` and `data_out`, initially `None`.
Data fields have no custom reducer: LangGraph uses its default `LastValue`
channel. An update replaces the previous value, including with `None`, and
allows at most one write per graph step. `data_in` accepts JSON values;
`data_out` accepts JSON values or a Pydantic model.

Custom state schemas remain supported. The executor needs `messages` with
stable message IDs: use `MessagesState` or annotate your own field with
`add_messages` so LangGraph assigns missing IDs when applying node updates.

| In the graph | The A2A client gets |
|---|---|
| the user's text | a `HumanMessage` in `messages` |
| the user's data part | the first one in `data_in` (`None` without one); if the graph has no `data_in`, all data parts are appended as JSON to the user's message |
| `get_stream_writer()("...")` / `({...})` | a WORKING update, text / data |
| `interrupt(question)` | INPUT_REQUIRED; reply with text or a `{"resume": value}` data part. `interrupt()` returns that inner value or the original text. Other parts update `messages` and `data_in`. Several pending at once are asked one per message |
| `interrupt(question, response_schema=...)` | the same question parts, plus the optional JSON Schema in `message.metadata.interrupt.response_schema` |
| final `AIMessage` whose ID was absent at request start | the answer, as a text part of the artifact |
| the state's `data_out` field | a data part next to the answer; reset to `None` on every incoming message, including interrupt replies, so make it `Optional` in a Pydantic state |
| an exception | FAILED, naming only the exception type |

Numbers in data parts arrive as floats (protobuf `Struct`): 7 becomes 7.0.

Response schemas describe the expected answer independently of the question's
text/data format. For example, `interrupt("Describe the change", response_schema=str)`
accepts a text reply and advertises `{"type": "string"}`. A Pydantic model
schema can describe a structured approval form. Questions without a schema
carry no schema metadata and continue to work as before.

`RemoteAgentResponse.response_schema` exposes the schema only while the remote
asks for input. Inside a graph, `send_request()` forwards the question and
schema to a local interrupt so they reach the final A2A client. `as_tool()`
uses the same path. The forwarded question is always text, combining the
remote's text and every data part serialized as JSON, in order.
Completed exchanges are checkpointed, so resuming an
interrupt continues the remote task without sending previous messages again.
Use a checkpointer for graphs that can interrupt.
Schemas received as JSON dictionaries are presentation hints in LangGraph;
the originating agent retains validation from its Pydantic/type declaration.

Final text must come from the last message, which must be an `AIMessage` whose
ID was absent at the start of the current request. The executor keeps that
initial set of IDs locally for each execution, including interrupt resumes.
History can be trimmed without hiding a new answer or republishing an old one.
A postprocessing node can append an adjusted answer or replace a message
created during the same request using its ID. Replacing a message whose ID
was already present at request start edits history and does not produce a new
answer. Nodes may return the full state: `add_messages` still assigns missing
IDs. A missing ID in the initial history or final text answer fails the task.
A request producing only `data_out` returns only data. Producing neither final
text nor data fails the task.

Graphs without `data_in` receive the original text and all data parts in
one `HumanMessage`, under an `Attached data (JSON):` label. Data-only
requests also create that message. This fallback applies to ordinary data
parts accompanying interrupt replies too. The resume control envelope is
removed before updating state; its value is available only through the
`interrupt()` return value. Graphs with `data_in` keep their application
payload separate from the text. The executor only writes data fields
declared by the graph.

Structured interrupt replies follow the [LangGraph Agent Server resume
envelope](https://docs.langchain.com/langsmith/server-a2a#interrupts):

```json
{"data": {"resume": {"approved": true}}}
```

Send it with the waiting task's `taskId` and `contextId`. The envelope can
appear anywhere among the parts; `false`, `0`, `""`, `null`, and empty arrays
or objects are valid values. Text-only replies remain supported. A resume
envelope takes precedence over accompanying text, which remains message
context. Application data goes in separate parts: `data_in` receives the
first ordinary data part, or `None` if there is none.

BAT reserves a top-level `resume` key for this control envelope, whose sole
key must be `resume`. Multiple envelopes, extra envelope keys, structured
replies without an envelope, empty text-only replies, and envelopes without
a pending interrupt are rejected. `BATAgentServer` validates before starting
task execution for both blocking and streaming requests, returning JSON-RPC
`-32602` without changing the waiting graph or task. Clients can correct
those requests. Validation of the inner value against a typed interrupt's
`response_schema` remains an open issue; it is separate from envelope
validation.

Outside a graph, `send_request()` returns the pending question. Answer it
with a resume envelope and the returned session:

```python
asked = await remote.send_request("Deploy the network")
done = await remote.send_request(
    {"resume": {"approved": True}}, asked.session
)
```

Inside a graph, `send_request()` and `as_tool()` forward interrupt answers
with this envelope automatically. Other dictionaries remain application
data; a pending session alone does not make them resume envelopes.

Interrupt replies replace `data_in`, so save any part of the original
request that a paused node still needs in a separate state field before
entering that node. The node restarts when resumed.

Custom data channels can also use the default behavior:

```python
from langgraph.graph import MessagesState
from pydantic import JsonValue

class State(MessagesState):
    data_in: JsonValue
    data_out: JsonValue
```

Put successive interrupts in separate sequential nodes when using these data
channels. After one node completes, the next question is in a new graph step.
Repeated interrupts in one node, or questions in parallel branches, can
accumulate several resume updates within one step and raise `InvalidUpdateError`
on `data_in` or `data_out`. A messages-only graph can still ask parallel
questions because `messages` already has its own `add_messages` reducer.
Ordinary parallel
work remains supported when each branch writes only its own fields or fields
with suitable aggregation reducers.

## Graph execution settings

Set the agent's `RunnableConfig` with LangGraph's `with_config()` before
passing the graph to the executor. It returns a graph with those defaults:

```python
graph = (await build_graph(config, checkpointer)).with_config(
    {
        "max_concurrency": 4,
        "recursion_limit": 50,
        "tags": ["network-agent"],
        "metadata": {"deployment": "production"},
    }
)
executor = bat.BATAgentExecutor(graph, card.name)
```

The defaults apply to requests and interrupt resumes. The executor supplies
the A2A `context_id` as `configurable.thread_id` for each request; other
custom `configurable` values remain available to nodes. `max_concurrency`
limits concurrent graph tasks within each execution, not the total number
of concurrent A2A requests served by the agent.

## Layout

```
src/bat/
├── __init__.py        # the public `bat.*` names
├── config.py          # BATAgentConfig and its sections, load_agent_config
├── executor.py        # BATAgentExecutor
├── state.py           # BATAgentState: messages, data_in, data_out
├── server.py          # BATAgentServer, load_agent_card
├── model.py           # init_model
├── mcp_tools.py       # init_mcp_clients, list_mcp_tools
├── remote_agents.py   # init_a2a_clients, RemoteAgentClient (send_request, as_tool, card)
├── a2a_stream.py      # values ↔ A2A parts; a remote's A2A stream → RemoteAgentResponse; RemoteTaskContext
├── persistence.py     # init_checkpointer
├── telemetry/         # from the previous API plus init_telemetry; used by both
├── logging/           # from the previous API, unchanged; used by both
└── agent/, prebuilt/, chat_model_client/, interop/   # the previous API
```

## Develop

```bash
uv sync                          # Python, dev tools, postgres + telemetry extras
.venv/bin/python -m pytest -q    # no network beyond localhost, no LLM
uvx ruff check src tests
```

The tests serve real agents and a real MCP server on free localhost ports.
Postgres is tested with a fake saver only.

## Documentation

The API documentation is generated from the docstrings with
[`pydoc-markdown`](https://pydoc-markdown.readthedocs.io/): `uv run
pydoc-markdown` writes it to `adk/build/docs/content/bat-adk`.

## Not here yet

- Deprecation of the old API, CLI templates and docs.
- A failure with a message for the user (`bat.AgentFailure`): still an open
  decision.
