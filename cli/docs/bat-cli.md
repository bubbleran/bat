# High Level documentation of BAT-CLI

## Introduction

**BAT-CLI (BubbleRAN Agentic Toolkit CLI)** is a command-line companion to [BAT-ADK](../../adk/docs/bat-adk.md). It gives you a single entry point to **scaffold, configure, containerize, and evaluate** BAT agents, so you can move from an empty directory to a deployed, benchmarked agent without writing boilerplate.

The CLI is distributed on PyPI as `bat-cli` and exposes a single executable: `bat`.

### Design Philosophy

BAT-CLI is designed to let you focus on **agent behavior**, not on the surrounding lifecycle plumbing. It encodes the conventions of a BAT agent project (its file layout, environment variables, Docker build, and evaluation harness) into a handful of commands, so every agent looks the same and ships the same way.

The tool is built with [Typer](https://typer.tiangolo.com/) and organized as a tree of subcommands, each owning one stage of the agent lifecycle.

## Preliminary Concepts

Before using **BAT-CLI**, you should be familiar with a few core ideas:

- **BAT-ADK basics**
  Understand what an Agent Application, Agent Card (`agent.json`), and Agent Configuration (`config.yaml`) are, since the CLI generates and operates on them.

- **The agent project layout**
  A BAT agent is a Python project (managed with [uv](https://docs.astral.sh/uv/)) with a conventional structure: `agent.json`, `config.yaml`, `pyproject.toml`, a `.env`, and a `src/` package containing the graph and LLM clients.

- **A2A fundamentals**
  Know that agents expose an **A2A server** over HTTP. The evaluation engine drives an agent purely through this A2A endpoint.

## Command Tree at a Glance

The CLI is a tree of subcommands, each mapping to one lifecycle stage:

```
bat
├── init
│   ├── agent
│   │   ├── <name>
│   │   ├── --clients, -c
│   │   ├── --output-dir, -o
│   │   ├── --force, -f
│   │   ├── --port
│   │   ├── --model
│   │   ├── --model-provider
│   │   ├── --privacy
│   │   ├── --reasoning-effort
│   │   └── --service-tier
│   └── blueprint
│       ├── <name>
│       ├── --output-dir, -o
│       ├── --force, -f
│       └── --model-provider
├── add
│   ├── client
│   │   ├── <clients>
│   │   └── --force, -f
│   └── agent
│       ├── <name>
│       ├── --clients, -c
│       ├── --port
│       ├── --model
│       ├── --model-provider
│       ├── --privacy
│       ├── --reasoning-effort
│       ├── --service-tier
│       └── --force, -f
├── set
│   ├── config
│   │   ├── [AGENT]
│   │   ├── --port
│   │   ├── --model
│   │   ├── --model-provider
│   │   ├── --reasoning-effort
│   │   └── --service-tier
│   └── image
│       ├── --docker-registry
│       └── --repo
├── eval
│   ├── init
│   │   ├── [AGENT]
│   │   └── --force, -f
│   ├── run
│   │   └── [AGENT]
│   ├── show
│   │   └── [AGENT]
│   └── plot
│       ├── --folder, -f
│       └── --filter, -F
├── manifests
│   ├── aifabric
│   │   ├── --output, -o
│   │   ├── --name
│   │   ├── --namespace
│   │   ├── --image-pull-secret
│   │   └── --telemetry-endpoint
│   └── composition-model
│       ├── --output, -o
│       ├── --name
│       ├── --namespace
│       ├── --docker-registry
│       ├── --repo
│       └── --version
├── build
│   ├── --docker-registry
│   ├── --repo
│   └── --version
├── push
│   ├── --docker-registry
│   ├── --repo
│   └── --version
└── version
```

Each top-level branch maps to one lifecycle stage: **create** (`init`), **extend** (`add`), **configure** (`set`), **evaluate** (`eval`), **describe for the cluster** (`manifests`), and **containerize/distribute** (`build`, `push`). The standalone `version` command reports the installed toolkit version.

Built-in help is available at every level (`bat --help`, `bat eval --help`, ...). For exact flags and examples, see the [README](../README.md); this document focuses on the concepts behind each stage.

## Two project shapes: agent and blueprint

`bat init agent` scaffolds a **standalone agent** — `config.yaml`, `agent.json` and `pyproject.toml` in one directory, started with `uv run .`. That shape is unchanged and still the right one for a single agent.

`bat init blueprint` scaffolds a **blueprint**: one uv project holding several agents. The pyproject, the virtualenv, the PyInstaller spec and the packaging live once at the root; each agent is a package right below it with its own `config.yaml` and `agent.json`. One binary serves them all, and `bat add agent <name>` adds one:

```
my-blueprint/
├── pyproject.toml          # one project: one venv, one uv.lock
├── __main__.py             # shared entrypoint; runs the agent named by argv[1]
├── my-blueprint.spec       # one frozen binary for every agent
├── Dockerfile · docker-compose.yaml · Makefile · .env
├── netops/     agent.json · app.py · config.yaml · src/
└── hermes/     agent.json · app.py · config.yaml · src/
```

There is no manifest and no blueprint-level `config.yaml`. **A blueprint is recognised by its layout**: a folder with `pyproject.toml` and `__main__.py` but no `agent.json` (which is what a standalone agent has on top), whose agents are the directories right below it holding `config.yaml` and `agent.json` and no `pyproject.toml` of their own. A directory that does have one — a nested MCP server, say — is a separate project, not one of the blueprint's agents. This is what `bat add agent` and `bat eval` look for, so they work in blueprints the CLI did not create, too.

**The entrypoint picks the agent from `argv[1]`** — locally, in the compose service's `command`, and in the deployment mode's `args` on a cluster alike. The process runs from the blueprint root, where the SDK's default `./config.yaml` is not the agent's, so the entrypoint points `CONFIG_PATH` at `<agent>/config.yaml` — unless `CONFIG_PATH` is already set (compose, `bat eval`) or a `config.yaml` sits at the root (the operator mounts the rendered one at `/app/config.yaml`):

```bash
uv run . netops    # reads netops/config.yaml
make netops        # the same, with .env loaded
```

An agent's name becomes a Python package the entrypoint imports, so it has to be a valid identifier: `cluster_view` is fine, `cluster-view` is rejected.

### What `bat add agent` keeps in sync

The agent folders on disk are the only registry, and almost nothing has to be told about a new one:

- **`__main__.py`** never changes. It imports the agent named on the command line (`importlib.import_module(argv[1]).run()`) — lazily, so one agent's dependencies stay off another's startup path — and accepts any folder holding an `agent.json` in the directory it runs from: the blueprint root locally, `/app` in the image, where the Dockerfile copies the cards.
- **The PyInstaller spec** cannot see imports made by name, so it finds the agent folders itself at build time (an `app.py` next to an `agent.json`) and bundles them.
- **The `Makefile` and the `Dockerfile`** discover agents from the filesystem too.
- **`docker-compose.yaml`** is the one file `bat add agent` edits: it adds a service for every agent that has none yet (an agent folder added by hand included), all on the same image, each mounting its own `config.yaml` and health-checked on its own port. Other services, and edits to an agent's own, are left as they are; the file is rewritten as YAML, so comments in it are not kept.

### Building the image

The image installs from the lockfile (`uv sync --frozen`), so a blueprint needs a committed `uv.lock`; `make build` writes one first when there is none. It carries the binary and each agent's `agent.json`, but no `config.yaml` — that is deployment-specific, mounted by `docker-compose.yaml` and provided by the platform on a cluster. It also sets `LANGGRAPH_STRICT_MSGPACK=true`, so checkpoints only deserialize the types the agent's graph declares, which is why the scaffolds pin `bat-adk>=2026.9.29a0`: that release restores checkpoints through the compiled graph, the only place those types are known.

## Scaffolding (`init` / `add`)

The **scaffolding commands** generate the conventional structure of a BAT agent from templates, so every agent starts from the same well-formed baseline.

### Creating an Agent

`bat init agent <name>` produces a complete, runnable agent project: the Agent Card (`agent.json`), the Agent Configuration (`config.yaml`), a `pyproject.toml`, a `Dockerfile` and `Makefile`, a `.env`, and a `src/` package containing the **AgentGraph** and the **LLM clients**.

The command parameterizes the generated files so the new agent is ready to run:

- `--clients` pre-generates one **ChatModelClient** scaffold per name you provide.
- `--port`, `--model`, and `--model-provider` are written directly into `config.yaml` (`endpoint.port`, `model.name`, `model.provider`).
- `--reasoning-effort` and `--service-tier` set `model.reasoning_effort` and `model.service_tier`; without them both stay commented out in `config.yaml`, so the provider's own defaults apply. A reasoning effort is supported by gpt-5 and later.

### Adding Clients Later

`bat add client <names>` adds new **ChatModelClient** scaffolds to an _existing_ agent. It must be run from the agent root (it expects `src/llm_clients/` to exist) and refuses to overwrite files unless `--force` is given.

This keeps the "one client per LLM role" pattern (e.g. `reformulator`, `planner`, `executor`) consistent whether the clients are created up front or added incrementally.

### Telemetry Privacy Floor

An agent's `telemetry.privacy` in `config.yaml` says how much of its internals may leave the process — `none` (the default) | `content` | `names` | `full`, each level redacting everything the one below it does. See the ADK's [bat-adk.md](../../adk/docs/bat-adk.md) for what each level covers.

That file is editable wherever the agent runs — a mounted ConfigMap, a replaced file, `CONFIG_PATH` pointing elsewhere — so for an agent shipped as a packaged artifact it is a default, not a guarantee. The **floor** closes the gap: a minimum that `config.yaml` can raise but never lower (the effective level is `max(floor, config.yaml)`). The ADK takes it as `AgentApplication(..., telemetry_privacy_floor=...)`.

Every scaffolded agent passes it where it builds its application, `none` unless asked for more:

```python
AgentApplication(
    AgentGraphType=NetopsAgentGraph,
    AgentStateType=NetopsAgentState,
    telemetry_privacy_floor="content",
)
```

`bat init agent --privacy LEVEL` writes the level into the standalone agent's `__main__.py`, and `bat add agent --privacy LEVEL` into the blueprint agent's `app.py`; without the flag both write `"none"`, AgentApplication's own default, so `config.yaml` alone decides until someone changes it. Inside a blueprint the floor is therefore **per agent**, even though all of them share one binary. Being code, it is frozen into the built binary: neither a mounted `config.yaml` nor an environment variable can lower it.

It applies wherever the agent runs, `bat eval` included. An agent with a floor of `content` exports its prompts, tool arguments and results as `__REDACTED__`, so the trajectory and the judge would see the steps but not what they carried; at `full` tool names go too, and tool-call checks could not match. `bat eval run` therefore refuses an agent whose code passes a floor above `none`, naming the file and line: set it to `"none"` while you evaluate, and put it back afterwards. A floor computed at runtime cannot be read from the source, and only gets a warning.

An unknown level is rejected before any file is created, since the ADK itself falls back to `none` on an unknown floor, and a typo silently degrading to `none` would ship an agent exporting in the clear precisely when someone meant to lock it down.

## Configuration (`set config` / `set image`)

The `set` commands make **in-place updates**, without regenerating any other file, and each names what it changes:

- **`bat set config`** writes an agent's runtime values (`--port`, `--model`, `--model-provider`, `--reasoning-effort`, `--service-tier`) into its `config.yaml` (`endpoint.port`, `model.name`, `model.provider`, `model.reasoning_effort`, `model.service_tier`), preserving comments and structure. It acts on the agent you are in, or — like `bat eval` — on the one named from the blueprint root: `bat set config netops --port 9309`.
- **`bat set image`** writes the image settings (`--docker-registry`, `--repo`) into the project's `Makefile` as `DOCKER_REGISTRY` / `REPO` — the blueprint's, when run from one of its agent folders — where `make build` reads them as well as `bat build`. Settings shared by the whole team belong there, committed, rather than in the gitignored `.env`, which holds secrets only. A Makefile with no `DOCKER_REGISTRY ?=` / `REPO ?=` line names its image some other way, so the command refuses rather than guess, and writes nothing.

Both are intentionally strict: they require at least one value to set, so they never silently do nothing.

## Containerization (`build` / `push`)

The **build and push commands** turn a project into a distributable container image. A scaffolded blueprint or agent has a `Makefile` that holds the recipe — for a blueprint, `make build` writes the `uv.lock` the image installs from, then runs `docker build` — and `bat build` / `bat push` run its `make build` / `make push` rather than repeating it. Run from an agent folder of a blueprint, they build the blueprint, which has the one image. A project with a `Dockerfile` but no `Makefile` is built with `docker build` directly.

### Image Reference

Both commands produce a single image reference of the form:

```
{registry}/{repo}:{version}
```

The version is used both as the **image tag** and as a `VERSION` build argument, which a blueprint's image stamps on its agent cards so a running agent reports the version it was built from.

The Makefile's own defaults make `make build` work anywhere: no registry, the project's name as the repo, and the git tag or commit as the version — outside a git checkout there is none, so the image is tagged `dev` (`demo:dev`, the image `docker-compose.yaml` runs by default) and the cards keep their version. `make push` refuses to run without a registry, since `docker push` would otherwise send the image to Docker Hub.

Before running a target, `bat build` / `bat push` dry-run it (`make -n`) to see the image it would produce — a Makefile may well have a default registry of its own, as the supervisor's does. An image can always be built, for this machine if there is no registry; `bat push` with no registry stops before anything runs, since the image would go to Docker Hub, and says that BubbleRAN can provide a registry. Following Docker's own rule, the first part of an image name is a registry only when it has a dot or a port, or is `localhost`: `orama/demo` has none.

### Configuration Precedence

`bat build` and `bat push` pass the Makefile only the flags they were given, so anything left unset is decided as with a plain `make build`. The registry and repository are looked up in a fixed order, which lets you set them once in the Makefile and override them per invocation:

1. CLI flag (`--docker-registry` / `--repo`)
2. Shell environment variable (`DOCKER_REGISTRY` / `REPO`), which make reads itself
3. The Makefile's `DOCKER_REGISTRY ?=` / `REPO ?=` (`bat set image` writes them)

A project without a Makefile reads the same two flags and shell variables; without a registry its image is `<name>:<version>`, for this machine only.

The version is `--version`, or the Makefile's default.

## Manifests (`manifests aifabric` / `manifests composition-model`)

The orama operator deploys agents from an **AIFabric**, and renders each internal agent's `config.yaml` from it: `model` from the agent's LLM in `spec.llms`, `remote-agents` from its dependencies, `mcp-servers` from `spec.mcp`, `telemetry` from `spec.telemetry`. `bat manifests aifabric` goes the other way: run from a blueprint (its root or one of its agents), it writes `aifabric.yaml` at the root from the agents' own `config.yaml` — and every time it is run, it updates that file.

For each agent the blueprint's `__main__.py` can run:

| AIFabric | from |
|---|---|
| `name` | the agent folder, as a Kubernetes name (`supervisor_v4` → `supervisor-v4`) |
| `internal.model` | the deployment mode of the blueprint's `composition-model.yaml` whose `args` (or `AGENT_CARD_PATH`) select the agent, else `<blueprint>/<agent>` |
| `internal.llm` | `model` — agents on the same provider and model share one `spec.llms` entry, with an `<provider>-api-key` secret reference |
| `internal.dependencies` | `remote-agents`, matched to the blueprint's agents by port, else by name |
| `internal.mcpServers` | `mcp-servers`, each a `spec.mcp` entry (internal when the CompositionModel has an `mcp` mode of that name) |
| `internal.role` | `none` on a new entry, so the dependencies are exactly the config's |
| `spec.telemetry` | a non-local `telemetry.output` collector, or `--telemetry-endpoint` |

`${VAR}`s in `config.yaml` are expanded first, as the ADK does when it loads the file.

**Updating, not overwriting.** Each run refreshes the `llm`, `dependencies` and `mcpServers` of the blueprint's own agents, adds the ones that are new, and keeps everything else: roles, `chattable`, a model, annotations and labels set by hand, secret names, `imagePullSecrets`, the telemetry collector, and the agents of other blueprints — one AIFabric often deploys several. Every change is listed (`kpi: llm openai-luna-6 -> openai-model`), because following the configs can replace a choice made for the cluster; an up-to-date file reports `No changes`. The file's leading comment block is kept; the rest is rewritten in one style (lists indented under their keys), so YAML anchors are renamed (`&id001`), flow lists become block lists and inline comments are lost.

**What cannot come from a dev config** is flagged rather than guessed: `localhost` URLs (an MCP server, a model's `base_url`, a local Phoenix), unset `${VAR}`s, remote agents that match no agent of the fabric, agent folders the dispatcher never names, and `telemetry.privacy` — the operator does not render it, so on the cluster only the agent's `telemetry_privacy_floor` applies.

### The CompositionModel (`manifests composition-model`)

An AIFabric's `internal.model` names a deployment mode of a **CompositionModel** (`<CompositionModel>/<mode>`), which says how to run it: the image, its `args`, `env`, RBAC `rules`. `bat manifests composition-model` writes `composition-model.yaml` at the blueprint root, where `manifests aifabric` reads it — so run it first. Each agent the dispatcher can run gets a mode named after its folder:

```yaml
netops:
  kind: agent
  name: netops
  imageTag: hub.bubbleran.com/orama/demo:1.2.3
  args: [netops]                  # the dispatcher runs the agent it names
  env:
    - name: AGENT_CARD_PATH       # the config.yaml the operator mounts has no agent_card
      value: netops/agent.json
```

The image is the one `bat build` tags with the same `--docker-registry`, `--repo` and `--version` (a dry run of `make build`); one with no registry is flagged, since the cluster can't pull it. Each run adds a mode for the agents that have none (an agent already run by a mode under another name keeps it) and moves every mode built from the blueprint's image — MCP modes included — to the new image, listing each change. Everything else is kept: `rules`, `resources`, `readinessProbe`, extra `env`, `spec.version`, other images' modes, and the leading comment block. RBAC rules are never guessed: add them by hand.

## Evaluation Engine (`eval`)

`bat eval` works in both shapes. It has to know *which* agent it is acting on, and there are two ways to tell it:

```bash
cd netops && bat eval run      # the agent is the directory you are in
bat eval run netops            # or name it, from the blueprint root
```

`init`, `run` and `show` all take that optional `AGENT` argument; it resolves against the enclosing blueprint and wins over the working directory, so it works from the blueprint root and from any of its agent directories. Naming an agent outside a blueprint is an error, since a standalone agent is just the directory you are in. Omit it at the blueprint root and the command says so, and lists the agents it found.

Either way the eval reads that agent's `eval/` folder, and inside a blueprint it starts the agent exactly as the Makefile does, from the blueprint root: `CONFIG_PATH=<agent>/config.yaml uv run . <agent>`, with the blueprint's `.env` loaded (a variable already exported in the shell wins, as under `make`). The `config.yaml` it patches for the run (to turn on the local span exporter) is the agent's own. Since that start command only works if the blueprint's `__main__.py` accepts the directory name as a selector, `bat eval run` warns when `__main__.py` never names the agent — automation's `logs_agent/`, selected as `logs`, is the case it catches.

The **evaluation engine** is the most substantial part of BAT-CLI. It runs a dataset of tasks against a live agent, judges the outcomes, and produces machine-readable artifacts and charts. It is exposed through four subcommands — `init`, `show`, `run`, and `plot` — and is implemented as a pipeline of cooperating components.

<p align="center">
  <img src="images/eval-pipeline.png" alt="The BAT-CLI evaluation pipeline: from dataset to evaluation result" width="600">
</p>

### Pipeline Overview

The diagram above shows how the engine flows from a static dataset to a judged result. Each stage corresponds to one of the components described in the sections that follow:

- **Dataset** — the full collection of tasks to evaluate. Each entry pairs a conversational `turn` with an `expected_outcome`, describing both what the agent is asked and what success looks like.
- **Task** — a single entry drawn from the dataset (a **`TaskSpec`**), enriched with its expectations: the `expected_outcome`, the `expected_tool_calls`, and the other checks the agent will be measured against.
- **Episodes (K Attempts)** — each task is run **`k`** times against the live agent. Every attempt is one **Episode** that records the agent's answers, token usage, and timing. Running multiple attempts lets the engine measure _reliability_, not just a single pass/fail.
- **Evaluation Result** — the episodes are judged along two complementary axes:
  - a **Deterministic Verdict** — rule-based, reproducible, and LLM-free (`passed` + a `reason`, e.g. _"Correct tool calls and final state"_).
  - an optional **Qualitative Scoring** — LLM judges that score dimensions such as `relevance` and `completion`, each with a free-text justification.

### Evaluation Configuration

Everything an evaluation needs is described in `eval/eval.yaml`, validated into an **`EvalConfig`** model. It declares:

- the **dataset** of tasks and the **output directory**,
- the **agent URL** plus startup/shutdown timeouts,
- **`k`** — how many attempts to run per task (for measuring reliability, not just pass/fail),
- whether **qualitative** (LLM-judge) scoring is enabled,
- the list of **models** to evaluate, and an optional **judge** model used for qualitative scoring,
- **`extra_spans`** — span files or directories to read besides the run's own (see [The Trajectory](#the-trajectory)).

`bat eval init` scaffolds this file (along with `eval/input/tasks.json` and `eval/output/`), and `bat eval show` prints the fully resolved configuration so you can confirm what will run before running it.

### Tasks and Expectations

Each entry in the dataset is a **`TaskSpec`**: an `id`, a list of conversational `turns`, and an **`TaskExpected`** block describing what success looks like. Expectations are intentionally multi-faceted:

- **`status`** — the expected final A2A task status (e.g. `completed`).
- **`output_must_contain`** — phrases that must appear verbatim in the final output.
- **`tool_calls`** — tools the agent is expected to call, matched by name and an **argument subset** (only the keys you specify must match), with a minimum call count. Called agents' tools count too.
- **`agent_calls`** — other agents the agent is expected to call (`agent` is the called agent's card name), with a minimum call count.
- **`max_model_calls`** / **`max_tokens`** — upper bounds for the whole conversation, called agents included; they catch a runaway loop that still ends well.
- **`no_errors`** — fail when any step failed (a model call, a tool call, an agent call, a node), even if the agent recovered and reached the expected status.
- **`expected_outcome`** — a free-text description of the desired result, evaluated _semantically_ by the LLM judge rather than by string matching.

`tool_calls`, `agent_calls`, the two limits and `no_errors` are read off the agent's spans. When a task asks for any of them and no span of the conversation was found, the episode fails with one `trace: no spans found` check instead of a misleading "called 0×".

```json
{
  "id": "draft_lab_network",
  "turns": ["Create a 5G network named lab-1 with two cells on band n78."],
  "expected": {
    "status": "completed",
    "expected_outcome": "A draft of lab-1 (two n78 cells) is saved, not applied.",
    "tool_calls": [{"name": "save_draft", "args_subset": {"name": "lab-1"}}],
    "agent_calls": [{"agent": "netops"}],
    "max_model_calls": 12,
    "no_errors": true
  }
}
```

### Running an Episode

For each task and each of its `k` attempts, the engine talks to the agent over A2A at its configured URL, one conversation per attempt. It records an **`EpisodeTrace`**: the status events of the stream, the wall time, the trajectory read from the agent's spans (tokens included) and the tool calls in it. The result of one attempt is an **`EpisodeResult`** carrying the final status, final output, and trace.

### The Trajectory

An agent streams over A2A only what its `to_task_result()` reports — often just "working" and the final answer. What it actually did lives in its OpenTelemetry spans, which `bat eval run` has the agent write to the run's spans directory. From them the adapter builds each episode's **trajectory** (`trace.trajectory` in the episode JSON): per turn, in the order it happened,

- **model** steps — the graph node, the model, tokens, what the model *said* and which tools it *decided to call*. The model's input is left out: it is the history the trajectory already holds, and it is what makes raw traces huge (a small two-agent conversation is ~350 KB of spans and ~2 KB of trajectory);
- **tool** steps — name, arguments, result, or the error. A tool that returns a LangGraph `Command` records the whole state update it carries; its result is the ToolMessage in that update, the rest being graph state rather than the tool's answer;
- **agent** steps — a call to another agent: what was asked, the called agent's own steps nested under it, and its answer or failure;
- **error** steps — a failure outside any of those, reported once.

Text fields are capped at 1,500 characters, marked `…[truncated: N chars]`. `trace.tool_calls` is derived from the same steps, so a ReAct loop's tool calls carry their arguments (its tool spans do not record them; the model's tool-call decision does).

**Called agents** write their spans into their own process's output. The run's spans directory is new for every run, so a called agent keeps a stable `telemetry.output` file, and the eval is told where it is; spans are picked by trace id, so a long-lived shared file is fine:

```yaml
evaluation:
  extra_spans:
    - ../netops/spans.jsonl     # netops' config.yaml: telemetry.output[].file_path
```

Without it, the call appears as an agent step with no steps under it.

### Deterministic Verdict

Once an episode completes, the engine produces a deterministic **`EpisodeVerdict`** (`passed` + a human-readable `reason`). It runs each declared expectation as an independent check — status match, each required phrase, each expected tool and agent call, the limits and `no_errors` — and the episode passes only if **all** checks pass. The `reason` string concatenates every check, so a failure tells you exactly which expectation was not met.

This stage is purely rule-based and reproducible: it does not call an LLM.

### Qualitative Scoring (optional)

When `qualitative: true`, an additional **LLM-judge** pass scores each episode along several dimensions:

- **response relevance**
- **task completion quality**
- **hallucination score**
- **tool-call appropriateness**

These scores (a **`QualitativeScores`** object, including the judge's reasoning) complement the deterministic verdict with a semantic assessment of quality. The scores are based on how the LLM has been instructed and may slightly vary between run and dependes on which model is used as a judge. Scoring runs concurrently across episodes for speed.

The judge is shown the **trajectory**, rendered as numbered lines with a legend, instead of the A2A messages (which it still gets when no spans were recorded). That lets it check claims against what happened: task completion requires a claimed action to be backed by a step that did it and did not fail, and groundedness treats tool results and other agents' answers as legitimate sources while flagging a response that contradicts them. Models and token counts are left out of what it is shown, since no rubric scores cost, and the tool-usage judge gets the calls as a list of names, the `[tool]` lines carrying their arguments. `judge.max_trajectory_chars` (default 24,000, roughly 6k tokens) bounds what it is shown; over budget, long fields are shortened first, then the middle steps of each turn left out.

`judge.mode: outcome` trades all of that for one cheap call per episode: the judge sees only what the user asked, the task's expected outcome and status, and how the conversation ended (final status and response), and scores leniently whether the task succeeded. Not seeing the steps, it takes the response's claims at face value, and the expectations about steps (`tool_calls`, `agent_calls`, `no_errors`) are left out of what it is told. The score is recorded as task completion (`task_completion_quality`), so metrics, summaries and plots read it as usual; the other scores stay empty, and `judge.prompts.task_completion` still applies. The default, `full`, is everything above.

```yaml
judge:
  provider: openai                   # vLLM / TGI / llama.cpp expose an OpenAI-compatible API
  model: openai/gpt-oss-120b
  base_url: http://judge-host:8000/v1
  api_key_env: BAT_JUDGE_API_KEY     # any value for a local server
  max_trajectory_chars: 24000
  prompts:                           # optional, per judge: agent-specific context appended to the rubric
    task_completion: "A draft is not a deployed network: only apply_network deploys."
```

Every judge answers with **structured output**: its server is sent the verdict schema — `reasoning`, then `score` between 0 and 1 — as Ollama's `format` or the OpenAI-compatible `response_format` (vLLM, TGI and llama.cpp support it), and decodes against it, so the answer is the verdict and nothing around it. An answer that does not fit is retried. A server or model that cannot answer to a schema gets one plain-text request instead, read even when it prints its thinking first or wraps the JSON in prose; that judge then stays on plain text for the rest of the run. A score that is still unusable is left empty, with the reason in `judge_reasoning`.

> **On-prem judges.** Serve the judge with a context window that fits the rubric, the trajectory and the model's own thinking — 16k tokens or more. Ollama's default is 4,096 and it silently cuts longer prompts: set `OLLAMA_CONTEXT_LENGTH` on the server, or lower `max_trajectory_chars`.
>
> Reasoning models think before they answer, and nothing in the eval caps that yet: `qwen3:4b` on a CPU took 7 to 27 minutes per judge call, and a `/no_think` in `judge.prompts` did not stop it. Serve a reasoning judge with a GPU and a low reasoning effort, or use a non-thinking model. Very small judges (3–4B) read the trajectory but do not follow the rubric reliably (a missed outcome scored above the 0.4 cap); use them to test the setup, not to score.

### Artifacts

Every run writes a structured set of artifacts under the output directory:

- **`episodes/`** — one JSON file per attempt (full `EpisodeResult`, including the trace).
- **`summary.json`** — per-task pass/fail counts, success percentage across the `k` attempts, and averaged qualitative scores.
- **`metrics.json`** — aggregated metrics for the run (e.g. pass@k style summaries).
- **`agent-<n>.log`** — what the agent printed while it ran for the n-th model. When it stops before it is ready, the eval reports the log's last line; for a remote agent or MCP server it requires that does not answer, it also says to start it or mark it `required: false`.

Because attempts are kept per-task and per-model, results from multiple models or runs sit side by side in the same output tree.

### Visualization (`plot`)

`bat eval plot` reads the `metrics.json` files produced by `run` and renders charts back into the output folder. Each sub-folder containing a `metrics.json` is treated as one run, so a single `plot` invocation can compare multiple models or runs. The `--filter` option narrows the _per-task_ charts to a subset of task ids, while summary charts always cover all runs.

## Version (`version`)

`bat version` prints the installed BAT-CLI version and exits. The value is read from the package metadata of the installed `bat-cli` distribution, so it always reflects the exact build in use — handy when reporting issues or confirming an upgrade. If the CLI is run from a source checkout that is not installed as a package, the command reports that the version is unavailable and exits non-zero.
