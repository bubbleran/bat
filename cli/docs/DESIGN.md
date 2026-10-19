# BAT-CLI Design

How the code is organised and why. For what the commands do, see [bat-cli.md](bat-cli.md).

## Layout

```
src/
├── cli.py          the Typer app: the command tree, flag parsing, nothing else
├── project.py      "where am I": a working directory → the agent a command acts on
├── create/         init / add: templates + rendering
├── set/            set config / set image: in-place edits
├── image.py        build / push: delegate to the project's Makefile
├── manifests/      aifabric / composition-model: read, merge, write
└── eval/
    ├── commands.py the harness: start the agent, run the engine, stop the agent
    └── engine/     the evaluation itself, independent of the CLI
```

`cli.py` stays thin: each command parses its flags and calls one function in the module that owns that stage.

## Principles

- **The filesystem is the registry.** There is no project manifest. A blueprint is recognised by its layout (`pyproject.toml` + `__main__.py`, no `agent.json`) and its agents by the folders below it.
- **Templates are plain files.** `create/templates/` holds real project files with `__KEY__` placeholders; `rendering.render()` substitutes them. To change what a new agent looks like, edit the template, not Python.
- **Delegate, don't duplicate.** `bat build` runs `make build` (dry-running it with `make -n` first to learn the image name); the eval reads `config.yaml` through the ADK's own `EndpointConfig`. Each recipe has one owner.
- **Merge, don't overwrite, files people edit.** `docker-compose.yaml`, `aifabric.yaml` and `composition-model.yaml` are loaded, updated only where the CLI owns the value, and written back with every change listed.
- **Validate at the boundary, refuse rather than guess.** CLI flags, `eval.yaml` and `tasks.json` are validated once, on entry, into typed models. When the CLI can't tell what the user meant (a Makefile without `REPO ?=`, an unknown privacy level), it stops with a message instead of picking something.

## The Parts

### Finding the project (`project.py`)

Every command first works out where it is. `find_blueprint_root()` returns the blueprint that the directory is, or is an agent folder of. `blueprint_agents()` lists that blueprint's agent folders. `resolve_agent_target()` combines them: from the working directory and an optional `AGENT` argument, it returns an `AgentTarget` with the project root, the agent's directory and how to start the agent (`uv run .`, or `uv run . <agent>` with `CONFIG_PATH`). `eval` and `set config` act on an `AgentTarget`. `add agent`, `build`, `push` and `manifests` only need the blueprint root.

### Scaffolding (`create/`)

`create/templates/agent/` and `create/templates/blueprint/` hold the files a new project gets, with `__KEY__` placeholders. `agent.py` and `blueprint.py` compute the values (names, port, model, the provider's `bat-adk` extra, the privacy floor, the graph's client fields), and `rendering.py` renders the files and writes them. A blueprint agent reuses the standalone agent's `agent.json` and graph templates, so both shapes generate the same agent code. Optional model settings (`reasoning_effort`, `service_tier`) aren't placeholders: after writing, the scaffold calls `set_config()`, the same function `bat set config` uses.

Each LLM client is its own `*_client.py`. `llm_clients/__init__.py` is regenerated from the files on disk, so clients added later by `bat add client` are exported too. In a blueprint, `bat add agent` only has to edit `docker-compose.yaml`: it adds a service for each agent folder that has none. The Makefile, Dockerfile, PyInstaller spec and `__main__.py` find agents on the filesystem themselves.

### Settings (`set/settings.py`)

`config.yaml` and the Makefile are edited as text, line by line, rather than loaded and dumped, so comments and layout survive. A YAML key is set on its own line, else on its commented-out line, else added at the end of its section. A Makefile variable is set only on an existing `NAME ?=` line; without one, the command refuses.

### Images (`image.py`)

The CLI does not know how to build an image; the project's Makefile does. `bat build` / `bat push` find the project (the blueprint root, from an agent folder), pass the flags given as `make` variables, and run `make build` / `make push`. Before that, `plan()` dry-runs the target (`make -n`) and reads the image name from the `docker` command it would run. That is how `push` catches a missing registry before anything runs, and how `manifests composition-model` learns the image tag. A project without a Makefile falls back to plain `docker build` / `docker push`.

### Manifests (`manifests/`)

Both manifests follow the same pattern: load, update, write. `update_aifabric()` and `update_composition_model()` take the blueprint root and the existing document (or `None`), and return a `ManifestUpdate`: the new document plus the entries added, the changes and the warnings. They only touch the fields that come from the agents' configs or the image, and keep everything else. `commands._update()` holds the file handling both share: read the file, keep its leading comment block (PyYAML drops comments), write the result and print what changed.

## Evaluation

The evaluation is the most involved part. It has two layers.

### The harness (`eval/commands.py`)

Everything that touches processes and files on the user's machine. For `bat eval run`:

1. Resolve the agent (`project.resolve_agent_target`), refuse it if its code sets a privacy floor above `none` (found by parsing the source with `ast`), load `eval.yaml` into an `EvalConfig` and `tasks.json` into `TaskSpec`s.
2. For each model in the config:
   - patch the agent's `config.yaml` so its spans go to `<run>/spans-<i>/agent.jsonl` (a context manager restores the file),
   - start the agent as a subprocess in its own process group, with the `.env`, `MODEL` / `MODEL_PROVIDER` / `BASE_URL`, and its output in `agent-<i>.log`,
   - wait for its port, call `run_evaluation()`, then stop the whole process group (`uv run` forks the server, so stopping uv alone would orphan it).

A model that fails is recorded and the loop moves on.

### The engine (`eval/engine/`)

Plain async functions over pydantic models; it knows nothing about Typer, subprocesses or blueprints.

```
tasks.json ─► TaskSpec[]
                │   for each task, k times
                ▼
 adapter.run_task ──A2A stream──► final status, output, events
        │
        └── read span files ──► trajectory.build_trajectory ──► Trajectory
                ▼
         EpisodeResult ──► evaluator.verdict ──► EpisodeVerdict      (deterministic)
                ▼
     episodes/*.json written
                ▼
 judge.score (optional) ──► QualitativeScores                         (LLM)
                ▼
 metrics.summary / metrics.metrics ──► summary.json, metrics.json
                ▼
 plotter (bat eval plot, later) ◄── metrics.json
```

| Module | Role |
|---|---|
| `contracts.py` | Every data shape: `TaskSpec`, `TaskExpected`, `Trajectory` and its steps, `EpisodeResult`, `EpisodeVerdict`, `QualitativeScores`, `EvalConfig`. Read this first. |
| `eval_config.py` | `eval.yaml` → `EvalConfig`, and the files `bat eval init` writes. |
| `orchestrator.py` | `run_evaluation()`: the task × k loop, then judging and writing results. |
| `adapter.py` | One conversation with the agent over A2A, plus its trajectory. |
| `trajectory.py` | Spans → `Trajectory`; renders it as text for the judge. |
| `evaluator.py` | `verdict()`: the deterministic checks. |
| `judge.py` | Rubric prompts, the `Judge` client, `score()`. |
| `metrics.py` | Aggregates episodes into `summary.json` and `metrics.json`. |
| `plotter.py` | Charts from `metrics.json` files. |

### Key ideas

**Behaviour comes from spans, not from A2A.** The A2A stream carries only what the agent chooses to report. The ADK writes OpenTelemetry spans as JSON lines, and those show every model, tool and agent call. The adapter therefore uses A2A only to drive the conversation and get the final answer, and reads everything else from the spans.

**Finding a conversation's spans.** Each attempt uses a unique conversation id (`<task>__try<n>`). Each turn is one request, so one trace, whose root span carries that id. `build_trajectory()` matches roots to turns in start order and collects every span with the root's `trace_id`. A called agent's spans join its caller's trace through the propagated `traceparent`, so they are found by trace id in whatever file they were written to (`extra_spans`). Since the last root span is written just after the answer is streamed, the adapter polls the files briefly until every turn has its root.

**Trajectory = a small tree of steps.** Spans are reduced to four step kinds (`model`, `tool`, `agent`, `error`), nested under the agent or tool call that contains them. The model's input is dropped (it is the history the trajectory already holds), which turns hundreds of KB of spans into a few KB. A tool span without arguments gets them from the model's tool-call decision.

**The verdict is a list of independent checks.** `verdict()` builds `(ok, reason)` pairs, one per expectation (status, each phrase, each tool or agent call, the limits, `no_errors`). The episode passes only if all pass, and the reason joins every check. If a task needs spans and none were found, the span checks collapse into one `no spans found` failure.

**The judge is a separate, optional stage.** Episodes are written to disk before judging, so a slow or broken judge never loses them. `score()` sends one prompt per rubric per episode, at most 16 at a time. Each judge answers with a fixed schema (`reasoning`, then `score`). If its server can't do structured output, that rubric switches to plain text for the rest of the run and the score is parsed from the answer. The judge sees the trajectory rendered as numbered lines within `max_trajectory_chars`: long fields are shortened first, then the middle steps of each turn are dropped, since the first and last steps are where the request is understood and the answer formed.

**Failures stay local.** An exception in one attempt becomes an `error` episode with a failed verdict; an exception for one model skips that model. Only Ctrl-C stops the whole sweep.

### Adding a check

- **A deterministic expectation:** add the field to `TaskExpected` (and to `needs_spans` if it reads the trajectory), then add a check in `evaluator.verdict()`.
- **A judge rubric:** add a prompt and an entry in `judge.RUBRICS` (rubric name → prompt, score field), plus the field in `QualitativeScores`. Metrics pick up the new score from `RUBRICS`; charts show only the scores listed in `plotter.QUALITATIVE`.
