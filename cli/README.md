# bat-cli

A CLI tool for creating, building, and evaluating BAT agent projects: standalone
agents and blueprints.

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/) installed
- Docker installed (required for `bat build` and `bat push`)
- For evaluation commands: a BAT agent — a standalone agent root (`agent.json`, `config.yaml` and `pyproject.toml`), or an agent folder of a blueprint

---

## Installation

### Option A — install system-wide with `uv tool` (recommended)

Installs `bat` into an isolated environment and puts the executable on your `PATH`,
so it is available from any directory.

````bash
# from PyPI
uv tool install bat-cli


Make sure the uv tools bin directory is on your `PATH` (uv prints the path on first
install; this is usually `~/.local/bin`):

```bash
uv tool update-shell      # adds the uv tools dir to your shell profile
````

Then verify:

```bash
bat --help
```

To upgrade or remove later:

```bash
uv tool upgrade bat-cli
uv tool uninstall bat-cli
```

### Option B — install into a virtual environment with `uv pip`

Use this when you want `bat` scoped to a specific project/venv rather than installed
globally.

```bash
uv venv                      # create .venv (skip if you already have one)
source .venv/bin/activate    # .venv\Scripts\activate on Windows

# from PyPI
uv pip install bat-cli

# or from a local checkout (run from the cli/ directory)
uv pip install .             # add -e for an editable/development install
```

`bat` is available whenever that virtual environment is active:

```bash
bat --help
```

### Option C — run without installing (development)

From the `cli/` directory:

```bash
uv sync --group dev
uv run bat --help
```

All examples below show `bat ...`; replace with `uv run bat ...` when using this option.

---

## Command Tree

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

Built-in help is available at every level:

```bash
bat --help
bat init agent --help
bat add agent --help
bat eval --help
bat manifests aifabric --help
bat manifests composition-model --help
```

---

## Workflows

### 1. Create a new agent

```bash
bat init agent my_agent

# specific output directory
bat init agent my_agent --output-dir .

# pre-generate LLM clients
bat init agent my_agent --clients reformulator,planner,executor

# set the port/model/provider written to config.yaml
bat init agent my_agent --port 9900 --model gpt-6-luna --model-provider openai

# the lowest telemetry privacy level the agent allows (none|content|names|full)
bat init agent my_agent --privacy content
```

### 2. Create a blueprint and add agents

Agents live in a blueprint: one uv project, one image, one folder per agent.

```bash
bat init blueprint my_blueprint
cd my_blueprint

# pre-generate LLM clients
bat add agent netops --clients reformulator,planner,executor

# set the port/model/provider written to netops/config.yaml
# (without --port: one past the highest port already used)
bat add agent hermes --port 9901 --model gpt-6-luna --model-provider openai

# the lowest telemetry privacy level the agent allows (none|content|names|full)
bat add agent kpi --privacy content

uv run . netops          # or: make netops
```

Agent names must be valid Python identifiers (`cluster_view`, not
`cluster-view`). `bat add agent` also gives every agent a service in
`docker-compose.yaml`.

### 3. Add clients to an existing agent

Run from the agent root (must contain `src/llm_clients/`):

```bash
bat add client planner,executor

# overwrite existing files
bat add client planner,executor --force
```

### 4. Update agent settings

`bat set config` writes an agent's runtime values into its `config.yaml`
(`endpoint.port`, `model.name`, `model.provider`, `model.reasoning_effort`,
`model.service_tier`). Run it from the agent root, or name the agent from a
blueprint's root:

```bash
bat set config --port 8080 --model gpt-6-luna --model-provider openai
bat set config netops --port 9309
bat set config netops --model gpt-5-mini --reasoning-effort low --service-tier flex
```

Reasoning effort and service tier have no environment variable, so
`config.yaml` is the only place to set them; their values depend on the
provider. A reasoning effort is supported by gpt-5 and later.

`bat set image` writes the image settings into the project's `Makefile`
(`DOCKER_REGISTRY`, `REPO`; the blueprint's, from an agent folder), where
`make build` reads them as well as `bat build`:

```bash
bat set image --docker-registry hub.bubbleran.com --repo orama/labs/my-agent
```

### 5. Build and push a Docker image

`bat build` and `bat push` run the project's `make build` / `make push`
(from an agent folder of a blueprint, the blueprint's), passing only the
settings they are given:

```bash
# the Makefile's defaults: demo:<git tag or commit>, or demo:dev outside git
bat build

# registry, repository and version (also the VERSION build arg)
bat build --docker-registry hub.bubbleran.com --repo orama/labs/demo --version 1.0.0

# builds, then pushes; a registry is required
bat push --docker-registry hub.bubbleran.com --repo orama/labs/demo --version 1.0.0
```

The same works with `make` directly:
`make push DOCKER_REGISTRY=hub.bubbleran.com REPO=orama/labs/demo VERSION=1.0.0`.

Without a registry (none given, none in the Makefile) the image is built for
this machine only, and `bat push` refuses, since the image would go to Docker
Hub. BubbleRAN can provide a registry for your images.

The image reference is always `{registry}/{repo}:{version}`.

Once `bat set image` has written them to the Makefile, `--docker-registry` and
`--repo` can be omitted.

**Precedence** (both `--docker-registry` / `--repo`):

1. CLI flag
2. Shell environment variable (`DOCKER_REGISTRY` / `REPO`), which make reads itself
3. The Makefile's `DOCKER_REGISTRY ?=` / `REPO ?=`

A project without a Makefile reads the same flags and shell variables; without
a registry its image is `<name>:latest`, for this machine only.

### 6. Run evaluation

Run `eval` commands from an agent root — a standalone agent's, or an agent
folder of a blueprint. From a blueprint's root, name the agent
(`bat eval run netops`):

```bash
# scaffold evaluation files
bat eval init

# inspect the resolved configuration
bat eval show

# run evaluation
bat eval run
```

`eval init` creates:

- `eval/eval.yaml`
- `eval/input/tasks.json`
- `eval/output/`

Minimal `eval/eval.yaml`:

```yaml
evaluation:
  dataset: eval/input/tasks.json # default path if omitted
  output_dir: eval/output # default path if omitted
  agent_startup_timeout_s: 45
  agent_shutdown_timeout_s: 10
  k: 1
  qualitative: false # set true to enable LLM judge scoring

models:
  - provider: openai
    model: your-model-name
  - provider: ollama
    model: your-local-model
    base_url: http://localhost:11434

# required only when qualitative: true
judge:
  provider: ollama
  model: local-judge-model
  base_url: http://localhost:11434
  # api_key_env: BAT_JUDGE_API_KEY      # env var (shell or .env) holding the judge's API key
  # mode: full                          # outcome: one lenient task-success score from the request and final response only
```

Notes:

- `bat eval run` starts the agent — `uv run .` from a standalone agent's root,
  `CONFIG_PATH=<agent>/config.yaml uv run . <agent>` from a blueprint's — and
  waits until the agent's `config.yaml` endpoint (`endpoint.url:port`) accepts a
  TCP connection. The agent's output goes to `agent-<n>.log` in the run's
  output folder; if it stops before it is ready, the eval prints the error.
- The agent gets the project's `.env` (the blueprint's, for a blueprint agent),
  as under `make <agent>`, so API keys go there (e.g. `OPENAI_API_KEY`); a
  variable already exported in the shell wins.
- `bat eval run` refuses an agent whose `telemetry_privacy_floor` is above
  `none`: its spans would hide what the eval reads. Set it to `"none"` while
  you evaluate.
- `models` entries may also be written as `"<provider>:<model>"` strings.

### 7. Plot evaluation metrics

`bat eval plot` reads the `metrics.json` files produced by `eval run` and renders
charts. Point `--folder` at an evaluation output directory; each sub-folder
containing a `metrics.json` is treated as one run.

```bash
# plot every run found under the output folder
bat eval plot --folder eval/output

# restrict the per-task charts to task ids containing a substring
bat eval plot --folder eval/output --filter smoke
```

Charts are saved back into the given folder. `--filter` only narrows the per-task
charts; summary charts always cover all runs.
