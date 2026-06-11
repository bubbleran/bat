# bat-cli

A CLI tool for creating, building, and evaluating BAT agent projects.

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/) installed
- Docker installed (required for `bat build` and `bat push`)
- For evaluation commands: an existing BAT agent root containing `agent.json`, `config.yaml`, and `pyproject.toml`

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
│   └── agent
│       ├── <name>
│       ├── --clients, -c
│       ├── --output-dir, -o
│       ├── --force, -f
│       ├── --port
│       ├── --model
│       └── --model-provider
├── add
│   └── client
│       ├── <clients>
│       └── --force, -f
├── set
│   └── env
│       ├── --port
│       ├── --model
│       ├── --model-provider
│       ├── --docker-registry
│       └── --repo
├── eval
│   ├── init
│   │   └── --force, -f
│   ├── run
│   ├── show
│   └── plot
│       ├── --folder, -f
│       └── --filter, -F
├── build
│   ├── --context, -C
│   ├── --docker-registry
│   ├── --repo
│   ├── --version
│   └── --no-cache
└── push
    ├── --context, -C
    ├── --docker-registry
    ├── --repo
    └── --version
```

Built-in help is available at every level:

```bash
bat --help
bat init agent --help
bat eval --help
bat build --help
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

# set the port/model/provider written to .env
bat init agent my_agent --port 9900 --model gpt-4o-mini --model-provider openai
```

### 2. Add clients to an existing agent

Run from the agent root (must contain `src/llm_clients/`):

```bash
bat add client planner,executor

# overwrite existing files
bat add client planner,executor --force
```

### 3. Update agent environment variables

Run from the agent root (updates an existing `.env`):

```bash
bat set env --port 8080 --model gpt-4o-mini --model-provider openai

# also set Docker defaults for build/push
bat set env --docker-registry hub.bubbleran.com --repo orama/labs/my-agent
```

### 4. Build and push a Docker image

```bash
# --version is used both as the image tag and as the VERSION build arg (default: latest)
bat build --context ./my_agent --docker-registry hub.bubbleran.com --repo orama/labs/my-agent --version latest

# no-cache build with a specific version
bat build --context ./my_agent --repo orama/labs/my-agent --version 1.0.0 --no-cache

bat push --context ./my_agent --docker-registry hub.bubbleran.com --repo orama/labs/my-agent --version latest
```

The image reference is always `{registry}/{repo}:{version}`.

If `BAT_DOCKER_REGISTRY` and `BAT_DOCKER_REPO` are already set in `.env` or the shell, `--docker-registry` and `--repo` can be omitted.

**Precedence** (both `--docker-registry` / `--repo`):

1. CLI flag
2. Shell environment variable (`BAT_DOCKER_REGISTRY` / `BAT_DOCKER_REPO`)
3. `.env` file in the current directory
4. Hardcoded default (`default_registry` / `default-repository/<project-name>`)

### 5. Run evaluation

Run all `eval` commands from an existing agent root (must contain `agent.json`,
`config.yaml`, and `pyproject.toml`):

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
  # api_key_env: BAT_JUDGE_API_KEY      # env var name holding the judge's API key
```

Notes:

- `bat eval run` starts the agent via `uv run .` from the agent root and waits until
  the agent's `config.yaml` endpoint (`endpoint.url:port`) accepts a TCP
  connection, so the agent project must have its dependencies installed (its own
  `.venv`). The eval reads that endpoint from the agent's `config.yaml`; it is no
  longer configured in `eval.yaml`.
- `models` entries may also be written as `"<provider>:<model>"` strings.
- For models that require an API key, set it in the agent's `.env` under
  `<PROVIDER>_API_KEY` (e.g. `OPENAI_API_KEY`).

### 6. Plot evaluation metrics

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
