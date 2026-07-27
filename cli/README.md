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

Run `eval` commands from the agent's eval directory. `bat eval init` and the
`local` target require a full agent root (`agent.json`, `config.yaml`,
`pyproject.toml` + `.venv`); the `docker` and `remote` targets only need
`eval/eval.yaml` and the dataset (see [Execution target](#execution-target)):

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
  agent_url: http://127.0.0.1:9900 # must include the scheme; this is the default
  agent_startup_timeout_s: 45
  agent_shutdown_timeout_s: 10
  k: 1
  qualitative: false # set true to enable LLM judge scoring
  target: local # local | docker | remote (default: local)

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

#### Execution target

`evaluation.target` selects how the agent under test is launched/reached. The
evaluation engine itself is identical across targets — only the agent's
lifecycle differs.

| Target | Who runs the agent | Model matrix | Requires |
| --- | --- | --- | --- |
| `local` (default) | CLI runs it from source via `uv run .`, restarting once per model | full `models:` list | agent root + `.venv` |
| `docker` | CLI starts one container per model, injecting the model via `-e` | full `models:` list | a built agent image + Docker |
| `remote` | Agent is already running at `agent_url`; CLI does not manage it | single model only (used as the result label) | only `eval.yaml` + dataset |

**`docker`** — evaluates a packaged agent image, preserving the model matrix
(the CLI restarts a container per model entry):

```yaml
evaluation:
  target: docker
  agent_url: http://127.0.0.1:9900
  # image defaults to the same {registry}/{repo}:{version} reference as
  # `bat build`/`bat push`; override explicitly if needed:
  # image: hub.bubbleran.com/orama/labs/my-agent:latest
  # image_version: latest      # used only when image is not set
  # docker_network: host       # host networking (Linux) reaches localhost providers
```

- The runtime image has no baked model/secrets, so the CLI passes
  `MODEL`/`MODEL_PROVIDER`/`BASE_URL`/`URL`/`PORT` (and any per-model `env:`) as
  `-e` flags, and forwards the agent's `.env` via `--env-file` for API keys.
  Explicit `-e` flags override the `.env`.
- With `docker_network: host` (Linux default) a model `base_url` of
  `http://localhost:11434` (e.g. Ollama on the host) is reachable from inside
  the container. With bridge networking the CLI publishes the port instead.
- Containers are named `bat-eval-<task_id>-<idx>` and removed on completion.

**`remote`** — evaluates an agent that is already running (a published Docker
container, a port-forward, an ingress). The CLI starts/stops nothing and
validates the agent by connecting to `agent_url`:

```yaml
evaluation:
  target: remote
  agent_url: http://127.0.0.1:9900   # the running agent's reachable URL
models:
  - provider: openai                 # label only — the deployed model is fixed
    model: gpt-4o-mini
```

The agent is evaluated **as-deployed**: its model is whatever it was started
with, so `remote` accepts exactly one `models:` entry, used only to label the
results. The judge (when `qualitative: true`) still runs locally in the CLI, so
judge config and keys work unchanged.

Notes:

- For `local`, `bat eval run` starts the agent via `uv run .` from the agent
  root and waits until `agent_url` accepts a TCP connection, so the agent
  project must have its dependencies installed (its own `.venv`). `docker` and
  `remote` only need `eval/eval.yaml` and the dataset.
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
