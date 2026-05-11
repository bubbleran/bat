# bat-cli

A CLI tool for creating, building, and evaluating BAT agent projects.

## Prerequisites

- Python and [uv](https://docs.astral.sh/uv/) installed
- Docker installed (required for `bat build` and `bat push`)
- For evaluation commands: an existing BAT agent root containing `agent.json`, `config.yaml`, and `src/graph.py`

---

## Installation

### Option A — build and install a standalone binary (Linux/macOS)

Run the helper script from the repo root:

```bash
bash cli/build_and_install.sh
```

This will:

1. Sync `dev` and `packaging` dependency groups via `uv`.
2. Build a one-file executable with PyInstaller.
3. Move it to `~/.local/bin/bat` (uses `sudo` only when necessary).

Make sure `~/.local/bin` is in your `PATH`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc   # or ~/.zshrc
source ~/.bashrc
```

Then verify:

```bash
bat --help
```

### Option B — build manually

```bash
uv sync --group dev --group packaging
uv run pyinstaller --clean --noconfirm bat_cli.spec
# binary is at dist/bat (Linux/macOS) or dist/bat.exe (Windows)
```

On **Windows**, copy `dist/bat.exe` to a folder on your `PATH` (e.g. `C:\tools\bat`) and open a new terminal.

> PyInstaller builds are OS-specific — build on each target OS.

### Option C — run without installing (development)

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
│       ├── [name=default]
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
│   └── run
├── build
│   ├── --context, -C
│   ├── --docker-registry
│   ├── --repo
│   ├── --tag
│   ├── --version
│   └── --no-cache
└── push
    ├── --context, -C
    ├── --docker-registry
    ├── --repo
    └── --tag
```

Built-in help is available at every level:

```bash
bat --help
bat init agent --help
bat build --help
```

---

## Workflows

### 1. Create a new agent

```bash
# default name
bat init agent

# custom name
bat init agent my_agent

# specific output directory
bat init agent my_agent --output-dir .

# pre-generate LLM clients
bat init agent my_agent --clients reformulator,planner,executor
```

### 2. Add clients to an existing agent

Run from the agent root (must contain `src/llm_clients/`):

```bash
bat add client planner,executor

# overwrite existing files
bat add client planner,executor --force
```

### 3. Update agent environment variables

Run from the agent root (updates or creates `.env`):

```bash
bat set env --port 8080 --model gpt-4.1-mini --model-provider openai

# also set Docker defaults for build/push
bat set env --docker-registry hub.bubbleran.com --repo orama/labs/my-agent
```

### 4. Build and push a Docker image

```bash
bat build --context ./my_agent --docker-registry hub.bubbleran.com --repo orama/labs/my-agent --tag latest

# no-cache build with version build-arg
bat build --context ./my_agent --repo orama/labs/my-agent --tag v1 --version 1.0.0 --no-cache

bat push --context ./my_agent --docker-registry hub.bubbleran.com --repo orama/labs/my-agent --tag latest
```

If `BAT_DOCKER_REGISTRY` and `BAT_DOCKER_REPO` are already set in `.env` or the shell, `--docker-registry` and `--repo` can be omitted.

**Precedence** (both `--docker-registry` / `--repo`):

1. CLI flag
2. Shell environment variable (`BAT_DOCKER_REGISTRY` / `BAT_DOCKER_REPO`)
3. `.env` file in the current directory
4. Hardcoded default (`default_registry` / `default-repository/<project-name>`)

### 5. Run evaluation

From an existing agent root:

```bash
# scaffold evaluation files
bat eval init

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
  dataset: eval/input/tasks.json
  output_dir: eval/output
  k: 1
  qualitative: true
  save_attempts: false

judge:
  provider: ollama
  model: your-judge-model
  base_url: http://localhost:11434

models:
  - provider: openai
    model: your-model-name
```

`eval run` requires the agent virtual environment at `.venv/bin/python` (`.venv/Scripts/python.exe` on Windows).
