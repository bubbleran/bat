# bat-cli

A CLI tool for creating and managing BAT agent projects.

## Quick Start

1. Install dependencies:

```bash
uv sync --group dev
```

If you also need packaging tools (PyInstaller):

```bash
uv sync --group dev --group packaging
```

2. Create a new agent scaffold:

```bash
uv run bat init agent
```

This creates a `default` folder by default.

Use a custom name:

```bash
uv run bat init agent my_agent
```

Choose an output directory:

```bash
uv run bat init agent my_agent --output-dir .
```

Generate custom LLM clients:

```bash
uv run bat init agent rng --clients talk,discuss
```

Set runtime and build environment variables from an existing agent root:

```bash
cd my_agent
uv run bat set env --port 8080 --model gpt-4.1-mini --model-provider openai --docker-registry hub.bubbleran.com --repo orama/labs/my-agent
```

This command updates `.env` (creating it if missing).

When running `init agent`, the scaffold creates `.env` as the runtime file.

Build a Docker image:

```bash
uv run bat build --context ./my_agent --docker-registry hub.bubbleran.com --repo orama/labs/my-agent --tag latest
```

Push a Docker image:

```bash
uv run bat push --context ./my_agent --docker-registry hub.bubbleran.com --repo orama/labs/my-agent --tag latest
```

Set a default repository for an agent via environment variable:

```bash
# in ./my_agent/.env
BAT_DOCKER_REPO=orama/labs/my-agent
BAT_DOCKER_REGISTRY=hub.bubbleran.com
```

Then you can omit `--repo`:

```bash
uv run bat build --context ./my_agent --docker-registry hub.bubbleran.com --tag latest
uv run bat push --context ./my_agent --docker-registry hub.bubbleran.com --tag latest
```

Repository precedence is:

1. `--repo`
2. `BAT_DOCKER_REPO` from shell environment
3. `BAT_DOCKER_REPO` from `.env` in current directory
4. auto-generated default: `default-repository/<project-name>`

Registry precedence is:

1. `--docker-registry`
2. `BAT_DOCKER_REGISTRY` from shell environment
3. `BAT_DOCKER_REGISTRY` from `.env` in current directory
4. fallback default: `default_registry`

## Evaluation Commands

Run these commands from an existing BAT agent root (the folder containing `agent.json`, `config.yaml`, and `src/graph.py`).

Initialize evaluation scaffold:

```bash
uv run bat eval init
```

This creates:

- `eval/eval.yaml`
- `eval/input/tasks.json`
- `eval/output/`

Run evaluation:

```bash
uv run bat eval run
```

How it works:

- Reads settings from `eval/eval.yaml`.
- Runs each configured model.
- Executes the agent in the agent's own virtual environment (`.venv/bin/python`).
- Writes artifacts under `eval/output/<task_id>/...`.

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

## Build Standalone Executable (PyInstaller)

Build a standalone executable for the current OS (on Windows this creates `bat.exe`):

```bash
uv sync --group packaging
uv run pyinstaller --clean --noconfirm bat_cli.spec
```

Or if you are not in a synchronized environment:

```bash
uv run --group packaging pyinstaller --clean --noconfirm bat_cli.spec
```

Output:

- One-file executable: `dist/bat` (Linux/macOS) or `dist/bat.exe` (Windows)

Install on your machine (Windows example):

1. Copy `dist/bat.exe` to a folder, for example `C:\tools\bat`.
2. Add that folder to your `PATH`.
3. Open a new terminal and run:

```bash
bat --help
```

Notes:

- PyInstaller builds are OS-specific. Build on each target OS.
- The spec includes `create` package data so scaffold templates are bundled in the executable.
