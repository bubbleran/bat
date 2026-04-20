# BAT CLI Guide

This guide covers the core BAT CLI commands and common workflows.

## Prerequisites

- Python and uv installed.
- Docker installed for image build and push commands.
- For evaluation commands, run inside an existing BAT agent root that contains:
  - `agent.json`
  - `config.yaml`
  - `src/graph.py`

## Install CLI Dependencies

```bash
uv sync --group dev
```

If you need PyInstaller packaging too:

```bash
uv sync --group dev --group packaging
```

## Command Tree

```text
bat
├── init
│   └── agent
│       ├── [name=default]
│       ├── --clients, -c
│       ├── --output-dir, -o
│       ├── --force, -f
│       ├── --port
│       ├── --model
│       └── --model-provider (--model_provider)
├── add
│   └── client
│       ├── <clients>
│       └── --force, -f
├── set
│   └── env
│       ├── --port
│       ├── --model
│       ├── --model-provider (--model_provider)
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

## Global Help

```bash
bat --help
```

Command-level help:

```bash
bat init --help
bat init agent --help
bat build --help
```

## Core Workflows

### 1) Create a New Agent

Create with default name (`default`):

```bash
bat init agent
```

Create with a custom name:

```bash
bat init agent my_agent
```

Create in a specific directory:

```bash
bat init agent my_agent --output-dir .
```

Generate multiple LLM clients in the scaffold:

```bash
bat init agent my_agent --clients reformulator,planner,executor
```

### 2) Add Clients to an Existing Agent

Run this from the agent root (must contain `src/llm_clients`):

```bash
bat add client planner,executor
```

Overwrite existing generated client files:

```bash
bat add client planner,executor --force
```

### 3) Update Agent .env Values

Run this from the agent root (must contain `.env`):

```bash
bat set env --port 8080 --model gpt-4.1-mini --model-provider openai
```

Set Docker defaults for build/push:

```bash
bat set env --docker-registry hub.example.com --repo team/my-agent
```

### 4) Build and Push Docker Image

Build:

```bash
bat build --context ./my_agent --docker-registry hub.example.com --repo team/my-agent --tag latest
```

Build without cache and with version build-arg:

```bash
bat build --context ./my_agent --repo team/my-agent --tag v1 --version 1.0.0 --no-cache
```

Push:

```bash
bat push --context ./my_agent --docker-registry hub.example.com --repo team/my-agent --tag latest
```

### 5) Run Evaluation

From an existing agent root:

```bash
bat eval init
```

This creates:

- `eval/eval.yaml`
- `eval/input/tasks.json`
- `eval/output/`

Then run:

```bash
bat eval run
```

## Notes

- `build` and `push` resolve image repo and registry using this precedence:
  - Registry: `--docker-registry` -> `BAT_DOCKER_REGISTRY` env -> `.env` -> `default_registry`
  - Repo: `--repo` -> `BAT_DOCKER_REPO` env -> `.env` -> `default-repository/<project-name>`
- `eval run` requires the agent virtual environment at `.venv/bin/python` (or `.venv/Scripts/python.exe` on Windows).
- If a command fails, run the matching `--help` command first to verify required arguments and execution context.
