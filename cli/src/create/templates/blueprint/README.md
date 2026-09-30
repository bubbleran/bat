# __BLUEPRINT_NAME__

A BAT blueprint: one uv project, one frozen binary, several agents.

## Layout

- `pyproject.toml` — the one project every agent shares (one venv, one
  `uv.lock`)
- `__main__.py` — shared entrypoint; runs the agent named by its first
  argument
- `__BLUEPRINT_NAME__.spec`, `Dockerfile` — freeze every agent into one binary
  and one image
- `docker-compose.yaml` — one service per agent, all on that image
- `<agent>/` — one directory per agent, added with `bat add agent`: `app.py`,
  `agent.json`, its own `config.yaml`, `src/`

## Running an agent

    make <agent>            # CONFIG_PATH=<agent>/config.yaml uv run . <agent>
    docker compose up --build

## Adding one

    bat add agent <name>

## Building the image

    make build              # writes uv.lock first if there is none; commit it
