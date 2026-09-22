# __BLUEPRINT_NAME__

A BAT blueprint: one uv project, one frozen binary, several agents.

## Layout

- `blueprint.yaml` — identity and the list of agents
- `__main__.py` — shared entrypoint; picks the agent from `argv[1]`, or from
  `AGENT_MODE` when there is none
- `config.yaml` — the blueprint's own config; each agent has its own
- `<agent>/` — one directory per agent, added with `bat add agent`

## Running an agent

    make <agent>            # CONFIG_PATH=<agent>/config.yaml uv run . <agent>
    docker compose up       # every agent at once

## Adding one

    bat add agent <name>
