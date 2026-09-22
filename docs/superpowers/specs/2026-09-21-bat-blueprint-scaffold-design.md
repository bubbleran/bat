# Blueprint scaffolding in bat-cli

**Status:** approved design, not yet implemented
**Date:** 2026-09-21

## Why

`bat-cli` scaffolds and evaluates one shape of project: a standalone agent, where
`config.yaml`, `agent.json` and `pyproject.toml` all sit in the same directory and
the agent is started with `uv run .`.

Real deployments no longer look like that. A **blueprint** — the shape used by
`orama-dev-enrico/blueprints/automation` — is a single uv project holding several
agents that share one virtualenv, one lockfile and one PyInstaller binary:

```
blueprints/automation/
├── blueprint.yaml          # name, namespace, provider, version, agents{}
├── pyproject.toml          # one project, one venv, one uv.lock
├── automation.spec         # one PyInstaller binary for every agent
├── __main__.py             # dispatcher: argv[1], else AGENT_MODE
├── config.yaml             # blueprint-level config
├── Dockerfile · docker-compose.yaml · Makefile · .env
├── netops/     agent.json · app.py · config.yaml · src/ · eval/
├── hermes/     agent.json · app.py · config.yaml · src/
└── cluster-view/           # nested MCP server: its own uv project
```

Two consequences the CLI does not handle:

1. **Starting an agent takes more than `uv run .`.** The blueprint's Makefile runs
   `CONFIG_PATH=<agent>/config.yaml uv run . <agent>`. The selector is `argv[1]`
   locally and `AGENT_MODE` on the cluster (the orama operator renders a Deployment
   with no command or args). `CONFIG_PATH` is needed because the SDK defaults to
   `./config.yaml`, which inside a blueprint is the *shared* one, not the agent's.
2. **`bat eval` refuses to run at all.** `cli/src/eval/commands.py` requires
   `config.yaml`, `agent.json` and `pyproject.toml` in one directory; in a blueprint
   the first two are in the agent directory and the third is at the blueprint root.
   `blueprints/automation/supervisor/eval/eval.yaml` already exists and cannot be run.

## Scope

In: `bat init blueprint`, `bat add agent`, and making `bat eval` blueprint-aware.
The generated blueprint includes local packaging (PyInstaller spec, Dockerfile,
`.dockerignore`, `docker-compose.yaml`, Makefile).

Out, for now: `bat build` / `bat push` stay per-agent and do not build the
blueprint's single image; cluster deployment descriptors (`aifabric.yaml`,
`composition-model.yaml`, RBAC `rules` per agent) are not generated;
`bat init agent` keeps working unchanged for the standalone shape.

## The target resolver

One new module, `cli/src/project.py`, owns "where am I":

```python
@dataclass(frozen=True)
class AgentTarget:
    project_root: Path      # where `uv run .` works (holds pyproject.toml)
    agent_dir: Path         # where config.yaml and agent.json live
    agent_name: str | None  # dispatcher selector; None when standalone
```

Resolution walks up from the working directory looking for `blueprint.yaml`.

- **Found** → `project_root` is the blueprint, `agent_dir` is the working directory,
  `agent_name` is its name relative to the blueprint. Called from the blueprint root
  itself, it fails with a message telling the caller to enter an agent directory or
  name one. Naming one is the optional `AGENT` argument the `bat eval` subcommands
  take (`bat eval run netops`): it resolves against the blueprint root and wins over
  the working directory.
- **Not found** → standalone: the working directory must hold `config.yaml`,
  `agent.json` and `pyproject.toml`, which is today's check, unchanged.

Everything else is derived from the target:

| | standalone | blueprint |
|---|---|---|
| launch command | `uv run .` | `uv run . <agent_name>` |
| `CONFIG_PATH` | unset | `<agent_name>/config.yaml` |
| config patched by eval | `./config.yaml` | `<agent_dir>/config.yaml` |
| agent URL read from | `./config.yaml` | `<agent_dir>/config.yaml` |

## `bat init blueprint <name>`

Creates an empty blueprint — no agents:

`blueprint.yaml` (`name`, `namespace`, `provider`, `version`, `agents: {}`),
`pyproject.toml`, `__main__.py`, `config.yaml`, `.env`, `.python-version`,
`.gitignore`, `README.md`, `<name>.spec`, `Dockerfile`, `.dockerignore`,
`docker-compose.yaml`, `Makefile`.

The `pyproject.toml` pins `bat-adk[<provider>,telemetry]>=<version>` using the same
`_bat_adk_extras` helper the agent scaffold now uses: without the `telemetry` extra
the SDK's tracers are no-ops and `bat eval` reports zero tokens for every episode.

## `bat add agent <name>`

Creates `<blueprint>/<name>/` — `__init__.py` (re-exports `run`), `app.py`,
`agent.json`, `config.yaml`, `src/graph.py`, `src/llm_clients/` — that is, today's
agent templates minus everything that moved up to the blueprint (pyproject,
Dockerfile, Makefile, spec, `__main__.py`).

The port comes from `--port`, or defaults to one past the highest already used in
the blueprint.

The agent is then registered in three places:

- **`blueprint.yaml`** — an entry in the `agents` map.
- **`__main__.py`** — a region between `# bat:agents:begin` / `# bat:agents:end`,
  regenerated from `blueprint.yaml`. Imports stay **static and explicit**, one
  `if`/`elif` branch per agent with the import inside the branch: PyInstaller has to
  see them to freeze them, and keeping them lazy avoids paying every agent's imports
  at startup. The markers exist so hand edits outside the region survive.
- **`docker-compose.yaml`** — one appended service.

The **Makefile needs no edit**. Instead of a static agent list it discovers them:

```make
AGENTS := $(patsubst %/config.yaml,%,$(wildcard */config.yaml))

$(AGENTS):
	CONFIG_PATH=$@/config.yaml uv run . $@
```

One fewer place to keep in sync.

## `bat eval`, blueprint-aware

`_start_agent_process`, `_patch_agent_config` and `_agent_url_from_config` take an
`AgentTarget` instead of an `agent_root`. `_start_agent_process` appends the agent
name to the command and sets `CONFIG_PATH` when `agent_name` is not `None`; the other
two read and write the agent's own `config.yaml`. The eval's own layout
(`eval/eval.yaml`, `eval/input`, `eval/output`) stays relative to the agent
directory, which is what `blueprints/automation/supervisor/eval/eval.yaml` already
assumes.

## Testing

Scaffolding, with `CliRunner` on `tmp_path` as in `cli/tests/test_create_new_agent.py`:
the file set `bat init blueprint` produces; `bat add agent` registering the agent in
all three places; **two `add agent` calls in a row** both surviving (the failure mode
of a marker rewrite is duplicating or dropping the previous entry); `bat add agent`
outside a blueprint failing clearly; port allocation.

Resolver, as plain unit tests: blueprint agent directory, standalone agent,
blueprint root, unrelated directory.

Eval, extending `cli/tests/test_eval_commands.py:198`, whose `fake_popen` already
captures the command, cwd and env:

```python
assert captured["popen_cmd"] == ["uv", "run", ".", "netops"]
assert captured["popen_cwd"] == blueprint_root
assert captured["popen_env"]["CONFIG_PATH"] == "netops/config.yaml"
```

plus: the patched file is the agent's `config.yaml` and is restored afterwards, and
the standalone path still launches `uv run .` with no `CONFIG_PATH`.

## Related

Landed separately, and independent of this design: the agent scaffold now pins
`bat-adk[<provider>,telemetry]>=2026.9.10a0` instead of `bat-adk>=2026.06`, which
PEP 440 never resolved to the span-emitting pre-release.

Known and not addressed here: the multi-turn span collection race in
`eval/engine/adapter.py`, and `_patch_agent_config` replacing the whole `telemetry`
block (dropping `privacy`) instead of merging into it.
