FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base
FROM debian:bookworm-slim AS runtime

# Stage 1: Builder
FROM base AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    binutils

WORKDIR /app
COPY . .

RUN uv lock
RUN uv sync --locked

RUN uv add pyinstaller
RUN uv run pyinstaller __BLUEPRINT_NAME__.spec

RUN strip dist/__BLUEPRINT_NAME__

# Collect each agent's runtime files, keeping the directory layout the
# CONFIG_PATH values in the compose file and the Deployment expect. Discovered
# rather than listed, so adding an agent needs no edit here.
RUN mkdir -p /out && \
    for config in */config.yaml; do \
        agent=$(dirname "$config"); \
        mkdir -p "/out/$agent"; \
        cp "$agent/config.yaml" "/out/$agent/"; \
        if [ -f "$agent/agent.json" ]; then cp "$agent/agent.json" "/out/$agent/"; fi; \
    done

# Stage 2: Runtime
FROM runtime

COPY --from=builder /app/dist/__BLUEPRINT_NAME__ /app/
COPY --from=builder /app/config.yaml /app/
COPY --from=builder /out/ /app/

WORKDIR /app
# The agent is the first argument, or AGENT_MODE when there is none:
#   docker run <image> netops
ENTRYPOINT ["./__BLUEPRINT_NAME__"]
