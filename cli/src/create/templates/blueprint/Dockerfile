FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS base
FROM debian:bookworm-slim AS runtime

# Stage 1: Builder
FROM base AS builder

# binutils provides `strip`
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    binutils

WORKDIR /app

# Dependencies first, from the committed lockfile and without the project
# source: this layer is rebuilt only when pyproject.toml or uv.lock change,
# not on every source edit. pyinstaller is in the dev group, which the
# default sync installs. (`make build` writes uv.lock when there is none.)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen

# Telemetry privacy floor for every agent in this image: config.yaml can raise
# it but never lower it. Written into a module right before freezing, so it is
# compiled into the binary rather than read at runtime -- neither a mounted
# config.yaml nor an env var can undo it. Override it per build with
# --build-arg TELEMETRY_PRIVACY_FLOOR=<none|content|names|full>.
ARG TELEMETRY_PRIVACY_FLOOR=__TELEMETRY_PRIVACY_FLOOR__
RUN case "$TELEMETRY_PRIVACY_FLOOR" in \
        none|content|names|full) ;; \
        *) echo "Unknown TELEMETRY_PRIVACY_FLOOR '$TELEMETRY_PRIVACY_FLOOR'; expected none|content|names|full." >&2; exit 1 ;; \
    esac && \
    printf 'TELEMETRY_PRIVACY_FLOOR = "%s"\n' "$TELEMETRY_PRIVACY_FLOOR" > telemetry_floor.py

RUN uv run pyinstaller __BLUEPRINT_NAME__.spec
RUN strip dist/__BLUEPRINT_NAME__

# Collect each agent's card, keeping the <agent>/agent.json layout the
# agent_card in its config.yaml points at. Discovered rather than listed, so
# adding an agent needs no edit here. config.yaml is not baked in: it is
# deployment-specific, mounted by docker-compose.yaml and provided by the
# platform on a cluster.
RUN mkdir -p /out && \
    for card in */agent.json; do \
        [ -f "$card" ] || continue; \
        mkdir -p "/out/$(dirname "$card")"; \
        cp "$card" "/out/$card"; \
    done

# Stage 2: Runtime
FROM runtime

# Release version (`make build` passes `git describe`). Labels the image and,
# when set, is stamped into every agent card served at
# /.well-known/agent-card.json.
ARG VERSION=
LABEL org.opencontainers.image.version="${VERSION}"

# Checkpoints may only deserialize the types the agent's graph declares,
# instead of importing whatever a stored checkpoint names.
ENV LANGGRAPH_STRICT_MSGPACK=true

# curl only serves the docker-compose.yaml healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# The binary lives outside /app because an agent directory there may share
# the blueprint's name.
COPY --from=builder /app/dist/__BLUEPRINT_NAME__ /usr/local/bin/__BLUEPRINT_NAME__
COPY --from=builder /out/ /app/

RUN if [ -n "$VERSION" ]; then \
        for card in /app/*/agent.json; do \
            [ -f "$card" ] || continue; \
            sed -i "s/\"version\": *\"[^\"]*\"/\"version\": \"${VERSION}\"/" "$card"; \
        done; \
    fi

WORKDIR /app
# The agent is the first argument: docker run <image> <agent>
ENTRYPOINT ["__BLUEPRINT_NAME__"]
