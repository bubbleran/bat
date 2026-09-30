FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base
FROM debian:bookworm-slim AS runtime

# Stage 1: Builder
FROM base AS builder

# Install objdump
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    binutils

# Copy app files
WORKDIR /app
COPY . .

# Install dependencies
RUN uv lock
RUN uv sync --locked

# Build the binary
RUN uv add pyinstaller

# Telemetry privacy floor for the agent in this image: config.yaml can raise
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

RUN uv run pyinstaller agent.spec

# Strip the binary
RUN strip dist/__PROJECT_NAME__

# Stage 2: Runtime
FROM runtime

# Copy built binary
COPY --from=builder /app/dist/__PROJECT_NAME__ /app/
COPY --from=builder /app/agent.json /app/
COPY --from=builder /app/config.yaml /app/

# Entrypoint
WORKDIR /app
ENTRYPOINT ["./__PROJECT_NAME__"]
