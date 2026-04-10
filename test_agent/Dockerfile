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
RUN uv run pyinstaller agent.spec

# Strip the binary
RUN strip dist/test-agent

# Stage 2: Runtime
FROM runtime

# Copy built binary
COPY --from=builder /app/dist/test-agent /app/
COPY --from=builder /app/agent.json /app/
COPY --from=builder /app/config.yaml /app/

# Entrypoint
WORKDIR /app
ENTRYPOINT ["./test-agent"]
