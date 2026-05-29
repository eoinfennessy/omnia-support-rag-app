FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages/models packages/models
COPY packages/mcp-server packages/mcp-server

RUN uv sync --frozen --no-dev --package omnia-support-mcp

EXPOSE 8081
ENTRYPOINT ["uv", "run", "--no-sync", "omnia-support-mcp"]
