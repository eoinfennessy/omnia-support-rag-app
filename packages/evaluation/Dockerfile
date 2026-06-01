FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY packages/models/pyproject.toml packages/models/pyproject.toml
COPY packages/store/pyproject.toml packages/store/pyproject.toml
COPY packages/rag/pyproject.toml packages/rag/pyproject.toml
COPY packages/agents/pyproject.toml packages/agents/pyproject.toml
COPY packages/mcp-server/pyproject.toml packages/mcp-server/pyproject.toml
COPY packages/evaluation/pyproject.toml packages/evaluation/pyproject.toml

RUN uv sync --frozen --no-dev --package omnia-support-eval --package omnia-support-mcp --no-install-workspace

COPY packages/models packages/models
COPY packages/store packages/store
COPY packages/rag packages/rag
COPY packages/agents packages/agents
COPY packages/mcp-server packages/mcp-server
COPY packages/evaluation packages/evaluation

RUN uv sync --frozen --no-dev --package omnia-support-eval

COPY dataset dataset
COPY prompts prompts
COPY packages/evaluation/src/omnia_support_eval/information/eval_config.yaml eval_config.yaml

ENTRYPOINT ["uv", "run", "--no-sync", "--package", "omnia-support-eval"]
CMD ["evaluate", "information", "--config", "eval_config.yaml"]
