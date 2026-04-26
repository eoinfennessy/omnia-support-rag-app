FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages/models packages/models
COPY packages/vectorstore packages/vectorstore
COPY packages/rag packages/rag
COPY packages/evaluation packages/evaluation

RUN uv sync --frozen --no-dev --package omnia-support-eval

COPY dataset dataset
COPY prompts prompts

ENTRYPOINT ["uv", "run", "--no-sync", "--package", "omnia-support-eval"]
CMD ["evaluate"]
