FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages/models packages/models
COPY packages/vectorstore packages/vectorstore
COPY packages/ingestion packages/ingestion

RUN uv sync --frozen --no-dev --package omnia-support-ingestion

COPY dataset dataset

ENTRYPOINT ["uv", "run", "--no-sync", "ingest-to-chroma"]
