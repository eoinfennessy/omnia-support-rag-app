FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
COPY packages/models packages/models
COPY packages/store packages/store
COPY packages/ingestion packages/ingestion

RUN uv sync --frozen --no-dev --package omnia-support-ingestion

COPY dataset dataset

ENTRYPOINT ["uv", "run", "--no-sync", "ingest-to-chroma"]
