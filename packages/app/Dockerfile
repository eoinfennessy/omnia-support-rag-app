FROM node:22-slim AS frontend

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages/models packages/models
COPY packages/vectorstore packages/vectorstore
COPY packages/rag packages/rag
COPY packages/app packages/app
COPY dataset/omnia_retail_knowledge_base.json dataset/omnia_retail_knowledge_base.json

RUN uv sync --frozen --no-dev --package omnia-support-app

COPY --from=frontend /frontend/build frontend/build

EXPOSE 8080
ENTRYPOINT ["uv", "run", "--no-sync", "omnia-support-app"]
