MLFLOW_TRACKING_URI ?= http://localhost:5000
RETRIEVAL_MODE ?= hybrid

.PHONY: help lint lint-fix format format-check type-check check fix install-hooks up down infra-local infra-docker ingest test register-prompt eval-local eval-docker frontend-build app

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$|^##@' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next} {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

##@ Docker
up: ## Start all services
	docker compose --profile app up --build

down: ## Stop all services
	docker compose --profile app --profile eval down

##@ Development
infra-local: ## Start ChromaDB + MLflow locally via uv
	@trap 'kill 0' EXIT; \
	uv run --package omnia-support-ingestion chroma run \
		--path ./data/chroma \
		--port 8000 & \
	uv run --package omnia-support-eval mlflow server \
		--backend-store-uri sqlite:///data/mlflow/mlflow.db \
		--default-artifact-root ./data/mlflow/artifacts \
		--host 0.0.0.0 --port 5000

infra-docker: ## Start ChromaDB + MLflow via Docker Compose
	docker compose up chromadb mlflow

ingest: ## Run ingestion pipeline locally via uv
	uv run ingest-to-chroma

test: ## Run tests
	MLFLOW_TRACKING_URI="sqlite:///:memory:" uv run --group test pytest

##@ Evaluation
register-prompt: ## Register all prompts to MLflow
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		uv run --package omnia-support-eval register-prompt
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		uv run --package omnia-support-eval register-prompt \
		--prompt-path prompts/interpret_query.yaml \
		--prompt-name interpret_query

eval-local: register-prompt ## Run evaluation locally
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	MLFLOW_GENAI_EVALUATION_MAX_WORKERS=1 \
		uv run --package omnia-support-eval evaluate --config packages/evaluation/eval_config.yaml --retrieval-mode $(RETRIEVAL_MODE)

eval-docker: ## Run evaluation in Docker
	docker compose --profile eval run --rm evaluation

##@ Application
frontend-build: ## Build the SvelteKit frontend
	cd frontend && npm ci && npm run build

app: frontend-build ## Start the app server locally
	RETRIEVAL_MODE=$(RETRIEVAL_MODE) uv run omnia-support-app

##@ Code quality
lint: ## Run ruff linter
	uv run ruff check .

lint-fix: ## Run ruff linter with auto-fix
	uv run ruff check --fix .

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check formatting without changes
	uv run ruff format --check .

type-check: ## Run ty type checker (strict)
	uv run ty check --error-on-warning .

check: lint format-check type-check test ## Run all checks

fix: lint-fix format ## Auto-fix lint and formatting issues

##@ Setup
install-hooks: ## Install git pre-commit hook
	ln -sf ../../.hooks/pre-commit .git/hooks/pre-commit
	@echo "Pre-commit hook installed"
