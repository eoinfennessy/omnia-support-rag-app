MLFLOW_TRACKING_URI ?= http://localhost:5000
RETRIEVAL_MODE ?= hybrid

.PHONY: help lint lint-fix format format-check type-check check fix install-hooks up down infra-local infra-docker ingest test register-prompt eval-information-local eval-triage-local eval-action-local eval-aggregator-local eval-information-docker eval-triage-docker eval-action-docker eval-aggregator-docker frontend-build app mcp-local mcp-docker mcp-inspect

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
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		uv run --package omnia-support-eval register-prompt \
		--prompt-path prompts/triage.yaml \
		--prompt-name triage
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		uv run --package omnia-support-eval register-prompt \
		--prompt-path prompts/action.yaml \
		--prompt-name action
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		uv run --package omnia-support-eval register-prompt \
		--prompt-path prompts/aggregate.yaml \
		--prompt-name aggregate

eval-information-local: register-prompt ## Run information agent evaluation locally
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	MLFLOW_GENAI_EVALUATION_MAX_WORKERS=1 \
		uv run --package omnia-support-eval evaluate information \
		--config packages/evaluation/src/omnia_support_eval/information/eval_config.yaml \
		--retrieval-mode $(RETRIEVAL_MODE)

eval-triage-local: register-prompt ## Run triage agent evaluation locally
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	MLFLOW_GENAI_EVALUATION_MAX_WORKERS=1 \
		uv run --package omnia-support-eval evaluate triage

eval-action-local: register-prompt ## Run action agent evaluation locally
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	MLFLOW_GENAI_EVALUATION_MAX_WORKERS=1 \
		uv run --package omnia-support-eval evaluate action

eval-aggregator-local: register-prompt ## Run aggregator agent evaluation locally
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
	MLFLOW_GENAI_EVALUATION_MAX_WORKERS=1 \
		uv run --package omnia-support-eval evaluate aggregator

eval-information-docker: ## Run information agent evaluation in Docker
	docker compose --profile eval run --rm evaluation-information

eval-triage-docker: ## Run triage agent evaluation in Docker
	docker compose --profile eval run --rm evaluation-triage

eval-action-docker: ## Run action agent evaluation in Docker
	docker compose --profile eval run --rm evaluation-action

eval-aggregator-docker: ## Run aggregator agent evaluation in Docker
	docker compose --profile eval run --rm evaluation-aggregator

##@ Application
frontend-build: ## Build the SvelteKit frontend
	cd frontend && npm ci && npm run build

app: frontend-build ## Start the app server locally
	RETRIEVAL_MODE=$(RETRIEVAL_MODE) uv run omnia-support-app

##@ MCP Server
mcp-local: ## Run the MCP server locally (stdio)
	uv run omnia-support-mcp

mcp-docker: ## Run the MCP server in Docker (SSE on port 8081)
	docker compose up --build mcp-server

mcp-inspect: ## Launch the MCP inspector
	npx @modelcontextprotocol/inspector -- uv run omnia-support-mcp

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
