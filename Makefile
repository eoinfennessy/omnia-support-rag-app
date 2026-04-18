.PHONY: help lint lint-fix format format-check type-check check fix install-hooks up down infra-local infra-docker ingest test

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$|^##@' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next} {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

##@ Docker
up: ## Start all services
	docker compose up --build

down: ## Stop all services
	docker compose down

##@ Development
infra-local: ## Start ChromaDB locally via uv
	uv run chroma run --path ./data/chroma --port 8000

infra-docker: ## Start ChromaDB via Docker Compose
	docker compose up chromadb

ingest: ## Run ingestion pipeline locally via uv
	uv run ingest-to-chroma

test: ## Run tests
	uv run --group test pytest

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
