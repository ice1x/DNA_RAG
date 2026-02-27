.PHONY: help install dev test lint typecheck check serve ui docker-build docker-up docker-down clean

PYTHON ?= python
PORT   ?= 8000

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:  ## Install the package (core only)
	$(PYTHON) -m pip install -e .

dev:  ## Install with dev + api + ui extras
	$(PYTHON) -m pip install -e ".[api,ui,dev]"

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

test:  ## Run the test suite with coverage
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:  ## Run ruff linter + formatter check
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m ruff format --check src/ tests/

typecheck:  ## Run mypy type checker
	$(PYTHON) -m mypy src/

check: lint typecheck test  ## Run all quality checks

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

serve:  ## Start the API server (development)
	uvicorn dna_rag.api.main:app --reload --port $(PORT)

ui:  ## Start the Streamlit UI
	$(PYTHON) -m streamlit run src/dna_rag/ui/app.py

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build:  ## Build the Docker image
	docker compose build

docker-up:  ## Start services via Docker Compose
	docker compose up -d

docker-down:  ## Stop services
	docker compose down

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:  ## Remove build artefacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .mypy_cache/
