.PHONY: help setup setup-local stack-up stack-down stack-logs \
       db-up db-down pipeline test lint api dashboard clean

# ─── Help ─────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Full Stack (Docker) ─────────────────────────────────────
stack-up:  ## Start full stack (DB + API + Dashboard) with Docker
	docker compose up --build -d
	@echo ""
	@echo "✓ Full stack running:"
	@echo "  • API:        http://localhost:8000/docs"
	@echo "  • Dashboard:  http://localhost:8501"
	@echo "  • Database:   localhost:5432"

stack-down:  ## Stop full stack
	docker compose down

stack-logs:  ## View container logs
	docker compose logs -f

# ─── Local Setup (no Docker) ─────────────────────────────────
setup-local:  ## Setup with SQLite (no Docker needed)
	pip install -r requirements.txt
	python -m src.data.ingest
	python -m src.data.generate_synthetic
	python -m src.data.feature_engineering
	@echo "✓ Local setup complete (SQLite)!"

# ─── Database ────────────────────────────────────────────────
db-up:  ## Start PostgreSQL container only
	docker compose up -d postgres

db-down:  ## Stop PostgreSQL container
	docker compose down

# ─── Data Pipeline ────────────────────────────────────────────
pipeline:  ## Run full data pipeline (ingest → synthetic → features)
	python -m src.data.ingest
	python -m src.data.generate_synthetic
	python -m src.data.feature_engineering

# ─── Services ─────────────────────────────────────────────────
api:  ## Run API locally (without Docker)
	uvicorn api.main:app --reload --port 8000

dashboard:  ## Run dashboard locally (without Docker)
	streamlit run dashboard/app.py

# ─── Quality ──────────────────────────────────────────────────
test:  ## Run tests
	python -m pytest tests/ -v

lint:  ## Run linter (ruff)
	python -m ruff check src/ api/ tests/

lint-fix:  ## Run linter with auto-fix
	python -m ruff check src/ api/ tests/ --fix

# ─── Cleanup ──────────────────────────────────────────────────
clean:  ## Remove generated files
	rm -rf data/processed/*
	rm -rf data/churn.db
	rm -rf mlruns/
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__
	@echo "✓ Cleaned generated files"
