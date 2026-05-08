.PHONY: help setup setup-local db-up db-down ingest generate-data features pipeline clean test lint

# ─── Help ─────────────────────────────────────────────────────
help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Setup ────────────────────────────────────────────────────
setup:  ## Full setup with Docker (PostgreSQL)
	docker compose up -d postgres
	@echo "Waiting for PostgreSQL to be ready..."
	@sleep 5
	pip install -r requirements.txt
	USE_POSTGRES=true python -m src.data.ingest
	USE_POSTGRES=true python -m src.data.generate_synthetic
	USE_POSTGRES=true python -m src.data.feature_engineering
	@echo "✓ Setup complete!"

setup-local:  ## Setup with SQLite (no Docker needed)
	pip install -r requirements.txt
	python -m src.data.ingest
	python -m src.data.generate_synthetic
	python -m src.data.feature_engineering
	@echo "✓ Local setup complete (SQLite)!"

# ─── Database ────────────────────────────────────────────────
db-up:  ## Start PostgreSQL container
	docker compose up -d postgres

db-down:  ## Stop PostgreSQL container
	docker compose down

# ─── Data Pipeline ────────────────────────────────────────────
ingest:  ## Run data ingestion
	python -m src.data.ingest

generate-data:  ## Generate synthetic engagement data
	python -m src.data.generate_synthetic

features:  ## Run feature engineering pipeline
	python -m src.data.feature_engineering

pipeline:  ## Run full data pipeline (ingest → synthetic → features)
	python -m src.data.ingest
	python -m src.data.generate_synthetic
	python -m src.data.feature_engineering

# ─── Quality ──────────────────────────────────────────────────
test:  ## Run tests
	python -m pytest tests/ -v

lint:  ## Run linter
	python -m ruff check src/ tests/ api/ dashboard/

# ─── Cleanup ──────────────────────────────────────────────────
clean:  ## Remove generated files
	rm -rf data/processed/*
	rm -rf data/churn.db
	rm -rf mlruns/
	rm -rf models/*.joblib
	rm -rf __pycache__ src/__pycache__ src/**/__pycache__
	@echo "✓ Cleaned generated files"
