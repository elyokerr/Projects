.PHONY: help setup-local stack-up stack-down stack-logs \
        db-up db-down pipeline train evaluate predict \
        test lint lint-fix api dashboard clean

# Default target: print available commands
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Full stack (Docker Compose) ------------------------------------------

stack-up:  ## Build and start the full stack (PostgreSQL + API + Dashboard)
	docker compose up --build -d
	@echo ""
	@echo "Stack running:"
	@echo "  API:        http://localhost:8000/docs"
	@echo "  Dashboard:  http://localhost:8501"
	@echo "  Database:   localhost:5432"

stack-down:  ## Stop the full stack
	docker compose down

stack-logs:  ## Tail container logs
	docker compose logs -f

# --- Database only --------------------------------------------------------

db-up:  ## Start the PostgreSQL container only
	docker compose up -d postgres

db-down:  ## Stop the PostgreSQL container
	docker compose down

# --- Local Python workflow (SQLite, no Docker) ----------------------------

setup-local:  ## Install dependencies and run the full data pipeline locally
	pip install -r requirements.txt
	python -m src.data.ingest
	python -m src.data.generate_synthetic
	python -m src.data.feature_engineering
	@echo "Local setup complete (SQLite backend)."

pipeline:  ## Run the data pipeline: ingest -> synthetic -> features
	python -m src.data.ingest
	python -m src.data.generate_synthetic
	python -m src.data.feature_engineering

train:  ## Train all models and select the best
	python -m src.models.train

evaluate:  ## Generate evaluation plots and SHAP explanations
	python -m src.models.evaluate

predict:  ## Score all customers and write scored_customers.csv
	python -m src.models.predict

# --- Services -------------------------------------------------------------

api:  ## Run the API locally (without Docker)
	uvicorn api.main:app --reload --port 8000

dashboard:  ## Run the dashboard locally (without Docker)
	streamlit run dashboard/app.py

# --- Quality --------------------------------------------------------------

test:  ## Run the test suite
	python -m pytest tests/ -v

lint:  ## Run the linter
	python -m ruff check src/ api/ tests/

lint-fix:  ## Run the linter with auto-fix enabled
	python -m ruff check src/ api/ tests/ --fix

# --- Cleanup --------------------------------------------------------------

clean:  ## Remove generated data, MLflow runs, and Python caches
	rm -rf data/processed/*
	rm -rf data/churn.db
	rm -rf mlruns/
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Cleaned generated files."
