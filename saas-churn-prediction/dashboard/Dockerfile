# Dashboard service container.
# Build context is the project root, so paths in COPY are relative to it.
FROM python:3.10-slim

WORKDIR /app

# System dependencies needed to install some Python packages from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dashboard dependencies. Copying requirements first means
# rebuilds skip dependency installation when only application code changes.
COPY dashboard/requirements.txt /app/dashboard/requirements.txt
RUN pip install --no-cache-dir -r /app/dashboard/requirements.txt

# Application code, trained model artifacts, and scored customer data.
COPY dashboard/ /app/dashboard/
COPY models/ /app/models/
COPY data/processed/ /app/data/processed/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Streamlit in headless mode (no browser auto-open) on 0.0.0.0 so the
# container is reachable from the host.
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
