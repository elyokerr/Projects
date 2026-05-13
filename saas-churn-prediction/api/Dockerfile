# API service container.
# Build context is the project root, so paths in COPY are relative to it.
FROM python:3.10-slim

WORKDIR /app

# System dependencies needed to install some Python packages from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies. Copy requirements first so dependency
# installation can be cached across rebuilds when only source changes.
COPY requirements.txt /app/requirements.txt
COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir -r /app/api/requirements.txt

# Copy the application code and the trained model artifacts.
COPY src/ /app/src/
COPY api/ /app/api/
COPY models/ /app/models/

# FastAPI's default port.
EXPOSE 8000

# Container-level health check used by Docker Compose for ordered startup.
HEALTHCHECK --interval=15s --timeout=10s --retries=3 --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the API with Uvicorn.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
