FROM python:3.11-slim

# libgomp1 provides OpenMP, needed by lightgbm, faiss, and implicit
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the committed sample fixture (the app trains on it at
# first request when no pre-built models/pipeline.pkl is present)
COPY src ./src
COPY app ./app
COPY tests/fixtures ./tests/fixtures
COPY README.md .

# Hugging Face Spaces (Docker SDK) serves on port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
