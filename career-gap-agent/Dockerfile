FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and committed data artifacts (ESCO index + Adzuna snapshot)
COPY src ./src
COPY app ./app
COPY data/esco ./data/esco
COPY data/fixtures ./data/fixtures
COPY README.md .

# Hugging Face Spaces (Docker SDK) serves on port 7860
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
