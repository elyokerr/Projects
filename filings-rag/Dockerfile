FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF + unstructured
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY data/vector/ data/vector/
COPY data/processed/ data/processed/
COPY data/filings_manifest.csv data/filings_manifest.csv

ENV STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    CHROMA_DIR=/app/data/vector/chroma \
    BM25_PATH=/app/data/vector/bm25.pkl \
    PROCESSED_DIR=/app/data/processed

EXPOSE 7860

CMD ["streamlit", "run", "app/streamlit_app.py"]
