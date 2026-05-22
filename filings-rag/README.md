# Filings-RAG — Question-Answering over UK FTSE 100 Annual Reports

A production-style retrieval-augmented question-answering system over the annual reports of UK FTSE 100 companies. Hybrid retrieval (BM25 + BGE dense vectors), cross-encoder re-ranking, forced citations, and a Ragas evaluation pipeline tracked in MLflow. Streamlit chat UI; designed to deploy to Hugging Face Spaces.

> **Demo:** _(URL added after Hugging Face Spaces deploy)_

---

## 1. Hero metrics

| | |
|---|---|
| Corpus | **23 FTSE 100 companies · 34 annual report PDFs · 10,722 pages** |
| Chunks indexed | **19,399 unique chunks** (page-aware, 512-token windows, 50-token overlap) |
| Retrieval stack | Hybrid BM25 + BGE dense, RRF-fused, BGE-reranker-v2-m3 top-5 |
| LLM | Groq Llama 3.3 70B (primary) + Google Gemini 2.0 Flash (fallback), both free tier |
| Citation validation | Every cited `[TICKER\|YEAR\|p.PAGE]` is verified against retrieved chunks; regenerated once if any is unverifiable |
| Eval framework | Ragas (faithfulness, answer-relevancy, context-precision, context-recall) + custom refusal accuracy on adversarial subset |
| Full Ragas ablation table | _pending — see `docs/eval_methodology.md` for methodology_ |

## 2. What it does

Given a question like *"What climate-related risks did BP discuss in 2024?"*, the system:

1. Optionally rewrites short/ambiguous queries via the LLM.
2. Retrieves the top-50 candidate chunks from a hybrid index (BM25 + BGE dense vectors fused via Reciprocal Rank Fusion).
3. Re-ranks them with a BGE cross-encoder, keeping the top 5.
4. Synthesises an answer with **Groq Llama 3.3 70B**, forcing inline `[TICKER|YEAR|p.PAGE]` citations.
5. Validates every cited (ticker, year, page) tuple against the retrieved chunks — regenerates once with a stricter prompt if any citation is unverifiable.

### Example response

> **Q:** What climate-related risks did BP discuss in 2024?
>
> **A:** BP discussed various climate-related risks in 2024, including physical risks such as extreme weather `[BP|2024|p.48]`, which may be affected or intensified by climate change, and transition risks associated with the energy transition `[BP|2024|p.46]`. Specifically, they considered risks related to severe weather events such as flooding or wind damage `[BP|2024|p.48]`, and the potential impact of increased water scarcity due to climate change on their operations `[BP|2024|p.48]`. Additionally, they assessed the resilience of their strategy to different climate-related scenarios, including 1.5°C-consistent scenarios `[BP|2024|p.52]`.

Each `[TICKER|YEAR|p.PAGE]` citation in the UI expands to show the verbatim source paragraph.

## 3. Architecture

See [`docs/2026-05-20-filings-rag-design.md`](docs/2026-05-20-filings-rag-design.md) for the full design (data flow, architecture diagram, retrieval/evaluation methodology, error-handling matrix, testing strategy, deployment, scaling notes). Companion: [`docs/architecture.md`](docs/architecture.md) for a prose walkthrough.

## 4. Stack

LangChain (LCEL) · ChromaDB (cosine, HNSW) · `bm25s` · `BAAI/bge-small-en-v1.5` via `fastembed` (ONNX, CPU-optimised) · `BAAI/bge-reranker-v2-m3` via sentence-transformers CrossEncoder · Groq Llama 3.3 70B + Google Gemini 2.0 Flash (free-tier, provider-agnostic) · PyMuPDF · Ragas · MLflow · Streamlit · Docker · GitHub Actions

## 5. Repo layout

```
filings-rag/
├── README.md              ← you are here
├── requirements.txt
├── Dockerfile · docker-compose.yml
├── .env.example
├── scripts/build_indices.py  ← local CPU index build (replaces the Colab step)
├── notebooks/             ← EDA, chunking, indexing, retrieval eval, end-to-end eval (01–05)
├── src/
│   ├── ingestion/         ← scraper, pdf_parser, chunker
│   ├── retrieval/         ← embeddings (fastembed), vector_store (Chroma), bm25_index, hybrid (RRF), reranker (CrossEncoder)
│   ├── generation/        ← llm_client (Groq+Gemini), query_rewriter, prompts + citation validator
│   ├── eval/              ← ragas_eval, mlflow_logger
│   └── chain.py           ← end-to-end orchestration
├── app/streamlit_app.py
├── data/
│   ├── filings_manifest.csv   ← committed (company → URL map, 49/60 URLs populated)
│   ├── qa_test_set.jsonl      ← committed (40 labelled QA pairs)
│   └── raw/ processed/ vector/   ← gitignored
├── tests/                 ← 18 unit tests
└── docs/
    ├── 2026-05-20-filings-rag-design.md
    ├── architecture.md
    └── eval_methodology.md
```

## 6. Quickstart (local — no GPU required)

```bash
# 1. Set up env
cp .env.example .env
# Edit .env: set GROQ_API_KEY (free: https://console.groq.com/keys)
# Optional fallback: set GOOGLE_API_KEY (free: https://aistudio.google.com/apikey)

# 2. Create venv + install deps
python -m venv .venv
.venv/Scripts/activate           # Windows; on macOS/Linux use: source .venv/bin/activate
pip install -r requirements.txt

# 3. Verify the code (8 test files, ~30s)
pytest tests -v

# 4. Acquire the corpus (~10 min for ~30 PDFs)
# First, fill any missing pdf_url cells in data/filings_manifest.csv from each company's IR page.
# Then:
python -m src.ingestion.scraper

# 5. Parse + chunk all downloaded PDFs (writes data/processed/{ticker}/{year}.jsonl)
jupyter notebook notebooks/02_chunking_experiments.ipynb    # run all cells
# or run inline:
python -c "
from pathlib import Path; import json
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import chunk_pages
for p in Path('data/raw').rglob('*.pdf'):
    ticker, year = p.parent.name, int(p.stem)
    chunks = list(chunk_pages(parse_pdf(p, ticker=ticker, year=year)))
    out = Path('data/processed') / ticker / f'{year}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(json.dumps(c) for c in chunks), encoding='utf-8')
"

# 6. Build the indices locally on CPU (~5-15 min for 19k chunks via fastembed/ONNX)
python -m scripts.build_indices
# (use --max-pdfs 5 to demo with a subset)

# 7. Run the app
streamlit run app/streamlit_app.py
# → http://localhost:8501
```

The first query in the Streamlit UI takes ~60s (one-time model warm-up); subsequent queries respond in 3–5s.

### Containerised alternative

```bash
docker compose up --build   # → http://localhost:8501
```

## 7. Evaluation

A 40-question hand-labelled test set (`data/qa_test_set.jsonl`) covers four categories:

- Single-company factual (10)
- Multi-section synthesis (10)
- Multi-company comparison (10)
- Adversarial / refusal (10) — system should refuse cleanly

The adversarial subset has definitive expected refusals (`"No relevant content found in the corpus."`); the other 30 questions have section-pointing placeholders to be filled by reading the source PDFs once. Ragas + custom refusal accuracy are logged to MLflow per run, tagged by retrieval configuration. Full methodology and the ablation-table template live in [`docs/eval_methodology.md`](docs/eval_methodology.md).

| Config | recall@5 | faithfulness | answer_relevancy | refusal_acc |
|---|---|---|---|---|
| Dense only | _pending_ | _pending_ | _pending_ | _pending_ |
| BM25 only | _pending_ | _pending_ | _pending_ | _pending_ |
| Hybrid (RRF) | _pending_ | _pending_ | _pending_ | _pending_ |
| Hybrid + re-ranker | _pending_ | _pending_ | _pending_ | _pending_ |
| + query rewriting | _pending_ | _pending_ | _pending_ | _pending_ |

Run notebooks `04_retrieval_evaluation.ipynb` and `05_end_to_end_eval.ipynb` to populate the table.

## 8. Testing

```bash
pytest tests -v
```

Unit-tested modules: chunker, vector store, BM25 index, RRF, LLM client (provider switch + retry), prompts (citation extraction + validation), scraper (URL validation + magic-byte verification), PDF parser.

## 9. Limitations

- **Corpus is static.** PDFs are fixed at indexing time; new filings need a re-index run.
- **Numerical-table extraction is text-only.** The system reads narrative around tables but does not parse cell values structurally. Marked as a v2 candidate.
- **Free-tier LLM caps.** Groq has request-per-minute limits; under sustained load the provider-agnostic client falls through to Gemini. Both are free but each has independent caps.
- **Refuses investment-advice and out-of-corpus queries.** This is intentional — the prompt + validator pipeline is designed for traceable analyst use, not speculation.
- **Complex synthesis questions occasionally underperform.** Multi-section / multi-company synthesis with bge-small (384-dim) embeddings and a top-5 reranker window can miss the most relevant passages for very abstract questions; bge-large + top-10 reranking would close most of the gap.

---

## Data attribution

Annual reports are statutorily public documents downloaded from each company's Investor Relations page. Source URLs are tracked in `data/filings_manifest.csv`.
