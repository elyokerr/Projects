# Filings-RAG — Question-Answering over UK FTSE 100 Annual Reports

A production-style retrieval-augmented question-answering system over the annual reports of UK FTSE 100 companies. Hybrid retrieval (BM25 + BGE dense vectors), cross-encoder re-ranking, forced citations, and a Ragas evaluation pipeline tracked in MLflow. Streamlit UI deployed to Hugging Face Spaces.

> **Demo:** _(URL added after Hugging Face Spaces deploy)_

---

## 1. Hero metrics

| | |
|---|---|
| Corpus | _(filled after indexing)_ FTSE 100 annual reports (≈ 60 PDFs across 2 fiscal years) |
| Retrieval recall@5 (hybrid + re-rank) | _(filled after eval run)_ |
| Ragas faithfulness | _(filled)_ |
| Ragas answer-relevancy | _(filled)_ |
| Refusal accuracy on adversarial queries | _(filled)_ |

## 2. What it does

Given a question like *"How does Tesco describe its principal climate-related risks in its 2024 annual report?"*, the system:

1. Optionally rewrites short/ambiguous queries via the LLM.
2. Retrieves candidate passages from a hybrid index (BM25 + BGE dense vectors fused via Reciprocal Rank Fusion).
3. Re-ranks the top candidates with a BGE cross-encoder.
4. Synthesises an answer with **Groq Llama 3.3 70B**, forcing inline `[TICKER|YEAR|p.PAGE]` citations.
5. Validates every cited (ticker, year, page) tuple against the retrieved chunks — regenerates with stricter prompting if any citation is unverifiable.

## 3. Architecture

See [`docs/2026-05-20-filings-rag-design.md`](docs/2026-05-20-filings-rag-design.md) for the full design (data flow, architecture diagram, retrieval/evaluation methodology, error-handling matrix, testing strategy, deployment, scaling notes).

## 4. Stack

LangChain · ChromaDB · `bm25s` · `BAAI/bge-small-en-v1.5` · `BAAI/bge-reranker-v2-m3` · Groq (Llama 3.3 70B) + Google AI Studio (Gemini 2.0 Flash) fallback · PyMuPDF · `unstructured` · Ragas · MLflow · Streamlit · Docker · GitHub Actions · Google Colab (T4) for one-shot indexing

## 5. Repo layout

```
filings-rag/
├── README.md              ← you are here
├── requirements.txt
├── Dockerfile · docker-compose.yml
├── .env.example
├── notebooks/             ← EDA, chunking, indexing, eval (notebooks 01–05)
├── src/
│   ├── ingestion/         ← scraper, pdf_parser, chunker
│   ├── retrieval/         ← embeddings, vector_store, bm25_index, hybrid, reranker
│   ├── generation/        ← llm_client, query_rewriter, prompts
│   ├── eval/              ← ragas_eval, mlflow_logger
│   └── chain.py           ← end-to-end orchestration
├── app/streamlit_app.py
├── data/
│   ├── filings_manifest.csv   ← committed (company → URL map)
│   ├── qa_test_set.jsonl      ← committed (40 labelled QA pairs)
│   └── raw/ processed/ vector/   ← gitignored
├── tests/
└── docs/
    ├── 2026-05-20-filings-rag-design.md
    ├── architecture.md
    └── eval_methodology.md
```

## 6. Quickstart (local)

```bash
# 1. Set up env
cp .env.example .env
# Fill in GROQ_API_KEY (or GOOGLE_API_KEY, with LLM_PROVIDER=gemini)

# 2. Install deps (one-time)
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt

# 3. Acquire corpus (populate the pdf_url column in data/filings_manifest.csv first)
python -m src.ingestion.scraper

# 4. Build the indices on Colab
#    Open notebooks/03_embedding_index_build.ipynb in Colab,
#    runtime: T4 GPU, mount Drive, execute top-to-bottom.
#    Download data/vector/ back to local.

# 5. Run the app
docker compose up --build
# → http://localhost:8501
```

## 7. Evaluation

A 40-question hand-labelled test set (`data/qa_test_set.jsonl`) covers four categories:

- Single-company factual (10 questions)
- Multi-section synthesis (10)
- Multi-company comparison (10)
- Adversarial / refusal (10) — system should refuse cleanly

Metrics (Ragas + custom refusal accuracy) are logged to MLflow on every retrieval-config change. The ablation table is maintained in [`docs/eval_methodology.md`](docs/eval_methodology.md):

| Config | recall@5 | faithfulness | answer_relevancy | refusal_acc |
|---|---|---|---|---|
| Dense only | _(filled)_ | _(filled)_ | _(filled)_ | _(filled)_ |
| BM25 only | _(filled)_ | _(filled)_ | _(filled)_ | _(filled)_ |
| Hybrid (RRF) | _(filled)_ | _(filled)_ | _(filled)_ | _(filled)_ |
| Hybrid + re-ranker | _(filled)_ | _(filled)_ | _(filled)_ | _(filled)_ |
| + query rewriting | _(filled)_ | _(filled)_ | _(filled)_ | _(filled)_ |

## 8. Testing

```bash
pytest tests -v
```

Unit-tested modules: chunker, vector store, BM25 index, RRF, LLM client (provider switch logic), prompts (citation extraction + validation), scraper (URL validation + download).

## 9. Limitations

- Corpus is static — current PDFs are fixed at indexing time; new filings require a re-index run.
- Numerical-table extraction is text-only — the system can read narrative around tables but does not parse cell values structurally. Documented as a v2 candidate.
- Free-tier Groq has rate limits — under sustained load the system falls through to the Gemini provider; both are free but each has independent caps.
- The system **refuses** investment-advice queries and queries outside the corpus rather than answering. This is intentional, not a bug.

---

## Data attribution

Annual reports are statutorily public documents downloaded from each company's Investor Relations page. Source URLs are tracked in `data/filings_manifest.csv`.
