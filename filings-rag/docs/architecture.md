# Architecture

A more detailed prose walkthrough of the Filings-RAG system. The design doc (`2026-05-20-filings-rag-design.md` §6) contains the ASCII diagram; this file expands on each component.

## Two pipelines, one query path

The system splits cleanly into an **offline pipeline** (build the indices once) and an **online query path** (every user question).

### Offline pipeline (one-shot, Colab)

```
manifest CSV  →  scraper  →  raw PDFs  →  parser  →  chunker  →  BGE embeddings  →  Chroma + BM25 (persisted)
```

Each step is a small Python module under `src/ingestion/` and `src/retrieval/`:

- **`scraper.py`** reads `data/filings_manifest.csv`, validates each row's `pdf_url` with a HEAD request, downloads to `data/raw/{ticker}/{year}.pdf` with a 1-second delay between requests, and records failures (Cloudflare-blocked sites, dead links) in the run report rather than aborting.
- **`pdf_parser.py`** uses PyMuPDF (`fitz`) to extract page-level text. Each page becomes a `{ticker, year, page, text}` record. Pages with empty or whitespace-only text are skipped.
- **`chunker.py`** does whitespace-token-based windowing with overlap (default: 512 tokens, 50 overlap). Chunks never span PDFs, every chunk carries page metadata, and each gets a stable 16-character SHA-256 `chunk_hash` used as its index ID. This hash is the bridge between the dense store, the BM25 store, and the chunk-text lookup at query time.
- **`embeddings.py`** wraps `sentence-transformers` for `BAAI/bge-small-en-v1.5` (384-dim, normalised for cosine).
- **`vector_store.py`** wraps Chroma with `hnsw:space=cosine` and a single collection `filings`.
- **`bm25_index.py`** wraps the `bm25s` library — pure-Python BM25 indexed over the same chunk corpus, persisted as a pickle. No external services.

The indexing step (notebook `03_embedding_index_build.ipynb`) runs on Colab T4 because BGE embedding of ~50k chunks is GPU-bound. Indices are persisted to Google Drive in 10-company batches so a Colab session timeout doesn't destroy progress, then synced to local `data/vector/` for the online path.

### Online query path (Streamlit + HF Spaces)

```
user query  →  query_rewriter  →  hybrid retrieval (BM25 + dense, RRF)  →  BGE re-ranker  →  LLM with citation prompt  →  citation validator  →  answer
```

- **`query_rewriter.py`** calls the LLM to expand short or ambiguous queries (e.g. "their climate stuff" → "climate-related risk disclosures"). Queries with ≥ 8 words pass through unchanged. On any LLM failure the rewriter falls back to the original — it never blocks the query path.
- **Hybrid retrieval** runs top-50 dense and top-50 BM25 in parallel, then fuses with **Reciprocal Rank Fusion** (`hybrid.py`). RRF is parameter-light (only `k=60` smoothing constant) and stable under ties because we break ties on first-seen rank. The fused top-20 candidates get passed to the re-ranker.
- **`reranker.py`** runs the BGE cross-encoder (`BAAI/bge-reranker-v2-m3`) on `(query, chunk_text)` pairs and returns the top-5 chunks by cross-encoder score. This is the most expensive step per query (~1-3s) but dramatically improves answer-relevancy.
- **`prompts.py` + `chain.py`** assemble the LLM call. The system prompt forces inline `[TICKER|YEAR|p.PAGE]` citations and the response *must* be either grounded-with-citations or the refusal sentinel `"No relevant content found in the corpus."`. After generation, the **citation validator** parses every cited tuple and confirms it exists in the retrieved chunk set; if any is unverifiable the chain regenerates once with a stricter follow-up prompt.

### Eval track (offline, on demand)

A third track runs the chain over a hand-labelled 40-question test set in `data/qa_test_set.jsonl` and feeds the results through Ragas. The metrics (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, plus custom `refusal_accuracy`) are logged to MLflow tagged by retrieval config. See `eval_methodology.md`.

## Why these choices

| Decision | Rationale |
|---|---|
| Hybrid retrieval (BM25 + dense) over dense-only | Annual reports use rare statutory phrases ("going concern", entity legal names, sectoral codes) where BM25 outperforms paraphrase-friendly dense embeddings. Hybrid wins on both lexical-heavy and paraphrased queries. |
| Cross-encoder re-ranking | Top-50 retrieval recall is high but ordering is noisy. A cross-encoder over (query, chunk) pairs is computationally cheap at top-50 size and produces dramatically better ordering than the bi-encoder alone. |
| Forced citations + validator | Finance-domain LLMs must be regulator-traceable. A grounded-citation prompt produces citations *most of the time*; the validator+regeneration loop converts that into *all of the time*. |
| Refusal on empty retrieval | The system refuses (returns the sentinel string) when no chunks clear the retrieval threshold rather than letting the LLM extrapolate. Refusal accuracy is measured directly on the adversarial QA subset. |
| Provider-agnostic LLM client | Single `build_llm()` factory switches between Groq Llama 3.3 70B and Google Gemini 2.0 Flash via the `LLM_PROVIDER` env var. No code change to swap providers — useful when one provider rate-limits during a Ragas eval run. |
| ChromaDB + bm25s (local, free) | No managed-service dependency. Both indices persist to disk and ride in the Docker image; the same container runs locally and on Hugging Face Spaces. |
| Page-aware over heading-aware chunking | The default chunker keeps each chunk inside a single PDF page so citations remain precise. Heading-aware chunking via `unstructured` is implemented in `02_chunking_experiments.ipynb` as an alternative — adopted only if the notebook proves a meaningful recall@5 win. |

## Failure modes and how they're handled

| Failure | Handling |
|---|---|
| LLM rate-limit / 429 | `tenacity` retry: 3 attempts with exponential backoff (1s, 2s, 4s). After 3 failures the chain returns an error to the UI. The fallback provider can be activated by changing `LLM_PROVIDER`. |
| Empty retrieval | The prompt forces refusal — LLM is not invoked. |
| Invalid citation generated | Validator rejects; chain regenerates once with stricter instruction. If still invalid, the answer is returned with a "citation could not be verified" UI warning. |
| PDF parse failure (corrupt or scanned image) | Parser silently skips empty pages. The full-PDF failure case is logged by the scraper's `run_from_manifest` and surfaces in the run report. |
| Context-window overflow after re-ranking | `chain.py` truncates to `rerank_top_k=5` chunks. With `bge-small` 384-dim embeddings and Llama 3.3 70B's 128k context, this is well within limits. |
| HF Spaces cold start | The Streamlit `load_components()` is wrapped in `@st.cache_resource(show_spinner="Warming up…")` so the first query surfaces a warm-up message rather than appearing to hang. |
