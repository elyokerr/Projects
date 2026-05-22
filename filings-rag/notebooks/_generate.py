"""Generate the 5 Jupyter notebooks for filings-rag.

Uses only stdlib (json, pathlib). Each notebook is a list of (cell_type, source_text)
tuples; we emit minimal nbformat 4 / minor 5 JSON.
"""

import json
from pathlib import Path


def to_source_lines(text: str) -> list[str]:
    """Split a source string into nbformat-style lines (each ending with \n
    except possibly the last)."""
    # Normalise: strip a single trailing newline if present so we don't add a
    # blank line at the end of the cell.
    if text.endswith("\n"):
        text = text[:-1]
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def build_notebook(cells: list[tuple[str, str]]) -> dict:
    nb_cells = []
    for cell_type, source_text in cells:
        source = to_source_lines(source_text)
        if cell_type == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": source,
            })
        elif cell_type == "code":
            nb_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            })
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------------------
# Notebook 1 — 01_eda_corpus.ipynb
# ---------------------------------------------------------------------------

NB1: list[tuple[str, str]] = [
    ("markdown", """# 01 — Corpus EDA

Stats over the FTSE 100 annual-report corpus. Run after `python -m src.ingestion.scraper` has downloaded PDFs into `data/raw/`.

Outputs `data/processed/corpus_stats.csv` for use in the README hero metrics."""),
    ("code", """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import csv
import fitz  # PyMuPDF
import pandas as pd"""),
    ("code", """RAW = Path("../data/raw")
MANIFEST = Path("../data/filings_manifest.csv")

manifest = pd.read_csv(MANIFEST)
print(f"Manifest rows: {len(manifest)}")
print(f"URLs populated: {manifest['pdf_url'].notna().sum()}")
print(f"By fiscal year: {manifest.groupby('fiscal_year').size().to_dict()}")"""),
    ("code", """# Discover downloaded PDFs
pdfs = sorted(RAW.rglob("*.pdf"))
print(f"PDFs on disk: {len(pdfs)}")
for p in pdfs[:5]:
    print(" ", p.relative_to(RAW))"""),
    ("code", """# Per-PDF stats: page count + file size
records = []
for p in pdfs:
    ticker = p.parent.name
    year = int(p.stem)
    size_mb = p.stat().st_size / (1024 * 1024)
    try:
        with fitz.open(p) as doc:
            page_count = doc.page_count
    except Exception as e:
        page_count = -1
        print(f"  parse failed: {p.name} — {e}")
    records.append({"ticker": ticker, "year": year, "pages": page_count, "size_mb": round(size_mb, 1)})

stats = pd.DataFrame(records)
stats"""),
    ("code", """# Distribution summary
if len(stats):
    print("Pages:", stats["pages"].describe().to_dict())
    print("Size MB:", stats["size_mb"].describe().to_dict())"""),
    ("code", """# Sample first 500 chars of 3 random PDFs (sanity check for parse quality)
import random
if pdfs:
    for p in random.sample(pdfs, min(3, len(pdfs))):
        with fitz.open(p) as doc:
            text = (doc[0].get_text() or "")[:500]
        print(f"--- {p.relative_to(RAW)} ---")
        print(text)
        print()"""),
    ("code", """# Persist for the README hero metrics
out = Path("../data/processed")
out.mkdir(parents=True, exist_ok=True)
stats.to_csv(out / "corpus_stats.csv", index=False)
print(f"Wrote {out / 'corpus_stats.csv'}")"""),
]


# ---------------------------------------------------------------------------
# Notebook 2 — 02_chunking_experiments.ipynb
# ---------------------------------------------------------------------------

NB2: list[tuple[str, str]] = [
    ("markdown", """# 02 — Chunking strategy experiment

Compare page-aware vs heading-aware chunking on a 5-PDF sample. Pick the winner, then chunk the full corpus into `data/processed/{ticker}/{year}.jsonl`."""),
    ("code", """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import json
import random
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.chunker import chunk_pages

random.seed(42)
RAW = Path("../data/raw")
PROCESSED = Path("../data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)"""),
    ("code", """# Pick a 5-PDF sample
all_pdfs = sorted(RAW.rglob("*.pdf"))
sample = random.sample(all_pdfs, min(5, len(all_pdfs)))
sample_meta = [(p.parent.name, int(p.stem), p) for p in sample]
sample_meta"""),
    ("code", """# Strategy A: page-aware (default chunker, max_tokens=512, overlap=50)
chunks_A = []
for ticker, year, p in sample_meta:
    pages = list(parse_pdf(p, ticker=ticker, year=year))
    chunks_A.extend(chunk_pages(pages, max_tokens=512, overlap=50))
print(f"Strategy A (page-aware 512/50): {len(chunks_A)} chunks")"""),
    ("code", """# Strategy B: heading-aware via unstructured (sketches — adopted only if it wins).
# Falls back gracefully if unstructured can't open a given PDF.
from unstructured.partition.pdf import partition_pdf

def heading_aware_chunks(pdf_path, ticker, year, max_tokens=512):
    chunks = []
    try:
        elements = partition_pdf(filename=str(pdf_path), strategy="fast")
    except Exception as e:
        print(f"  unstructured failed on {pdf_path.name}: {e}")
        return []
    current_heading = None
    buf = []
    page = 1
    for el in elements:
        text = el.text or ""
        if el.category in ("Title", "Header"):
            if buf:
                chunks.append({"text": " ".join(buf), "ticker": ticker, "year": year, "page": page, "section": current_heading or ""})
                buf = []
            current_heading = text
        else:
            buf.append(text)
            page = getattr(el.metadata, "page_number", page) or page
            if sum(len(x.split()) for x in buf) >= max_tokens:
                chunks.append({"text": " ".join(buf), "ticker": ticker, "year": year, "page": page, "section": current_heading or ""})
                buf = []
    if buf:
        chunks.append({"text": " ".join(buf), "ticker": ticker, "year": year, "page": page, "section": current_heading or ""})
    return chunks

chunks_B = []
for ticker, year, p in sample_meta:
    chunks_B.extend(heading_aware_chunks(p, ticker, year))
print(f"Strategy B (heading-aware): {len(chunks_B)} chunks")"""),
    ("markdown", """## Choice

Empirical comparison on a small manual set (qa_pilot.jsonl) belongs in notebook 04. For now we go with **Strategy A (page-aware, 512 tokens, 50 overlap)** as the default — it preserves citations cleanly without depending on `unstructured`'s heading detection (which is unreliable on financial-filing layouts). Switch to B only if notebook 04 demonstrates a meaningful recall@5 win."""),
    ("code", """# Materialise the full corpus with Strategy A — one JSONL per PDF
written = 0
for p in sorted(RAW.rglob("*.pdf")):
    ticker = p.parent.name
    year = int(p.stem)
    pages = list(parse_pdf(p, ticker=ticker, year=year))
    chunks = list(chunk_pages(pages, max_tokens=512, overlap=50))
    out_dir = PROCESSED / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\\n")
    written += 1

print(f"Wrote chunks for {written} PDFs to {PROCESSED}")"""),
]


# ---------------------------------------------------------------------------
# Notebook 3 — 03_embedding_index_build.ipynb (Colab-first)
# ---------------------------------------------------------------------------

NB3: list[tuple[str, str]] = [
    ("markdown", """# 03 — Embedding index build (Colab T4)

Run this notebook **on Google Colab with T4 GPU**. Embeds all processed chunks with `BAAI/bge-small-en-v1.5`, builds ChromaDB and BM25 indices, persists to Google Drive, then downloads to local `data/vector/`.

Setup:
1. Upload `filings-rag/data/processed/` to your Google Drive at `MyDrive/filings-rag/data/processed/`
2. Open this notebook in Colab → Runtime → Change runtime type → T4 GPU
3. Run all cells"""),
    ("code", """# Install deps in Colab
!pip install -q sentence-transformers chromadb bm25s"""),
    ("code", """# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = "/content/drive/MyDrive/filings-rag"
PROCESSED = f"{DRIVE_ROOT}/data/processed"
VECTOR = f"{DRIVE_ROOT}/data/vector"

import os
os.makedirs(VECTOR, exist_ok=True)"""),
    ("code", """# Load all processed JSONL chunks
import json
from pathlib import Path

all_chunks = []
for jsonl_path in sorted(Path(PROCESSED).rglob("*.jsonl")):
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            all_chunks.append(json.loads(line))

print(f"Loaded {len(all_chunks)} chunks across {len(set((c['ticker'], c['year']) for c in all_chunks))} (ticker, year) pairs")"""),
    ("code", """# Embed with BGE-small on T4 GPU, batched
from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = SentenceTransformer("BAAI/bge-small-en-v1.5", device=device)
texts = [c["text"] for c in all_chunks]

embeddings = model.encode(
    texts,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
)
print(f"Shape: {embeddings.shape}")"""),
    ("code", """# Persist to Chroma (cosine), batched in 10-company chunks to survive any timeout
import chromadb

client = chromadb.PersistentClient(path=f"{VECTOR}/chroma")
coll = client.get_or_create_collection("filings", metadata={"hnsw:space": "cosine"})

BATCH = 1000
for i in range(0, len(all_chunks), BATCH):
    batch = all_chunks[i:i + BATCH]
    coll.add(
        documents=[c["text"] for c in batch],
        metadatas=[{"ticker": c["ticker"], "year": c["year"], "page": c["page"]} for c in batch],
        ids=[c["chunk_hash"] for c in batch],
        embeddings=embeddings[i:i + BATCH].tolist(),
    )
    print(f"  Indexed {i + len(batch)}/{len(all_chunks)}")
print("Chroma persistence complete.")"""),
    ("code", """# Build BM25 index over the same chunk IDs
import bm25s
import pickle

corpus_tokens = bm25s.tokenize([c["text"] for c in all_chunks], stopwords="en")
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

with open(f"{VECTOR}/bm25.pkl", "wb") as f:
    pickle.dump({"retriever": retriever, "ids": [c["chunk_hash"] for c in all_chunks]}, f)
print("BM25 index saved.")"""),
    ("code", """# Smoke-test both indices with 3 sanity queries
sanity_qs = [
    "climate-related risks",
    "going concern statement",
    "executive remuneration policy",
]

for q in sanity_qs:
    q_vec = model.encode([q], normalize_embeddings=True).tolist()
    res = coll.query(query_embeddings=q_vec, n_results=3)
    print(f"\\n--- {q!r} ---")
    for _id, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
        print(f"  {meta['ticker']}|{meta['year']}|p.{meta['page']}  score={1-dist:.3f}")"""),
    ("markdown", """## Download the indices to local

After this notebook finishes, run on your local machine:

```
gdown --folder https://drive.google.com/drive/folders/<id-of-filings-rag/data/vector>
```

Or zip the `data/vector/` folder in Drive and download manually. Place it at `filings-rag/data/vector/` locally."""),
]


# ---------------------------------------------------------------------------
# Notebook 4 — 04_retrieval_evaluation.ipynb
# ---------------------------------------------------------------------------

NB4: list[tuple[str, str]] = [
    ("markdown", """# 04 — Retrieval-only evaluation

Compare dense / BM25 / hybrid (RRF) / hybrid+reranker on a 15-question pilot QA set with known ground-truth chunks. Metrics: recall@5, recall@10, MRR.

Outputs the retrieval ablation block of `docs/eval_methodology.md`."""),
    ("code", """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import json
import os
from dotenv import load_dotenv
load_dotenv(Path("../.env"))

from src.retrieval.embeddings import BGEEmbedder
from src.retrieval.vector_store import ChromaStore
from src.retrieval.bm25_index import BM25Index
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import BGEReranker"""),
    ("code", """# Load indices and pilot QA set
embedder = BGEEmbedder()
vec_store = ChromaStore(collection="filings", persist_dir=Path(os.environ.get("CHROMA_DIR", "../data/vector/chroma")))
bm25 = BM25Index.load(Path(os.environ.get("BM25_PATH", "../data/vector/bm25.pkl")))
reranker = BGEReranker()

with Path("../data/qa_pilot.jsonl").open(encoding="utf-8") as f:
    pilot = [json.loads(line) for line in f]

print(f"Pilot questions: {len(pilot)}")"""),
    ("code", """# Build chunk_lookup (id -> chunk record) from processed JSONLs
PROCESSED = Path("../data/processed")
chunk_lookup = {}
for jsonl in PROCESSED.rglob("*.jsonl"):
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_lookup[c["chunk_hash"]] = {
                "text": c["text"],
                "metadata": {"ticker": c["ticker"], "year": c["year"], "page": c["page"]},
            }
print(f"Chunks in lookup: {len(chunk_lookup)}")"""),
    ("code", """def recall_at_k(retrieved_ids, gold_ids, k):
    \"\"\"Fraction of gold IDs found in top-k retrieved.\"\"\"
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return sum(1 for g in gold_ids if g in top_k) / len(gold_ids)

def mrr(retrieved_ids, gold_ids):
    \"\"\"Mean reciprocal rank of the first gold ID in the ranked list.\"\"\"
    for rank, _id in enumerate(retrieved_ids, start=1):
        if _id in gold_ids:
            return 1.0 / rank
    return 0.0

def aggregate(metrics_per_q):
    return {k: sum(m[k] for m in metrics_per_q) / len(metrics_per_q) for k in metrics_per_q[0]}"""),
    ("code", """# Strategy 1: Dense only
dense_metrics = []
for row in pilot:
    q_vec = embedder.embed([row["question"]])[0]
    hits = vec_store.query(q_vec, k=20)
    ids = [h["id"] for h in hits]
    dense_metrics.append({
        "recall@5": recall_at_k(ids, row["ground_truth_chunk_ids"], 5),
        "recall@10": recall_at_k(ids, row["ground_truth_chunk_ids"], 10),
        "mrr": mrr(ids, row["ground_truth_chunk_ids"]),
    })
print("Dense only:", aggregate(dense_metrics))"""),
    ("code", """# Strategy 2: BM25 only
bm25_metrics = []
for row in pilot:
    hits = bm25.query(row["question"], k=20)
    ids = [h["id"] for h in hits]
    bm25_metrics.append({
        "recall@5": recall_at_k(ids, row["ground_truth_chunk_ids"], 5),
        "recall@10": recall_at_k(ids, row["ground_truth_chunk_ids"], 10),
        "mrr": mrr(ids, row["ground_truth_chunk_ids"]),
    })
print("BM25 only:", aggregate(bm25_metrics))"""),
    ("code", """# Strategy 3: Hybrid (RRF over top-50 from each)
hybrid_metrics = []
for row in pilot:
    q_vec = embedder.embed([row["question"]])[0]
    dense_hits = vec_store.query(q_vec, k=50)
    bm25_hits = bm25.query(row["question"], k=50)
    fused = reciprocal_rank_fusion([dense_hits, bm25_hits])
    ids = [h["id"] for h in fused][:20]
    hybrid_metrics.append({
        "recall@5": recall_at_k(ids, row["ground_truth_chunk_ids"], 5),
        "recall@10": recall_at_k(ids, row["ground_truth_chunk_ids"], 10),
        "mrr": mrr(ids, row["ground_truth_chunk_ids"]),
    })
print("Hybrid (RRF):", aggregate(hybrid_metrics))"""),
    ("code", """# Strategy 4: Hybrid + reranker
hyrr_metrics = []
for row in pilot:
    q_vec = embedder.embed([row["question"]])[0]
    dense_hits = vec_store.query(q_vec, k=50)
    bm25_hits = bm25.query(row["question"], k=50)
    fused = reciprocal_rank_fusion([dense_hits, bm25_hits])[:20]
    candidates = [chunk_lookup[h["id"]] for h in fused if h["id"] in chunk_lookup]
    for c, h in zip(candidates, fused):
        c["id"] = h["id"]
    reranked = reranker.rerank(row["question"], candidates, top_k=10)
    ids = [c["id"] for c in reranked]
    hyrr_metrics.append({
        "recall@5": recall_at_k(ids, row["ground_truth_chunk_ids"], 5),
        "recall@10": recall_at_k(ids, row["ground_truth_chunk_ids"], 10),
        "mrr": mrr(ids, row["ground_truth_chunk_ids"]),
    })
print("Hybrid + re-ranker:", aggregate(hyrr_metrics))"""),
    ("code", """# Summary table for eval_methodology.md
import pandas as pd
table = pd.DataFrame({
    "Dense only": aggregate(dense_metrics),
    "BM25 only": aggregate(bm25_metrics),
    "Hybrid (RRF)": aggregate(hybrid_metrics),
    "Hybrid + re-ranker": aggregate(hyrr_metrics),
}).T
table = table[["recall@5", "recall@10", "mrr"]].round(3)
print(table.to_markdown())
table.to_csv("../docs/retrieval_ablation.csv")"""),
]


# ---------------------------------------------------------------------------
# Notebook 5 — 05_end_to_end_eval.ipynb
# ---------------------------------------------------------------------------

NB5: list[tuple[str, str]] = [
    ("markdown", """# 05 — End-to-end RAG eval (Ragas + MLflow)

Run the full chain over the 40-question test set; compute Ragas faithfulness / answer_relevancy / context_precision / context_recall + custom refusal_accuracy on the adversarial subset; log to MLflow."""),
    ("code", """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import json
import os
from dotenv import load_dotenv
load_dotenv(Path("../.env"))

from src.retrieval.embeddings import BGEEmbedder
from src.retrieval.vector_store import ChromaStore
from src.retrieval.bm25_index import BM25Index
from src.retrieval.reranker import BGEReranker
from src.generation.llm_client import build_llm
from src.chain import answer_question
from src.eval.ragas_eval import load_qa_set, run_chain_on_qa, run_ragas_eval
from src.eval.mlflow_logger import log_eval_run"""),
    ("code", """# Load everything once
embedder = BGEEmbedder()
vec_store = ChromaStore(collection="filings", persist_dir=Path(os.environ["CHROMA_DIR"]))
bm25 = BM25Index.load(Path(os.environ["BM25_PATH"]))
reranker = BGEReranker()
llm = build_llm()

PROCESSED = Path("../data/processed")
chunk_lookup = {}
for jsonl in PROCESSED.rglob("*.jsonl"):
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk_lookup[c["chunk_hash"]] = {
                "text": c["text"],
                "metadata": {"ticker": c["ticker"], "year": c["year"], "page": c["page"]},
            }

qa_set = load_qa_set(Path("../data/qa_test_set.jsonl"))
print(f"QA set: {len(qa_set)} questions; chunk_lookup: {len(chunk_lookup)} chunks")"""),
    ("code", """# Bound chain function for the eval pipeline
def chain_fn(question: str) -> dict:
    return answer_question(
        question,
        embedder=embedder,
        vector_store=vec_store,
        bm25=bm25,
        reranker=reranker,
        llm=llm,
        chunk_lookup=chunk_lookup,
    )"""),
    ("code", """# Execute the chain over the 40 questions (this is the slow step — ~10 min on free Groq)
rows = run_chain_on_qa(qa_set, chain_fn)
print(f"Completed {len(rows)} runs")"""),
    ("code", """# Ragas metrics + custom refusal_accuracy
metrics = run_ragas_eval(rows)
metrics"""),
    ("code", """# Log to MLflow tagged by retrieval config (hybrid + reranker + query_rewrite is the default)
run_id = log_eval_run(
    metrics,
    config={
        "retrieval": "hybrid+rerank",
        "query_rewriter": "on",
        "embedder": "BAAI/bge-small-en-v1.5",
        "reranker": "BAAI/bge-reranker-v2-m3",
        "llm": os.environ.get("LLM_PROVIDER", "groq"),
    },
)
print(f"MLflow run: {run_id}")"""),
    ("code", """# Persist a snapshot of the per-question results for inspection
import pandas as pd
df = pd.DataFrame(rows)
df.to_csv("../docs/end_to_end_eval_run.csv", index=False)
print("Wrote ../docs/end_to_end_eval_run.csv")"""),
]


NOTEBOOKS = {
    "01_eda_corpus.ipynb": NB1,
    "02_chunking_experiments.ipynb": NB2,
    "03_embedding_index_build.ipynb": NB3,
    "04_retrieval_evaluation.ipynb": NB4,
    "05_end_to_end_eval.ipynb": NB5,
}


def main() -> None:
    out_dir = Path(__file__).parent
    for name, cells in NOTEBOOKS.items():
        nb = build_notebook(cells)
        path = out_dir / name
        with path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
            f.write("\n")
        print(f"Wrote {name} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
