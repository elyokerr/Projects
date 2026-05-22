"""Build Chroma + BM25 indices locally on CPU using fastembed.

Replaces the Colab T4 step. Runs in ~5 minutes on a basic Windows laptop
thanks to ONNX-optimised inference.

Usage (from filings-rag/ root):
    python -m scripts.build_indices                 # full corpus
    python -m scripts.build_indices --max-pdfs 5    # demo subset (faster)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import time
from pathlib import Path
from typing import List

import bm25s
import chromadb
from dotenv import load_dotenv

from src.retrieval.embeddings import BGEEmbedder


def load_chunks(processed_dir: Path, max_pdfs: int | None = None) -> List[dict]:
    """Load chunks from every JSONL in processed_dir; optionally cap PDF count."""
    jsonls = sorted(processed_dir.rglob("*.jsonl"))
    if max_pdfs is not None:
        jsonls = jsonls[:max_pdfs]
        print(f"Subset mode: using {len(jsonls)} PDFs")
    chunks: List[dict] = []
    for jsonl_path in jsonls:
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def build_chroma(chunks: List[dict], embeddings: List[List[float]], persist_dir: Path) -> None:
    """Persist `embeddings` + chunk metadata into a Chroma collection."""
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    client = chromadb.PersistentClient(path=str(persist_dir))
    coll = client.get_or_create_collection("filings", metadata={"hnsw:space": "cosine"})
    BATCH = 1000
    for i in range(0, len(chunks), BATCH):
        b_chunks = chunks[i : i + BATCH]
        b_embs = embeddings[i : i + BATCH]
        coll.add(
            documents=[c["text"] for c in b_chunks],
            metadatas=[{"ticker": c["ticker"], "year": c["year"], "page": c["page"]} for c in b_chunks],
            ids=[c["chunk_hash"] for c in b_chunks],
            embeddings=b_embs,
        )
        print(f"    chroma: {min(i + BATCH, len(chunks))}/{len(chunks)}")


def build_bm25(chunks: List[dict], out_path: Path) -> None:
    corpus_tokens = bm25s.tokenize([c["text"] for c in chunks], stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    with out_path.open("wb") as f:
        pickle.dump({"retriever": retriever, "ids": [c["chunk_hash"] for c in chunks]}, f)


def smoke_test(chroma_dir: Path, embedder: BGEEmbedder) -> None:
    client = chromadb.PersistentClient(path=str(chroma_dir))
    coll = client.get_collection("filings")
    queries = [
        "climate-related risks",
        "going concern statement",
        "executive remuneration policy",
    ]
    for q in queries:
        q_vec = embedder.embed([q])[0]
        res = coll.query(query_embeddings=[q_vec], n_results=3)
        print(f"\n  Query: {q!r}")
        for _id, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
            print(f"    {meta['ticker']}|{meta['year']}|p.{meta['page']}  score={1-dist:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pdfs", type=int, default=None, help="Subset to first N PDFs (alphabetical)")
    args = parser.parse_args()

    load_dotenv()
    processed = Path(os.environ.get("PROCESSED_DIR", "./data/processed"))
    chroma_dir = Path(os.environ.get("CHROMA_DIR", "./data/vector/chroma"))
    bm25_path = Path(os.environ.get("BM25_PATH", "./data/vector/bm25.pkl"))

    chroma_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"PROCESSED: {processed}")
    print(f"CHROMA:    {chroma_dir}")
    print(f"BM25:      {bm25_path}")
    print()

    t0 = time.time()
    chunks = load_chunks(processed, max_pdfs=args.max_pdfs)
    pairs = sorted({(c["ticker"], c["year"]) for c in chunks})
    print(f"Loaded {len(chunks):,} chunks across {len(pairs)} (ticker, year) pairs")

    print("\nLoading fastembed BGE-small...")
    t1 = time.time()
    embedder = BGEEmbedder()
    print(f"  loaded in {time.time() - t1:.1f}s")

    print(f"\nEmbedding {len(chunks):,} chunks (live progress per batch)...", flush=True)
    t1 = time.time()
    BATCH = 256
    embeddings: List[List[float]] = []
    for start in range(0, len(chunks), BATCH):
        batch_texts = [c["text"] for c in chunks[start : start + BATCH]]
        embeddings.extend(embedder.embed(batch_texts))
        done = start + len(batch_texts)
        rate = done / (time.time() - t1)
        eta_min = (len(chunks) - done) / rate / 60 if rate > 0 else 0
        print(f"    {done:>6,}/{len(chunks):,}  ({rate:.1f} chunks/s, ETA {eta_min:.1f} min)", flush=True)
    print(f"  done in {time.time() - t1:.1f}s ({len(chunks) / (time.time() - t1):.1f} chunks/sec)", flush=True)

    print("\nBuilding Chroma...")
    t1 = time.time()
    build_chroma(chunks, embeddings, chroma_dir)
    print(f"  done in {time.time() - t1:.1f}s")

    print("\nBuilding BM25...")
    t1 = time.time()
    build_bm25(chunks, bm25_path)
    print(f"  done in {time.time() - t1:.1f}s")

    print("\nSmoke test:")
    smoke_test(chroma_dir, embedder)

    print(f"\nTotal: {time.time() - t0:.1f}s")
    print(f"\nDone. Indices written to {chroma_dir.parent}")


if __name__ == "__main__":
    main()
