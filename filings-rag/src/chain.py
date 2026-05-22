"""End-to-end RAG chain assembling rewriter, hybrid retrieval, re-rank, LLM, and citation validation."""

from typing import Dict, Iterable

from src.generation.llm_client import llm_invoke
from src.generation.prompts import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    extract_citations,
    validate_citations,
)
from src.generation.query_rewriter import rewrite_query
from src.retrieval.hybrid import reciprocal_rank_fusion


def _build_context(chunks: Iterable[dict]) -> str:
    """Format retrieved chunks into a numbered, citation-tagged context block."""
    return "\n\n---\n\n".join(
        f"[{c['metadata']['ticker']}|{c['metadata']['year']}|p.{c['metadata']['page']}] {c['text']}"
        for c in chunks
    )


def answer_question(
    question: str,
    *,
    embedder,
    vector_store,
    bm25,
    reranker,
    llm,
    chunk_lookup: Dict[str, dict],
    top_k_dense: int = 50,
    top_k_bm25: int = 50,
    fuse_top_n: int = 20,
    rerank_top_k: int = 5,
) -> dict:
    """Run a single RAG query end-to-end and return answer + citations + supporting chunks.

    `chunk_lookup` maps the chunk ID stored in both the Chroma collection and BM25 index
    to the full chunk record (with text + metadata). It's the bridge between retrieval
    (which returns IDs) and the LLM prompt (which needs text + metadata).
    """
    # 1. Optional query rewriting for short queries.
    query = rewrite_query(question, llm)

    # 2. Hybrid retrieval — dense + BM25 fused via Reciprocal Rank Fusion.
    q_vec = embedder.embed([query])[0]
    dense_hits = vector_store.query(q_vec, k=top_k_dense)
    bm25_hits = bm25.query(query, k=top_k_bm25)
    fused = reciprocal_rank_fusion([dense_hits, bm25_hits])[:fuse_top_n]
    candidates = [chunk_lookup[h["id"]] for h in fused if h["id"] in chunk_lookup]

    if not candidates:
        return {
            "answer": "No relevant content found in the corpus.",
            "citations": [],
            "chunks": [],
            "rewritten_query": query,
        }

    # 3. Cross-encoder re-ranking → top-k chunks.
    top_chunks = reranker.rerank(query, candidates, top_k=rerank_top_k)
    if not top_chunks:
        return {
            "answer": "No relevant content found in the corpus.",
            "citations": [],
            "chunks": [],
            "rewritten_query": query,
        }

    # 4. LLM call with grounded-citation prompt.
    context = _build_context(top_chunks)
    msgs = [
        ("system", SYSTEM_PROMPT),
        ("user", USER_TEMPLATE.format(context=context, question=question)),
    ]
    resp = llm_invoke(llm, msgs)
    answer = resp.content
    cites = extract_citations(answer)

    # 5. Citation validation — regenerate once if any citation is unverifiable.
    if cites and not validate_citations(cites, top_chunks):
        msgs.append(
            (
                "user",
                "Your previous answer had unverifiable citations. "
                "Re-answer using ONLY valid [TICKER|YEAR|p.PAGE] citations drawn from the context.",
            )
        )
        resp = llm_invoke(llm, msgs)
        answer = resp.content
        cites = extract_citations(answer)

    return {
        "answer": answer,
        "citations": cites,
        "chunks": top_chunks,
        "rewritten_query": query,
    }
