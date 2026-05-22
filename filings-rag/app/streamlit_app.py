"""Streamlit chat UI for Filings-RAG.

Lazy-loads the persisted retrieval indices once per session; runs queries
through the LCEL chain; renders answers with click-through citations.
"""

import json
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from src.x import y` works when
# Streamlit launches the script from inside `app/`. Must precede any `src.*` import.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

from src.chain import answer_question
from src.generation.llm_client import build_llm
from src.retrieval.bm25_index import BM25Index
from src.retrieval.embeddings import BGEEmbedder
from src.retrieval.reranker import BGEReranker
from src.retrieval.vector_store import ChromaStore


load_dotenv()

st.set_page_config(
    page_title="Filings-RAG — FTSE 100 Annual Reports QA",
    layout="wide",
)
st.title("Filings-RAG")
st.caption(
    "Question-answering over UK FTSE 100 annual reports, with grounded citations."
)


@st.cache_resource(
    show_spinner="Warming up retrieval indices — first query may take ~20s..."
)
def load_components():
    chroma_dir = Path(os.environ["CHROMA_DIR"])
    bm25_path = Path(os.environ["BM25_PATH"])
    processed_dir = Path(os.environ["PROCESSED_DIR"])

    embedder = BGEEmbedder()
    vector_store = ChromaStore(collection="filings", persist_dir=chroma_dir)
    bm25 = BM25Index.load(bm25_path)
    reranker = BGEReranker()
    llm = build_llm()

    # Build chunk_lookup from all processed JSONL files.
    chunk_lookup: dict = {}
    for jsonl in processed_dir.rglob("*.jsonl"):
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                chunk_lookup[chunk["chunk_hash"]] = {
                    "text": chunk["text"],
                    "metadata": {
                        "ticker": chunk["ticker"],
                        "year": chunk["year"],
                        "page": chunk["page"],
                    },
                }
    return embedder, vector_store, bm25, reranker, llm, chunk_lookup


def _render_chunk_expander(chunk: dict) -> None:
    meta = chunk["metadata"]
    label = f"[{meta['ticker']}|{meta['year']}|p.{meta['page']}]"
    with st.expander(label):
        st.write(chunk["text"])


def main() -> None:
    with st.sidebar:
        st.header("About")
        st.markdown(
            "Ask questions about FTSE 100 annual reports. Answers include "
            "verbatim citations as `[TICKER|YEAR|p.PAGE]`. Click any citation "
            "to expand the source paragraph."
        )
        st.markdown("---")
        st.caption(
            "Built with LangChain, ChromaDB, BGE embeddings + re-ranker, "
            "and Groq Llama 3.3 70B."
        )

    try:
        embedder, vector_store, bm25, reranker, llm, chunk_lookup = load_components()
    except Exception as e:  # noqa: BLE001 — surface load failures to the user
        st.error(
            "Failed to load retrieval indices. Have you run the Colab indexing "
            f"notebook? Underlying error: `{e}`"
        )
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render prior history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for chunk in msg.get("chunks", []):
                _render_chunk_expander(chunk)

    if prompt := st.chat_input("Ask about a FTSE 100 annual report..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching filings..."):
                try:
                    result = answer_question(
                        prompt,
                        embedder=embedder,
                        vector_store=vector_store,
                        bm25=bm25,
                        reranker=reranker,
                        llm=llm,
                        chunk_lookup=chunk_lookup,
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Query failed: `{e}`")
                    return

            st.markdown(result["answer"])
            for chunk in result["chunks"]:
                _render_chunk_expander(chunk)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "chunks": result["chunks"],
                }
            )


if __name__ == "__main__":
    main()
