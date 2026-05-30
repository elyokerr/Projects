"""FastAPI web application for the UK Retail Recommender.

The app imports and boots with no secrets — the pipeline is built lazily on the
first real recommendation request via _get_pipeline(). Health checks and the
index page work without any trained model present.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pickle
from functools import lru_cache

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="UK Retail Recommender")

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pipeline.pkl"
_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_transactions.parquet"
)

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Lazy pipeline singleton
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_pipeline():
    """Return the trained RecommenderPipeline.

    If models/pipeline.pkl exists, unpickle it. Otherwise train on the sample
    fixture parquet (a few seconds; ALS iters kept low).
    """
    from src.pipeline import RecommenderPipeline

    if _MODEL_PATH.exists():
        with open(_MODEL_PATH, "rb") as f:
            return pickle.load(f)  # noqa: S301

    import pandas as pd

    df = pd.read_parquet(_FIXTURE_PATH)
    return RecommenderPipeline.train(df, als_iters=3)


@lru_cache(maxsize=1)
def _get_item2item():
    """Return an ItemToItem model fit on the same data as the pipeline."""
    import pandas as pd

    from src.retrieval.item2item import ItemToItem

    df = pd.read_parquet(_FIXTURE_PATH)
    return ItemToItem().fit(df)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, customer_id: str = "", k: str = "10"):
    try:
        cid = int(customer_id)
        k_int = max(1, min(int(k), 50))
    except (ValueError, TypeError):
        return templates.TemplateResponse(
            request,
            "_results.html",
            {"error": f"Invalid input: customer_id must be an integer, got '{customer_id}'."},
        )

    pipeline = _get_pipeline()
    recs = pipeline.recommend(cid, k_int)
    is_cold = bool(recs) and recs[0]["source"] == "cold_start"
    return templates.TemplateResponse(
        request,
        "_results.html",
        {"error": None, "recs": recs, "is_cold": is_cold},
    )


@app.get("/similar", response_class=HTMLResponse)
async def similar(request: Request, item_id: str = "", k: str = "10"):
    try:
        k_int = max(1, min(int(k), 50))
    except (ValueError, TypeError):
        k_int = 10

    i2i = _get_item2item()
    results = i2i.similar(str(item_id), k_int)
    return templates.TemplateResponse(
        request,
        "_similar.html",
        {"item_id": item_id, "results": results},
    )
