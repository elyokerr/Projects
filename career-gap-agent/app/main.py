"""FastAPI web application for the career-gap agent.

The app imports and boots with no secrets — the agent and matcher are built
lazily on the first real analysis request via _get_runtime(). This means
health checks, the index page, and empty-CV tests all work without any API
keys or ESCO data files present.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Career Gap Agent")

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_STATIC_DIR = Path(__file__).resolve().parent / "static"

templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Lazy runtime singleton (matcher + graph deps)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_runtime():
    """Build and cache the EscoMatcher.

    Called only on first real analysis; raises if ESCO data files are missing.
    """
    from src.data.esco_loader import EscoIndex
    from src.skills.esco_matcher import EscoMatcher

    index = EscoIndex.load()
    return EscoMatcher(index)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    cv_text: Annotated[str, Form()] = "",
    role: Annotated[str, Form()] = "",
    location: Annotated[str, Form()] = "",
    cv_pdf: UploadFile | None = None,
):
    def _error_fragment(message: str) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "_report.html",
            {"error": message},
        )

    # ------------------------------------------------------------------
    # 1. Parse CV
    # ------------------------------------------------------------------
    from src.data.cv_parser import CvParseError, parse_cv

    pdf_bytes: bytes | None = None
    if cv_pdf and cv_pdf.filename:
        pdf_bytes = await cv_pdf.read()
        if not pdf_bytes:
            pdf_bytes = None

    try:
        parsed_cv = parse_cv(
            text=cv_text if not pdf_bytes else None,
            pdf_bytes=pdf_bytes,
        )
    except CvParseError as exc:
        return _error_fragment(
            f"Could not read your CV: {exc}. Please paste your CV text or upload a valid PDF."
        )

    # ------------------------------------------------------------------
    # 2. Build LLM (raises RuntimeError if no key set)
    # ------------------------------------------------------------------
    try:
        from src.generation.llm_client import build_chat_model, simple_complete

        llm = build_chat_model()
    except RuntimeError:
        return _error_fragment(
            "Analysis needs a free API key. Set GROQ_API_KEY (Groq) or "
            "GOOGLE_API_KEY (Gemini) in your environment and restart the app."
        )

    # ------------------------------------------------------------------
    # 3. Build deps / graph and run the agent
    # ------------------------------------------------------------------
    from langchain_core.messages import HumanMessage

    from src.agent.graph import build_graph
    from src.agent.tools import AgentDeps
    from src.eval.langfuse_setup import get_callbacks
    from src.generation.llm_client import simple_complete  # noqa: F811

    matcher = _get_runtime()

    deps = AgentDeps(
        app_id=os.getenv("ADZUNA_APP_ID"),
        app_key=os.getenv("ADZUNA_APP_KEY"),
        matcher=matcher,
        llm=lambda p: simple_complete(p, model=llm),
    )

    graph = build_graph(llm, deps)

    user_instruction = (
        f"I am looking for a '{role}' role in '{location}'. "
        f"Here is my CV:\n\n{parsed_cv}\n\n"
        "Please search for relevant job postings and analyse the skill gaps "
        "between my CV and the current market demand."
    )

    initial_state = {
        "messages": [HumanMessage(content=user_instruction)],
        "role": role,
        "location": location,
        "cv_text": parsed_cv,
        "iterations": 0,
    }

    final = graph.invoke(
        initial_state,
        config={"callbacks": get_callbacks(), "recursion_limit": 50},
    )

    report = final.get("report")
    if report is None:
        return _error_fragment(
            "The agent did not produce a report. Check your API keys and try again."
        )

    return templates.TemplateResponse(
        request,
        "_report.html",
        {"error": None, "report": report},
    )
