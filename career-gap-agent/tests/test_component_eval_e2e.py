"""End-to-end component eval: real ESCO index + real LLM extraction.

Gated by RUN_SLOW=1 and a valid GROQ_API_KEY or GOOGLE_API_KEY.
Run with:
    RUN_SLOW=1 GROQ_API_KEY=<key> python -m pytest tests/test_component_eval_e2e.py -v
"""
from __future__ import annotations

import os

import pytest

GOLD_PATH = "data/fixtures/gold_skills.json"


@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="slow: set RUN_SLOW=1 to enable")
def test_e2e_component_eval():
    if not os.getenv("GROQ_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("No LLM key available (set GROQ_API_KEY or GOOGLE_API_KEY)")

    from src.data.esco_loader import EscoIndex
    from src.eval.component_eval import evaluate
    from src.generation.llm_client import simple_complete
    from src.skills.esco_matcher import EscoMatcher
    from src.skills.extractor import extract_skill_phrases

    index = EscoIndex.load()
    matcher = EscoMatcher(index=index)

    def extract(text: str) -> list[str]:
        return extract_skill_phrases(text, llm=lambda p: simple_complete(p))

    result = evaluate(GOLD_PATH, extract, matcher.match)

    assert result["n"] > 0, "Gold set loaded no entries"
    assert result["macro_f1"] > 0.3, (
        f"macro_f1={result['macro_f1']:.3f} is below the 0.3 sanity floor; "
        f"precision={result['macro_precision']:.3f}, recall={result['macro_recall']:.3f}"
    )
