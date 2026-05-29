import json

from src.eval.component_eval import evaluate, prf1

# ---------------------------------------------------------------------------
# prf1 unit tests
# ---------------------------------------------------------------------------


def test_prf1_perfect():
    p, r, f = prf1(pred={"python", "sql"}, gold={"python", "sql"})
    assert (p, r, f) == (1.0, 1.0, 1.0)


def test_prf1_partial():
    p, r, f = prf1(pred={"python"}, gold={"python", "sql"})
    assert p == 1.0 and r == 0.5


def test_prf1_both_empty():
    assert prf1(pred=set(), gold=set()) == (1.0, 1.0, 1.0)


def test_prf1_pred_empty_gold_nonempty():
    p, r, f = prf1(pred=set(), gold={"python"})
    assert p == 0.0 and r == 0.0 and f == 0.0


def test_prf1_pred_nonempty_gold_empty():
    # gold empty: recall = 1.0 (nothing to miss), precision = 0/1 = 0, f1 = 0
    p, r, f = prf1(pred={"python"}, gold=set())
    assert p == 0.0 and r == 1.0 and f == 0.0


def test_prf1_no_overlap():
    p, r, f = prf1(pred={"docker"}, gold={"python"})
    assert p == 0.0 and r == 0.0 and f == 0.0


# ---------------------------------------------------------------------------
# evaluate aggregation test (no LLM, no fastembed)
# ---------------------------------------------------------------------------


def test_evaluate_aggregation(tmp_path):
    """Verify macro aggregation math using trivial extract/match stubs."""
    # Two gold entries:
    #   entry 0: gold={"python","sql"}, pred={"python","sql"} -> p=1, r=1, f=1
    #   entry 1: gold={"python","sql"}, pred={"python"}       -> p=1, r=0.5, f=2/3
    # macro_precision = (1 + 1) / 2 = 1.0
    # macro_recall    = (1 + 0.5) / 2 = 0.75
    # macro_f1        = (1 + 2/3) / 2 = 5/6 ≈ 0.8333

    gold_data = [
        {"description": "desc_a", "esco_skills": ["python", "sql"]},
        {"description": "desc_b", "esco_skills": ["python", "sql"]},
    ]
    gold_path = tmp_path / "gold.json"
    gold_path.write_text(json.dumps(gold_data), encoding="utf-8")

    # For desc_a extract returns ["python","sql"], match passes through
    # For desc_b extract returns ["python"], match passes through
    def extract_fn(text: str) -> list[str]:
        if text == "desc_a":
            return ["python", "sql"]
        return ["python"]

    def match_fn(phrases: list[str]) -> list[str]:
        return phrases  # identity

    result = evaluate(gold_path, extract_fn, match_fn)

    assert result["n"] == 2
    assert result["macro_precision"] == 1.0
    assert abs(result["macro_recall"] - 0.75) < 1e-9
    expected_f1 = (1.0 + 2 / 3) / 2
    assert abs(result["macro_f1"] - expected_f1) < 1e-9
