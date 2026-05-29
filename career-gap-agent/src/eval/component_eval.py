from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path


def prf1(pred: set, gold: set) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for two sets of skill labels.

    Edge cases:
    - Both empty: (1.0, 1.0, 1.0)
    - pred empty, gold non-empty: (0.0, 0.0, 0.0)  -- recall is zero
    - pred non-empty, gold empty: (0.0, 1.0, 0.0)  -- nothing missed but precision is 0/n
    """
    tp = len(pred & gold)

    if not pred and not gold:
        return (1.0, 1.0, 1.0)

    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 1.0

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, f1)


def evaluate(
    gold_path: str | Path,
    extract_fn: Callable[[str], list[str]],
    match_fn: Callable[[list[str]], list[str]],
) -> dict:
    """Compute macro precision, recall and F1 over a gold set file.

    Args:
        gold_path: path to a JSON file, a list of
                   {"description": str, "esco_skills": list[str]}.
        extract_fn: maps a job description text -> list of raw skill phrases.
        match_fn: maps raw phrases -> list of ESCO preferredLabel strings.

    Returns:
        {"macro_precision": float, "macro_recall": float, "macro_f1": float, "n": int}
    """
    gold_entries: list[dict] = json.loads(Path(gold_path).read_text(encoding="utf-8"))

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for entry in gold_entries:
        pred = set(match_fn(extract_fn(entry["description"])))
        gold = set(entry["esco_skills"])
        p, r, f = prf1(pred, gold)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    n = len(gold_entries)
    return {
        "macro_precision": sum(precisions) / n if n else 0.0,
        "macro_recall": sum(recalls) / n if n else 0.0,
        "macro_f1": sum(f1s) / n if n else 0.0,
        "n": n,
    }
