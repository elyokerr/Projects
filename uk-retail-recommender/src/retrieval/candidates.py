from __future__ import annotations

# Canonical candidate-source name keys, reused by Phase 3 features and the
# Phase 5 pipeline. Do not rename: pop (popularity), i2i (item-to-item),
# als (matrix factorisation), tt (two-tower).
SOURCE_NAMES: tuple[str, ...] = ("pop", "i2i", "als", "tt")


def union_candidates(
    sources: dict[str, list[tuple[str, float]]],
    k_per_source: int = 200,
) -> list[dict]:
    """Union candidate lists from multiple retrieval sources.

    Each source maps a name (one of ``SOURCE_NAMES``) to a ranked list of
    ``(item_id, score)``. The top ``k_per_source`` from each source are kept.
    Returns one dict per distinct item with ``item_id`` plus one key per
    canonical source name; a source that did not surface an item scores 0.0.
    """
    scores: dict[str, dict[str, float]] = {}
    for name, items in sources.items():
        for item_id, score in items[:k_per_source]:
            row = scores.setdefault(item_id, {})
            row[name] = float(score)

    out: list[dict] = []
    for item_id, per_source in scores.items():
        record = {"item_id": item_id}
        for name in SOURCE_NAMES:
            record[name] = per_source.get(name, 0.0)
        out.append(record)
    return out
