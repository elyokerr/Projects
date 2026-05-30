import pytest

from src.retrieval.candidates import union_candidates


def test_union_rejects_unknown_source_name():
    with pytest.raises(ValueError, match="Unknown candidate source: bogus"):
        union_candidates({"bogus": [("z", 1.0)]})


def test_union_dedups_and_keeps_per_source_scores():
    sources = {
        "pop": [("a", 0.9), ("b", 0.5)],
        "als": [("b", 2.0), ("c", 1.0)],
    }
    cands = union_candidates(sources, k_per_source=10)
    items = {c["item_id"] for c in cands}
    assert items == {"a", "b", "c"}
    b = next(c for c in cands if c["item_id"] == "b")
    assert b["pop"] == 0.5 and b["als"] == 2.0  # per-source score retained
    a = next(c for c in cands if c["item_id"] == "a")
    assert a["als"] == 0.0  # missing source -> 0.0


def test_union_truncates_per_source():
    sources = {
        "pop": [("a", 0.9), ("b", 0.5), ("c", 0.1)],
        "i2i": [("d", 1.0)],
    }
    cands = union_candidates(sources, k_per_source=2)
    items = {c["item_id"] for c in cands}
    assert items == {"a", "b", "d"}  # 'c' truncated from pop


def test_union_has_all_canonical_keys():
    sources = {"pop": [("a", 0.9)]}
    cands = union_candidates(sources, k_per_source=10)
    row = cands[0]
    for key in ("pop", "i2i", "als", "tt"):
        assert key in row
    assert row["i2i"] == 0.0 and row["als"] == 0.0 and row["tt"] == 0.0
