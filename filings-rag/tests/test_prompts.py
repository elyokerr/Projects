from src.generation.prompts import extract_citations, validate_citations


def test_extract_well_formed_citations():
    answer = "BP reported X [BP|2024|p.42]. Also Y [BP|2024|p.43]."
    cites = extract_citations(answer)
    assert cites == [("BP", 2024, 42), ("BP", 2024, 43)]


def test_extract_handles_malformed_gracefully():
    answer = "no citations here at all"
    assert extract_citations(answer) == []


def test_validator_rejects_unseen_pages():
    retrieved = [{"metadata": {"ticker": "BP", "year": 2024, "page": 42}}]
    assert validate_citations([("BP", 2024, 42)], retrieved) is True
    assert validate_citations([("BP", 2024, 99)], retrieved) is False
