from src.skills.extractor import extract_skill_phrases


def test_parses_json_list():
    fake_llm = lambda prompt: '["Python", "SQL", "machine learning"]'  # noqa: E731
    out = extract_skill_phrases("We need Python, SQL and ML.", llm=fake_llm)
    assert out == ["python", "sql", "machine learning"]


def test_tolerates_codefenced_json():
    fake_llm = lambda prompt: '```json\n["Docker"]\n```'  # noqa: E731
    assert extract_skill_phrases("Docker shop", llm=fake_llm) == ["docker"]


def test_empty_on_unparseable():
    assert extract_skill_phrases("x", llm=lambda p: "not json") == []
