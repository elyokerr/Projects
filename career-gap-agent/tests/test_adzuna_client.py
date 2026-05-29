from pathlib import Path

from src.data.adzuna_client import _parse_results, search_jobs

SNAPSHOT = Path("data/fixtures/adzuna_snapshot.json")


def test_snapshot_fallback_when_no_credentials():
    postings = search_jobs("data scientist", "london", n=5, app_id=None, app_key=None)
    assert len(postings) > 0
    assert all(p.title and p.description for p in postings)


def test_parse_results_maps_fields():
    raw = {
        "results": [
            {
                "title": "Data Scientist",
                "description": "We use Python and SQL.",
                "company": {"display_name": "Acme"},
                "location": {"display_name": "London, UK"},
                "salary_min": 50000,
                "salary_max": 70000,
            }
        ]
    }
    postings = _parse_results(raw)
    p = postings[0]
    assert p.title == "Data Scientist"
    assert p.company == "Acme"
    assert p.location == "London, UK"
    assert p.salary_min == 50000


def test_live_uses_injected_transport():
    def fake_transport(url, params):
        return {
            "results": [
                {
                    "title": "ML Engineer",
                    "description": "PyTorch.",
                    "company": {"display_name": "X"},
                    "location": {"display_name": "Leeds"},
                }
            ]
        }

    postings = search_jobs(
        "ml engineer", "leeds", n=1, app_id="id", app_key="key", transport=fake_transport
    )
    assert postings[0].title == "ML Engineer"
