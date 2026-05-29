from src.agent.state import Gap
from src.skills.gap import aggregate_demand, compute_gap


def test_aggregate_counts_across_postings():
    per_posting = [["python", "sql"], ["python", "docker"], ["python"]]
    demand = aggregate_demand(per_posting)
    assert demand["python"] == 3
    assert demand["sql"] == 1


def test_compute_gap_ranks_by_demand_and_excludes_known():
    demand = {"python": 3, "sql": 2, "docker": 1}
    cv_skills = ["python"]
    gaps = compute_gap(cv_skills, demand, n_postings=3)
    assert [g.skill for g in gaps] == ["sql", "docker"]  # python excluded, sorted by count
    assert gaps[0] == Gap(skill="sql", demand_count=2, demand_fraction=2 / 3)
