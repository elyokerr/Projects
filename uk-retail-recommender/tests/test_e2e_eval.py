"""End-to-end ladder evaluation test.

Run with RUN_SLOW=1 to execute the full ladder; skipped otherwise so the
default suite stays fast.
"""
from __future__ import annotations

import os

import pytest

from src.eval.ladder import evaluate_ladder


@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="slow")
def test_e2e_ladder(sample_df):
    results = evaluate_ladder(sample_df, k=10)

    print("\n=== Ladder evaluation results ===")
    for rung in ("popularity", "item2item", "als", "two_stage", "two_tower"):
        if rung in results:
            m = results[rung]
            print(
                f"  {rung:<12}  recall@10={m['recall@k']:.4f}  "
                f"ndcg@10={m['ndcg@k']:.4f}  map@10={m['map@k']:.4f}  "
                f"coverage={m['coverage']:.4f}"
            )
    print(f"  retrieval_ceiling = {results['retrieval_ceiling']:.4f}")
    print(f"  k={results['k']}  n_users={results['n_users']}")

    metric_keys = ("recall@k", "ndcg@k", "map@k", "coverage")
    for rung in ("popularity", "item2item", "als", "two_stage"):
        assert rung in results, f"Missing rung: {rung}"
        for key in metric_keys:
            val = results[rung][key]
            assert 0.0 <= val <= 1.0, f"{rung} {key}={val} not in [0,1]"

    # Sanity floor: ALS must produce some valid (non-negative) recall
    assert results["als"]["recall@k"] >= 0.0

    # The two-stage pipeline should at least match 90% of popularity recall.
    # On small data the LambdaMART ranker can add variance, but the cluster
    # structure in the fixture gives collaborative models a clear advantage,
    # so this floor is intentionally loose.  The real signal is in the printed
    # numbers above.
    pop_recall = results["popularity"]["recall@k"]
    two_stage_recall = results["two_stage"]["recall@k"]
    assert two_stage_recall >= 0.9 * pop_recall, (
        f"two_stage recall@10={two_stage_recall:.4f} < 0.9 * popularity "
        f"recall@10={pop_recall:.4f}.  Check cluster structure in _synth()."
    )

    # Retrieval ceiling must be a valid probability
    assert 0.0 <= results["retrieval_ceiling"] <= 1.0
