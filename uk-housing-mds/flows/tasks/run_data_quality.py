"""Great Expectations checkpoint runner.

Halts the Prefect flow when any suite in the checkpoint fails.

The plain function ``run_landing_checkpoint`` is what tests call directly
(avoids the Prefect ``@task`` engine spin-up that has caused pytest timeouts
in earlier phases). ``run_landing_checkpoint_task`` is the Prefect-wrapped
sibling for use inside flows.
"""

from __future__ import annotations

from pathlib import Path

from prefect import task

import great_expectations as gx

_GE_DIR = Path(__file__).resolve().parents[2] / "great_expectations"


class DataQualityFailed(RuntimeError):
    """Raised when a GE checkpoint reports at least one failing suite."""


def run_landing_checkpoint(checkpoint_name: str) -> dict:
    """Run a GE checkpoint; raise on failure, otherwise return a summary."""
    ctx = gx.get_context(context_root_dir=str(_GE_DIR))
    result = ctx.run_checkpoint(checkpoint_name=checkpoint_name)
    summary = {
        "success": result.success,
        "results": result.list_validation_results(),
    }
    if not result.success:
        failed_suites = [
            vr["meta"].get("expectation_suite_name", "<unknown>")
            for vr in summary["results"]
            if not vr.get("success", False)
        ]
        raise DataQualityFailed(
            f"Checkpoint {checkpoint_name!r} failed; suites: {failed_suites}"
        )
    return summary


run_landing_checkpoint_task = task(
    run_landing_checkpoint,
    name="run_landing_checkpoint",
)
