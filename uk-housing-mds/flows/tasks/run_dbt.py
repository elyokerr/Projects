"""dbt CLI runner task.

Shells out to ``python -m dbt.cli.main`` so we use the dbt installed in the
active virtualenv without depending on the ``dbt`` executable being on PATH.

The plain function ``run_dbt`` is what tests call directly; ``run_dbt_task``
is the Prefect-wrapped sibling for use inside flows.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from prefect import task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROJECT_DIR = PROJECT_ROOT / "dbt_project"


def _log(message: str) -> None:
    try:
        from prefect import get_run_logger

        get_run_logger().info(message)
    except Exception:
        print(message, flush=True)


def run_dbt(
    command: str,
    target: str = "duckdb",
    project_dir: Path | None = None,
) -> None:
    """Run a dbt command against the given target.

    Raises ``RuntimeError`` on non-zero exit, including the tail of stderr.
    """
    effective_project_dir = Path(project_dir) if project_dir else _DEFAULT_PROJECT_DIR

    cmd = [
        sys.executable,
        "-m",
        "dbt.cli.main",
        command,
        "--target",
        target,
        "--profiles-dir",
        str(effective_project_dir),
        "--project-dir",
        str(effective_project_dir),
    ]

    _log(f"[run_dbt] invoking: {' '.join(cmd)}")
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if completed.stdout:
        _log(completed.stdout)
    if completed.stderr:
        _log(completed.stderr)

    if completed.returncode != 0:
        stderr_tail = "\n".join((completed.stderr or "").splitlines()[-50:])
        raise RuntimeError(
            f"dbt {command} failed with exit {completed.returncode}\n{stderr_tail}"
        )


run_dbt_task = task(run_dbt, retries=1)
