"""Evidence.dev static-site build task.

Evidence init lands in Phase 6. Until then this is a no-op: if
``evidence_dir/package.json`` is missing we log a warning and return ``None``.

The plain function ``build_evidence`` is what tests call directly;
``build_evidence_task`` is the Prefect-wrapped sibling.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from prefect import task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_EVIDENCE_DIR = PROJECT_ROOT / "evidence"
_DEFAULT_WAREHOUSE_PATH = PROJECT_ROOT / "data" / "duckdb" / "housing.duckdb"


def _log(message: str) -> None:
    try:
        from prefect import get_run_logger

        get_run_logger().warning(message)
    except Exception:
        print(message, flush=True)


def build_evidence(
    evidence_dir: Path | None = None,
    warehouse_path: Path | None = None,
) -> Path | None:
    """Build the Evidence static site.

    Evidence init in Phase 6. Until then this is a no-op when there is no
    ``package.json`` in ``evidence_dir``.

    Returns the build output dir on success, or ``None`` if Evidence is not
    yet initialised.
    """
    effective_evidence_dir = Path(evidence_dir) if evidence_dir else _DEFAULT_EVIDENCE_DIR
    effective_warehouse = (
        Path(warehouse_path) if warehouse_path else _DEFAULT_WAREHOUSE_PATH
    )

    pkg = effective_evidence_dir / "package.json"
    if not pkg.exists():
        _log(
            f"[build_evidence] skip: no package.json at {pkg} "
            f"(Evidence init in Phase 6)"
        )
        return None

    env = os.environ.copy()
    env["DUCKDB_PATH"] = str(effective_warehouse)

    # Resolve npm explicitly. On Windows, `npm` is `npm.cmd` (a batch file);
    # `subprocess.run(["npm", ...])` without shell=True fails because
    # CreateProcess only finds .exe, not .cmd. shutil.which finds both.
    npm = shutil.which("npm") or "npm"
    completed = subprocess.run(
        [npm, "run", "build", "--prefix", str(effective_evidence_dir)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        shell=False,
    )
    if completed.stdout:
        print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, flush=True)

    if completed.returncode != 0:
        stderr_tail = "\n".join((completed.stderr or "").splitlines()[-50:])
        raise RuntimeError(
            f"npm run build failed with exit {completed.returncode}\n{stderr_tail}"
        )

    build_dir = effective_evidence_dir / "build"
    index = build_dir / "index.html"
    if not index.exists():
        raise RuntimeError(f"Evidence build produced no index.html at {index}")
    return build_dir


build_evidence_task = task(build_evidence)
