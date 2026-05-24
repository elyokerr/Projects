"""Unit tests for flows.tasks.build_evidence."""

from __future__ import annotations

from pathlib import Path

from flows.tasks.build_evidence import build_evidence


def test_build_evidence_soft_skip_when_not_initialised(tmp_path: Path) -> None:
    # tmp_path has no package.json -> Evidence is not initialised yet.
    result = build_evidence(evidence_dir=tmp_path, warehouse_path=tmp_path / "x.duckdb")
    assert result is None
