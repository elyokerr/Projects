"""Unit tests for flows.tasks.run_dbt."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from flows.tasks.run_dbt import run_dbt


def _fake_completed(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_run_dbt_invokes_cli_with_expected_args() -> None:
    with patch("flows.tasks.run_dbt.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(stdout="Configuration parses cleanly")
        run_dbt("parse", target="duckdb")
    args = mock_run.call_args[0][0]
    assert "dbt.cli.main" in " ".join(args)
    assert "parse" in args
    assert "--target" in args
    assert "duckdb" in args


def test_run_dbt_raises_on_nonzero_exit() -> None:
    with patch("flows.tasks.run_dbt.subprocess.run") as mock_run:
        mock_run.return_value = _fake_completed(returncode=2, stderr="boom")
        with pytest.raises(RuntimeError, match="dbt parse failed with exit 2"):
            run_dbt("parse", target="duckdb")
