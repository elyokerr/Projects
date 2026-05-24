from unittest.mock import MagicMock, patch

import pytest

from flows.tasks.run_data_quality import DataQualityFailed, run_landing_checkpoint


def test_failed_suite_raises():
    fake_result = MagicMock(success=False)
    fake_result.list_validation_results.return_value = [
        {"success": False, "meta": {"expectation_suite_name": "ppd_landing"}}
    ]
    with patch("flows.tasks.run_data_quality.gx.get_context") as ctx:
        ctx.return_value.run_checkpoint.return_value = fake_result
        with pytest.raises(DataQualityFailed):
            run_landing_checkpoint("landing_all")


def test_passed_suite_returns_summary():
    fake_result = MagicMock(success=True)
    fake_result.list_validation_results.return_value = [
        {"success": True, "meta": {"expectation_suite_name": "ppd_landing"}}
    ]
    with patch("flows.tasks.run_data_quality.gx.get_context") as ctx:
        ctx.return_value.run_checkpoint.return_value = fake_result
        out = run_landing_checkpoint("landing_all")
        assert out["success"] is True
