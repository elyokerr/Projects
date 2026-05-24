import pandas as pd
import pytest

from src.abtest.data import ExperimentData


def _df():
    return pd.DataFrame({
        "unit_id": range(6),
        "variant": ["control", "control", "control",
                    "treatment", "treatment", "treatment"],
        "converted": [0, 1, 0, 1, 1, 0],
    })


def test_valid_construction():
    ed = ExperimentData(_df(), variant_col="variant", metric_cols=["converted"])
    assert ed.n_control == 3 and ed.n_treatment == 3


def test_rejects_unknown_variant_label():
    df = _df()
    df.loc[0, "variant"] = "foo"
    with pytest.raises(ValueError, match="variant labels"):
        ExperimentData(df, variant_col="variant", metric_cols=["converted"],
                       control_label="control", treatment_label="treatment")


def test_rejects_single_variant():
    df = _df()
    df["variant"] = "control"
    with pytest.raises(ValueError, match="two variants"):
        ExperimentData(df, variant_col="variant", metric_cols=["converted"])


def test_rejects_all_nan_metric():
    df = _df()
    df["converted"] = float("nan")
    with pytest.raises(ValueError, match="all-missing"):
        ExperimentData(df, variant_col="variant", metric_cols=["converted"])
