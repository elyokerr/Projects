from __future__ import annotations

import pandas as pd


class ExperimentData:
    def __init__(self, df: pd.DataFrame, *, variant_col: str = "variant",
                 metric_cols: list[str], control_label: str = "control",
                 treatment_label: str = "treatment",
                 covariate_col: str | None = None,
                 timestamp_col: str | None = None):
        self.df = df.reset_index(drop=True)
        self.variant_col = variant_col
        self.metric_cols = list(metric_cols)
        self.control_label = control_label
        self.treatment_label = treatment_label
        self.covariate_col = covariate_col
        self.timestamp_col = timestamp_col
        self._validate()

    def _validate(self) -> None:
        for c in [self.variant_col, *self.metric_cols]:
            if c not in self.df.columns:
                raise ValueError(f"missing column: {c}")
        labels = set(self.df[self.variant_col].unique())
        allowed = {self.control_label, self.treatment_label}
        if not labels.issubset(allowed):
            raise ValueError(f"unexpected variant labels: {labels - allowed}")
        if labels != allowed:
            raise ValueError("data must contain exactly two variants")
        for m in self.metric_cols:
            if self.df[m].isna().all():
                raise ValueError(f"metric column '{m}' is all-missing")
        if self.covariate_col and self.covariate_col not in self.df.columns:
            raise ValueError(f"covariate column missing: {self.covariate_col}")

    def _arm(self, label: str) -> pd.DataFrame:
        return self.df[self.df[self.variant_col] == label]

    @property
    def control(self) -> pd.DataFrame:
        return self._arm(self.control_label)

    @property
    def treatment(self) -> pd.DataFrame:
        return self._arm(self.treatment_label)

    @property
    def n_control(self) -> int:
        return len(self.control)

    @property
    def n_treatment(self) -> int:
        return len(self.treatment)

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "ExperimentData":
        return cls(pd.read_csv(path), **kwargs)
