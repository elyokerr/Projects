"""Rolling-origin backtest harness (Phase 5).

Public API
----------
generate_origins(time_index, start, step="1D", end=None) -> list[pd.Timestamp]
    Enumerate forecast origins at successive midnight boundaries.

BacktestResult
    Dataclass holding actuals/forecasts for a completed backtest run.
    Exposes a `.metrics()` method.

run_backtest(model, bundle, origins, horizon=48, quantiles=(0.1,0.5,0.9),
             refit=False, model_name=None) -> BacktestResult
    Execute a rolling-origin evaluation loop.

build_ablation_table(results, baseline="seasonal_naive") -> pd.DataFrame
    Summarise multiple BacktestResult objects into a skill-scored DataFrame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics.coverage import interval_coverage
from src.metrics.crps import crps_from_quantiles
from src.metrics.pinball import mean_pinball
from src.metrics.point_metrics import mae, rmse, skill_vs_baseline, smape


# ---------------------------------------------------------------------------
# Module 1 — generate_origins
# ---------------------------------------------------------------------------


def generate_origins(
    time_index: pd.DatetimeIndex,
    start: pd.Timestamp,
    step: str = "1D",
    end: pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """Return forecast origins at successive midnight boundaries.

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        The full time index of the panel (tz-naive or tz-aware).
    start : pd.Timestamp
        Earliest desired origin.  The first returned origin is the first
        midnight in *time_index* that is >= start.
    step : str
        Pandas frequency string for the spacing between consecutive origins.
        Default ``"1D"`` (daily).
    end : pd.Timestamp or None
        Latest desired origin (inclusive).  If None, uses the last timestamp
        in *time_index*.

    Returns
    -------
    list[pd.Timestamp]
        Timestamps that are midnight (00:00) entries in *time_index*, spaced
        by *step*, within [start, end].
    """
    if end is None:
        end = time_index[-1]

    # All midnight timestamps present in the index
    midnights = time_index[(time_index.hour == 0) & (time_index.minute == 0)]

    # Keep those within [start, end]
    midnights = midnights[(midnights >= start) & (midnights <= end)]

    if len(midnights) == 0:
        return []

    # Build the desired grid starting from the first valid midnight
    grid = pd.date_range(start=midnights[0], end=end, freq=step)

    # Keep only grid entries that are actually in the midnight set
    midnight_set = set(midnights)
    return [ts for ts in grid if ts in midnight_set]


# ---------------------------------------------------------------------------
# Module 2 — BacktestResult + run_backtest
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Results from a rolling-origin backtest evaluation.

    Attributes
    ----------
    model_name : str
        Identifier for the model.
    origins : list
        Forecast origin timestamps used in the evaluation.
    quantiles : tuple
        Quantile levels that were requested.
    actuals : np.ndarray
        Shape (n_origins, horizon). Observed values for each origin window.
    forecasts : dict[float, np.ndarray]
        Mapping quantile -> array of shape (n_origins, horizon).
    horizon : int
        Number of forecast steps per origin.
    """

    model_name: str
    origins: list
    quantiles: tuple
    actuals: np.ndarray
    forecasts: dict
    horizon: int

    def metrics(self) -> dict[str, float]:
        """Compute evaluation metrics by flattening actuals and forecasts.

        Returns
        -------
        dict[str, float]
            Keys: pinball, coverage_80, coverage_nominal, crps, mae, rmse, smape.
        """
        actuals_flat = self.actuals.flatten()
        fc_flat = {q: arr.flatten() for q, arr in self.forecasts.items()}

        # Pinball (mean over all quantiles)
        pinball_val = mean_pinball(actuals_flat, fc_flat)

        # Coverage: use min/max quantile as the nominal interval bounds
        q_low = min(self.quantiles)
        q_high = max(self.quantiles)
        cov = interval_coverage(actuals_flat, fc_flat[q_low], fc_flat[q_high])
        cov_nominal = q_high - q_low

        # CRPS
        crps_val = crps_from_quantiles(actuals_flat, fc_flat)

        # Point metrics on median quantile (closest to 0.5)
        q_median = min(self.quantiles, key=lambda q: abs(q - 0.5))
        median_fc = fc_flat[q_median]
        mae_val = mae(actuals_flat, median_fc)
        rmse_val = rmse(actuals_flat, median_fc)
        smape_val = smape(actuals_flat, median_fc)

        return {
            "pinball": pinball_val,
            "coverage_80": cov,
            "coverage_nominal": cov_nominal,
            "crps": crps_val,
            "mae": mae_val,
            "rmse": rmse_val,
            "smape": smape_val,
        }


def run_backtest(
    model,
    bundle,
    origins: list,
    horizon: int = 48,
    quantiles: tuple = (0.1, 0.5, 0.9),
    refit: bool = False,
    model_name: str | None = None,
) -> BacktestResult:
    """Execute a rolling-origin backtest evaluation loop.

    Parameters
    ----------
    model
        A model with ``.fit(bundle, train_end)`` and
        ``.predict_quantiles(bundle, origin, horizon, quantiles)`` methods.
    bundle : PanelBundle
        The full data bundle.
    origins : list[pd.Timestamp]
        Forecast origins to evaluate.
    horizon : int
        Steps to forecast after each origin.
    quantiles : tuple of float
        Quantile levels to evaluate.
    refit : bool
        If False (default), fit once before the loop using ``origins[0]`` as
        train_end.  If True, re-fit inside the loop at each origin.
    model_name : str or None
        Label for this model.  Defaults to ``type(model).__name__``.

    Returns
    -------
    BacktestResult
    """
    if model_name is None:
        model_name = type(model).__name__

    target_index = bundle.target.time_index
    target_values = bundle.target.values().flatten()

    # Fit once if refit is False
    if not refit and len(origins) > 0:
        model.fit(bundle, train_end=origins[0])

    actuals_list: list[np.ndarray] = []
    forecasts_list: dict[float, list[np.ndarray]] = {q: [] for q in quantiles}
    valid_origins: list = []

    for origin in origins:
        # Locate origin position in the target index
        try:
            pos = target_index.get_loc(origin)
        except KeyError:
            # Origin not in index; skip
            continue

        # Need at least `horizon` steps after the origin
        if pos + horizon >= len(target_values):
            continue

        # Re-fit if requested (local models)
        if refit:
            model.fit(bundle, train_end=origin)

        preds = model.predict_quantiles(bundle, origin, horizon, quantiles)

        actual_window = target_values[pos + 1 : pos + 1 + horizon]
        actuals_list.append(actual_window)
        for q in quantiles:
            forecasts_list[q].append(preds[q])
        valid_origins.append(origin)

    # Stack into 2D arrays (n_origins x horizon)
    actuals_arr = np.stack(actuals_list, axis=0) if actuals_list else np.empty((0, horizon))
    forecasts_arr = {
        q: np.stack(forecasts_list[q], axis=0) if forecasts_list[q] else np.empty((0, horizon))
        for q in quantiles
    }

    return BacktestResult(
        model_name=model_name,
        origins=valid_origins,
        quantiles=tuple(quantiles),
        actuals=actuals_arr,
        forecasts=forecasts_arr,
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# Module 3 — build_ablation_table
# ---------------------------------------------------------------------------


def build_ablation_table(
    results: dict[str, "BacktestResult"],
    baseline: str = "seasonal_naive",
) -> pd.DataFrame:
    """Summarise multiple BacktestResult objects into a skill-scored DataFrame.

    Parameters
    ----------
    results : dict[str, BacktestResult]
        Mapping of model name -> BacktestResult.
    baseline : str
        Key of the baseline model in *results*.  Skill scores are computed
        relative to this model; the baseline row itself has skill == 0.

    Returns
    -------
    pd.DataFrame
        Index = model names.
        Columns: pinball, coverage_80, crps, mae, skill_pinball, skill_mae.
    """
    rows: dict[str, dict[str, float]] = {}
    for name, res in results.items():
        m = res.metrics()
        rows[name] = m

    # Extract baseline metrics for skill computation
    baseline_metrics = rows.get(baseline, {})
    baseline_pinball = baseline_metrics.get("pinball", 0.0)
    baseline_mae = baseline_metrics.get("mae", 0.0)

    records = []
    index = []
    for name, m in rows.items():
        skill_p = skill_vs_baseline(m["pinball"], baseline_pinball) if name != baseline else 0.0
        skill_m = skill_vs_baseline(m["mae"], baseline_mae) if name != baseline else 0.0
        records.append(
            {
                "pinball": m["pinball"],
                "coverage_80": m["coverage_80"],
                "crps": m["crps"],
                "mae": m["mae"],
                "skill_pinball": skill_p,
                "skill_mae": skill_m,
            }
        )
        index.append(name)

    return pd.DataFrame(records, index=index)
