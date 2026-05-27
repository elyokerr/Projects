"""Baseline forecasting models with the shared Phase-4/5 interface.

Shared model interface (both classes below implement this):

    fit(self, bundle, train_end=None) -> self
        Fit on data up to ``train_end`` (a pd.Timestamp); if None, use the
        whole bundle. Stateless baselines may no-op.

    predict_quantiles(self, bundle, origin, horizon, quantiles=(0.1, 0.5, 0.9))
        -> dict[float, np.ndarray]
        Forecast the ``horizon`` steps AFTER ``origin``, returning
        ``{q: array_of_length_horizon}``.  MUST only use information up to
        ``origin`` from the target/past covariates (leakage-safe); future
        covariates are calendar-only so the full horizon window is always safe.

Classes
-------
SeasonalNaive
    Repeats the same season-ago values as the point forecast.  All quantiles
    equal the point forecast (zero interval width).
ThetaBaseline
    Wraps Darts' Theta model.  Deterministic point forecast replicated across
    quantiles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from darts.models import Theta


class SeasonalNaive:
    """Seasonal-naive baseline: forecast[k] = target[origin_pos - season + k].

    Parameters
    ----------
    season : int
        Seasonal period in timesteps.  Defaults to 48 (one calendar day of
        half-hourly settlement periods).
    """

    def __init__(self, season: int = 48) -> None:
        self.season = season

    def fit(self, bundle, train_end: pd.Timestamp | None = None) -> "SeasonalNaive":
        """No-op — SeasonalNaive requires no fitting."""
        return self

    def predict_quantiles(
        self,
        bundle,
        origin: pd.Timestamp,
        horizon: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> dict[float, np.ndarray]:
        """Return a seasonal-naive point forecast replicated across all quantiles.

        All quantile arrays are identical (zero interval width).

        Parameters
        ----------
        bundle : PanelBundle
            Only ``bundle.target`` is used.
        origin : pd.Timestamp
            Last observed timestep; forecast starts at origin + 1 step.
        horizon : int
            Number of steps to forecast.
        quantiles : tuple of float
            Quantile levels to return.

        Returns
        -------
        dict[float, np.ndarray]
            ``{q: array of length horizon}`` for each q in quantiles.
        """
        target_values = bundle.target.values().flatten()
        target_index = bundle.target.time_index

        # Find the integer position of `origin` in the target time index.
        pos = target_index.get_loc(origin)

        # forecast[k-1] = target_values[pos - season + k], k = 1 .. horizon
        forecast = np.array(
            [target_values[pos - self.season + k] for k in range(1, horizon + 1)]
        )

        return {q: forecast.copy() for q in quantiles}


class ThetaBaseline:
    """Theta method baseline wrapping Darts' :class:`darts.models.Theta`.

    Deterministic point forecast; all quantiles receive the same array.

    Parameters
    ----------
    theta : float
        Theta parameter passed to :class:`darts.models.Theta`. Default 2.
    """

    def __init__(self, theta: float = 2) -> None:
        self.theta = theta
        self._model: Theta | None = None

    def fit(self, bundle, train_end: pd.Timestamp | None = None) -> "ThetaBaseline":
        """Fit Theta on ``bundle.target`` up to ``train_end``.

        Parameters
        ----------
        bundle : PanelBundle
        train_end : pd.Timestamp or None
            If provided, fit only on the series up to this timestamp.
        """
        target = bundle.target if train_end is None else bundle.target[:train_end]
        self._model = Theta(theta=self.theta)
        self._model.fit(target)
        return self

    def predict_quantiles(
        self,
        bundle,
        origin: pd.Timestamp,
        horizon: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> dict[float, np.ndarray]:
        """Return Theta point forecast replicated across quantiles.

        If the model was not fit before calling this method, it is fit on
        ``bundle.target[:origin]`` on the fly.

        Parameters
        ----------
        bundle : PanelBundle
        origin : pd.Timestamp
            Last observed timestep.
        horizon : int
            Number of steps to forecast.
        quantiles : tuple of float

        Returns
        -------
        dict[float, np.ndarray]
        """
        if self._model is None:
            self.fit(bundle, train_end=origin)

        pred = self._model.predict(n=horizon)
        forecast = pred.values().flatten()

        return {q: forecast.copy() for q in quantiles}
