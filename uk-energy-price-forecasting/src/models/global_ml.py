"""Global LightGBM quantile model with the shared Phase-4/5 interface.

Shared model interface (implemented here):

    fit(self, bundle, train_end=None) -> self
        Fit on data up to ``train_end`` (a pd.Timestamp); if None, use the
        whole bundle.

    predict_quantiles(self, bundle, origin, horizon, quantiles=(0.1, 0.5, 0.9))
        -> dict[float, np.ndarray]
        Forecast the ``horizon`` steps AFTER ``origin``, returning
        ``{q: array_of_length_horizon}``.  Only uses information up to
        ``origin`` from the target/past covariates (leakage-safe).

Darts API notes (v0.44)
-----------------------
* ``LightGBMModel(likelihood='quantile', quantiles=[...])`` is trained with
  one LightGBM regressor per quantile via the pinball loss.
* ``model.predict(..., predict_likelihood_parameters=True)`` returns a
  deterministic ``TimeSeries`` with one component per quantile, named
  ``<target>_q0.100``, ``<target>_q0.500``, etc.  Values are extracted
  directly as a ``(horizon, n_quantiles)`` array.
* ``output_chunk_length=horizon`` prevents auto-regression and avoids the
  Darts requirement that future covariates extend ``horizon`` steps beyond
  the last target step.  This means ``lags_future_covariates`` only needs
  to look back into already-observed calendar features.
* Quantile crossing: LightGBM quantile regressors trained independently can
  produce crossing quantiles.  ``predict_quantiles`` sorts each timestep's
  quantile array so that the returned values are weakly monotone increasing.
  This is documented as a known artefact of independent quantile regression.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from darts.models import LightGBMModel


class GlobalLGBM:
    """Global LightGBM model producing calibrated quantile forecasts.

    Parameters
    ----------
    quantiles : tuple of float
        Quantile levels to train and predict.  Defaults to ``(0.1, 0.5, 0.9)``.
    lags : int
        Number of lagged target values to use as features.  Defaults to 48.
    lags_past_covariates : int
        Number of lagged past-covariate steps.  Defaults to 48.
    lags_future_covariates : tuple[int, int] or None
        ``(past_lags, future_lags)`` for future (calendar) covariates.
        Defaults to ``None`` (disabled).  Enabling this requires that
        ``bundle.future_covariates`` extends at least ``horizon`` steps past
        ``origin``; in practice the full calendar series covers the window.
        Set to ``(48, 0)`` to use 48 look-back calendar lags, no horizon lags.
    **kwargs
        Passed through to :class:`darts.models.LightGBMModel` (e.g.
        ``n_estimators``, ``num_leaves``).
    """

    # Sparse seasonal lags: recent steps + 1-day (48) + 2-day (96) history.
    # Using explicit lag LISTS instead of 48 contiguous lags keeps the feature
    # matrix small, which matters because output_chunk_length=48 with
    # multi_models=True trains one LightGBM per (horizon-step x quantile) =
    # 48 x 3 = 144 sub-models. Dense 48-lag features made this ~10 min/fit.
    _DEFAULT_LAGS = [-1, -2, -3, -48, -96]
    _DEFAULT_PAST_COV_LAGS = [-1, -2, -3, -48]

    def __init__(
        self,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        lags: int | list[int] | None = None,
        lags_past_covariates: int | list[int] | None = None,
        lags_future_covariates: tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        if lags is None:
            lags = list(self._DEFAULT_LAGS)
        if lags_past_covariates is None:
            lags_past_covariates = list(self._DEFAULT_PAST_COV_LAGS)
        self.quantiles = tuple(sorted(quantiles))
        self.lags = lags
        self.lags_past_covariates = lags_past_covariates
        self.lags_future_covariates = lags_future_covariates
        self.kwargs = kwargs
        self.model: LightGBMModel | None = None
        self._fit_horizon: int | None = None

    def fit(self, bundle, train_end: pd.Timestamp | None = None) -> "GlobalLGBM":
        """Fit the LightGBM quantile model on ``bundle.target[:train_end]``.

        Parameters
        ----------
        bundle : PanelBundle
            Training data.
        train_end : pd.Timestamp or None
            If provided, fit only on the series up to (and including) this
            timestamp.

        Returns
        -------
        GlobalLGBM
            self, for method chaining.
        """
        target = bundle.target if train_end is None else bundle.target[:train_end]
        past_cov = bundle.past_covariates
        future_cov = bundle.future_covariates

        # Modest defaults keep the 144 sub-models cheap; override via kwargs.
        kwargs = {"n_estimators": 50, "num_leaves": 15, "verbose": -1}
        kwargs.update(self.kwargs)

        # output_chunk_length must equal n at predict time when using
        # predict_likelihood_parameters=True (Darts 0.44 constraint).
        # We default to 48 (one full day of SPs) which covers the standard
        # horizon.  The caller can override via kwargs if needed.
        output_chunk_length = kwargs.pop("output_chunk_length", 48)

        self.model = LightGBMModel(
            lags=self.lags,
            lags_past_covariates=self.lags_past_covariates,
            lags_future_covariates=self.lags_future_covariates,
            likelihood="quantile",
            quantiles=list(self.quantiles),
            output_chunk_length=output_chunk_length,
            **kwargs,
        )
        self._output_chunk_length = output_chunk_length

        fit_kwargs: dict = {"past_covariates": past_cov}
        if self.lags_future_covariates is not None:
            fit_kwargs["future_covariates"] = future_cov

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(target, **fit_kwargs)

        return self

    def predict_quantiles(
        self,
        bundle,
        origin: pd.Timestamp,
        horizon: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> dict[float, np.ndarray]:
        """Forecast ``horizon`` steps after ``origin`` for each quantile.

        Uses only information up to ``origin`` from target and past covariates
        (leakage-safe). Future covariates are calendar-only and may be passed
        in full.

        Quantile monotonicity: independently trained quantile regressors can
        produce crossing quantiles.  Each timestep's quantile vector is sorted
        in ascending order before returning so that q0.1 <= q0.5 <= q0.9 is
        guaranteed.

        Parameters
        ----------
        bundle : PanelBundle
        origin : pd.Timestamp
            Last observed timestep.
        horizon : int
            Forecast horizon in timesteps.
        quantiles : tuple of float
            Subset of the fitted quantiles to return.

        Returns
        -------
        dict[float, np.ndarray]
            ``{q: array_of_length_horizon}`` for each q in ``quantiles``.

        Raises
        ------
        ValueError
            If a requested quantile was not fitted.
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_quantiles().")

        missing = set(quantiles) - set(self.quantiles)
        if missing:
            raise ValueError(
                f"Quantiles {missing} were not trained. "
                f"Fitted quantiles: {self.quantiles}"
            )

        predict_kwargs: dict = {
            "series": bundle.target[:origin],
            "past_covariates": bundle.past_covariates,
            "predict_likelihood_parameters": True,
            "show_warnings": False,
        }
        if self.lags_future_covariates is not None:
            predict_kwargs["future_covariates"] = bundle.future_covariates

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.model.predict(n=horizon, **predict_kwargs)

        # pred is a deterministic TimeSeries with one component per quantile:
        # e.g. ['price_q0.100', 'price_q0.500', 'price_q0.900']
        # shape: (horizon, n_fitted_quantiles)
        values = pred.values()  # shape (horizon, n_fitted_quantiles)

        # Sort each timestep's quantile values to guarantee monotonicity.
        # (Crossing quantiles are a known side-effect of per-quantile pinball
        # regression; sorting per-timestep is a standard post-processing fix.)
        values = np.sort(values, axis=1)  # axis=1 = quantile dimension

        # Map sorted columns back to quantile levels.
        sorted_fitted_quantiles = sorted(self.quantiles)
        q_to_array = {
            q: values[:, i] for i, q in enumerate(sorted_fitted_quantiles)
        }

        return {q: q_to_array[q].copy() for q in quantiles}
