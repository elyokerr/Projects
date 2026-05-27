"""Global deep-learning quantile models with the shared Phase-4/5 interface.

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
* ``TiDEModel`` and ``TFTModel`` accept ``likelihood=QuantileRegression(...)``
  which makes them output stochastic (multi-sample) forecasts.
* ``model.predict(..., num_samples=200)`` returns a stochastic ``TimeSeries``
  with ``n_samples=200``.  Per-timestep quantiles are extracted via
  ``pred.quantile(q).values().flatten()`` which returns a 1-D array of length
  ``horizon``.
* Quantile monotonicity is enforced by sorting each timestep's quantile vector
  after extraction.  With ``num_samples=200`` the empirical quantiles are
  usually already monotone, but the sort is a cheap safety net.
* TFT requires ``future_covariates`` at both fit and predict time.
  TiDE supports both past and future covariates; both are passed.
* ``pl_trainer_kwargs={"accelerator": "cpu"}`` forces CPU training — no GPU
  required.  Set ``"accelerator": "gpu"`` in the constructor on Colab T4.
* ``enable_progress_bar=False`` suppresses the Lightning per-batch bar.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from darts.models import TFTModel, TiDEModel
from darts.utils.likelihood_models import QuantileRegression

# Default tiny config: fast on CPU for smoke tests and local dev.
_DEFAULT_INPUT_CHUNK = 48
_DEFAULT_OUTPUT_CHUNK = 48
_DEFAULT_EPOCHS = 3
_DEFAULT_BATCH = 32
_DEFAULT_QUANTILES = [0.1, 0.5, 0.9]

_TIDE_DEFAULTS = {
    "hidden_size": 16,
    "num_encoder_layers": 1,
    "num_decoder_layers": 1,
}
_TFT_DEFAULTS = {
    "hidden_size": 8,
    "lstm_layers": 1,
    "num_attention_heads": 1,
}


class GlobalTiDE:
    """Global TiDE (Time-series Dense Encoder) model producing quantile forecasts.

    Wraps :class:`darts.models.TiDEModel` with the shared fit/predict_quantiles
    interface.  Default hyperparameters are intentionally small for fast CPU
    training; override via constructor kwargs.

    Parameters
    ----------
    quantiles : list of float
        Quantile levels to train and predict.
    input_chunk_length : int
        Look-back window in timesteps.
    output_chunk_length : int
        Forecast horizon used during training.
    n_epochs : int
        Training epochs.
    batch_size : int
    random_state : int
    pl_trainer_kwargs : dict or None
        Passed to PyTorch Lightning Trainer.  Defaults to CPU, no progress bar.
    **kwargs
        Additional keyword arguments forwarded to :class:`darts.models.TiDEModel`.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        input_chunk_length: int = _DEFAULT_INPUT_CHUNK,
        output_chunk_length: int = _DEFAULT_OUTPUT_CHUNK,
        n_epochs: int = _DEFAULT_EPOCHS,
        batch_size: int = _DEFAULT_BATCH,
        random_state: int = 42,
        pl_trainer_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        self.quantiles = list(sorted(quantiles or _DEFAULT_QUANTILES))
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.pl_trainer_kwargs = pl_trainer_kwargs or {
            "enable_progress_bar": False,
            "accelerator": "cpu",
        }
        self.extra_kwargs = kwargs
        self.model: TiDEModel | None = None

    def fit(self, bundle, train_end: pd.Timestamp | None = None) -> "GlobalTiDE":
        """Fit TiDE on ``bundle.target[:train_end]``.

        Parameters
        ----------
        bundle : PanelBundle
        train_end : pd.Timestamp or None

        Returns
        -------
        GlobalTiDE
            self, for method chaining.
        """
        target = bundle.target if train_end is None else bundle.target[:train_end]

        model_kwargs = {**_TIDE_DEFAULTS, **self.extra_kwargs}
        self.model = TiDEModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            random_state=self.random_state,
            likelihood=QuantileRegression(quantiles=self.quantiles),
            pl_trainer_kwargs=self.pl_trainer_kwargs,
            **model_kwargs,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                target,
                past_covariates=bundle.past_covariates,
                future_covariates=bundle.future_covariates,
            )

        return self

    def predict_quantiles(
        self,
        bundle,
        origin: pd.Timestamp,
        horizon: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> dict[float, np.ndarray]:
        """Forecast ``horizon`` steps after ``origin`` for each quantile.

        Leakage-safe: only target/past-covariate history up to ``origin`` is
        used.  Future covariates are calendar-only and are passed in full.

        Parameters
        ----------
        bundle : PanelBundle
        origin : pd.Timestamp
            Last observed timestep (inclusive).
        horizon : int
            Forecast horizon in timesteps.
        quantiles : tuple of float
            Subset of the fitted quantiles to return.

        Returns
        -------
        dict[float, np.ndarray]
            ``{q: array_of_length_horizon}`` for each q.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If a requested quantile was not fitted.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_quantiles().")

        missing = set(quantiles) - set(self.quantiles)
        if missing:
            raise ValueError(
                f"Quantiles {missing} were not trained. "
                f"Fitted quantiles: {self.quantiles}"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.model.predict(
                n=horizon,
                series=bundle.target[:origin],
                past_covariates=bundle.past_covariates,
                future_covariates=bundle.future_covariates,
                num_samples=200,
            )

        result: dict[float, np.ndarray] = {}
        for q in quantiles:
            arr = pred.quantile(q).values().flatten()
            result[q] = arr

        # Enforce monotonicity per-timestep (safety net; usually already holds).
        sorted_qs = sorted(quantiles)
        if len(sorted_qs) > 1:
            stacked = np.column_stack([result[q] for q in sorted_qs])
            stacked = np.sort(stacked, axis=1)
            for i, q in enumerate(sorted_qs):
                result[q] = stacked[:, i]

        return {q: result[q].copy() for q in quantiles}


class GlobalTFT:
    """Global TFT (Temporal Fusion Transformer) model producing quantile forecasts.

    Wraps :class:`darts.models.TFTModel` with the shared fit/predict_quantiles
    interface.  TFT requires ``future_covariates`` at both fit and predict time;
    they are taken from ``bundle.future_covariates`` automatically.

    Parameters
    ----------
    quantiles : list of float
        Quantile levels to train and predict.
    input_chunk_length : int
        Look-back window in timesteps.
    output_chunk_length : int
        Forecast horizon used during training.
    n_epochs : int
        Training epochs.
    batch_size : int
    random_state : int
    pl_trainer_kwargs : dict or None
        Passed to PyTorch Lightning Trainer.  Defaults to CPU, no progress bar.
    **kwargs
        Additional keyword arguments forwarded to :class:`darts.models.TFTModel`.
    """

    def __init__(
        self,
        quantiles: list[float] | None = None,
        input_chunk_length: int = _DEFAULT_INPUT_CHUNK,
        output_chunk_length: int = _DEFAULT_OUTPUT_CHUNK,
        n_epochs: int = _DEFAULT_EPOCHS,
        batch_size: int = _DEFAULT_BATCH,
        random_state: int = 42,
        pl_trainer_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        self.quantiles = list(sorted(quantiles or _DEFAULT_QUANTILES))
        self.input_chunk_length = input_chunk_length
        self.output_chunk_length = output_chunk_length
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.pl_trainer_kwargs = pl_trainer_kwargs or {
            "enable_progress_bar": False,
            "accelerator": "cpu",
        }
        self.extra_kwargs = kwargs
        self.model: TFTModel | None = None

    def fit(self, bundle, train_end: pd.Timestamp | None = None) -> "GlobalTFT":
        """Fit TFT on ``bundle.target[:train_end]``.

        TFT requires future covariates, taken from ``bundle.future_covariates``.

        Parameters
        ----------
        bundle : PanelBundle
        train_end : pd.Timestamp or None

        Returns
        -------
        GlobalTFT
            self, for method chaining.
        """
        target = bundle.target if train_end is None else bundle.target[:train_end]

        model_kwargs = {**_TFT_DEFAULTS, **self.extra_kwargs}
        self.model = TFTModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.output_chunk_length,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            random_state=self.random_state,
            likelihood=QuantileRegression(quantiles=self.quantiles),
            pl_trainer_kwargs=self.pl_trainer_kwargs,
            **model_kwargs,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                target,
                past_covariates=bundle.past_covariates,
                future_covariates=bundle.future_covariates,
            )

        return self

    def predict_quantiles(
        self,
        bundle,
        origin: pd.Timestamp,
        horizon: int,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    ) -> dict[float, np.ndarray]:
        """Forecast ``horizon`` steps after ``origin`` for each quantile.

        Leakage-safe: only target/past-covariate history up to ``origin`` is
        used.  Future covariates (calendar) are passed in full as they contain
        no leakage.

        Parameters
        ----------
        bundle : PanelBundle
        origin : pd.Timestamp
            Last observed timestep (inclusive).
        horizon : int
            Forecast horizon in timesteps.
        quantiles : tuple of float
            Subset of the fitted quantiles to return.

        Returns
        -------
        dict[float, np.ndarray]
            ``{q: array_of_length_horizon}`` for each q.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If a requested quantile was not fitted.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_quantiles().")

        missing = set(quantiles) - set(self.quantiles)
        if missing:
            raise ValueError(
                f"Quantiles {missing} were not trained. "
                f"Fitted quantiles: {self.quantiles}"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred = self.model.predict(
                n=horizon,
                series=bundle.target[:origin],
                past_covariates=bundle.past_covariates,
                future_covariates=bundle.future_covariates,
                num_samples=200,
            )

        result: dict[float, np.ndarray] = {}
        for q in quantiles:
            arr = pred.quantile(q).values().flatten()
            result[q] = arr

        # Enforce monotonicity per-timestep (safety net; usually already holds).
        sorted_qs = sorted(quantiles)
        if len(sorted_qs) > 1:
            stacked = np.column_stack([result[q] for q in sorted_qs])
            stacked = np.sort(stacked, axis=1)
            for i, q in enumerate(sorted_qs):
                result[q] = stacked[:, i]

        return {q: result[q].copy() for q in quantiles}
