"""Zero-shot foundation model baseline interfaces (Colab-only).

These functions provide a uniform interface to large pretrained time-series
foundation models (Chronos, TimesFM).  The libraries are NOT installed in the
local project venv — they are heavy GPU dependencies used only inside the
dedicated Colab training notebook.  Calling these functions locally will raise
a clear ImportError so the caller knows exactly what to do.

Colab usage notes
-----------------
* On Colab T4, wrap inference in a try/except for CUDA OOM and fall back to
  either CPU inference or a shorter context window:

    try:
        result = chronos_forecast(series, horizon=48)
    except RuntimeError as exc:          # CUDA OOM
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            result = chronos_forecast(series[-336:], horizon=48)  # shorter ctx
        else:
            raise

* TimesFM may similarly hit Colab free-tier quota limits (GPU memory or RAM).
  Reduce ``context_len`` or switch to the 200m variant if the 500m model OOMs.
"""
from __future__ import annotations

import numpy as np


def chronos_forecast(  # Colab-only
    series: np.ndarray,
    horizon: int,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    model_name: str = "amazon/chronos-t5-small",
    device: str = "cuda",
) -> dict[float, np.ndarray]:
    """Zero-shot quantile forecast using Amazon Chronos.

    Parameters
    ----------
    series : np.ndarray
        1-D array of historical values (context window).
    horizon : int
        Number of steps to forecast.
    quantiles : tuple of float
        Quantile levels to return.
    model_name : str
        HuggingFace model ID, e.g. ``"amazon/chronos-t5-small"``.
    device : str
        ``"cuda"`` for GPU (Colab T4) or ``"cpu"`` for local fallback.

    Returns
    -------
    dict[float, np.ndarray]
        ``{q: array_of_length_horizon}`` for each q.

    Raises
    ------
    ImportError
        If ``chronos-forecasting`` is not installed.

    Notes
    -----
    Colab-only: on CUDA OOM, catch ``RuntimeError`` and retry with a shorter
    context window (e.g. ``series[-336:]``) or fall back to ``device="cpu"``.
    """
    try:
        import chronos  # noqa: F401  # Colab-only
    except ImportError as exc:
        raise ImportError(
            "chronos-forecasting not installed; it is used only in the Colab "
            "training notebook.  pip install chronos-forecasting to run locally."
        ) from exc

    import torch
    from chronos import ChronosPipeline

    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    context = torch.tensor(series[np.newaxis, :], dtype=torch.float32)
    # num_samples controls the MC samples used to estimate quantiles.
    forecast = pipeline.predict(context, prediction_length=horizon, num_samples=200)
    # forecast shape: (1, num_samples, horizon)
    samples = forecast[0].numpy()  # (num_samples, horizon)

    return {q: np.quantile(samples, q, axis=0) for q in quantiles}


def timesfm_forecast(  # Colab-only
    series: np.ndarray,
    horizon: int,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    model_name: str = "google/timesfm-1.0-200m",
    context_len: int = 512,
) -> dict[float, np.ndarray]:
    """Zero-shot quantile forecast using Google TimesFM.

    Parameters
    ----------
    series : np.ndarray
        1-D array of historical values (context window).
    horizon : int
        Number of steps to forecast.
    quantiles : tuple of float
        Quantile levels to return.
    model_name : str
        HuggingFace model ID, e.g. ``"google/timesfm-1.0-200m"``.
    context_len : int
        Maximum context length to pass to the model.  Reduce to ``256`` if the
        500m variant hits Colab RAM/GPU quota limits.

    Returns
    -------
    dict[float, np.ndarray]
        ``{q: array_of_length_horizon}`` for each q.

    Raises
    ------
    ImportError
        If ``timesfm`` is not installed.

    Notes
    -----
    Colab-only: on CUDA OOM or Colab quota exhaustion, reduce ``context_len``
    or switch to ``"google/timesfm-1.0-200m"`` from the 500m variant.
    Catch ``RuntimeError`` for CUDA OOM and ``torch.cuda.empty_cache()`` before
    retrying.
    """
    try:
        import timesfm  # noqa: F401  # Colab-only
    except ImportError as exc:
        raise ImportError(
            "timesfm not installed; it is used only in the Colab training "
            "notebook.  pip install timesfm to run locally."
        ) from exc

    import timesfm as tfm

    tfm_model = tfm.TimesFm(
        hparams=tfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            horizon_len=horizon,
            context_len=context_len,
        ),
        checkpoint=tfm.TimesFmCheckpoint(huggingface_repo_id=model_name),
    )

    ctx = series[-context_len:]
    # forecast_on_df returns (point_forecast, quantile_forecast)
    point, quantile_preds = tfm_model.forecast(
        [ctx],
        freq=[0],
        quantile_levels=list(quantiles),
    )
    # quantile_preds shape: (1, horizon, n_quantiles)
    q_arr = quantile_preds[0]  # (horizon, n_quantiles)

    return {q: q_arr[:, i] for i, q in enumerate(quantiles)}
