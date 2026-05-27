"""
Leakage guard for UK energy price forecasting.

Enforces that future covariates contain ONLY deterministic calendar features
and never contain out-turn (observed) data such as generation, demand, or price.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class LeakageError(Exception):
    """Raised when a forbidden (out-turn) component is found in future_covariates."""


# ---------------------------------------------------------------------------
# Allowed / forbidden patterns
# ---------------------------------------------------------------------------

# Exact allowed names for future covariate components.
_ALLOWED_EXACT: frozenset[str] = frozenset({
    "dow",
    "is_weekend",
    "is_holiday",
    "sp_of_day",
})

# Allowed prefixes (column name starts with one of these).
_ALLOWED_PREFIXES: tuple[str, ...] = ("sin_", "cos_", "windsolar_forecast")

# Forbidden patterns — any component matching these is out-turn / target data.
_FORBIDDEN_PREFIXES: tuple[str, ...] = ("gen_", "demand_")
_FORBIDDEN_EXACT: frozenset[str] = frozenset({"price"})


def _is_allowed(name: str) -> bool:
    """Return True if *name* is a permitted future-covariate component."""
    if name in _ALLOWED_EXACT:
        return True
    if any(name.startswith(p) for p in _ALLOWED_PREFIXES):
        return True
    return False


def _is_forbidden(name: str) -> bool:
    """Return True if *name* matches a forbidden (out-turn) pattern."""
    if name in _FORBIDDEN_EXACT:
        return True
    if any(name.startswith(p) for p in _FORBIDDEN_PREFIXES):
        return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assert_no_future_leakage(bundle) -> bool:
    """Raise :class:`LeakageError` if *bundle.future_covariates* contains out-turn data.

    A component is rejected if it matches a *forbidden* pattern (starts with
    ``gen_`` or ``demand_``, or is exactly ``price``) OR if it is simply not in
    the allowed set (exact names: ``dow``, ``is_weekend``, ``is_holiday``,
    ``sp_of_day``; prefixes: ``sin_``, ``cos_``, ``windsolar_forecast``).

    Parameters
    ----------
    bundle
        Any object with a ``future_covariates`` attribute that is a Darts
        ``TimeSeries`` (exposes ``.components``).

    Returns
    -------
    bool
        ``True`` if all components are clean.

    Raises
    ------
    LeakageError
        On the first forbidden or unrecognised component.
    """
    for name in bundle.future_covariates.components:
        if _is_forbidden(name):
            raise LeakageError(
                f"Forbidden out-turn component '{name}' found in future_covariates. "
                "Out-turn data must never be used as a future covariate."
            )
        if not _is_allowed(name):
            raise LeakageError(
                f"Unrecognised component '{name}' in future_covariates. "
                "Only deterministic calendar features are permitted as future covariates."
            )
    return True


def slice_future_covariates(future_cov, origin, horizon):
    """Slice future covariates for the window [origin+1 ... origin+horizon].

    Calendar features are deterministic, so this window is leakage-safe by
    construction — we are not revealing any out-turn information.

    Parameters
    ----------
    future_cov : darts.TimeSeries
        The full future-covariate series.
    origin : pd.Timestamp
        The last known timestamp (forecast origin).
    horizon : int
        Number of steps ahead to include (inclusive).

    Returns
    -------
    darts.TimeSeries
        Sliced series covering [origin + 1 freq … origin + horizon freq].
    """

    freq = future_cov.freq
    start = origin + freq
    end = origin + horizon * freq
    return future_cov.slice(start, end)
