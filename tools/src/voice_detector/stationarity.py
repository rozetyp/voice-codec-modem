from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import levene
from statsmodels.tsa.stattools import adfuller, kpss


@dataclass
class StationarityResult:
    feature: str
    n: int
    adf_p: float | None       # low p  -> reject unit root -> stationary
    kpss_p: float | None      # low p  -> reject stationarity -> non-stationary
    levene_p: float | None    # low p  -> variance differs halves
    mk_p: float | None        # low p  -> monotonic trend
    is_non_stationary: bool   # our combined verdict

    def as_dict(self) -> dict:
        return {
            "feature": self.feature,
            "n": self.n,
            "adf_p": self.adf_p,
            "kpss_p": self.kpss_p,
            "levene_p": self.levene_p,
            "mk_p": self.mk_p,
            "is_non_stationary": self.is_non_stationary,
        }


def _safe_adf(x: np.ndarray) -> float | None:
    try:
        return float(adfuller(x, autolag="AIC")[1])
    except Exception:
        return None


def _safe_kpss(x: np.ndarray) -> float | None:
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(kpss(x, regression="c", nlags="auto")[1])
    except Exception:
        return None


def _safe_levene(x: np.ndarray) -> float | None:
    mid = len(x) // 2
    if mid < 4 or len(x) - mid < 4:
        return None
    try:
        return float(levene(x[:mid], x[mid:])[1])
    except Exception:
        return None


def _safe_mk(x: np.ndarray) -> float | None:
    try:
        import pymannkendall as mk

        return float(mk.original_test(x).p)
    except Exception:
        return None


def test_feature(series: pl.Series | np.ndarray, feature_name: str) -> StationarityResult:
    x = np.asarray(series.to_numpy() if isinstance(series, pl.Series) else series, dtype=float)
    x = x[~np.isnan(x)]
    n = int(len(x))
    if n < 10:
        return StationarityResult(feature_name, n, None, None, None, None, False)

    adf_p = _safe_adf(x)
    kpss_p = _safe_kpss(x)
    lev_p = _safe_levene(x)
    mk_p = _safe_mk(x)

    # Combined verdict: non-stationary if any of these fire
    #   - ADF fails to reject unit root (adf_p > 0.05)
    #   - KPSS rejects stationarity (kpss_p < 0.05)
    #   - Variance differs first vs second half (lev_p < 0.05)
    #   - Monotonic trend present (mk_p < 0.05)
    votes = 0
    if adf_p is not None and adf_p > 0.05:
        votes += 1
    if kpss_p is not None and kpss_p < 0.05:
        votes += 1
    if lev_p is not None and lev_p < 0.05:
        votes += 1
    if mk_p is not None and mk_p < 0.05:
        votes += 1
    is_non_stationary = votes >= 2

    return StationarityResult(feature_name, n, adf_p, kpss_p, lev_p, mk_p, is_non_stationary)


def score_recording(df: pl.DataFrame, features: list[str]) -> tuple[int, list[StationarityResult]]:
    """Return (count of non-stationary features, per-feature results)."""
    results = [test_feature(df[f], f) for f in features]
    return sum(r.is_non_stationary for r in results), results


def rolling_std(series: pl.Series, window: int) -> pl.Series:
    return series.rolling_std(window_size=window, min_samples=max(3, window // 2))
