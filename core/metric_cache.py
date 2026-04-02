"""
core/metric_cache.py
Disk-based metric cache for computed risk metrics.

Persists TurbulenceResult, AbsorptionResult, and DynamicFactorResult to
parquet files under data/_cache/metrics/ so Streamlit restarts and hot-reloads
do not trigger full recomputation.

Cache key convention:
  turb_panel_{data_key}_{window}_{vol_std_int}
  turb_fx_{data_key}_{window}_{vol_std_int}
  turb_eq_{data_key}_{window}_{vol_std_int}
  turb_{country}_{data_key}_{window}_{vol_std_int}
  ar_{data_key}_{window}_{lam_int}        (lam_int = int(lam * 100))
  dyn_{data_key}_{window}_{lam_int}

Architecture notes:
- Thresholds are stored as constant-value float columns (thresh_elevated,
  thresh_turbulent, thresh_crisis) alongside the turbulence time series.
  No mixed-index issues; single parquet file per result.
- CachedDynamicResult exposes a .pca proxy so dashboard code calling
  dyn.pca.innovation_norm continues to work without modification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

METRIC_CACHE_DIR = Path(__file__).parent.parent / "data" / "_cache" / "metrics"
METRIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Low-level cache primitives
# ---------------------------------------------------------------------------

def make_key(*parts) -> str:
    """
    Build a cache key string from positional parts.

    Parameters
    ----------
    *parts : any scalar — joined with underscores

    Returns
    -------
    str cache key safe for use as a filename stem
    """
    return "_".join(str(p) for p in parts)


def cache_path(key: str) -> Path:
    return METRIC_CACHE_DIR / f"{key}.parquet"


def exists(key: str) -> bool:
    """Return True if a cache file exists for the given key."""
    return cache_path(key).exists()


def save(key: str, df: pd.DataFrame) -> None:
    """Persist a DataFrame to parquet under the metric cache directory."""
    cache_path(key).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path(key))
    log.debug("Saved metric cache: %s  shape=%s", key, df.shape)


def load(key: str) -> pd.DataFrame:
    """Load a cached DataFrame from parquet."""
    return pd.read_parquet(cache_path(key))


# ---------------------------------------------------------------------------
# Lightweight result wrappers (duck-typed against the real dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class CachedTurbulenceResult:
    """
    Reconstructed TurbulenceResult from disk.

    Exposes the same attributes and methods the dashboard calls so the
    Streamlit rendering code needs no conditional branches.
    """
    turbulence: pd.Series
    log_turbulence: pd.Series
    regime: pd.Series
    regime_code: pd.Series
    magnitude_component: pd.Series
    correlation_component: pd.Series
    thresholds: dict
    chi2_pvalue: pd.Series

    def current_regime(self) -> str:
        return str(self.regime.dropna().iloc[-1])

    def current_score(self) -> float:
        return float(self.turbulence.dropna().iloc[-1])

    def current_pctile(self) -> float:
        from scipy import stats as sp_stats
        s = self.turbulence.dropna()
        return float(sp_stats.percentileofscore(s.values, float(s.iloc[-1])))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "turbulence":    self.turbulence,
            "log_turbulence": self.log_turbulence,
            "regime":        self.regime,
            "regime_code":   self.regime_code,
            "magnitude":     self.magnitude_component,
            "correlation":   self.correlation_component,
            "chi2_pvalue":   self.chi2_pvalue,
        })


@dataclass
class CachedAbsorptionResult:
    """Reconstructed AbsorptionResult from disk."""
    absorption_ratio: pd.Series
    delta_ar: pd.Series
    standardized_delta: pd.Series

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "absorption_ratio":  self.absorption_ratio,
            "delta_ar":          self.delta_ar,
            "standardized_delta": self.standardized_delta,
        })


@dataclass
class _PCAProxy:
    """Thin proxy so dyn.pca.innovation_norm works on CachedDynamicResult."""
    innovation_norm: pd.Series


@dataclass
class CachedDynamicResult:
    """Reconstructed DynamicFactorResult from disk."""
    factor_scores: pd.DataFrame
    innovation_norm: pd.Series
    composite_stress: pd.Series

    @property
    def pca(self) -> _PCAProxy:
        """Proxy object for dashboard compatibility (dyn.pca.innovation_norm)."""
        return _PCAProxy(innovation_norm=self.innovation_norm)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            **{f"F{i+1}_score": self.factor_scores.iloc[:, i]
               for i in range(self.factor_scores.shape[1])},
            "innovation_norm": self.innovation_norm,
            "composite_stress": self.composite_stress,
        })


# ---------------------------------------------------------------------------
# Turbulence save / load
# ---------------------------------------------------------------------------

def save_turbulence(key: str, result) -> None:
    """
    Persist a TurbulenceResult (or CachedTurbulenceResult) to parquet.

    Thresholds are embedded as constant-value float columns
    (thresh_elevated, thresh_turbulent, thresh_crisis) so a single file
    stores the complete state.

    Parameters
    ----------
    key    : cache key string
    result : TurbulenceResult or CachedTurbulenceResult
    """
    df = result.to_frame().copy()
    # Embed scalar thresholds as constant columns — avoids mixed-index parquet.
    for label, val in result.thresholds.items():
        df[f"thresh_{label}"] = float(val)
    save(key, df)


def load_turbulence(key: str) -> CachedTurbulenceResult:
    """
    Load a TurbulenceResult from parquet and reconstruct a
    CachedTurbulenceResult.

    Parameters
    ----------
    key : cache key string

    Returns
    -------
    CachedTurbulenceResult
    """
    df = load(key)
    thresh_cols = [c for c in df.columns if c.startswith("thresh_")]
    thresholds  = {
        c.replace("thresh_", ""): float(df[c].iloc[0])
        for c in thresh_cols
    }
    df = df.drop(columns=thresh_cols)

    return CachedTurbulenceResult(
        turbulence=df["turbulence"],
        log_turbulence=df["log_turbulence"],
        regime=df["regime"].astype(str),
        regime_code=df["regime_code"],
        magnitude_component=df["magnitude"],
        correlation_component=df["correlation"],
        thresholds=thresholds,
        chi2_pvalue=df["chi2_pvalue"],
    )


# ---------------------------------------------------------------------------
# Absorption save / load
# ---------------------------------------------------------------------------

def save_absorption(key: str, result) -> None:
    """
    Persist an AbsorptionResult (or CachedAbsorptionResult) to parquet.

    Parameters
    ----------
    key    : cache key string
    result : AbsorptionResult or CachedAbsorptionResult
    """
    save(key, result.to_frame())


def load_absorption(key: str) -> CachedAbsorptionResult:
    """
    Load an AbsorptionResult from parquet.

    Parameters
    ----------
    key : cache key string

    Returns
    -------
    CachedAbsorptionResult
    """
    df = load(key)
    return CachedAbsorptionResult(
        absorption_ratio=df["absorption_ratio"],
        delta_ar=df["delta_ar"],
        standardized_delta=df["standardized_delta"],
    )


# ---------------------------------------------------------------------------
# Dynamic factor save / load
# ---------------------------------------------------------------------------

def save_dynamic(key: str, result) -> None:
    """
    Persist a DynamicFactorResult (or CachedDynamicResult) to parquet.

    innovation_norm (pca.innovation_norm) is added as an explicit column
    since DynamicFactorResult.to_frame() omits it.

    Parameters
    ----------
    key    : cache key string
    result : DynamicFactorResult or CachedDynamicResult
    """
    df = result.to_frame().copy()
    # to_frame() on a real DynamicFactorResult doesn't include innovation_norm;
    # on CachedDynamicResult it does.  Overwrite unconditionally from source.
    inorm = (
        result.innovation_norm
        if isinstance(result, CachedDynamicResult)
        else result.pca.innovation_norm
    )
    df["innovation_norm"] = inorm
    save(key, df)


def load_dynamic(key: str) -> CachedDynamicResult:
    """
    Load a DynamicFactorResult from parquet.

    Parameters
    ----------
    key : cache key string

    Returns
    -------
    CachedDynamicResult
    """
    df = load(key)
    factor_cols = sorted([c for c in df.columns if c.endswith("_score")])
    return CachedDynamicResult(
        factor_scores=df[factor_cols].copy(),
        innovation_norm=df["innovation_norm"],
        composite_stress=df["composite_stress"],
    )
