"""
modules/turbulence.py
==============================
Turbulence Index — Core Building Block
==============================

Implementation of the Kritzman & Li (2010) Turbulence Index, extended with:
  - Mahalanobis decomposition: magnitude vs. correlation components
  - Ledoit-Wolf shrinkage for covariance estimation
  - Rolling computation with configurable window
  - Regime classification (Calm / Elevated / Turbulent / Crisis)
  - Turbulence-Corr matrix (Chow, Jacquier, Kritzman, Lowry 1999 variant)

References:
  Kritzman, M., & Li, Y. (2010). Skulls, Financial Turbulence, and Expected Returns.
  Financial Analysts Journal, 66(5), 30–41.
  
  State Street Global Markets (2010). Turbulence and the Risk of Investing.

Mathematical core:
  τ_t = (r_t - μ)' Σ⁻¹ (r_t - μ)

  where r_t is the N-dimensional return vector at time t,
  μ is the rolling mean vector, and Σ is the rolling covariance matrix.

  Under multivariate Gaussian, τ_t ~ χ²(N), so χ²(N) percentiles give
  regime thresholds with clean statistical interpretation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Internal imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.returns import (
    estimate_covariance,
    pseudo_inverse,
    clean_returns,
    log_returns,
    CovMethod,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TurbulenceResult:
    """
    Container for all outputs of the Turbulence Index computation.

    Attributes
    ----------
    turbulence : pd.Series
        Raw Mahalanobis distance squared (τ_t) for each date.
    log_turbulence : pd.Series
        log(τ_t + 1) — more normally distributed, useful for z-scoring.
    regime : pd.Series
        Categorical regime label per date.
    regime_code : pd.Series
        Integer code: 0=Calm, 1=Elevated, 2=Turbulent, 3=Crisis.
    magnitude_component : pd.Series
        Component of τ due to return magnitudes (diagonal of Σ⁻¹ terms).
    correlation_component : pd.Series
        Component of τ due to unusual correlations (off-diagonal terms).
    thresholds : dict
        Actual τ values at regime transition percentiles.
    chi2_pvalue : pd.Series
        Right-tail p-value under χ²(N) distribution (lower = more turbulent).
    window : int
        Rolling window used.
    n_assets : int
        Number of assets in the return vector.
    asset_names : list[str]
        Column names from input returns.
    """
    turbulence: pd.Series
    log_turbulence: pd.Series
    regime: pd.Series
    regime_code: pd.Series
    magnitude_component: pd.Series
    correlation_component: pd.Series
    thresholds: dict
    chi2_pvalue: pd.Series
    window: int
    n_assets: int
    asset_names: list[str]
    _cov_estimates: dict = field(default_factory=dict, repr=False)

    def summary(self) -> pd.DataFrame:
        """Compact summary stats by regime."""
        df = pd.DataFrame({
            "turbulence": self.turbulence,
            "regime": self.regime,
        }).dropna()
        return (
            df.groupby("regime")["turbulence"]
            .agg(["count", "mean", "median", "max"])
            .rename(columns={"count": "N", "mean": "Mean τ", "median": "Median τ", "max": "Max τ"})
        )

    def current_regime(self) -> str:
        """Most recent non-NaN regime."""
        return self.regime.dropna().iloc[-1]

    def current_score(self) -> float:
        """Most recent turbulence score."""
        return float(self.turbulence.dropna().iloc[-1])

    def current_pctile(self) -> float:
        """Percentile of most recent score in the full history."""
        score = self.current_score()
        all_scores = self.turbulence.dropna().values
        return float(sp_stats.percentileofscore(all_scores, score))

    def rolling_percentile(self, window: int = 252) -> pd.Series:
        """Rolling percentile rank of turbulence score (0–100)."""
        return self.turbulence.rolling(window, min_periods=30).apply(
            lambda x: sp_stats.percentileofscore(x[:-1], x[-1]) if len(x) > 1 else np.nan
        )

    def to_frame(self) -> pd.DataFrame:
        """All main series in one DataFrame."""
        return pd.DataFrame({
            "turbulence": self.turbulence,
            "log_turbulence": self.log_turbulence,
            "regime": self.regime,
            "regime_code": self.regime_code,
            "magnitude": self.magnitude_component,
            "correlation": self.correlation_component,
            "chi2_pvalue": self.chi2_pvalue,
        })


REGIME_LABELS = {0: "Calm", 1: "Elevated", 2: "Turbulent", 3: "Crisis"}
REGIME_COLORS = {
    "Calm":      "#2ecc71",
    "Elevated":  "#f39c12",
    "Turbulent": "#e67e22",
    "Crisis":    "#e74c3c",
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _mahalanobis_sq(
    r: np.ndarray,
    mu: np.ndarray,
    sigma_inv: np.ndarray,
) -> float:
    """
    Compute squared Mahalanobis distance for a single observation.
    τ = (r - μ)' Σ⁻¹ (r - μ)
    """
    diff = r - mu
    return float(diff @ sigma_inv @ diff)


def _decompose_mahalanobis(
    r: np.ndarray,
    mu: np.ndarray,
    sigma_inv: np.ndarray,
) -> Tuple[float, float]:
    """
    Decompose τ into:
      - magnitude: uses only diagonal of Σ⁻¹ (individual asset volatility scaling)
      - correlation: τ_total - τ_magnitude (residual due to cross-asset correlations)

    This decomposition is useful for distinguishing:
      - Volatility spikes (magnitude dominates) → typical bear markets
      - Correlation breaks (correlation dominates) → true systemic events
    """
    diff = r - mu

    # Magnitude-only: diagonal Σ⁻¹
    diag_inv = np.diag(np.diag(sigma_inv))
    tau_mag = float(diff @ diag_inv @ diff)

    # Total
    tau_total = float(diff @ sigma_inv @ diff)

    # Correlation = residual
    tau_corr = tau_total - tau_mag

    return tau_mag, max(0.0, tau_corr)  # numerical floor at 0


# ---------------------------------------------------------------------------
# Main turbulence index function
# ---------------------------------------------------------------------------

def compute_turbulence_index(
    returns: pd.DataFrame,
    window: int = 252,
    min_periods: int = 60,
    method: CovMethod = "ledoit_wolf",
    regime_thresholds: Tuple[float, float, float] = (0.75, 0.90, 0.95),
    winsorize: float = 5.0,
    use_log: bool = True,
) -> TurbulenceResult:
    """
    Compute the rolling Turbulence Index (Kritzman & Li, 2010).

    The turbulence score at time t is the squared Mahalanobis distance of the
    current return vector r_t from the historical mean, scaled by the inverse
    covariance matrix estimated on the trailing `window` observations.

    Parameters
    ----------
    returns : pd.DataFrame
        T × N DataFrame of returns (log or simple). Dates as index.
    window : int
        Rolling window for estimating μ and Σ. Default 252 (1 trading year).
    min_periods : int
        Minimum observations before computing. Default 60.
    method : CovMethod
        Covariance estimator: 'ledoit_wolf' (default), 'oas', 'sample', 'ewm'.
    regime_thresholds : tuple of 3 floats
        Percentile cutoffs for (Elevated, Turbulent, Crisis) regimes.
        Applied to the full history of turbulence scores.
    winsorize : float
        Winsorize returns at ±N standard deviations. Set 0 to disable.
    use_log : bool
        Whether to use log(τ) for regime classification. Raw τ for display.

    Returns
    -------
    TurbulenceResult
        Dataclass with all outputs. See TurbulenceResult docstring.

    Notes
    -----
    For N assets, τ_t ~ χ²(N) under Gaussian returns. This gives a clean
    statistical interpretation: a score at the 95th percentile of χ²(N) means
    the return vector would occur only 5% of the time if markets were normal.

    The Mahalanobis decomposition (magnitude vs. correlation) is non-standard
    but highly diagnostic: correlation breaks are the hallmark of systemic crises.
    """
    # --- Input validation and cleaning ---
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pd.DataFrame")

    returns = clean_returns(returns, winsorize_sigma=winsorize if winsorize > 0 else None)
    N = returns.shape[1]
    T = returns.shape[0]
    asset_names = list(returns.columns)

    if T < min_periods:
        raise ValueError(f"Insufficient data: {T} obs, need at least {min_periods}")

    # --- Rolling computation ---
    tau_values = np.full(T, np.nan)
    mag_values = np.full(T, np.nan)
    corr_values = np.full(T, np.nan)

    for i in range(T):
        if i < min_periods:
            continue

        # Rolling window (expanding until we hit `window`)
        start_i = max(0, i - window)
        window_data = returns.iloc[start_i:i]  # exclude current observation

        if len(window_data) < min_periods:
            continue

        try:
            mu, sigma = estimate_covariance(window_data, method=method)
            sigma_inv = pseudo_inverse(sigma)

            r_t = returns.iloc[i].values
            tau = _mahalanobis_sq(r_t, mu, sigma_inv)
            tau_mag, tau_corr = _decompose_mahalanobis(r_t, mu, sigma_inv)

            tau_values[i] = tau
            mag_values[i] = tau_mag
            corr_values[i] = tau_corr

        except np.linalg.LinAlgError:
            warnings.warn(f"Linear algebra error at index {i}; skipping")
            continue
        except Exception as e:
            warnings.warn(f"Unexpected error at index {i}: {e}")
            continue

    # --- Build Series ---
    idx = returns.index
    turbulence = pd.Series(tau_values, index=idx, name="turbulence")
    magnitude = pd.Series(mag_values, index=idx, name="magnitude")
    correlation = pd.Series(corr_values, index=idx, name="correlation")
    log_turbulence = np.log1p(turbulence)
    log_turbulence.name = "log_turbulence"

    # Chi-squared p-values (under H0: returns are multivariate normal)
    chi2_pvalue = turbulence.apply(
        lambda x: 1 - sp_stats.chi2.cdf(x, df=N) if not np.isnan(x) else np.nan
    )
    chi2_pvalue.name = "chi2_pvalue"

    # --- Regime classification ---
    # Use the full (non-rolling) history for percentile thresholds
    valid_tau = turbulence.dropna()
    if use_log:
        valid_scores = np.log1p(valid_tau.values)
    else:
        valid_scores = valid_tau.values

    th_elev, th_turb, th_crisis = np.percentile(
        valid_scores, [q * 100 for q in regime_thresholds]
    )

    # Map back to raw τ if needed
    if use_log:
        thresh_raw = {
            "elevated":  np.expm1(th_elev),
            "turbulent": np.expm1(th_turb),
            "crisis":    np.expm1(th_crisis),
        }
    else:
        thresh_raw = {
            "elevated":  th_elev,
            "turbulent": th_turb,
            "crisis":    th_crisis,
        }

    def _classify(x: float) -> Tuple[str, int]:
        if np.isnan(x):
            return np.nan, np.nan
        if x >= thresh_raw["crisis"]:
            return "Crisis", 3
        elif x >= thresh_raw["turbulent"]:
            return "Turbulent", 2
        elif x >= thresh_raw["elevated"]:
            return "Elevated", 1
        else:
            return "Calm", 0

    classified = turbulence.apply(_classify)
    regime = pd.Series(
        [c[0] for c in classified], index=idx, name="regime", dtype=object
    )
    regime_code = pd.Series(
        [c[1] for c in classified], index=idx, name="regime_code"
    )

    return TurbulenceResult(
        turbulence=turbulence,
        log_turbulence=log_turbulence,
        regime=regime,
        regime_code=regime_code,
        magnitude_component=magnitude,
        correlation_component=correlation,
        thresholds=thresh_raw,
        chi2_pvalue=chi2_pvalue,
        window=window,
        n_assets=N,
        asset_names=asset_names,
    )


# ---------------------------------------------------------------------------
# Country-level turbulence (single country, all signals)
# ---------------------------------------------------------------------------

def compute_country_turbulence(
    country_returns: dict[str, pd.DataFrame],
    **kwargs,
) -> dict[str, TurbulenceResult]:
    """
    Convenience wrapper: compute turbulence for each country separately.

    Parameters
    ----------
    country_returns : dict mapping country code → returns DataFrame
    **kwargs : passed to compute_turbulence_index

    Returns
    -------
    dict mapping country code → TurbulenceResult
    """
    results = {}
    for country, rets in country_returns.items():
        try:
            results[country] = compute_turbulence_index(rets, **kwargs)
        except Exception as e:
            warnings.warn(f"Turbulence computation failed for {country}: {e}")
    return results


# ---------------------------------------------------------------------------
# Panel-level turbulence (all countries jointly)
# ---------------------------------------------------------------------------

def compute_panel_turbulence(
    returns_panel: pd.DataFrame,
    **kwargs,
) -> TurbulenceResult:
    """
    Compute turbulence on the full EM panel (all countries × all signals).
    This captures cross-country contagion and global EM stress.

    Parameters
    ----------
    returns_panel : pd.DataFrame with all country/asset returns as columns

    Returns
    -------
    TurbulenceResult for the panel
    """
    return compute_turbulence_index(returns_panel, **kwargs)


# ---------------------------------------------------------------------------
# Convenience: crisis calendar from turbulence history
# ---------------------------------------------------------------------------

def crisis_episodes(
    result: TurbulenceResult,
    regime: str = "Crisis",
    min_duration_days: int = 5,
) -> pd.DataFrame:
    """
    Extract contiguous crisis episodes from turbulence regime series.

    Returns a DataFrame with columns: start, end, duration_days, peak_tau.
    """
    r = result.regime.dropna()
    in_episode = (r == regime).astype(int)
    transitions = in_episode.diff().fillna(in_episode)

    episodes = []
    start_date = None

    for date, val in in_episode.items():
        if val == 1 and start_date is None:
            start_date = date
        elif val == 0 and start_date is not None:
            duration = (date - start_date).days
            if duration >= min_duration_days:
                episode_tau = result.turbulence.loc[start_date:date]
                episodes.append({
                    "start": start_date,
                    "end": date,
                    "duration_days": duration,
                    "peak_tau": float(episode_tau.max()),
                    "mean_tau": float(episode_tau.mean()),
                })
            start_date = None

    return pd.DataFrame(episodes)
