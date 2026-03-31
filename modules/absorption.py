"""
modules/absorption.py
==============================
Absorption Ratio — Spectral Fragility Metric
==============================

Computes the Absorption Ratio (AR) as defined in:
  Kritzman, M., Li, Y., Page, S., & Rigobon, R. (2011).
  "Principal Components as a Measure of Systemic Risk."
  The Journal of Portfolio Management, 37(4), 112–126.

Definition:
  AR = Σᵢ₌₁ᵏ λᵢ / Σᵢ₌₁ᴺ λᵢ

where λᵢ are eigenvalues of the correlation matrix R (sorted descending),
and k is the smallest number of eigenvectors explaining F% of total variance.

Key insight:
  - High AR → few dominant factors drive the system (fragile, correlated)
  - Low AR → variance is well-distributed (diversified, resilient)
  - ΔAR (first difference) is the actionable signal; spikes precede crises
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import eigh  # More numerically stable than eig for symmetric matrices

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.returns import clean_returns


@dataclass
class AbsorptionResult:
    absorption_ratio: pd.Series       # AR_t for each date
    delta_ar: pd.Series                # First difference ΔAR_t
    standardized_delta: pd.Series      # ΔAR standardized by trailing std
    top_k_eigenvalues: pd.DataFrame    # Top-3 eigenvalues over time (diagnostic)
    window: int
    variance_fraction: float           # F parameter (e.g. 0.20 = top 20%)
    n_assets: int

    def fragility_state(self, threshold: float = 0.01) -> pd.Series:
        """Binary fragility signal: 1 if ΔAR > threshold (sudden concentration)."""
        return (self.delta_ar > threshold).astype(int)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "absorption_ratio": self.absorption_ratio,
            "delta_ar": self.delta_ar,
            "standardized_delta": self.standardized_delta,
        })


def _compute_ar(
    corr_matrix: np.ndarray,
    variance_fraction: float = 0.20,
) -> Tuple[float, np.ndarray]:
    """
    Compute the Absorption Ratio for a single correlation matrix.

    Parameters
    ----------
    corr_matrix : N × N correlation matrix
    variance_fraction : fraction of total variance for numerator (default 0.20)

    Returns
    -------
    ar : absorption ratio scalar
    eigenvalues : sorted descending eigenvalues
    """
    N = corr_matrix.shape[0]

    # eigh is faster and more stable than eig for symmetric positive semidefinite
    eigenvalues = eigh(corr_matrix, eigvals_only=True)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending
    eigenvalues = np.maximum(eigenvalues, 0)  # numerical floor

    total_var = eigenvalues.sum()
    if total_var < 1e-10:
        return np.nan, eigenvalues

    # Find k: smallest k s.t. cumsum / total >= variance_fraction
    cumvar = np.cumsum(eigenvalues) / total_var
    k = int(np.searchsorted(cumvar, variance_fraction) + 1)
    k = max(1, min(k, N))

    ar = eigenvalues[:k].sum() / total_var
    return float(ar), eigenvalues


def compute_absorption_ratio(
    returns: pd.DataFrame,
    window: int = 252,
    min_periods: int = 60,
    variance_fraction: float = 0.20,
    winsorize: float = 5.0,
) -> AbsorptionResult:
    """
    Compute the rolling Absorption Ratio.

    Parameters
    ----------
    returns : T × N return DataFrame
    window : rolling estimation window
    min_periods : minimum observations
    variance_fraction : fraction of variance for numerator (Kritzman et al. use 1/5)
    winsorize : clip returns at ±N sigma

    Returns
    -------
    AbsorptionResult
    """
    returns = clean_returns(returns, winsorize_sigma=winsorize if winsorize > 0 else None)
    T, N = returns.shape
    idx = returns.index

    ar_values = np.full(T, np.nan)
    top_eigs = np.full((T, min(3, N)), np.nan)

    for i in range(T):
        if i < min_periods:
            continue
        start_i = max(0, i - window)
        window_data = returns.iloc[start_i:i + 1]

        if len(window_data) < min_periods:
            continue

        try:
            corr_matrix = window_data.corr().values
            if np.any(np.isnan(corr_matrix)):
                continue
            ar, eigs = _compute_ar(corr_matrix, variance_fraction)
            ar_values[i] = ar
            top_eigs[i, :] = eigs[:min(3, N)]
        except Exception as e:
            warnings.warn(f"AR computation failed at index {i}: {e}")

    absorption_ratio = pd.Series(ar_values, index=idx, name="absorption_ratio")
    delta_ar = absorption_ratio.diff()
    delta_ar.name = "delta_ar"

    # Standardize ΔAR by trailing 252-day std
    standardized_delta = delta_ar / delta_ar.rolling(252, min_periods=30).std()
    standardized_delta.name = "standardized_delta_ar"

    top_eig_df = pd.DataFrame(
        top_eigs,
        index=idx,
        columns=[f"lambda_{i+1}" for i in range(min(3, N))],
    )

    return AbsorptionResult(
        absorption_ratio=absorption_ratio,
        delta_ar=delta_ar,
        standardized_delta=standardized_delta,
        top_k_eigenvalues=top_eig_df,
        window=window,
        variance_fraction=variance_fraction,
        n_assets=N,
    )
