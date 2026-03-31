"""
core/returns.py
Return computation, cleaning, and covariance estimation.

Covariance estimators available:
  - sample: plain MLE (biased for large N)
  - ledoit_wolf: shrinkage (Oracle Approximating, sklearn)
  - oas: Oracle Approximating Shrinkage
  - ewm: exponentially weighted (lambda-parameterized)
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def log_returns(prices: pd.DataFrame, fill_limit: int = 5) -> pd.DataFrame:
    """
    Compute log returns from price DataFrame.
    Forward-fills up to `fill_limit` NaN prices before differencing
    (handles market holidays where some markets are open, others closed).
    """
    prices = prices.ffill(limit=fill_limit)
    return np.log(prices).diff().iloc[1:]


def simple_returns(prices: pd.DataFrame, fill_limit: int = 5) -> pd.DataFrame:
    prices = prices.ffill(limit=fill_limit)
    return prices.pct_change().iloc[1:]


def align_returns(
    *dfs: pd.DataFrame,
    method: Literal["inner", "outer"] = "inner",
    fill: bool = True,
) -> Tuple[pd.DataFrame, ...]:
    """
    Align multiple return DataFrames to a common DatetimeIndex.
    With method='inner', only dates present in ALL frames are kept.
    """
    combined = pd.concat(dfs, axis=1, join=method)
    if fill:
        combined = combined.ffill(limit=3)
    return tuple(combined[df.columns] for df in dfs)


def clean_returns(
    returns: pd.DataFrame,
    winsorize_sigma: Optional[float] = 5.0,
    drop_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Clean return series:
    1. Drop columns with more than (1 - drop_threshold) fraction of NaNs
    2. Winsorize extreme values (optional) — useful for EM data with fixing jumps
    3. Fill remaining NaNs with 0 (for Mahalanobis: means 'no observation that day')
    """
    # Drop columns too sparse
    min_obs = int(drop_threshold * len(returns))
    returns = returns.dropna(axis=1, thresh=min_obs)

    # Winsorize
    if winsorize_sigma:
        mu = returns.mean()
        sigma = returns.std()
        lower = mu - winsorize_sigma * sigma
        upper = mu + winsorize_sigma * sigma
        returns = returns.clip(lower=lower, upper=upper, axis=1)

    # Fill residual NaNs
    returns = returns.fillna(0.0)

    return returns


# ---------------------------------------------------------------------------
# Covariance estimation
# ---------------------------------------------------------------------------

CovMethod = Literal["sample", "ledoit_wolf", "oas", "ewm"]


def estimate_covariance(
    returns: pd.DataFrame,
    method: CovMethod = "ledoit_wolf",
    ewm_halflife: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate covariance matrix and mean vector from a return DataFrame.

    Parameters
    ----------
    returns : T × N DataFrame of returns (already cleaned)
    method : covariance estimator
    ewm_halflife : half-life in trading days for EWM estimator

    Returns
    -------
    mu : (N,) mean vector
    sigma : (N, N) covariance matrix
    """
    X = returns.values
    mu = X.mean(axis=0)

    if method == "sample":
        sigma = np.cov(X.T, bias=False)

    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(X)
        sigma = lw.covariance_

    elif method == "oas":
        from sklearn.covariance import OAS
        oas = OAS().fit(X)
        sigma = oas.covariance_

    elif method == "ewm":
        # Exponentially weighted covariance
        alpha = 1 - np.exp(-np.log(2) / ewm_halflife)
        T, N = X.shape
        weights = (1 - alpha) ** np.arange(T - 1, -1, -1)
        weights /= weights.sum()
        mu_ew = (weights[:, None] * X).sum(axis=0)
        Xc = X - mu_ew
        sigma = (weights[:, None] * Xc).T @ Xc

    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure symmetric (numerical precision)
    sigma = (sigma + sigma.T) / 2

    return mu, sigma


def pseudo_inverse(sigma: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """
    Moore-Penrose pseudoinverse with threshold `rcond`.
    More stable than np.linalg.inv for near-singular covariance matrices.
    Always regularize for EM data — correlation spikes can create near-singularity.
    """
    return np.linalg.pinv(sigma, rcond=rcond)


def rolling_stats(
    returns: pd.DataFrame,
    window: int = 252,
    method: CovMethod = "ledoit_wolf",
    min_periods: int = 60,
) -> dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute rolling (mu, sigma) for each date in returns.index.
    Returns a dict {date_index: (mu, sigma)} for all dates with sufficient history.

    Note: This is O(T * window * N²) and can be slow for large T.
    For production, consider incremental updates.
    """
    stats = {}
    idx = returns.index

    for i in range(len(returns)):
        if i < min_periods:
            continue
        start_i = max(0, i - window)
        window_data = returns.iloc[start_i:i]

        # Skip if too many zeros (market closed)
        if window_data.shape[0] < min_periods:
            continue

        try:
            mu, sigma = estimate_covariance(window_data, method=method)
            stats[idx[i]] = (mu, sigma)
        except Exception:
            continue

    return stats
