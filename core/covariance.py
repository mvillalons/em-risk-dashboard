"""
core/covariance.py
Two-track covariance estimation for the EM risk dashboard.

SLOW track (SlowCovariance): Ledoit-Wolf shrinkage over a long window (default 504 days).
  Feeds turbulence.py. Measures deviation from long-run structural normality.

FAST track (FastCovariance): EWMA covariance, decomposed into D_t @ R_t @ D_t.
  Feeds absorption.py and pca_kalman.py. Tracks current correlation architecture.

VolStandardizer: EWMA vol pre-whitening. Applied to returns before turbulence computation
  so that tau captures correlation anomalies, not pure vol spikes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SlowCovarianceResult:
    """Output of SlowCovariance.fit()."""
    mu: np.ndarray        # (N,) mean vector
    sigma: np.ndarray     # (N, N) covariance matrix
    sigma_inv: np.ndarray # (N, N) pseudoinverse of sigma


@dataclass
class FastCovarianceResult:
    """Output of FastCovariance.fit(). Terminal (most recent) state."""
    R_t: np.ndarray      # (N, N) EWMA correlation matrix, PSD, unit diagonal
    D_t: np.ndarray      # (N, N) diagonal matrix of conditional vols
    sigma_t: np.ndarray  # (N, N) EWMA covariance matrix = D_t @ R_t @ D_t


# ---------------------------------------------------------------------------
# SlowCovariance
# ---------------------------------------------------------------------------

class SlowCovariance:
    """
    Long-run structural covariance via Ledoit-Wolf shrinkage.

    Intended for turbulence computation where tau measures deviation from
    the long-run correlation regime, not the current one.
    """

    def fit(
        self,
        returns: pd.DataFrame,
        window: int = 504,
        method: str = "ledoit_wolf",
    ) -> SlowCovarianceResult:
        """
        Fit covariance on the most recent `window` observations.

        Parameters
        ----------
        returns : T × N DataFrame of returns (DatetimeIndex)
        window  : lookback in trading days; uses all rows if T < window
        method  : only 'ledoit_wolf' supported (placeholder for extension)

        Returns
        -------
        SlowCovarianceResult

        Notes
        -----
        Raises ValueError if n_samples < n_features — Ledoit-Wolf degenerates.
        Uses np.linalg.pinv(rcond=1e-10) for pseudoinverse to handle
        near-singular cases from correlated EM assets.
        """
        data = returns.fillna(0.0).iloc[-window:]
        X = data.values
        n_samples, n_features = X.shape

        if n_samples < n_features:
            raise ValueError(
                f"SlowCovariance requires n_samples ({n_samples}) >= "
                f"n_features ({n_features}). Expand the window or reduce the asset universe."
            )

        mu = X.mean(axis=0)

        if method == "ledoit_wolf":
            lw = LedoitWolf()
            lw.fit(X)
            sigma = lw.covariance_
        else:
            raise ValueError(f"Unknown method: {method!r}. Only 'ledoit_wolf' is supported.")

        # Enforce symmetry (numerical precision)
        sigma = (sigma + sigma.T) / 2.0

        sigma_inv = np.linalg.pinv(sigma, rcond=1e-10)

        return SlowCovarianceResult(mu=mu, sigma=sigma, sigma_inv=sigma_inv)


# ---------------------------------------------------------------------------
# FastCovariance
# ---------------------------------------------------------------------------

def _enforce_psd_correlation(R: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """
    Project a symmetric matrix onto the PSD cone and re-normalize to correlation.

    Parameters
    ----------
    R     : symmetric N × N matrix (approximately a correlation matrix)
    floor : minimum eigenvalue

    Returns
    -------
    R_psd : PSD correlation matrix with unit diagonal and off-diagonal in [-1, 1]
    """
    R = (R + R.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, floor)
    R_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Re-normalize to correlation
    d = np.sqrt(np.diag(R_psd))
    d_safe = np.where(d > 1e-10, d, 1.0)
    R_psd = R_psd / np.outer(d_safe, d_safe)
    np.fill_diagonal(R_psd, 1.0)
    R_psd = np.clip(R_psd, -1.0, 1.0)
    np.fill_diagonal(R_psd, 1.0)

    return R_psd


class FastCovariance:
    """
    EWMA covariance estimator decomposed as sigma_t = D_t @ R_t @ D_t.

    D_t: diagonal matrix of conditional vols (sqrt of EWMA variance diagonal).
    R_t: EWMA correlation matrix — unit diagonal, off-diagonals in [-1, 1], PSD.

    Returns the terminal state (most recent date). For a rolling series of R_t,
    call fit() on successive windows or use the internal loop externally.
    """

    def fit(
        self,
        returns: pd.DataFrame,
        lam: float = 0.94,
    ) -> FastCovarianceResult:
        """
        Compute EWMA covariance from returns, return terminal state.

        Parameters
        ----------
        returns : T × N DataFrame of returns (DatetimeIndex)
        lam     : EWMA decay factor (RiskMetrics default: 0.94)

        Returns
        -------
        FastCovarianceResult with R_t, D_t, sigma_t at the final date.

        Notes
        -----
        sigma_t = lam * sigma_{t-1} + (1-lam) * r_t r_t'   (zero-mean EWMA)
        D_t     = diag(sqrt(diag(sigma_t)))
        R_t     = D_t^{-1} sigma_t D_t^{-1}
        PSD is enforced on R_t via eigenvalue floor at 1e-6.
        """
        X = returns.fillna(0.0).values
        T, N = X.shape

        # Initialize sigma from a short burn-in window
        burn = min(63, max(T // 4, N + 1))
        X_burn = X[:burn]
        if burn >= N:
            sigma_t = np.cov(X_burn.T, bias=False)
        else:
            sigma_t = np.diag(np.var(X_burn, axis=0) + 1e-8)

        sigma_t = (sigma_t + sigma_t.T) / 2.0

        for t in range(T):
            r = X[t]
            sigma_t = lam * sigma_t + (1.0 - lam) * np.outer(r, r)

        # Decompose into D_t and R_t
        var_diag = np.diag(sigma_t)
        d_vec = np.sqrt(np.maximum(var_diag, 1e-12))
        D_t = np.diag(d_vec)
        d_inv = 1.0 / d_vec
        D_inv = np.diag(d_inv)

        R_t = D_inv @ sigma_t @ D_inv
        np.fill_diagonal(R_t, 1.0)
        R_t = _enforce_psd_correlation(R_t)

        # Reconstruct sigma from cleaned R_t
        sigma_t = D_t @ R_t @ D_t

        return FastCovarianceResult(R_t=R_t, D_t=D_t, sigma_t=sigma_t)

    def fit_rolling(
        self,
        returns: pd.DataFrame,
        lam: float = 0.94,
    ) -> List[np.ndarray]:
        """
        Compute EWMA correlation matrix R_t at every date in `returns`.

        Parameters
        ----------
        returns : T × N DataFrame of returns (DatetimeIndex)
        lam     : EWMA decay factor

        Returns
        -------
        List of T N×N correlation matrices (one per row of returns).
        Each matrix is PSD with unit diagonal and off-diagonal in [-1, 1].
        """
        X = returns.fillna(0.0).values
        T, N = X.shape

        burn = min(63, max(T // 4, N + 1))
        X_burn = X[:burn]
        if burn >= N:
            sigma_t = np.cov(X_burn.T, bias=False)
        else:
            sigma_t = np.diag(np.var(X_burn, axis=0) + 1e-8)
        sigma_t = (sigma_t + sigma_t.T) / 2.0

        R_series = []
        for t in range(T):
            r = X[t]
            sigma_t = lam * sigma_t + (1.0 - lam) * np.outer(r, r)

            var_diag = np.diag(sigma_t)
            d_vec = np.sqrt(np.maximum(var_diag, 1e-12))
            d_inv = 1.0 / d_vec
            D_inv = np.diag(d_inv)
            R_t = D_inv @ sigma_t @ D_inv
            np.fill_diagonal(R_t, 1.0)
            R_t = _enforce_psd_correlation(R_t)
            R_series.append(R_t)

        return R_series


# ---------------------------------------------------------------------------
# VolStandardizer
# ---------------------------------------------------------------------------

class VolStandardizer:
    """
    EWMA vol pre-whitening: divides each return r_t by its conditional vol v_t.

    Used before turbulence computation so that tau measures correlation
    anomalies rather than pure vol spikes. A pure vol spike with unchanged
    correlations should not register as turbulent after standardization.
    """

    def fit_transform(
        self,
        returns: pd.DataFrame,
        lam: float = 0.94,
        min_periods: int = 30,
    ) -> pd.DataFrame:
        """
        Compute EWMA vol series and return vol-standardized returns.

        Parameters
        ----------
        returns     : T × N DataFrame of returns (DatetimeIndex)
        lam         : EWMA decay factor
        min_periods : first `min_periods` rows returned as NaN (insufficient history)

        Returns
        -------
        pd.DataFrame with same index and columns as `returns`.
        Rows 0 .. min_periods-1 are NaN. Remaining rows have ~ unit variance.

        Notes
        -----
        v_t^2[i] = lam * v_{t-1}^2[i] + (1-lam) * r_t[i]^2
        r_tilde_t[i] = r_t[i] / v_t[i]
        Initialized from sample variance of the first min(63, T//4) observations.
        """
        X = returns.fillna(0.0).values
        T, N = X.shape

        # Initialize EWMA variance from short burn-in
        burn = min(63, max(T // 4, 2))
        v2 = np.var(X[:burn], axis=0)
        v2 = np.maximum(v2, 1e-12)

        result = np.full((T, N), np.nan)

        for t in range(T):
            v2 = lam * v2 + (1.0 - lam) * X[t] ** 2
            v2 = np.maximum(v2, 1e-12)
            if t >= min_periods:
                result[t] = X[t] / np.sqrt(v2)

        return pd.DataFrame(result, index=returns.index, columns=returns.columns)


# ---------------------------------------------------------------------------
# Helpers for vech (lower triangle excluding diagonal)
# ---------------------------------------------------------------------------

def _vech(R: np.ndarray) -> np.ndarray:
    """Extract lower-triangle (excluding diagonal) of a symmetric matrix."""
    N = R.shape[0]
    idx = np.tril_indices(N, k=-1)
    return R[idx].copy()


def _unvech(v: np.ndarray, N: int) -> np.ndarray:
    """Reconstruct symmetric N×N matrix from vech vector; diagonal set to 1."""
    R = np.zeros((N, N))
    idx = np.tril_indices(N, k=-1)
    R[idx] = v
    R += R.T
    np.fill_diagonal(R, 1.0)
    return R


# ---------------------------------------------------------------------------
# KalmanCorrelation
# ---------------------------------------------------------------------------

@dataclass
class KalmanCorrelationResult:
    """Output of KalmanCorrelation.fit()."""
    filtered_R: List[np.ndarray]      # T-length list of smoothed N×N correlation matrices
    innovations_norm: List[float]     # T-length list of Frobenius norms ||R_hat_t - R_{t|t-1}||_F


class KalmanCorrelation:
    """
    Kalman filter on the vectorized correlation matrix vech(R_t).

    Treats the time-varying correlation matrix as a latent state evolving
    as a random walk. The EWMA-estimated R_hat_t is the noisy observation.
    The Kalman filter produces smoothed R_t and the innovation norm
    ||R_hat_t - R_{t|t-1}||_F, a leading indicator of correlation regime shifts.

    State vector: vech(R_t) — lower triangle excluding diagonal, length M = N(N-1)/2.
    State model:  x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)   [random walk]
    Obs model:    y_t = x_t + v_t,        v_t ~ N(0, R_n) [direct observation]

    Both Q and R_n are diagonal so the filter operates element-wise.
    """

    def fit(
        self,
        R_hat_series: List[np.ndarray],
        q_scale: float = 0.01,
    ) -> KalmanCorrelationResult:
        """
        Fit Kalman filter on a series of EWMA correlation matrices.

        Parameters
        ----------
        R_hat_series : list of T N×N correlation matrices (from FastCovariance)
        q_scale      : process noise scale; Q = q_scale * Var(vech(R_hat))
            Controls how fast the latent R_t can move. Larger = more responsive.

        Returns
        -------
        KalmanCorrelationResult with:
          filtered_R       : list of T smoothed N×N correlation matrices
          innovations_norm : list of T Frobenius norms ||R_hat_t - R_{t|t-1}||_F

        Notes
        -----
        Q   = diag(q_scale * empirical_var(vech element across T obs))
        R_n = diag(rolling 63-day variance of vech innovations), updated each step.
        After each update, R_t is reconstructed, diagonal forced to 1,
        off-diagonals clipped to [-0.999, 0.999], then PSD-enforced.
        """
        T = len(R_hat_series)
        N = R_hat_series[0].shape[0]
        M = N * (N - 1) // 2      # vech length

        # Vectorize all observed correlation matrices
        vech_obs = np.array([_vech(R) for R in R_hat_series])   # T × M

        # --- Process noise Q: q_scale * empirical variance of each vech element ---
        var_vech = np.var(vech_obs, axis=0)
        Q_diag = np.maximum(q_scale * var_vech, 1e-8)

        # --- Initial observation noise R_n: variance of first-differences ---
        burn = min(63, T)
        if burn > 1:
            R_noise_diag = np.var(np.diff(vech_obs[:burn], axis=0), axis=0) + 1e-8
        else:
            R_noise_diag = np.ones(M) * 1e-4

        # --- Initialize Kalman state ---
        x_hat = vech_obs[0].copy()
        P_diag = R_noise_diag.copy()     # Initial state variance

        # Rolling buffer for updating R_noise (63-day window of innovations)
        innov_buf = np.zeros((63, M))

        filtered_R: List[np.ndarray] = []
        innovations_norm: List[float] = []

        for t in range(T):
            y_t = vech_obs[t]

            # --- Predict ---
            x_pred = x_hat.copy()
            P_pred = P_diag + Q_diag

            # --- Innovation (Frobenius norm on full matrix) ---
            R_pred = _unvech(np.clip(x_pred, -0.999, 0.999), N)
            innov_matrix = R_hat_series[t] - R_pred
            innovations_norm.append(float(np.linalg.norm(innov_matrix, 'fro')))

            innov_vec = y_t - x_pred   # vech of innovation

            # --- Update rolling R_noise from 63-day window ---
            innov_buf[t % 63] = innov_vec
            window_size = min(t + 1, 63)
            if window_size > 1:
                R_noise_diag = np.var(innov_buf[:window_size], axis=0) + 1e-8

            # --- Kalman gain (element-wise, diagonal covariance) ---
            K_diag = P_pred / (P_pred + R_noise_diag)

            # --- Update state ---
            x_hat = x_pred + K_diag * innov_vec
            P_diag = (1.0 - K_diag) * P_pred

            # --- Reconstruct and enforce valid correlation matrix ---
            x_clipped = np.clip(x_hat, -0.999, 0.999)
            R_filtered = _unvech(x_clipped, N)
            np.fill_diagonal(R_filtered, 1.0)
            R_filtered = _enforce_psd_correlation(R_filtered)

            filtered_R.append(R_filtered)

            # Sync state vector with PSD-projected matrix
            x_hat = _vech(R_filtered)

        return KalmanCorrelationResult(
            filtered_R=filtered_R,
            innovations_norm=innovations_norm,
        )
