"""
modules/pca_kalman.py
==============================
Dynamic Factor Structure: PCA + Kalman Filter
==============================

Purpose: Track the time-varying latent factor structure of EM risk.

Two-stage approach:
  Stage 1 — Rolling PCA:
    Decompose the return panel into K principal components.
    Track eigenvalues, loadings, and factor scores over time.
    Detect shifts in factor structure (rotation of risk drivers).

  Stage 2 — Kalman Filter on factor scores:
    Apply a local-level (random walk + noise) Kalman filter to each factor.
    Produces smoothed factor estimates and one-step-ahead forecasts.
    The innovation variance is a real-time signal: large innovations = structural break.

This gives you:
  - Which latent risk factor is driving EM stress at any moment
  - How the factor loadings (which countries contribute to factor 1) are shifting
  - A smoothed, noise-reduced signal for regime detection

References:
  Stock, J.H., Watson, M.W. (2002). "Forecasting Using Principal Components
    from a Large Number of Predictors." JASA 97(460).

  Harvey, A.C. (1989). "Forecasting, Structural Time Series Models and the
    Kalman Filter." Cambridge University Press.

  Kritzman, M., Page, S., Turkington, D. (2012). "Regime Shifts: Implications
    for Dynamic Strategies." Financial Analysts Journal.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.covariance import FastCovariance, KalmanCorrelation


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RollingPCAResult:
    """
    Output of rolling PCA decomposition.

    factor_scores : T × K DataFrame — principal component scores at each date
    loadings_history : dict {date → (N × K) loading matrix} — sampled loadings
    explained_variance : T × K DataFrame — fraction of variance per component
    eigenvalues : T × K DataFrame — eigenvalue magnitudes
    n_components : K
    asset_names : list of N asset names
    """
    factor_scores: pd.DataFrame
    explained_variance: pd.DataFrame
    eigenvalues: pd.DataFrame
    loadings_latest: pd.DataFrame        # Most recent N × K loading matrix
    loadings_history: dict               # Sampled snapshots {date: DataFrame}
    factor_correlation: pd.DataFrame     # Corr of factor scores with original assets
    window: int
    n_components: int
    asset_names: list[str]
    innovation_norm: pd.Series = field(default_factory=pd.Series)  # ||R_hat_t - R_{t|t-1}||_F

    def dominant_factor_exposure(self, country: str) -> pd.Series:
        """Rolling exposure (loading) of `country` to Factor 1."""
        return pd.Series(
            {dt: ldf.loc[country, "F1"] for dt, ldf in self.loadings_history.items()
             if country in ldf.index},
            name=f"{country}_F1_loading",
        ).sort_index()

    def factor_regime(self, threshold_sigma: float = 1.5) -> pd.DataFrame:
        """
        Flag dates where any factor score exceeds ±threshold_sigma.
        Returns boolean DataFrame T × K.
        """
        zscores = (self.factor_scores - self.factor_scores.mean()) / self.factor_scores.std()
        return zscores.abs() > threshold_sigma


@dataclass
class KalmanResult:
    """
    Output of Kalman filter applied to a single factor score series.

    filtered : smoothed factor estimate (x_t|t)
    predicted : one-step-ahead prediction (x_t|t-1)
    innovations : prediction errors (y_t - x_t|t-1)
    innovation_variance : rolling variance of innovations (structural break signal)
    gain : Kalman gain over time
    """
    filtered: pd.Series
    predicted: pd.Series
    innovations: pd.Series
    innovation_variance: pd.Series
    gain: pd.Series
    signal_noise_ratio: float           # Estimated Q/R ratio
    factor_name: str


@dataclass
class DynamicFactorResult:
    """Combined PCA + Kalman output."""
    pca: RollingPCAResult
    kalman: dict[str, KalmanResult]     # {factor_name: KalmanResult}
    composite_stress: pd.Series         # Weighted combination of filtered factors
    factor_regime_flags: pd.DataFrame   # Bool flags per factor per date

    def stress_percentile(self) -> float:
        """Current composite stress percentile."""
        s = self.composite_stress.dropna()
        if len(s) == 0:
            return np.nan
        from scipy import stats
        return float(stats.percentileofscore(s.values, s.iloc[-1]))

    def to_frame(self) -> pd.DataFrame:
        kf_cols = {f"kf_{k}": v.filtered for k, v in self.kalman.items()}
        inno_cols = {f"inno_{k}": v.innovations for k, v in self.kalman.items()}
        return pd.DataFrame({
            **{f"F{i+1}_score": self.pca.factor_scores.iloc[:, i]
               for i in range(self.pca.n_components)},
            **kf_cols,
            **inno_cols,
            "composite_stress": self.composite_stress,
        })


# ---------------------------------------------------------------------------
# Rolling PCA
# ---------------------------------------------------------------------------

def _pca_window(
    X: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA on a single window of data X (T_window × N).

    Returns
    -------
    scores : (T_window, K) — factor scores for each observation in window
    loadings : (N, K) — eigenvectors (loading matrix), columns = components
    explained_var : (K,) — fraction of variance explained per component
    eigenvalues : (K,) — eigenvalue magnitudes
    """
    X = X - X.mean(axis=0)  # demean

    # Use SVD for numerical stability (equivalent to covariance eigendecomp)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    eigenvalues = (s ** 2) / (X.shape[0] - 1)
    total_var = eigenvalues.sum()
    explained_var = eigenvalues / (total_var + 1e-12)

    K = min(n_components, len(s))
    loadings = Vt[:K].T           # N × K
    scores   = X @ loadings        # T_window × K

    return scores, loadings, explained_var[:K], eigenvalues[:K]


def compute_rolling_pca(
    returns: pd.DataFrame,
    window: int = 252,
    n_components: int = 3,
    min_periods: int = 60,
    snapshot_freq: int = 21,    # Save loading snapshot every N days (for history)
    standardize: bool = True,
    R_t_series: Optional[List[np.ndarray]] = None,
) -> RollingPCAResult:
    """
    Rolling PCA on the return panel.

    At each date t, we:
      1. Take the trailing window of returns (t-window : t)
      2. Standardize (optional) and demean
      3. Compute SVD → extract K principal components
      4. Project the CURRENT observation onto the loadings → factor score for t

    The resulting factor scores form a T × K time series of latent risk drivers.

    Parameters
    ----------
    returns : T × N return DataFrame
    window : rolling estimation window
    n_components : K, number of factors to extract
    min_periods : minimum obs before computing
    snapshot_freq : save loading snapshot every N periods (for loading evolution)
    standardize : whether to scale columns to unit variance before PCA
    R_t_series : optional list of T N×N correlation matrices.
        If provided, eigendecompose R_t_series[i] directly instead of computing
        sample correlation from the window. Returns are still used for factor
        score projection (current observation projected onto eigenvectors).

    Returns
    -------
    RollingPCAResult
    """
    returns = returns.ffill(limit=3).fillna(0.0)
    T, N = returns.shape
    idx = returns.index
    K = min(n_components, N)

    score_arr = np.full((T, K), np.nan)
    ev_arr    = np.full((T, K), np.nan)
    eig_arr   = np.full((T, K), np.nan)
    loadings_history = {}
    latest_loadings  = None
    asset_names      = list(returns.columns)

    scaler = StandardScaler() if standardize else None

    for i in range(T):
        if i < min_periods:
            continue

        start_i = max(0, i - window)
        X_raw = returns.iloc[start_i:i].values

        if X_raw.shape[0] < min_periods:
            continue

        if standardize:
            scaler.fit(X_raw)
            X = scaler.transform(X_raw)
            x_now = scaler.transform(returns.iloc[i:i+1].values)
        else:
            X = X_raw
            x_now = returns.iloc[i:i+1].values

        try:
            if R_t_series is not None:
                # Use provided correlation matrix: eigendecompose directly
                R_i = R_t_series[i]
                eigvals, eigvecs = np.linalg.eigh(R_i)
                # eigh returns ascending order; reverse for descending
                eigvals = eigvals[::-1]
                eigvecs = eigvecs[:, ::-1]
                eigvals = np.maximum(eigvals, 0.0)
                total_var = eigvals.sum()
                explained_var = (eigvals[:K] / (total_var + 1e-12))
                eigenvalues = eigvals[:K]
                loadings = eigvecs[:, :K]    # N × K
            else:
                _, loadings, explained_var, eigenvalues = _pca_window(X, K)

            # Project current observation onto loadings
            x_demeaned = x_now - X.mean(axis=0)
            scores_now = (x_demeaned @ loadings).ravel()

            score_arr[i, :K] = scores_now
            ev_arr[i, :K]    = explained_var
            eig_arr[i, :K]   = eigenvalues
            latest_loadings  = loadings

            # Save snapshot
            if i % snapshot_freq == 0:
                ldf = pd.DataFrame(
                    loadings,
                    index=asset_names,
                    columns=[f"F{k+1}" for k in range(K)],
                )
                loadings_history[idx[i]] = ldf

        except np.linalg.LinAlgError as e:
            warnings.warn(f"PCA failed at index {i}: {e}")
            continue

    # Save final snapshot regardless of freq
    if latest_loadings is not None:
        ldf_final = pd.DataFrame(
            latest_loadings,
            index=asset_names,
            columns=[f"F{k+1}" for k in range(K)],
        )
        loadings_history[idx[-1]] = ldf_final
        loadings_latest = ldf_final
    else:
        loadings_latest = pd.DataFrame(index=asset_names, columns=[f"F{k+1}" for k in range(K)])

    factor_scores = pd.DataFrame(
        score_arr, index=idx, columns=[f"F{k+1}" for k in range(K)]
    )
    explained_variance = pd.DataFrame(
        ev_arr, index=idx, columns=[f"F{k+1}" for k in range(K)]
    )
    eigenvalues_df = pd.DataFrame(
        eig_arr, index=idx, columns=[f"F{k+1}" for k in range(K)]
    )

    # Correlation of factor scores with original assets (interpretation aid)
    # factor_corr: rows=assets, columns=factors (for easy per-asset lookup)
    factor_corr = pd.DataFrame(index=asset_names, columns=[f"F{k+1}" for k in range(K)])
    for k in range(K):
        fs = factor_scores.iloc[:, k].dropna()
        for asset in asset_names:
            ret_s = returns[asset]
            aligned = pd.concat([fs, ret_s], axis=1).dropna()
            if len(aligned) > 30:
                factor_corr.loc[asset, f"F{k+1}"] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

    return RollingPCAResult(
        factor_scores=factor_scores,
        explained_variance=explained_variance,
        eigenvalues=eigenvalues_df,
        loadings_latest=loadings_latest,
        loadings_history=loadings_history,
        factor_correlation=factor_corr.astype(float),
        window=window,
        n_components=K,
        asset_names=asset_names,
    )


# ---------------------------------------------------------------------------
# Kalman Filter (local level model)
# ---------------------------------------------------------------------------

def kalman_filter_local_level(
    y: pd.Series,
    Q: Optional[float] = None,    # State noise variance (process noise)
    R: Optional[float] = None,    # Observation noise variance
    auto_tune: bool = True,       # Estimate Q/R from data if not provided
    n_burn: int = 30,             # Burn-in period before using innovations
) -> KalmanResult:
    """
    Local-level Kalman Filter for a univariate time series.

    State space model:
      x_t = x_{t-1} + η_t,   η_t ~ N(0, Q)   [state equation: random walk]
      y_t = x_t + ε_t,        ε_t ~ N(0, R)   [observation equation]

    The signal-to-noise ratio Q/R controls the filter's responsiveness:
      - High Q/R: filter tracks the signal closely (low smoothing)
      - Low Q/R: filter is more stable, larger lag (high smoothing)

    For factor scores, moderate Q/R ≈ 0.01–0.10 is typical.

    Parameters
    ----------
    y : pd.Series (factor score time series, may contain NaN)
    Q : process noise variance (auto-estimated if None and auto_tune=True)
    R : observation noise variance (auto-estimated if None and auto_tune=True)
    auto_tune : estimate Q and R from sample variance of y
    n_burn : number of initial periods to exclude from innovation statistics

    Returns
    -------
    KalmanResult
    """
    y_clean = y.dropna()
    n = len(y_clean)

    if n < 10:
        empty = pd.Series(np.nan, index=y.index)
        return KalmanResult(
            filtered=empty, predicted=empty, innovations=empty,
            innovation_variance=empty, gain=empty,
            signal_noise_ratio=np.nan, factor_name=y.name or "F?",
        )

    if auto_tune and (Q is None or R is None):
        # Simple moment estimator: R from short-diff variance, Q from long-run
        r_est = y_clean.diff().var() / 2   # Observation noise ~ half of short-diff var
        q_est = r_est * 0.05               # Signal-to-noise = 0.05 (moderate smoothing)
        Q = Q or max(q_est, 1e-8)
        R = R or max(r_est, 1e-8)

    Q = Q or 1e-4
    R = R or 1e-2

    # Initialize
    x_hat = float(y_clean.iloc[0])   # Initial state = first observation
    P     = float(R)                   # Initial state variance

    filtered_vals   = np.full(n, np.nan)
    predicted_vals  = np.full(n, np.nan)
    innovations_arr = np.full(n, np.nan)
    gain_arr        = np.full(n, np.nan)

    for t, yt in enumerate(y_clean.values):
        # Predict
        x_pred = x_hat
        P_pred = P + Q

        # Update
        K_gain = P_pred / (P_pred + R)   # Kalman gain
        innov  = yt - x_pred              # Innovation (prediction error)
        x_hat  = x_pred + K_gain * innov
        P      = (1 - K_gain) * P_pred

        predicted_vals[t]  = x_pred
        filtered_vals[t]   = x_hat
        innovations_arr[t] = innov
        gain_arr[t]        = K_gain

    # Reconstruct on original (possibly gapped) index
    def _reindex(arr):
        s = pd.Series(arr, index=y_clean.index)
        return s.reindex(y.index)

    filtered   = _reindex(filtered_vals);   filtered.name   = f"{y.name}_filtered"
    predicted  = _reindex(predicted_vals);  predicted.name  = f"{y.name}_predicted"
    innovations = _reindex(innovations_arr); innovations.name = f"{y.name}_innovation"
    gain        = _reindex(gain_arr);         gain.name        = f"{y.name}_gain"

    # Rolling innovation variance (structural break signal)
    inno_var = innovations.rolling(63, min_periods=20).var()
    inno_var.name = f"{y.name}_inno_variance"

    return KalmanResult(
        filtered=filtered,
        predicted=predicted,
        innovations=innovations,
        innovation_variance=inno_var,
        gain=gain,
        signal_noise_ratio=float(Q / R),
        factor_name=y.name or "F?",
    )


# ---------------------------------------------------------------------------
# Combined dynamic factor computation
# ---------------------------------------------------------------------------

def compute_dynamic_factors(
    returns: pd.DataFrame,
    window: int = 252,
    n_components: int = 3,
    min_periods: int = 60,
    kalman_auto_tune: bool = True,
    snapshot_freq: int = 21,
    standardize: bool = True,
) -> DynamicFactorResult:
    """
    Full pipeline: Rolling PCA → Kalman Filter → Composite Stress Index.

    The composite stress index is a variance-weighted combination of
    Kalman-filtered factor scores (weighted by their rolling explained variance).

    Parameters
    ----------
    returns : T × N return DataFrame
    window : rolling estimation window for PCA
    n_components : number of latent factors to extract
    min_periods : minimum observations before computation
    kalman_auto_tune : estimate Kalman parameters from data
    snapshot_freq : frequency of loading snapshots (days)
    standardize : standardize columns before PCA

    Returns
    -------
    DynamicFactorResult
    """
    # Stage 1: Rolling PCA
    pca_result = compute_rolling_pca(
        returns,
        window=window,
        n_components=n_components,
        min_periods=min_periods,
        snapshot_freq=snapshot_freq,
        standardize=standardize,
    )

    # Stage 2: Kalman filter on each factor score
    kalman_results = {}
    for k in range(pca_result.n_components):
        fname = f"F{k+1}"
        fs = pca_result.factor_scores[fname]
        kf = kalman_filter_local_level(fs, auto_tune=kalman_auto_tune)
        kalman_results[fname] = kf

    # Stage 3: Composite stress index
    # Weight = average explained variance of each factor
    ev = pca_result.explained_variance.rolling(window, min_periods=30).mean()

    composite_parts = []
    weights_used = []
    for k in range(pca_result.n_components):
        fname = f"F{k+1}"
        kf_filtered = kalman_results[fname].filtered
        weight = ev[fname].fillna(1.0 / pca_result.n_components)
        composite_parts.append(kf_filtered.abs() * weight)
        weights_used.append(weight)

    composite_stress = sum(composite_parts)
    if composite_stress is not int:
        composite_stress = composite_stress / (sum(weights_used) + 1e-12)
    composite_stress.name = "composite_stress"

    # Factor regime flags
    factor_regime_flags = pca_result.factor_regime(threshold_sigma=1.5)

    return DynamicFactorResult(
        pca=pca_result,
        kalman=kalman_results,
        composite_stress=composite_stress,
        factor_regime_flags=factor_regime_flags,
    )


# ---------------------------------------------------------------------------
# V2: Kalman filter on R_t (not factor scores)
# ---------------------------------------------------------------------------

def compute_dynamic_factors_v2(
    returns: pd.DataFrame,
    window: int = 252,
    n_components: int = 3,
    min_periods: int = 60,
    lam: float = 0.94,
    q_scale: float = 0.01,
    kalman_auto_tune: bool = True,
    snapshot_freq: int = 21,
    standardize: bool = True,
) -> DynamicFactorResult:
    """
    Full pipeline v2: FastCovariance EWMA R_t → KalmanCorrelation → Rolling PCA
    → Kalman Filter on factor scores → Composite Stress Index.

    Differences from v1:
    - Correlation matrix is EWMA-based (FastCovariance, lam=0.94) instead of
      rolling sample correlation inside PCA.
    - KalmanCorrelation smooths R_hat_t → smoothed R_t (latent correlation state).
    - compute_rolling_pca uses the smoothed R_t via R_t_series parameter.
    - RollingPCAResult.innovation_norm is populated with ||R_hat_t - R_{t|t-1}||_F.

    Parameters
    ----------
    returns : T × N return DataFrame
    window : rolling window passed to compute_rolling_pca (for window_data stats)
    n_components : number of latent factors
    min_periods : minimum observations before computation
    lam : EWMA decay factor for FastCovariance
    q_scale : process noise scale for KalmanCorrelation
    kalman_auto_tune : estimate Kalman parameters from data for factor score filter
    snapshot_freq : frequency of loading snapshots (days)
    standardize : standardize columns before PCA

    Returns
    -------
    DynamicFactorResult with pca.innovation_norm populated.

    Notes
    -----
    innovation_norm[t] = ||R_hat_t - R_{t|t-1}||_F
    Large values signal rapid correlation regime transitions.
    """
    from core.returns import clean_returns
    returns_clean = returns.ffill(limit=3).fillna(0.0)

    # Stage 1: Get EWMA R_hat_t series for every date
    R_hat_series = FastCovariance().fit_rolling(returns_clean, lam=lam)

    # Stage 2: Kalman filter on R_hat_series → smoothed R_t + innovation norms
    kalman_corr_result = KalmanCorrelation().fit(R_hat_series, q_scale=q_scale)
    filtered_R = kalman_corr_result.filtered_R
    inno_norms = kalman_corr_result.innovations_norm

    # Stage 3: Rolling PCA using smoothed R_t
    pca_result = compute_rolling_pca(
        returns_clean,
        window=window,
        n_components=n_components,
        min_periods=min_periods,
        snapshot_freq=snapshot_freq,
        standardize=standardize,
        R_t_series=filtered_R,
    )

    # Attach innovation_norm as a pd.Series on the PCA result
    pca_result.innovation_norm = pd.Series(
        inno_norms, index=returns_clean.index, name="innovation_norm"
    )

    # Stage 4: Kalman filter on each factor score (same as v1)
    kalman_results = {}
    for k in range(pca_result.n_components):
        fname = f"F{k+1}"
        fs = pca_result.factor_scores[fname]
        kf = kalman_filter_local_level(fs, auto_tune=kalman_auto_tune)
        kalman_results[fname] = kf

    # Stage 5: Composite stress index (variance-weighted, same as v1)
    ev = pca_result.explained_variance.rolling(window, min_periods=30).mean()
    composite_parts = []
    weights_used = []
    for k in range(pca_result.n_components):
        fname = f"F{k+1}"
        kf_filtered = kalman_results[fname].filtered
        weight = ev[fname].fillna(1.0 / pca_result.n_components)
        composite_parts.append(kf_filtered.abs() * weight)
        weights_used.append(weight)

    composite_stress = sum(composite_parts)
    if composite_stress is not int:
        composite_stress = composite_stress / (sum(weights_used) + 1e-12)
    composite_stress.name = "composite_stress"

    factor_regime_flags = pca_result.factor_regime(threshold_sigma=1.5)

    return DynamicFactorResult(
        pca=pca_result,
        kalman=kalman_results,
        composite_stress=composite_stress,
        factor_regime_flags=factor_regime_flags,
    )


# ---------------------------------------------------------------------------
# Interpretation utilities
# ---------------------------------------------------------------------------

def loading_heatmap_data(pca: RollingPCAResult) -> pd.DataFrame:
    """
    Return the latest loadings as a DataFrame ready for heatmap plotting.
    Rows = assets, Columns = factors, Values = loadings.
    """
    return pca.loadings_latest.copy()


def factor_attribution(
    pca: RollingPCAResult,
    date: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    At a given date, which assets contribute most to Factor 1?
    Returns absolute loadings sorted descending.
    """
    if date is None:
        ldf = pca.loadings_latest
    else:
        # Find nearest snapshot
        snap_dates = sorted(pca.loadings_history.keys())
        nearest = min(snap_dates, key=lambda d: abs((d - date).days))
        ldf = pca.loadings_history[nearest]

    return ldf["F1"].abs().sort_values(ascending=False)


def innovation_spikes(
    kalman: KalmanResult,
    threshold_sigma: float = 2.0,
) -> pd.DatetimeIndex:
    """
    Dates where Kalman innovations exceed threshold_sigma standard deviations.
    These are candidate structural break dates.
    """
    innov = kalman.innovations.dropna()
    z = (innov - innov.mean()) / innov.std()
    return innov.index[z.abs() > threshold_sigma]
