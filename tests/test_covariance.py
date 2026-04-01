"""
tests/test_covariance.py
Unit tests for core/covariance.py (Session 1).

All tests use deterministic synthetic data (seed=42).
No live data. No external API calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.synthetic import generate_em_universe
from core.covariance import SlowCovariance, FastCovariance, VolStandardizer, KalmanCorrelation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def panel_500() -> pd.DataFrame:
    """500-observation panel from synthetic EM universe, seed=42."""
    universe = generate_em_universe(seed=42)
    return universe.panel.iloc[:500]


@pytest.fixture(scope="module")
def panel_with_nans(panel_500) -> pd.DataFrame:
    """panel_500 with ~5% of values replaced by NaN."""
    rng = np.random.default_rng(0)
    df = panel_500.copy().astype(float)
    mask = rng.random(df.shape) < 0.05
    df.values[mask] = np.nan
    return df


@pytest.fixture(scope="module")
def slow_result(panel_500):
    return SlowCovariance().fit(panel_500)


@pytest.fixture(scope="module")
def fast_result(panel_500):
    return FastCovariance().fit(panel_500)


@pytest.fixture(scope="module")
def vol_std_result(panel_500):
    return VolStandardizer().fit_transform(panel_500)


# ---------------------------------------------------------------------------
# SlowCovariance tests
# ---------------------------------------------------------------------------

class TestSlowCovariance:

    def test_sigma_psd(self, slow_result):
        """All eigenvalues of sigma must be > -1e-10 (positive semi-definite)."""
        eigvals = np.linalg.eigvalsh(slow_result.sigma)
        assert np.all(eigvals > -1e-10), (
            f"sigma has negative eigenvalues: {eigvals[eigvals <= -1e-10]}"
        )

    def test_sigma_inv_is_left_inverse(self, slow_result):
        """sigma_inv @ sigma should be close to identity (atol=1e-6)."""
        N = slow_result.sigma.shape[0]
        product = slow_result.sigma_inv @ slow_result.sigma
        assert np.allclose(product, np.eye(N), atol=1e-6), (
            f"Max deviation from identity: {np.abs(product - np.eye(N)).max():.2e}"
        )

    def test_mu_shape(self, slow_result, panel_500):
        N = panel_500.shape[1]
        assert slow_result.mu.shape == (N,)

    def test_sigma_shape(self, slow_result, panel_500):
        N = panel_500.shape[1]
        assert slow_result.sigma.shape == (N, N)
        assert slow_result.sigma_inv.shape == (N, N)

    def test_raises_on_insufficient_samples(self, panel_500):
        """Raise ValueError when n_samples < n_features."""
        N = panel_500.shape[1]
        tiny = panel_500.iloc[: N - 1]  # fewer rows than columns
        with pytest.raises(ValueError, match="n_samples"):
            SlowCovariance().fit(tiny, window=N - 1)

    def test_no_crash_with_nans(self, panel_with_nans):
        """Should not raise on input with ~5% NaN."""
        result = SlowCovariance().fit(panel_with_nans)
        assert result.sigma is not None

    def test_sigma_symmetric(self, slow_result):
        assert np.allclose(slow_result.sigma, slow_result.sigma.T, atol=1e-12)


# ---------------------------------------------------------------------------
# FastCovariance tests
# ---------------------------------------------------------------------------

class TestFastCovariance:

    def test_R_t_unit_diagonal(self, fast_result):
        """R_t must have unit diagonal."""
        N = fast_result.R_t.shape[0]
        assert np.allclose(np.diag(fast_result.R_t), np.ones(N), atol=1e-10), (
            f"Diagonal deviates from 1: {np.diag(fast_result.R_t)}"
        )

    def test_R_t_off_diagonal_bounded(self, fast_result):
        """All off-diagonal elements of R_t must be in [-1, 1]."""
        R = fast_result.R_t.copy()
        np.fill_diagonal(R, 0.0)
        assert np.all(R >= -1.0) and np.all(R <= 1.0), (
            f"Off-diagonal out of bounds: min={R.min():.4f}, max={R.max():.4f}"
        )

    def test_R_t_psd(self, fast_result):
        """R_t must be positive semi-definite."""
        eigvals = np.linalg.eigvalsh(fast_result.R_t)
        assert np.all(eigvals >= -1e-10), (
            f"R_t has negative eigenvalues: {eigvals[eigvals < -1e-10]}"
        )

    def test_D_t_nonnegative_diagonal(self, fast_result):
        """D_t diagonal entries (conditional vols) must be non-negative."""
        d = np.diag(fast_result.D_t)
        assert np.all(d >= 0.0)

    def test_sigma_t_decomposition(self, fast_result):
        """sigma_t should equal D_t @ R_t @ D_t (up to float precision)."""
        D = fast_result.D_t
        R = fast_result.R_t
        reconstructed = D @ R @ D
        assert np.allclose(fast_result.sigma_t, reconstructed, atol=1e-10), (
            f"Max deviation: {np.abs(fast_result.sigma_t - reconstructed).max():.2e}"
        )

    def test_no_crash_with_nans(self, panel_with_nans):
        """Should not raise on input with ~5% NaN."""
        result = FastCovariance().fit(panel_with_nans)
        assert result.R_t is not None


# ---------------------------------------------------------------------------
# VolStandardizer tests
# ---------------------------------------------------------------------------

class TestVolStandardizer:

    def test_output_shape_and_index(self, vol_std_result, panel_500):
        assert vol_std_result.shape == panel_500.shape
        assert list(vol_std_result.columns) == list(panel_500.columns)
        assert (vol_std_result.index == panel_500.index).all()

    def test_first_min_periods_are_nan(self, vol_std_result):
        """First 30 rows should be NaN."""
        assert vol_std_result.iloc[:30].isna().all().all(), (
            "Expected first 30 rows to be NaN."
        )

    def test_remaining_rows_not_nan(self, vol_std_result):
        """Rows 30+ should have no NaN."""
        tail = vol_std_result.iloc[30:]
        assert not tail.isna().any().any(), (
            "Unexpected NaN in standardized returns after min_periods."
        )

    def test_column_stds_near_unity(self, vol_std_result):
        """
        After vol-standardization, each column std should be in [0.8, 1.2]
        on 500-obs synthetic data.
        """
        stds = vol_std_result.iloc[30:].std()
        lo, hi = 0.8, 1.2
        out_of_range = stds[(stds < lo) | (stds > hi)]
        assert len(out_of_range) == 0, (
            f"Columns with std outside [{lo}, {hi}]:\n{out_of_range}"
        )

    def test_no_crash_with_nans(self, panel_with_nans):
        """Should not raise on input with ~5% NaN."""
        result = VolStandardizer().fit_transform(panel_with_nans)
        assert result is not None
        assert result.shape == panel_with_nans.shape


# ---------------------------------------------------------------------------
# KalmanCorrelation tests (Session 4)
# ---------------------------------------------------------------------------

class TestKalmanCorrelation:
    """
    Tests for KalmanCorrelation on synthetic EM data (seed=42).

    Uses FastCovariance.fit_rolling to generate the R_hat_t input series,
    then verifies structural properties and empirical stress detection.
    """

    @pytest.fixture(scope="class")
    def universe(self):
        return generate_em_universe(seed=42)

    @pytest.fixture(scope="class")
    def R_hat_series(self, universe):
        """EWMA R_t series for the full panel, lam=0.94."""
        return FastCovariance().fit_rolling(universe.panel, lam=0.94)

    @pytest.fixture(scope="class")
    def kalman_result(self, R_hat_series):
        return KalmanCorrelation().fit(R_hat_series, q_scale=0.01)

    # --- Structural tests ---

    def test_output_length(self, kalman_result, universe):
        T = len(universe.panel)
        assert len(kalman_result.filtered_R) == T
        assert len(kalman_result.innovations_norm) == T

    def test_filtered_R_unit_diagonal(self, kalman_result):
        """All filtered correlation matrices must have diagonal = 1.0."""
        for t, R in enumerate(kalman_result.filtered_R):
            diag = np.diag(R)
            assert np.allclose(diag, np.ones(len(diag)), atol=1e-10), (
                f"Diagonal ≠ 1 at t={t}: {diag}"
            )

    def test_filtered_R_psd(self, kalman_result):
        """All filtered correlation matrices must be PSD."""
        failures = []
        for t, R in enumerate(kalman_result.filtered_R):
            eigvals = np.linalg.eigvalsh(R)
            if np.any(eigvals < -1e-10):
                failures.append((t, eigvals.min()))
        assert len(failures) == 0, (
            f"PSD violated at {len(failures)} dates, e.g. t={failures[0][0]}, "
            f"min_eigval={failures[0][1]:.2e}"
        )

    def test_filtered_R_off_diagonal_bounded(self, kalman_result):
        """Off-diagonal elements must be in [-1, 1]."""
        N = kalman_result.filtered_R[0].shape[0]
        for t, R in enumerate(kalman_result.filtered_R):
            R_off = R.copy()
            np.fill_diagonal(R_off, 0.0)
            assert np.all(R_off >= -1.0) and np.all(R_off <= 1.0), (
                f"Off-diagonal out of bounds at t={t}"
            )

    def test_innovations_norm_non_negative(self, kalman_result):
        """Frobenius norm is always non-negative."""
        norms = np.array(kalman_result.innovations_norm)
        assert np.all(norms >= 0.0), (
            f"Negative innovation norms: {norms[norms < 0]}"
        )

    def test_innovations_norm_positive_floats(self, kalman_result):
        """All innovation norms should be strictly positive (norm of non-zero matrix)."""
        norms = np.array(kalman_result.innovations_norm)
        assert np.all(norms >= 0.0)
        # Almost all should be strictly positive; allow at most 1 exact zero
        assert np.sum(norms == 0.0) <= 1, (
            f"Too many zero innovation norms: {np.sum(norms == 0.0)}"
        )

    # --- Empirical test: stress episodes have higher innovation norm ---

    def test_innovation_norm_elevated_during_stress(self):
        """
        KalmanCorrelation should show elevated innovation_norm during a
        correlation regime shift.

        Setup: constructed R_hat_series (N=5, T=300):
          - t=0..99:   calm, all off-diagonals ≈ 0.20 + small noise
          - t=100..199: stress, all off-diagonals ≈ 0.80 + small noise
          - t=200..299: calm again, all off-diagonals ≈ 0.20 + small noise

        The Kalman filter initializes near the calm state. The stress-episode
        matrices are far from the filter's one-step-ahead prediction, so
        innovation_norm should be large during the transition.

        This tests KalmanCorrelation in isolation from EWMA noise:
        the EWMA output (from FastCovariance on synthetic data) adds daily
        t-distributed noise that dilutes the calm/stress contrast.
        """
        rng = np.random.default_rng(7)
        N = 5
        T_calm = 100
        T_stress = 100

        def make_corr(rho: float, noise_scale: float = 0.01) -> np.ndarray:
            R = np.full((N, N), rho)
            np.fill_diagonal(R, 1.0)
            noise = rng.normal(0, noise_scale, (N, N))
            noise = (noise + noise.T) / 2.0
            np.fill_diagonal(noise, 0.0)
            R = R + noise
            R = np.clip(R, -0.999, 0.999)
            np.fill_diagonal(R, 1.0)
            from core.covariance import _enforce_psd_correlation
            return _enforce_psd_correlation(R)

        calm_mats   = [make_corr(0.20) for _ in range(T_calm)]
        stress_mats = [make_corr(0.80) for _ in range(T_stress)]
        calm2_mats  = [make_corr(0.20) for _ in range(T_calm)]

        R_hat_series = calm_mats + stress_mats + calm2_mats

        result = KalmanCorrelation().fit(R_hat_series, q_scale=0.01)
        norms = np.array(result.innovations_norm)

        mean_calm   = norms[:T_calm].mean()
        mean_stress = norms[T_calm : T_calm + T_stress].mean()

        assert mean_stress > 2.0 * mean_calm, (
            f"Expected stress mean ({mean_stress:.4f}) > 2x calm mean ({mean_calm:.4f}). "
            f"Ratio: {mean_stress / mean_calm:.2f}x. "
            "KalmanCorrelation should detect the 0.2→0.8 correlation regime shift."
        )
