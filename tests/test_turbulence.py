"""
tests/test_turbulence.py
Unit tests for modules/turbulence.py.

All tests use deterministic synthetic data (seed=42). No live data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules.turbulence import compute_turbulence_index, TurbulenceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(T: int = 300, N: int = 5, daily_vol: float = 0.01, seed: int = 42) -> pd.DataFrame:
    """Gaussian returns with given shape and vol."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=T)
    data = rng.normal(0, daily_vol, size=(T, N))
    cols = [f"A{i}" for i in range(N)]
    return pd.DataFrame(data, index=dates, columns=cols)


# ---------------------------------------------------------------------------
# Session 2 — pure vol spike test
# ---------------------------------------------------------------------------

class TestVolStandardizationEffect:
    """
    Confirm that VolStandardizer suppresses pure vol spikes in turbulence.

    Setup: 700 days of 5-asset Gaussian returns (seed=42, daily vol=0.01).
    Shock: ALL assets from index 600 onwards have 5x vol (sustained regime shift,
      no change in cross-asset correlations).

    Test point: index 650 (50 days into the spike).

    Rationale for T=700 and slow_window=504:
    - EWMA(lam=0.94) is 95% adapted after 50 days → standardized return ≈ 1 sigma
      at day 650 → tau_std ≈ chi2(N) → NOT Crisis.
    - slow_window=504 window at day 650 sees days 146-650 = 454 calm + 50 spike
      (≈10% spike data) → LedoitWolf Sigma stays near calm calibration.
      Raw return (5x) vs. calm-calibrated Sigma → tau_unstd >> threshold.
    - A shorter window (e.g., 200) would include 25%+ spike data, diluting the
      Sigma enough that tau_unstd falls below the Crisis/Turbulent threshold.
    """

    @pytest.fixture(scope="class")
    def returns_with_vol_spike(self) -> pd.DataFrame:
        rets = _make_returns(T=700, N=5, daily_vol=0.01, seed=42)
        # Sustained 5x vol block: days 600-699, same correlation structure
        rets.iloc[600:] = rets.iloc[600:] * 5.0
        return rets

    def test_vol_spike_suppressed_when_standardized(self, returns_with_vol_spike):
        """
        At day 650 (50 days into the spike), EWMA vol is fully adapted to 5x regime
        → standardized returns ≈ 1 sigma → tau ≈ chi2(N) → NOT Crisis.
        """
        result = compute_turbulence_index(
            returns_with_vol_spike,
            vol_standardize=True,
            slow_window=504,
            min_periods=30,
        )
        spike_date = returns_with_vol_spike.index[650]
        regime_at_spike = result.regime.loc[spike_date]
        assert regime_at_spike != "Crisis", (
            f"Expected sustained vol spike to not be 'Crisis' with vol_standardize=True "
            f"after 50 days of EWMA adaptation, got '{regime_at_spike}' "
            f"(tau={result.turbulence.loc[spike_date]:.2f})"
        )

    def test_vol_spike_detected_without_standardization(self, returns_with_vol_spike):
        """
        Without standardization, 5x returns vs. mostly-calm SlowCovariance Sigma
        → tau >> threshold → Turbulent or Crisis.
        """
        result = compute_turbulence_index(
            returns_with_vol_spike,
            vol_standardize=False,
            slow_window=504,
            min_periods=30,
        )
        spike_date = returns_with_vol_spike.index[650]
        regime_at_spike = result.regime.loc[spike_date]
        assert regime_at_spike in ("Crisis", "Turbulent"), (
            f"Expected vol spike to register as 'Crisis' or 'Turbulent' "
            f"with vol_standardize=False, got '{regime_at_spike}' "
            f"(tau={result.turbulence.loc[spike_date]:.2f})"
        )


# ---------------------------------------------------------------------------
# Smoke tests — basic output integrity
# ---------------------------------------------------------------------------

class TestTurbulenceBasic:

    @pytest.fixture(scope="class")
    def result(self) -> TurbulenceResult:
        rets = _make_returns(T=300, N=5, seed=42)
        return compute_turbulence_index(rets, slow_window=200, min_periods=60)

    def test_returns_turbulence_result(self, result):
        assert isinstance(result, TurbulenceResult)

    def test_turbulence_series_length(self, result):
        assert len(result.turbulence) == 300

    def test_non_nan_after_min_periods(self, result):
        """Should have valid scores after min_periods."""
        valid = result.turbulence.dropna()
        assert len(valid) > 0

    def test_all_regime_values_known(self, result):
        known = {"Calm", "Elevated", "Turbulent", "Crisis", float("nan")}
        unique = set(result.regime.dropna().unique())
        assert unique.issubset({"Calm", "Elevated", "Turbulent", "Crisis"})

    def test_turbulence_non_negative(self, result):
        """τ = (r-μ)'Σ⁻¹(r-μ) is a quadratic form — must be non-negative."""
        valid = result.turbulence.dropna()
        assert (valid >= 0).all(), f"Negative τ values found: {valid[valid < 0]}"

    def test_decomposition_components_non_negative(self, result):
        """
        Both magnitude and correlation components must be non-negative after floor.

        Note: magnitude + correlation ≠ turbulence in general. For positively-
        correlated EM assets, Σ⁻¹ has negative off-diagonal elements. The
        off-diagonal quadratic form diff'(Σ⁻¹ - diag(Σ⁻¹))diff can be negative
        (when returns all move in the same direction), so tau_mag can exceed
        tau_total. The floor at 0 in _decompose_mahalanobis ensures the reported
        correlation component is non-negative, but breaks additive identity.
        """
        valid_idx = result.magnitude_component.dropna().index
        assert (result.magnitude_component.loc[valid_idx] >= 0).all(), (
            "magnitude_component has negative values"
        )
        corr_idx = result.correlation_component.dropna().index
        assert (result.correlation_component.loc[corr_idx] >= 0).all(), (
            "correlation_component has negative values (floor should prevent this)"
        )

    def test_regime_code_consistent_with_regime(self, result):
        mapping = {"Calm": 0, "Elevated": 1, "Turbulent": 2, "Crisis": 3}
        for date, label in result.regime.dropna().items():
            assert result.regime_code.loc[date] == mapping[label]

    def test_n_assets_correct(self, result):
        assert result.n_assets == 5

    def test_asset_names_correct(self, result):
        assert result.asset_names == [f"A{i}" for i in range(5)]

    def test_to_frame_shape(self, result):
        df = result.to_frame()
        assert df.shape[0] == 300
        assert "turbulence" in df.columns
        assert "regime" in df.columns
