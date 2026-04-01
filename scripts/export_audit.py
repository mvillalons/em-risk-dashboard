"""
scripts/export_audit.py
Export all computed EM risk time series to CSV for manual inspection.

Usage
-----
  python scripts/export_audit.py [options]

Examples
--------
  python scripts/export_audit.py --mode both
  python scripts/export_audit.py --mode live --start 2018-01-01 --lam 0.97
  python scripts/export_audit.py --mode synthetic --slow-window 252 --no-vol-std
  python scripts/export_audit.py --mode both --output-dir my_exports/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from core.covariance import VolStandardizer
from core.returns import log_returns
from data.synthetic import generate_em_universe
from modules.absorption import compute_absorption_ratio, AbsorptionResult
from modules.pca_kalman import compute_dynamic_factors_v2, DynamicFactorResult
from modules.turbulence import compute_turbulence_index, TurbulenceResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class AuditParams:
    """All CLI arguments in one typed container."""
    mode: str
    start: str
    slow_window: int
    min_periods: int
    lam: float
    vol_std: bool
    output_dir: Path

    def as_json(self) -> str:
        """Serialize to JSON string for embedding in CSV metadata."""
        return json.dumps({
            "mode":        self.mode,
            "start":       self.start,
            "slow_window": self.slow_window,
            "min_periods": self.min_periods,
            "lam":         self.lam,
            "vol_std":     self.vol_std,
            "output_dir":  str(self.output_dir),
        })


# ---------------------------------------------------------------------------
# Computed results container
# ---------------------------------------------------------------------------

@dataclass
class ComputedResults:
    """All outputs from one mode's pipeline run."""
    data_source: str
    fx_ret:        pd.DataFrame
    eq_ret:        pd.DataFrame
    panel:         pd.DataFrame
    prices_fx:     pd.DataFrame
    prices_eq:     pd.DataFrame
    vol_std_ret:   Optional[pd.DataFrame]
    turb_panel:    TurbulenceResult
    turb_fx:       TurbulenceResult
    turb_eq:       TurbulenceResult
    ar_fx:         AbsorptionResult
    dyn:           DynamicFactorResult
    country_turb:  dict[str, TurbulenceResult]
    ticker_quality: Optional[pd.DataFrame]     # live mode only
    summary_vals:  dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_synthetic(params: AuditParams) -> ComputedResults:
    """Load data from generate_em_universe(seed=42) and run full pipeline."""
    logger.info("Loading synthetic data  start=%s", params.start)
    uni = generate_em_universe(start=params.start, seed=42)
    fx_ret    = uni.fx
    eq_ret    = uni.equity
    panel     = uni.panel
    prices_fx = uni.prices_fx
    prices_eq = uni.prices_eq

    return _run_pipeline(
        data_source="synthetic",
        fx_ret=fx_ret,
        eq_ret=eq_ret,
        panel=panel,
        prices_fx=prices_fx,
        prices_eq=prices_eq,
        ticker_quality=None,
        params=params,
    )


def _load_live(params: AuditParams) -> Optional[ComputedResults]:
    """
    Load live data via data/fetcher.py, compute quality stats, run pipeline.

    Returns None if the top-level fetch fails entirely. Individual ticker
    failures are handled gracefully — columns with all-NaN are dropped with
    a warning; remaining columns proceed normally.
    """
    logger.info("Loading live data  start=%s", params.start)

    try:
        from data.fetcher import load_em_universe
        raw = load_em_universe(start=params.start)
    except Exception as exc:
        logger.warning("Live fetch failed entirely: %s", exc)
        return None

    # ---- Compute log returns with graceful per-column failure handling ----
    def _safe_log_returns(prices: pd.DataFrame, asset_class: str) -> pd.DataFrame:
        ret = log_returns(prices)
        bad = [c for c in ret.columns if ret[c].isna().all()]
        if bad:
            logger.warning("Dropping all-NaN %s columns: %s", asset_class, bad)
            ret = ret.drop(columns=bad)
        return ret

    fx_prices  = raw.get("fx",     pd.DataFrame())
    eq_prices  = raw.get("equity", pd.DataFrame())
    gl_prices  = raw.get("global", pd.DataFrame())

    fx_ret = _safe_log_returns(fx_prices,  "FX")     if not fx_prices.empty  else pd.DataFrame()
    eq_ret = _safe_log_returns(eq_prices,  "equity") if not eq_prices.empty  else pd.DataFrame()
    gl_ret = _safe_log_returns(gl_prices,  "global") if not gl_prices.empty  else pd.DataFrame()

    if fx_ret.empty and eq_ret.empty:
        logger.warning("Live data produced no usable FX or equity returns. Skipping live mode.")
        return None

    panel_parts = [df for df in [fx_ret, eq_ret, gl_ret] if not df.empty]
    panel = pd.concat(panel_parts, axis=1).dropna(how="all")

    # Use price series as proxy for rebased levels
    prices_fx = (1 + fx_ret).cumprod() * 100 if not fx_ret.empty else pd.DataFrame()
    prices_eq = (1 + eq_ret).cumprod() * 100 if not eq_ret.empty else pd.DataFrame()

    # ---- Data quality stats ----
    all_prices: dict[str, tuple[pd.Series, str]] = {}
    for col in fx_prices.columns:
        all_prices[col] = (fx_prices[col], "fx")
    for col in eq_prices.columns:
        all_prices[col] = (eq_prices[col], "equity")
    for col in gl_prices.columns:
        all_prices[col] = (gl_prices[col], "global")

    all_rets: dict[str, pd.Series] = {}
    for col in fx_ret.columns:
        all_rets[col] = fx_ret[col]
    for col in eq_ret.columns:
        all_rets[col] = eq_ret[col]
    for col in gl_ret.columns:
        all_rets[col] = gl_ret[col]

    ticker_quality = _compute_ticker_quality(all_prices, all_rets)

    return _run_pipeline(
        data_source="live",
        fx_ret=fx_ret,
        eq_ret=eq_ret,
        panel=panel,
        prices_fx=prices_fx,
        prices_eq=prices_eq,
        ticker_quality=ticker_quality,
        params=params,
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(
    data_source: str,
    fx_ret: pd.DataFrame,
    eq_ret: pd.DataFrame,
    panel: pd.DataFrame,
    prices_fx: pd.DataFrame,
    prices_eq: pd.DataFrame,
    ticker_quality: Optional[pd.DataFrame],
    params: AuditParams,
) -> ComputedResults:
    """Run the full compute pipeline on pre-loaded return DataFrames."""
    slow_w = max(params.slow_window, params.min_periods + 1)

    # ---- Vol-standardized returns (full panel) ----
    vol_std_ret: Optional[pd.DataFrame] = None
    if params.vol_std and not panel.empty:
        logger.info("  Computing VolStandardizer on panel (%d×%d)", *panel.shape)
        vol_std_ret = VolStandardizer().fit_transform(panel, lam=params.lam)

    # ---- Turbulence ----
    def _turb(df: pd.DataFrame, label: str) -> Optional[TurbulenceResult]:
        if df.empty:
            logger.warning("  Skipping turbulence for %s (empty DataFrame)", label)
            return None
        logger.info("  Computing turbulence [%s]  shape=%s", label, df.shape)
        try:
            return compute_turbulence_index(
                df,
                window=params.slow_window,
                min_periods=params.min_periods,
                vol_standardize=params.vol_std,
                slow_window=slow_w,
            )
        except Exception as exc:
            logger.warning("  Turbulence [%s] failed: %s", label, exc)
            return None

    turb_panel = _turb(panel,  "panel")
    turb_fx    = _turb(fx_ret, "FX")
    turb_eq    = _turb(eq_ret, "equity")

    # ---- Absorption Ratio (FX) ----
    ar_fx: Optional[AbsorptionResult] = None
    if not fx_ret.empty:
        logger.info("  Computing Absorption Ratio [FX]")
        try:
            ar_fx = compute_absorption_ratio(
                fx_ret,
                window=params.slow_window,
                min_periods=params.min_periods,
                lam=params.lam,
            )
        except Exception as exc:
            logger.warning("  Absorption Ratio failed: %s", exc)

    # ---- Dynamic factors v2 (KalmanCorrelation on panel) ----
    dyn: Optional[DynamicFactorResult] = None
    if not panel.empty and panel.shape[1] >= 3:
        logger.info("  Computing dynamic factors v2  panel=%s", panel.shape)
        try:
            dyn = compute_dynamic_factors_v2(
                panel,
                window=params.slow_window,
                n_components=3,
                min_periods=params.min_periods,
                lam=params.lam,
            )
        except Exception as exc:
            logger.warning("  Dynamic factors v2 failed: %s", exc)

    # ---- Per-country turbulence ----
    country_turb: dict[str, TurbulenceResult] = {}
    if not fx_ret.empty and not eq_ret.empty:
        fx_cols = list(fx_ret.columns)
        eq_cols = list(eq_ret.columns)
        for i, ctry in enumerate(fx_cols):
            if i >= len(eq_cols):
                break
            pair = pd.concat([fx_ret[[ctry]], eq_ret[[eq_cols[i]]]], axis=1)
            try:
                ct = compute_turbulence_index(
                    pair,
                    window=params.slow_window,
                    min_periods=params.min_periods,
                    vol_standardize=params.vol_std,
                    slow_window=slow_w,
                )
                country_turb[ctry] = ct
                logger.info("  Country turbulence [%s] OK", ctry)
            except Exception as exc:
                logger.warning("  Country turbulence [%s] failed: %s", ctry, exc)

    results = ComputedResults(
        data_source=data_source,
        fx_ret=fx_ret,
        eq_ret=eq_ret,
        panel=panel,
        prices_fx=prices_fx,
        prices_eq=prices_eq,
        vol_std_ret=vol_std_ret,
        turb_panel=turb_panel,
        turb_fx=turb_fx,
        turb_eq=turb_eq,
        ar_fx=ar_fx,
        dyn=dyn,
        country_turb=country_turb,
        ticker_quality=ticker_quality,
    )
    results.summary_vals = _collect_summary_vals(results)
    return results


# ---------------------------------------------------------------------------
# Ticker quality (live mode)
# ---------------------------------------------------------------------------

def _max_consecutive_gap_days(series: pd.Series) -> int:
    """
    Compute the maximum calendar-day span covered by a consecutive NaN run.

    Parameters
    ----------
    series : price series with DatetimeIndex

    Returns
    -------
    Maximum gap in calendar days (0 if no NaN values).
    """
    if not series.isna().any():
        return 0
    max_gap = 0
    gap_start: Optional[pd.Timestamp] = None
    for ts, val in series.items():
        if pd.isna(val):
            if gap_start is None:
                gap_start = ts
        else:
            if gap_start is not None:
                gap_days = (ts - gap_start).days
                max_gap = max(max_gap, gap_days)
                gap_start = None
    if gap_start is not None:
        gap_days = (series.index[-1] - gap_start).days
        max_gap = max(max_gap, gap_days)
    return max_gap


def _compute_ticker_quality(
    all_prices: dict[str, tuple[pd.Series, str]],
    all_rets: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Compute per-ticker data quality statistics and flag anomalies.

    Parameters
    ----------
    all_prices : {ticker: (price_series, asset_class)}
    all_rets   : {ticker: return_series} — may be empty if fetch failed

    Returns
    -------
    pd.DataFrame with one row per ticker and quality columns.
    """
    rows = []
    for ticker, (prices, asset_class) in all_prices.items():
        n_total   = len(prices)
        nan_count = int(prices.isna().sum())
        nan_pct   = nan_count / n_total if n_total > 0 else 1.0
        valid     = prices.dropna()
        n_obs     = len(valid)
        first_dt  = str(valid.index[0].date()) if n_obs > 0 else ""
        last_dt   = str(valid.index[-1].date()) if n_obs > 0 else ""

        if nan_count == n_total:
            fetch_status = "all_nan"
        elif nan_count > 0:
            fetch_status = "partial_nan"
        else:
            fetch_status = "ok"

        max_gap = _max_consecutive_gap_days(prices)

        ret_series = all_rets.get(ticker, pd.Series(dtype=float))
        ret_clean  = ret_series.dropna()
        if len(ret_clean) > 10:
            ann_vol   = float(ret_clean.std() * np.sqrt(252))
            skewness  = float(sp_stats.skew(ret_clean.values))
            exc_kurt  = float(sp_stats.kurtosis(ret_clean.values))
        else:
            ann_vol  = float("nan")
            skewness = float("nan")
            exc_kurt = float("nan")

        rows.append({
            "ticker":                  ticker,
            "asset_class":             asset_class,
            "fetch_status":            fetch_status,
            "n_obs":                   n_obs,
            "first_date":              first_dt,
            "last_date":               last_dt,
            "nan_count":               nan_count,
            "nan_pct":                 round(nan_pct * 100, 3),
            "max_consecutive_gap_days": max_gap,
            "annualized_vol":          round(ann_vol * 100, 2) if not np.isnan(ann_vol) else float("nan"),
            "skewness":                round(skewness, 4) if not np.isnan(skewness) else float("nan"),
            "excess_kurtosis":         round(exc_kurt, 4) if not np.isnan(exc_kurt) else float("nan"),
        })

    df = pd.DataFrame(rows)

    # ---- Flag anomalies to terminal ----
    for _, row in df.iterrows():
        t  = row["ticker"]
        ac = row["asset_class"]
        flags: list[str] = []

        if row["nan_pct"] > 2.0:
            flags.append(f"nan_pct={row['nan_pct']:.1f}% > 2%")
        if row["max_consecutive_gap_days"] > 5:
            flags.append(f"max_gap={row['max_consecutive_gap_days']}d > 5d")

        vol = row["annualized_vol"]
        if not np.isnan(vol):
            if ac == "fx" and not (5.0 <= vol <= 35.0):
                flags.append(f"FX vol={vol:.1f}% outside [5,35]%")
            if ac == "equity" and not (10.0 <= vol <= 50.0):
                flags.append(f"equity vol={vol:.1f}% outside [10,50]%")

        if flags:
            print(f"  [DATA QUALITY FLAG]  {t}: {' | '.join(flags)}")

    return df


# ---------------------------------------------------------------------------
# Summary value collector
# ---------------------------------------------------------------------------

def _collect_summary_vals(r: ComputedResults) -> dict[str, float]:
    """
    Extract a flat dict of current values for comparison table.

    Parameters
    ----------
    r : ComputedResults

    Returns
    -------
    dict mapping metric_name → float (NaN where unavailable)
    """
    vals: dict[str, float] = {}

    def _safe_last(s: Optional[pd.Series]) -> float:
        if s is None:
            return float("nan")
        clean = s.dropna()
        return float(clean.iloc[-1]) if len(clean) > 0 else float("nan")

    def _safe_mean(s: Optional[pd.Series]) -> float:
        if s is None:
            return float("nan")
        clean = s.dropna()
        return float(clean.mean()) if len(clean) > 0 else float("nan")

    for label, tr in [
        ("turbulence_panel", r.turb_panel),
        ("turbulence_fx",    r.turb_fx),
        ("turbulence_eq",    r.turb_eq),
    ]:
        if tr is not None:
            vals[f"{label}_current"] = _safe_last(tr.turbulence)
            vals[f"{label}_mean"]    = _safe_mean(tr.turbulence)
        else:
            vals[f"{label}_current"] = float("nan")
            vals[f"{label}_mean"]    = float("nan")

    if r.ar_fx is not None:
        vals["ar_fx_current"] = _safe_last(r.ar_fx.absorption_ratio)
        vals["ar_fx_mean"]    = _safe_mean(r.ar_fx.absorption_ratio)
    else:
        vals["ar_fx_current"] = float("nan")
        vals["ar_fx_mean"]    = float("nan")

    if r.dyn is not None:
        inorm = r.dyn.pca.innovation_norm
        vals["innovation_norm_current"] = _safe_last(inorm)
        vals["innovation_norm_mean"]    = _safe_mean(inorm)
    else:
        vals["innovation_norm_current"] = float("nan")
        vals["innovation_norm_mean"]    = float("nan")

    return vals


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _series_stats(s: pd.Series) -> dict[str, float]:
    """Return mean/std/min/max over non-NaN values."""
    clean = s.dropna()
    if len(clean) == 0:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(clean.mean()),
        "std":  float(clean.std()),
        "min":  float(clean.min()),
        "max":  float(clean.max()),
    }


def _export_summary(
    results: ComputedResults,
    out_dir: Path,
    params: AuditParams,
    timestamp: str,
) -> None:
    """Export 00_summary.csv — one row per metric."""
    rows = []
    params_json = params.as_json()

    def _add(
        metric: str,
        series: Optional[pd.Series],
        current_regime: str = "",
        current_pctile: float = float("nan"),
    ) -> None:
        if series is None or len(series.dropna()) == 0:
            rows.append({
                "metric":              metric,
                "current_value":       float("nan"),
                "current_regime":      current_regime,
                "current_percentile":  float("nan"),
                "series_mean":         float("nan"),
                "series_std":          float("nan"),
                "series_min":          float("nan"),
                "series_max":          float("nan"),
                "data_source":         results.data_source,
                "parameters_used":     params_json,
                "export_timestamp":    timestamp,
            })
            return
        clean = series.dropna()
        st    = _series_stats(clean)
        rows.append({
            "metric":              metric,
            "current_value":       float(clean.iloc[-1]),
            "current_regime":      current_regime,
            "current_percentile":  current_pctile,
            "series_mean":         st["mean"],
            "series_std":          st["std"],
            "series_min":          st["min"],
            "series_max":          st["max"],
            "data_source":         results.data_source,
            "parameters_used":     params_json,
            "export_timestamp":    timestamp,
        })

    for label, tr in [
        ("turbulence_panel", results.turb_panel),
        ("turbulence_fx",    results.turb_fx),
        ("turbulence_eq",    results.turb_eq),
    ]:
        if tr is not None:
            _add(label, tr.turbulence,
                 current_regime=tr.current_regime(),
                 current_pctile=tr.current_pctile())
            _add(f"{label}_log",   tr.log_turbulence)
            _add(f"{label}_mag",   tr.magnitude_component)
            _add(f"{label}_corr",  tr.correlation_component)
        else:
            _add(label, None)

    if results.ar_fx is not None:
        _add("absorption_ratio_fx",  results.ar_fx.absorption_ratio)
        _add("delta_ar_fx",          results.ar_fx.delta_ar)
        _add("std_delta_ar_fx",      results.ar_fx.standardized_delta)

    if results.dyn is not None:
        _add("innovation_norm", results.dyn.pca.innovation_norm)
        for k in range(results.dyn.pca.n_components):
            fname = f"F{k+1}"
            _add(f"factor_score_{fname}",
                 results.dyn.pca.factor_scores[fname]
                 if fname in results.dyn.pca.factor_scores.columns else None)

    pd.DataFrame(rows).to_csv(out_dir / "00_summary.csv", index=False)
    logger.info("  → 00_summary.csv  (%d rows)", len(rows))


def _export_data_quality(results: ComputedResults, out_dir: Path) -> None:
    """Export 00_data_quality.csv (live mode only)."""
    if results.ticker_quality is None:
        return
    results.ticker_quality.to_csv(out_dir / "00_data_quality.csv", index=False)
    logger.info("  → 00_data_quality.csv  (%d tickers)", len(results.ticker_quality))


def _export_returns(results: ComputedResults, out_dir: Path) -> None:
    """Export 01/02 raw return CSVs."""
    if not results.fx_ret.empty:
        results.fx_ret.to_csv(out_dir / "01_raw_returns_fx.csv")
        logger.info("  → 01_raw_returns_fx.csv  %s", results.fx_ret.shape)

    if not results.eq_ret.empty:
        results.eq_ret.to_csv(out_dir / "02_raw_returns_equity.csv")
        logger.info("  → 02_raw_returns_equity.csv  %s", results.eq_ret.shape)


def _export_vol_std(results: ComputedResults, out_dir: Path) -> None:
    """Export 03_vol_standardized_returns.csv if vol-std was active."""
    if results.vol_std_ret is None:
        return
    results.vol_std_ret.to_csv(out_dir / "03_vol_standardized_returns.csv")
    logger.info("  → 03_vol_standardized_returns.csv  %s", results.vol_std_ret.shape)


def _export_turbulence(results: ComputedResults, out_dir: Path) -> None:
    """Export 04/05/06 turbulence CSVs."""
    for fname, tr in [
        ("04_turbulence_panel.csv",  results.turb_panel),
        ("05_turbulence_fx.csv",     results.turb_fx),
        ("06_turbulence_equity.csv", results.turb_eq),
    ]:
        if tr is None:
            continue
        df = pd.DataFrame({
            "tau":          tr.turbulence,
            "log_tau":      tr.log_turbulence,
            "magnitude":    tr.magnitude_component,
            "correlation":  tr.correlation_component,
            "regime":       tr.regime,
            "regime_code":  tr.regime_code,
            "chi2_pvalue":  tr.chi2_pvalue,
        })
        df.index.name = "date"
        df.to_csv(out_dir / fname)
        logger.info("  → %s  %s", fname, df.shape)


def _export_absorption(results: ComputedResults, out_dir: Path) -> None:
    """Export 07_absorption_ratio.csv."""
    if results.ar_fx is None:
        return
    df = pd.DataFrame({
        "absorption_ratio":     results.ar_fx.absorption_ratio,
        "delta_ar":             results.ar_fx.delta_ar,
        "standardized_delta_ar": results.ar_fx.standardized_delta,
    })
    df.index.name = "date"
    df.to_csv(out_dir / "07_absorption_ratio.csv")
    logger.info("  → 07_absorption_ratio.csv  %s", df.shape)


def _export_pca_kalman(results: ComputedResults, out_dir: Path) -> None:
    """Export 08/09/10 PCA and Kalman CSVs."""
    if results.dyn is None:
        return

    pca = results.dyn.pca
    K   = pca.n_components

    # 08_pca_factor_scores.csv
    scores_df = pca.factor_scores.copy()
    scores_df.index.name = "date"
    scores_df.to_csv(out_dir / "08_pca_factor_scores.csv")
    logger.info("  → 08_pca_factor_scores.csv  %s", scores_df.shape)

    # 09_kalman_filtered_factors.csv
    kf_cols: dict[str, pd.Series] = {}
    for k in range(K):
        fname = f"F{k+1}"
        kf    = results.dyn.kalman.get(fname)
        if kf is not None:
            kf_cols[f"{fname}_filtered"]   = kf.filtered
            kf_cols[f"{fname}_innovation"]  = kf.innovations
    if kf_cols:
        kf_df = pd.DataFrame(kf_cols)
        kf_df.index.name = "date"
        kf_df.to_csv(out_dir / "09_kalman_filtered_factors.csv")
        logger.info("  → 09_kalman_filtered_factors.csv  %s", kf_df.shape)

    # 10_innovation_norm.csv
    inorm = pca.innovation_norm
    if inorm is not None and len(inorm) > 0:
        inorm_df = inorm.rename("innovation_norm").to_frame()
        inorm_df.index.name = "date"
        inorm_df.to_csv(out_dir / "10_innovation_norm.csv")
        logger.info("  → 10_innovation_norm.csv  %s", inorm_df.shape)


def _export_country_turbulence(results: ComputedResults, out_dir: Path) -> None:
    """Export 11_country_turbulence.csv."""
    if not results.country_turb:
        return
    frames: dict[str, pd.Series] = {}
    for ctry, ct in results.country_turb.items():
        frames[f"{ctry}_tau"]    = ct.turbulence
        frames[f"{ctry}_regime"] = ct.regime
    df = pd.DataFrame(frames)
    df.index.name = "date"
    df.to_csv(out_dir / "11_country_turbulence.csv")
    logger.info("  → 11_country_turbulence.csv  %s", df.shape)


def export_all(
    results: ComputedResults,
    out_dir: Path,
    params: AuditParams,
    timestamp: str,
) -> None:
    """
    Export all CSV files for one mode to `out_dir`.

    Parameters
    ----------
    results   : ComputedResults from the pipeline
    out_dir   : output directory (created if missing)
    params    : CLI parameters (written into summary metadata)
    timestamp : ISO timestamp string for all files in this run
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting to %s", out_dir)
    _export_summary(results, out_dir, params, timestamp)
    _export_data_quality(results, out_dir)
    _export_returns(results, out_dir)
    _export_vol_std(results, out_dir)
    _export_turbulence(results, out_dir)
    _export_absorption(results, out_dir)
    _export_pca_kalman(results, out_dir)
    _export_country_turbulence(results, out_dir)


# ---------------------------------------------------------------------------
# Comparison table (mode=both)
# ---------------------------------------------------------------------------

def print_comparison(
    syn: ComputedResults,
    live: ComputedResults,
) -> None:
    """
    Print a side-by-side metric comparison table to terminal.

    Parameters
    ----------
    syn  : synthetic ComputedResults
    live : live ComputedResults
    """
    # Merge on shared metric keys
    all_keys = sorted(set(syn.summary_vals) | set(live.summary_vals))

    col_w = [38, 14, 14, 12, 13]
    hdr = (
        f"{'metric':<{col_w[0]}}  "
        f"{'synthetic':>{col_w[1]}}  "
        f"{'live':>{col_w[2]}}  "
        f"{'difference':>{col_w[3]}}  "
        f"{'pct_diff':>{col_w[4]}}"
    )
    sep = "-" * len(hdr)

    print()
    print("=" * len(hdr))
    print("  SYNTHETIC vs LIVE — key metric comparison")
    print("=" * len(hdr))
    print(hdr)
    print(sep)

    for key in all_keys:
        sv = syn.summary_vals.get(key, float("nan"))
        lv = live.summary_vals.get(key, float("nan"))

        if np.isnan(sv) or np.isnan(lv):
            diff_str = "n/a"
            pct_str  = "n/a"
        else:
            diff     = lv - sv
            pct      = (diff / abs(sv) * 100) if abs(sv) > 1e-12 else float("nan")
            diff_str = f"{diff:+.4f}"
            pct_str  = f"{pct:+.1f}%" if not np.isnan(pct) else "n/a"

        sv_str = f"{sv:.4f}" if not np.isnan(sv) else "n/a"
        lv_str = f"{lv:.4f}" if not np.isnan(lv) else "n/a"

        print(
            f"{key:<{col_w[0]}}  "
            f"{sv_str:>{col_w[1]}}  "
            f"{lv_str:>{col_w[2]}}  "
            f"{diff_str:>{col_w[3]}}  "
            f"{pct_str:>{col_w[4]}}"
        )

    print(sep)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export all EM risk time series to CSV for manual inspection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["synthetic", "live", "both"], default="both",
        help="Data source(s) to export",
    )
    p.add_argument(
        "--start", default="2015-01-01",
        help="Start date YYYY-MM-DD",
    )
    p.add_argument(
        "--slow-window", type=int, default=504,
        help="SlowCovariance window in trading days",
    )
    p.add_argument(
        "--min-periods", type=int, default=60,
        help="Minimum observations before computing",
    )
    p.add_argument(
        "--lam", type=float, default=0.94,
        help="EWMA decay lambda for FastCovariance",
    )
    p.add_argument(
        "--vol-std", action=argparse.BooleanOptionalAction, default=True,
        help="Apply VolStandardizer to turbulence input (--no-vol-std to disable)",
    )
    p.add_argument(
        "--output-dir", default="audit_exports/",
        help="Base folder for CSV exports",
    )
    return p


def main() -> None:
    """Entry point: parse CLI args, run pipeline(s), export CSVs, print comparison."""
    parser = _build_parser()
    args   = parser.parse_args()

    params = AuditParams(
        mode=args.mode,
        start=args.start,
        slow_window=args.slow_window,
        min_periods=args.min_periods,
        lam=args.lam,
        vol_std=args.vol_std,
        output_dir=Path(args.output_dir),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print()
    print("EM Risk Dashboard — Audit Export")
    print(f"  mode={params.mode}  start={params.start}  slow_window={params.slow_window}")
    print(f"  lam={params.lam}  vol_std={params.vol_std}  output_dir={params.output_dir}")
    print(f"  timestamp={timestamp}")
    print()

    syn_results:  Optional[ComputedResults] = None
    live_results: Optional[ComputedResults] = None

    if params.mode in ("synthetic", "both"):
        print("── Synthetic mode ────────────────────────────────")
        syn_results = _load_synthetic(params)
        export_all(syn_results, params.output_dir / "synthetic", params, timestamp)
        print(f"  Done. Files in: {params.output_dir / 'synthetic'}")
        print()

    if params.mode in ("live", "both"):
        print("── Live mode ─────────────────────────────────────")
        live_results = _load_live(params)
        if live_results is None:
            print("  Live data unavailable. Skipping live export.")
        else:
            export_all(live_results, params.output_dir / "live", params, timestamp)
            print(f"  Done. Files in: {params.output_dir / 'live'}")
        print()

    if params.mode == "both" and syn_results is not None and live_results is not None:
        print_comparison(syn_results, live_results)

    print("Export complete.")


if __name__ == "__main__":
    main()
