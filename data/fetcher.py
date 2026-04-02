"""
data/fetcher.py
Unified data fetcher: yfinance + FRED with local parquet cache.

Design decisions:
- All price series returned as pd.DataFrame with DatetimeIndex, columns = ticker strings
- Returns are computed in core/returns.py, not here
- Cache invalidation: if file is older than `max_age_hours`, re-fetch
- FRED key is optional for prototype (some series have yfinance proxies)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "_cache"
CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    safe_key = key.replace("/", "_").replace("^", "").replace("=", "")
    return CACHE_DIR / f"{safe_key}.parquet"


def _is_stale(path: Path, max_age_hours: float = 12.0) -> bool:
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return datetime.now() - mtime > timedelta(hours=max_age_hours)


def _load_cache(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning(f"Cache read failed for {path}: {e}")
        return None


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path)
    except Exception as e:
        logger.warning(f"Cache write failed for {path}: {e}")


# ---------------------------------------------------------------------------
# yfinance fetcher
# ---------------------------------------------------------------------------

def fetch_yfinance(
    tickers: list[str],
    start: str = "2010-01-01",
    end: Optional[str] = None,
    price_col: str = "Close",
    use_cache: bool = True,
    max_age_hours: float = 12.0,
) -> pd.DataFrame:
    """
    Fetch adjusted closing prices for a list of tickers from yfinance.

    Parameters
    ----------
    tickers : list of ticker strings (yfinance-compatible)
    start : start date string 'YYYY-MM-DD'
    end : end date string (default: today)
    price_col : which OHLCV column to extract
    use_cache : whether to use local parquet cache
    max_age_hours : cache staleness threshold in hours

    Returns
    -------
    pd.DataFrame with DatetimeIndex and one column per ticker
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    cache_key = f"yf_{'_'.join(sorted(tickers))}_{start}_{end}"
    cache_path = _cache_path(cache_key)

    if use_cache and not _is_stale(cache_path, max_age_hours):
        cached = _load_cache(cache_path)
        if cached is not None:
            logger.debug(f"Cache hit: {cache_key}")
            return cached

    logger.info(f"Fetching from yfinance: {tickers}")
    try:
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        # yfinance returns MultiIndex columns when multiple tickers
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[price_col]
        else:
            df = raw[[price_col]].rename(columns={price_col: tickers[0]})

        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        if use_cache:
            _save_cache(df, cache_path)

        return df

    except Exception as e:
        logger.error(f"yfinance fetch failed: {e}")
        raise


# ---------------------------------------------------------------------------
# FRED fetcher (optional, requires fredapi)
# ---------------------------------------------------------------------------

def fetch_fred(
    series_ids: list[str],
    start: str = "2010-01-01",
    end: Optional[str] = None,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    max_age_hours: float = 24.0,
) -> pd.DataFrame:
    """
    Fetch series from FRED via fredapi.

    Falls back gracefully if fredapi is not installed or API key is missing.
    FRED_API_KEY can also be set as environment variable.
    """
    end = end or datetime.today().strftime("%Y-%m-%d")
    cache_key = f"fred_{'_'.join(sorted(series_ids))}_{start}_{end}"
    cache_path = _cache_path(cache_key)

    if use_cache and not _is_stale(cache_path, max_age_hours):
        cached = _load_cache(cache_path)
        if cached is not None:
            return cached

    api_key = api_key or os.environ.get("FRED_API_KEY")

    try:
        from fredapi import Fred
        if not api_key:
            raise ValueError("FRED_API_KEY not set. Export it or pass api_key=")
        fred = Fred(api_key=api_key)
    except ImportError:
        raise ImportError("fredapi not installed. Run: pip install fredapi")

    frames = {}
    for sid in series_ids:
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            s.name = sid
            frames[sid] = s
        except Exception as e:
            logger.warning(f"FRED series {sid} failed: {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if use_cache:
        _save_cache(df, cache_path)

    return df


# ---------------------------------------------------------------------------
# Convenience: EM universe loader
# ---------------------------------------------------------------------------

def load_em_universe(
    start: str = "2015-01-01",
    include_equity: bool = True,
    include_fx: bool = True,
    include_global: bool = True,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Load the full EM universe as defined in config/universe.yaml.

    Returns a dict with keys: 'fx', 'equity', 'global'
    Each value is a DataFrame with one column per country/asset.
    """
    result = {}

    if include_fx:
        # Fetch each FX ticker individually to avoid yfinance alphabetical column
        # reordering in multi-ticker downloads.  All series stored as local CCY per USD.
        #
        # Ticker notes:
        #   CLPUSD=X  → USD per CLP  (invert to get CLP per USD, expected ~600-1100)
        #   USDBRL=X  → BRL per USD  (direct,  expected ~2.5-6.5)
        #   USDMXN=X  → MXN per USD  (direct,  expected ~13-22)
        #   USDCOP=X  → COP per USD  (direct,  expected ~1800-5000)
        #   PEN=X     → PEN per USD  (direct,  expected ~2.5-4.5)
        fx_spec: list[tuple[str, str, bool]] = [
            ("CLPUSD=X", "CLP", True),   # invert: 1/raw
            ("USDBRL=X", "BRL", False),
            ("USDMXN=X", "MXN", False),
            ("USDCOP=X", "COP", False),
            ("PEN=X",    "PEN", False),
        ]

        FX_RANGES: dict[str, tuple[float, float]] = {
            "CLP": (500.0,  1200.0),
            "BRL": (2.0,    8.0),
            "MXN": (10.0,   25.0),
            "COP": (1500.0, 6000.0),
            "PEN": (2.5,    4.5),
        }

        fx_frames: list[pd.Series] = []
        for ticker, name, invert in fx_spec:
            raw_df = fetch_yfinance([ticker], start=start, use_cache=use_cache)
            s = raw_df.iloc[:, 0].copy()
            if invert:
                s = (1.0 / s.replace(0.0, float("nan"))).ffill(limit=3)
            else:
                s = s.ffill(limit=3)
            s.name = name

            # Range sanity check on most-recent valid observation
            last_valid = s.dropna()
            if not last_valid.empty:
                latest = float(last_valid.iloc[-1])
                lo, hi = FX_RANGES[name]
                if not (lo <= latest <= hi):
                    logger.warning(
                        "FX sanity check failed: %s latest=%.4f expected [%.1f, %.1f]",
                        name, latest, lo, hi,
                    )

            fx_frames.append(s)

        df_fx = pd.concat(fx_frames, axis=1)
        result["fx"] = df_fx

    if include_equity:
        eq_tickers = ["ECH", "EWZ", "EWW", "GXG", "EPU"]
        eq_names = ["CHL", "BRA", "MEX", "COL", "PER"]
        df_eq = fetch_yfinance(eq_tickers, start=start, use_cache=use_cache)
        df_eq.columns = eq_names
        result["equity"] = df_eq

    if include_global:
        global_tickers = ["^VIX", "DX-Y.NYB", "SPY", "EEM"]
        global_names = ["VIX", "DXY", "SPY", "EEM"]
        df_gl = fetch_yfinance(global_tickers, start=start, use_cache=use_cache)
        df_gl.columns = global_names
        result["global"] = df_gl

    return result
