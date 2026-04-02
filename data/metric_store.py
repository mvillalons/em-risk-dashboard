"""
data/metric_store.py
Thin read interface over the parquet-based metric cache (core/metric_cache.py).

Provides a named-series API compatible with the daily master notebook.
The ``db_path`` constructor arg is accepted for future SQLite migration
but the current implementation reads from parquet files.

Cache key convention (mirrors core/metric_cache.make_key):
  turbulence_panel  → turb_panel_{data_key}_{window}_{vs_int}
  turbulence_{CCY}  → turb_{CCY}_{data_key}_{window}_{vs_int}
  absorption_ratio  → ar_{data_key}_{window}_{lam_int}
  dynamic_factors   → dyn_{data_key}_{window}_{lam_int}

params string format:
  "w{window}_vs{vol_std_int}"   for turbulence series
  "w{window}_lam{lam_int}"      for fragility series  (lam_int = int(lam*100))
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


class MetricStore:
    """
    Read-only wrapper over the parquet metric cache.

    Parameters
    ----------
    db_path : str
        Accepted for API compatibility; ignored in this implementation.
        Future versions may write to SQLite at this path.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    # ------------------------------------------------------------------
    def load_series(
        self,
        metric_name: str,
        data_key: str,
        params: str,
    ) -> pd.DataFrame:
        """
        Load a named metric DataFrame from the parquet cache.

        Parameters
        ----------
        metric_name : one of
            "turbulence_panel", "turbulence_{CCY}",
            "absorption_ratio", "dynamic_factors"
        data_key : 16-char md5 hex of panel hash
        params   : "w{window}_vs{vs_int}" or "w{window}_lam{lam_int}"

        Returns
        -------
        pd.DataFrame loaded from parquet

        Raises
        ------
        FileNotFoundError if the cache file does not exist
        """
        from core.metric_cache import exists, load  # lazy import — avoids circular dep

        key = self._build_key(metric_name, data_key, params)
        if not exists(key):
            raise FileNotFoundError(
                f"Metric not in parquet cache: key={key!r}\n"
                "  Populate the cache by running the dashboard once (Synthetic mode)\n"
                "  or:  python scripts/export_audit.py --mode synthetic"
            )
        df = load(key)
        log.debug("MetricStore loaded %s  shape=%s", key, df.shape)
        return df

    # ------------------------------------------------------------------
    def _build_key(self, metric_name: str, data_key: str, params: str) -> str:
        """
        Translate (metric_name, data_key, params) to a parquet cache key.

        Notes
        -----
        params = "w252_vs1"   → window=252, suffix=1
        params = "w252_lam94" → window=252, suffix=94
        """
        parts  = params.split("_")
        window = int(parts[0][1:])                                      # "w252" → 252
        suffix = int("".join(c for c in parts[1] if c.isdigit())) if len(parts) > 1 else 0

        if metric_name == "turbulence_panel":
            return f"turb_panel_{data_key}_{window}_{suffix}"
        if metric_name == "absorption_ratio":
            return f"ar_{data_key}_{window}_{suffix}"
        if metric_name == "dynamic_factors":
            return f"dyn_{data_key}_{window}_{suffix}"
        if metric_name.startswith("turbulence_"):
            country = metric_name[len("turbulence_"):]
            return f"turb_{country}_{data_key}_{window}_{suffix}"
        raise ValueError(f"Unknown metric_name: {metric_name!r}")
