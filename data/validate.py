"""
data/validate.py
==============================
Live Data Validation Script
==============================
Run this locally to verify your yfinance data before switching DATA_MODE=live.

Usage:
    python data/validate.py                     # full validation
    python data/validate.py --quick             # first 3 tickers only
    python data/validate.py --start 2020-01-01  # shorter history
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

def validate_series(df: pd.DataFrame, name: str) -> dict:
    issues = []
    stats = {}
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            issues.append(f"  {col}: ALL NaN — ticker may be wrong or delisted")
            continue
        nan_pct = df[col].isna().mean()
        if nan_pct > 0.10:
            issues.append(f"  {col}: {nan_pct:.1%} NaN — check ticker / market hours")
        if len(s) > 5:
            extreme = (s.abs() > 0.15).sum()
            if extreme > 5:
                issues.append(f"  {col}: {extreme} returns >15% — verify not split-adjusted")
        stats[col] = {
            "obs": len(s),
            "nan_pct": f"{nan_pct:.1%}",
            "mean": f"{s.mean()*252*100:.1f}% ann",
            "vol": f"{s.std()*np.sqrt(252)*100:.1f}% ann",
            "min": f"{s.min()*100:.2f}%",
            "max": f"{s.max()*100:.2f}%",
        }
    return {"name": name, "stats": stats, "issues": issues}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--start", default="2015-01-01")
    args = parser.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("EM Risk Dashboard — Live Data Validation")
    print(f"{'='*60}\n")

    universes = {
        "FX": {
            "tickers": ["CLPUSD=X", "BRL=X", "MXN=X", "COP=X", "PEN=X"],
            "names":   ["CLP",      "BRL",   "MXN",   "COP",   "PEN"],
            "notes": {
                "CLPUSD=X": "USD/CLP — may need inversion. Alternative: try 'CLP=X'",
                "COP=X":    "COP often has gaps — consider BIS data as backup",
                "PEN=X":    "PEN is less liquid; more gaps expected",
            }
        },
        "Equity ETFs": {
            "tickers": ["ECH", "EWZ", "EWW", "GXG", "EPU"],
            "names":   ["ECH", "EWZ", "EWW", "GXG", "EPU"],
            "notes": {
                "GXG": "Colombia ETF — lower AUM, wider spreads, possible illiquidity dates",
                "EPU": "Peru ETF — verify not suspended",
            }
        },
        "Global Risk": {
            "tickers": ["^VIX", "DX-Y.NYB", "SPY", "EEM"],
            "names":   ["VIX",  "DXY",       "SPY", "EEM"],
            "notes": {}
        },
    }

    all_passed = True

    for universe_name, cfg in universes.items():
        tickers = cfg["tickers"][:3] if args.quick else cfg["tickers"]
        names   = cfg["names"][:3]   if args.quick else cfg["names"]
        notes   = cfg["notes"]

        print(f"\n{'─'*50}")
        print(f"  {universe_name}")
        print(f"{'─'*50}")

        try:
            raw = yf.download(tickers, start=args.start, auto_adjust=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
            prices.columns = names
            returns = np.log(prices).diff().dropna(how="all")
        except Exception as e:
            print(f"  FETCH ERROR: {e}")
            all_passed = False
            continue

        result = validate_series(returns, universe_name)

        for col, stat in result["stats"].items():
            print(f"\n  {col}")
            for k, v in stat.items():
                print(f"    {k:10s}: {v}")
            if col in notes:
                print(f"    NOTE: {notes[col]}")

        if result["issues"]:
            print(f"\n  ⚠️  Issues detected:")
            for issue in result["issues"]:
                print(f"    {issue}")
            all_passed = False
        else:
            print(f"\n  ✓ {universe_name} looks clean")

    print(f"\n{'='*60}")
    if all_passed:
        print("✅  All checks passed — set DATA_MODE=live to use live data")
    else:
        print("⚠️   Some issues found — review above before switching to live mode")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
