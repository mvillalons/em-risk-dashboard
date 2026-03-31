"""
data/synthetic.py — Realistic Synthetic EM Data Generator
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class EMUniverse:
    fx: pd.DataFrame
    equity: pd.DataFrame
    global_: pd.DataFrame
    panel: pd.DataFrame
    prices_fx: pd.DataFrame
    prices_eq: pd.DataFrame

    @property
    def dates(self): return self.panel.index
    @property
    def n_obs(self): return len(self.panel)

COUNTRY_PARAMS = {
    "CLP": (0.0055, 0.0140, -0.0001, -0.0008, 5.0),
    "BRL": (0.0095, 0.0220, -0.0002, -0.0015, 4.5),
    "MXN": (0.0080, 0.0180, -0.0002, -0.0012, 5.0),
    "COP": (0.0070, 0.0160, -0.0001, -0.0010, 5.5),
    "PEN": (0.0040, 0.0100, -0.0001, -0.0005, 6.0),
}
EQUITY_PARAMS = {
    "ECH": (0.0100, 0.0220, 0.0003, -0.0010, 5.0),
    "EWZ": (0.0160, 0.0320, 0.0003, -0.0015, 4.0),
    "EWW": (0.0130, 0.0260, 0.0003, -0.0012, 4.5),
    "GXG": (0.0120, 0.0250, 0.0002, -0.0010, 5.5),
    "EPU": (0.0090, 0.0200, 0.0003, -0.0008, 5.0),
}
GLOBAL_PARAMS = {
    "VIX_chg": (0.05, 0.15, -0.001, 0.005),
    "DXY":     (0.003, 0.007, 0.0001, 0.0004),
    "SPY":     (0.006, 0.014, 0.0004, -0.0008),
    "EEM":     (0.008, 0.018, 0.0002, -0.0012),
}

CORR_FX_CALM = np.array([
    [1.00, 0.45, 0.42, 0.38, 0.30],
    [0.45, 1.00, 0.55, 0.50, 0.35],
    [0.42, 0.55, 1.00, 0.48, 0.32],
    [0.38, 0.50, 0.48, 1.00, 0.38],
    [0.30, 0.35, 0.32, 0.38, 1.00],
])
CORR_FX_STRESS = np.array([
    [1.00, 0.72, 0.70, 0.65, 0.55],
    [0.72, 1.00, 0.80, 0.75, 0.60],
    [0.70, 0.80, 1.00, 0.72, 0.58],
    [0.65, 0.75, 0.72, 1.00, 0.62],
    [0.55, 0.60, 0.58, 0.62, 1.00],
])
CORR_EQ_CALM = np.array([
    [1.00, 0.55, 0.52, 0.48, 0.50],
    [0.55, 1.00, 0.62, 0.58, 0.52],
    [0.52, 0.62, 1.00, 0.55, 0.50],
    [0.48, 0.58, 0.55, 1.00, 0.48],
    [0.50, 0.52, 0.50, 0.48, 1.00],
])
CORR_EQ_STRESS = np.array([
    [1.00, 0.82, 0.80, 0.75, 0.78],
    [0.82, 1.00, 0.85, 0.80, 0.78],
    [0.80, 0.85, 1.00, 0.78, 0.75],
    [0.75, 0.80, 0.78, 1.00, 0.72],
    [0.78, 0.78, 0.75, 0.72, 1.00],
])

def _chol(corr):
    c = (corr + corr.T) / 2 + np.eye(len(corr)) * 1e-8
    return np.linalg.cholesky(c)

def _t_innov(rng, T, N, df=5.0):
    z = rng.standard_t(df=df, size=(T, N))
    return z / np.sqrt(df / (df - 2))

def generate_em_universe(start="2015-01-01", end="2024-12-31", seed=42, include_known_crises=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)

    # Regime process
    regime = np.zeros(T, dtype=int)
    for t in range(1, T):
        regime[t] = int(rng.random() < (0.03 if regime[t-1] == 0 else 0.10 - 1.0))
        # Fix the logic
    regime = np.zeros(T, dtype=int)
    for t in range(1, T):
        if regime[t-1] == 0:
            regime[t] = int(rng.random() < 0.03)
        else:
            regime[t] = int(rng.random() < 0.90)

    if include_known_crises:
        for s, e in [("2018-08-01","2018-12-31"),("2020-02-20","2020-04-30"),
                     ("2022-01-15","2022-06-30"),("2023-03-10","2023-03-31")]:
            mask = (dates >= s) & (dates <= e)
            regime[mask] = 1

    # FX returns
    fx_cols = list(COUNTRY_PARAMS.keys())
    N_fx = len(fx_cols)
    Lc = _chol(CORR_FX_CALM); Ls = _chol(CORR_FX_STRESS)
    ic = _t_innov(rng, T, N_fx) @ Lc.T
    is_ = _t_innov(rng, T, N_fx, df=4.0) @ Ls.T
    fx_ret = np.zeros((T, N_fx))
    for t in range(T):
        if regime[t] == 0:
            vols  = np.array([COUNTRY_PARAMS[c][0] for c in fx_cols])
            means = np.array([COUNTRY_PARAMS[c][2] for c in fx_cols])
            fx_ret[t] = means + vols * ic[t]
        else:
            vols  = np.array([COUNTRY_PARAMS[c][1] for c in fx_cols])
            means = np.array([COUNTRY_PARAMS[c][3] for c in fx_cols])
            fx_ret[t] = means + vols * is_[t]

    # Idiosyncratic country episodes
    def _shock(arr, start_d, end_d, col, extra_vol):
        mask = (dates >= start_d) & (dates <= end_d)
        n = mask.sum()
        if n: arr[mask, col] += rng.normal(-extra_vol*0.3, extra_vol, n)
    _shock(fx_ret, "2019-10-18", "2019-11-15", 0, 0.03)
    _shock(fx_ret, "2022-10-01", "2022-11-01", 1, 0.015)

    # Equity returns
    eq_cols = list(EQUITY_PARAMS.keys())
    N_eq = len(eq_cols)
    Lce = _chol(CORR_EQ_CALM); Lse = _chol(CORR_EQ_STRESS)
    ice = _t_innov(rng, T, N_eq) @ Lce.T
    ise = _t_innov(rng, T, N_eq, df=4.0) @ Lse.T
    eq_ret = np.zeros((T, N_eq))
    for t in range(T):
        if regime[t] == 0:
            vols  = np.array([EQUITY_PARAMS[c][0] for c in eq_cols])
            means = np.array([EQUITY_PARAMS[c][2] for c in eq_cols])
            eq_ret[t] = means + vols * ice[t]
        else:
            vols  = np.array([EQUITY_PARAMS[c][1] for c in eq_cols])
            means = np.array([EQUITY_PARAMS[c][3] for c in eq_cols])
            eq_ret[t] = means + vols * ise[t]

    # Global signals
    gl_cols = list(GLOBAL_PARAMS.keys())
    N_gl = len(gl_cols)
    gl_ret = np.zeros((T, N_gl))
    for i, col in enumerate(gl_cols):
        vc, vs, mc, ms = GLOBAL_PARAMS[col]
        innov = rng.standard_normal(T)
        calm = regime == 0
        gl_ret[calm, i]  = mc + vc * innov[calm]
        gl_ret[~calm, i] = ms + vs * innov[~calm]

    df_fx = pd.DataFrame(fx_ret, index=dates, columns=fx_cols)
    df_eq = pd.DataFrame(eq_ret, index=dates, columns=eq_cols)
    df_gl = pd.DataFrame(gl_ret, index=dates, columns=gl_cols)
    panel = pd.concat([df_fx, df_eq, df_gl], axis=1)
    panel.attrs["regime"] = pd.Series(regime, index=dates, name="regime")
    panel.attrs["stress_fraction"] = regime.mean()

    return EMUniverse(
        fx=df_fx, equity=df_eq, global_=df_gl, panel=panel,
        prices_fx=(1 + df_fx).cumprod() * 100,
        prices_eq=(1 + df_eq).cumprod() * 100,
    )

def get_regime_series(universe):
    return universe.panel.attrs.get("regime", pd.Series(dtype=int))
