"""
dashboard/app.py
==============================
EM Risk Dashboard — Streamlit Application
==============================

Run with: streamlit run dashboard/app.py

Caching architecture (three independent layers):
  load_raw_data          ttl=12h   key: data_mode + start + end
  compute_turbulence_metrics  no ttl   key: data_key + window + vol_standardize
  compute_fragility_metrics   no ttl   key: data_key + window + lam
Moving the lam slider recomputes ONLY fragility (~3s).
Toggling vol_standardize recomputes ONLY turbulence (~2s).
"""

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from modules.turbulence import compute_turbulence_index, crisis_episodes, REGIME_COLORS
from modules.absorption import compute_absorption_ratio
from modules.pca_kalman import compute_dynamic_factors_v2
from data.synthetic import generate_em_universe, get_regime_series
from core.metric_cache import (
    exists, make_key,
    save_turbulence, load_turbulence,
    save_absorption, load_absorption,
    save_dynamic,    load_dynamic,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COUNTRY_NAMES = {
    "CLP": "Chile",
    "BRL": "Brazil",
    "MXN": "Mexico",
    "COP": "Colombia",
    "PEN": "Peru",
}

FLAG = {
    "CLP": "🇨🇱",
    "BRL": "🇧🇷",
    "MXN": "🇲🇽",
    "COP": "🇨🇴",
    "PEN": "🇵🇪",
}

REGIME_BADGE_CSS = {
    "Calm":      "background:#1a3a2a;color:#2ecc71",
    "Elevated":  "background:#3a2e0a;color:#f39c12",
    "Turbulent": "background:#3a1e0a;color:#e67e22",
    "Crisis":    "background:#3a0a0a;color:#e74c3c",
}

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT = "#00d4aa"
DANGER = "#e74c3c"
WARN   = "#e67e22"
MUTED  = "#8892a0"


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="EM Risk Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.stMetric label { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #8892a0; }
.stMetric [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }

.regime-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

.country-card {
    border: 1px solid #1e2a35;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    background: #0d1117;
}

.section-label {
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #8892a0;
    margin-bottom: 6px;
}

div[data-testid="stSidebar"] { background: #080d12; }
.stPlotlyChart { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached layer 1: Raw data (TTL = 12 h)
# Cache key: data_mode + start + end  — no metric params here
# ---------------------------------------------------------------------------

@st.cache_data(ttl=43200, show_spinner="Fetching market data...")
def load_raw_data(data_mode: str, start: str, end: str) -> dict:
    """
    Load raw price levels and log-returns.  No metric computation.

    Parameters
    ----------
    data_mode : "Synthetic" or "Live"
    start     : start date string 'YYYY-MM-DD'
    end       : end date string   'YYYY-MM-DD' (used for synthetic only)

    Returns
    -------
    dict with keys: fx_ret, eq_ret, panel, prices_fx, prices_eq,
                    data_mode (actual), last_updated (UTC timestamp)
    """
    actual_mode = data_mode

    if data_mode == "Live":
        try:
            from data.fetcher import load_em_universe
            from core.returns import log_returns
            raw        = load_em_universe(start=start)
            fx_ret     = log_returns(raw["fx"])
            eq_ret     = log_returns(raw["equity"])
            global_ret = log_returns(raw["global"])
            panel      = pd.concat([fx_ret, eq_ret, global_ret], axis=1).dropna(how="all")
            prices_fx  = raw["fx"]
            prices_eq  = raw["equity"]
        except Exception as e:
            st.warning(f"Live fetch failed ({e}), using synthetic data.")
            actual_mode = "Synthetic"

    if actual_mode == "Synthetic":
        uni       = generate_em_universe(start=start, end=end, seed=42)
        fx_ret    = uni.fx
        eq_ret    = uni.equity
        panel     = uni.panel
        prices_fx = uni.prices_fx
        prices_eq = uni.prices_eq

    return {
        "fx_ret":       fx_ret,
        "eq_ret":       eq_ret,
        "panel":        panel,
        "prices_fx":    prices_fx,
        "prices_eq":    prices_eq,
        "data_mode":    actual_mode,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


# ---------------------------------------------------------------------------
# Metric layer 2: Turbulence  (disk cache — survives Streamlit restarts)
# Cache key: data_key + window + vol_standardize
# lam is NOT in the key — moving the lam slider will NOT bust this cache.
# ---------------------------------------------------------------------------

def get_turbulence_metrics(
    data_key: str,
    window: int,
    vol_standardize: bool,
    raw: dict,
) -> dict:
    """
    Return turbulence metrics, loading from disk cache when available.

    Cache key encodes data_key + window + vol_standardize only.
    Changing lam will NOT bust this cache.

    Parameters
    ----------
    data_key       : 16-char md5 of panel hash
    window         : rolling estimation window (trading days)
    vol_standardize: EWMA vol pre-whitening toggle
    raw            : dict from load_raw_data

    Returns
    -------
    dict with keys: turb_panel, turb_fx, turb_eq, country_turb
    """
    vs = int(vol_standardize)
    fx_ret = raw["fx_ret"]
    eq_ret = raw["eq_ret"]
    panel  = raw["panel"]

    keys = {
        "panel": make_key("turb_panel", data_key, window, vs),
        "fx":    make_key("turb_fx",    data_key, window, vs),
        "eq":    make_key("turb_eq",    data_key, window, vs),
        **{c:    make_key("turb", c, data_key, window, vs) for c in fx_ret.columns},
    }

    if all(exists(k) for k in keys.values()):
        return {
            "turb_panel":   load_turbulence(keys["panel"]),
            "turb_fx":      load_turbulence(keys["fx"]),
            "turb_eq":      load_turbulence(keys["eq"]),
            "country_turb": {c: load_turbulence(keys[c]) for c in fx_ret.columns},
        }

    # Cache miss — compute all turbulence metrics and persist.
    with st.spinner("Computing turbulence metrics..."):
        slow_w = max(window, 252)
        kw = dict(window=window, min_periods=60,
                  vol_standardize=vol_standardize, slow_window=slow_w)

        turb_panel = compute_turbulence_index(panel,  **kw)
        turb_fx    = compute_turbulence_index(fx_ret, **kw)
        turb_eq    = compute_turbulence_index(eq_ret, **kw)

        cols_fx = list(fx_ret.columns)
        cols_eq = list(eq_ret.columns)
        country_turb: dict = {}
        for i, col in enumerate(cols_fx):
            ct = pd.concat([fx_ret[[col]], eq_ret[[cols_eq[i]]]], axis=1)
            country_turb[col] = compute_turbulence_index(ct, **kw)

        save_turbulence(keys["panel"], turb_panel)
        save_turbulence(keys["fx"],    turb_fx)
        save_turbulence(keys["eq"],    turb_eq)
        for col in cols_fx:
            save_turbulence(keys[col], country_turb[col])

    return {
        "turb_panel":   turb_panel,
        "turb_fx":      turb_fx,
        "turb_eq":      turb_eq,
        "country_turb": country_turb,
    }


# ---------------------------------------------------------------------------
# Metric layer 3: Fragility  (disk cache — survives Streamlit restarts)
# Cache key: data_key + window + lam
# vol_standardize is NOT in the key — toggling it will NOT bust this cache.
# ---------------------------------------------------------------------------

def get_fragility_metrics(
    data_key: str,
    window: int,
    lam: float,
    raw: dict,
) -> dict:
    """
    Return fragility metrics (AR + dynamic factors), loading from disk cache
    when available.

    Cache key encodes data_key + window + lam only.
    Changing vol_standardize will NOT bust this cache.

    Parameters
    ----------
    data_key : 16-char md5 of panel hash
    window   : rolling estimation window (trading days)
    lam      : EWMA decay lambda (0.90–0.99); stored as int(lam*100) in key
    raw      : dict from load_raw_data

    Returns
    -------
    dict with keys: ar_fx, dyn
    """
    lam_int = int(round(lam * 100))
    key_ar  = make_key("ar",  data_key, window, lam_int)
    key_dyn = make_key("dyn", data_key, window, lam_int)

    if exists(key_ar) and exists(key_dyn):
        return {
            "ar_fx": load_absorption(key_ar),
            "dyn":   load_dynamic(key_dyn),
        }

    # Cache miss — compute and persist.
    with st.spinner("Computing fragility metrics..."):
        ar_fx = compute_absorption_ratio(
            raw["fx_ret"], window=window, min_periods=60, lam=lam,
        )
        dyn = compute_dynamic_factors_v2(
            raw["panel"], window=window, n_components=3, min_periods=60, lam=lam,
        )
        save_absorption(key_ar,  ar_fx)
        save_dynamic(key_dyn, dyn)

    return {
        "ar_fx": ar_fx,
        "dyn":   dyn,
    }


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    st.markdown("---")

    data_mode = st.radio("Data source", ["Synthetic", "Live"], index=0)

    st.markdown("---")
    start_date = st.date_input("Start date", value=pd.Timestamp("2015-01-01"))
    end_date   = st.date_input("End date",   value=pd.Timestamp("2024-12-31"))

    st.markdown("---")
    window = st.slider(
        "Rolling window (trading days)", 60, 504, 252, step=21,
        help="Window for estimating μ and Σ in the Mahalanobis distance",
    )
    lam = st.slider(
        "EWMA decay lambda", 0.90, 0.99, 0.94, step=0.01,
        help="Controls EWMA responsiveness: higher = slower decay, smoother estimates",
    )
    vol_standardize = st.toggle("Vol-standardize turbulence", value=True,
        help="Pre-whiten returns with EWMA vol before Mahalanobis computation. "
             "Suppresses pure vol spikes; tau captures correlation anomalies.")

    signal_layer = st.selectbox(
        "Primary signal layer",
        ["FX Returns", "Equity Returns", "Full Panel (FX + Equity + Global)"],
        index=0,
    )
    st.markdown("---")
    show_decomp  = st.checkbox("Show Mahalanobis decomposition", value=True)
    show_ar      = st.checkbox("Show Absorption Ratio", value=True)
    show_inorm   = st.checkbox("Show Correlation Regime Shock", value=True)
    show_heatmap = st.checkbox("Show correlation heatmap", value=True)


# ---------------------------------------------------------------------------
# Load and compute (three cached layers)
# ---------------------------------------------------------------------------

raw      = load_raw_data(data_mode, str(start_date), str(end_date))
panel    = raw["panel"]

# Compact hash of panel content — used as cache key for metric layers without
# hashing the full DataFrame on every Streamlit re-render.
# pd.util.hash_pandas_object is faster and avoids serialization overhead.
_panel_hash = pd.util.hash_pandas_object(panel, index=True).sum()
data_key    = hashlib.md5(str(_panel_hash).encode()).hexdigest()[:16]

turb      = get_turbulence_metrics(data_key, window, vol_standardize, raw)
fragility = get_fragility_metrics(data_key, window, lam, raw)


# ---------------------------------------------------------------------------
# Sidebar cache status (appended after data is available)
# ---------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.caption(
    f"📊 Data: {raw['last_updated']}\n\n"
    f"Cache key: `{data_key}`"
)
st.sidebar.caption("EM Risk Dashboard v0.2 · Prototype")


# ---------------------------------------------------------------------------
# Resolve active turbulence result
# ---------------------------------------------------------------------------

signal_map = {
    "FX Returns":                           "turb_fx",
    "Equity Returns":                       "turb_eq",
    "Full Panel (FX + Equity + Global)":    "turb_panel",
}
active_turb = turb[signal_map[signal_layer]]
ar          = fragility["ar_fx"]
dyn         = fragility["dyn"]
fx_ret      = raw["fx_ret"]
eq_ret      = raw["eq_ret"]
actual_mode = raw["data_mode"]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

current_regime = active_turb.current_regime()
current_score  = active_turb.current_score()
current_pctile = active_turb.current_pctile()
last_date      = active_turb.turbulence.dropna().index[-1]
badge_css      = REGIME_BADGE_CSS.get(current_regime, "")

mode_badge = "🟢 Live" if actual_mode == "Live" else "🟡 Synthetic"

col_title, col_regime, col_spacer = st.columns([3, 1.5, 2])

with col_title:
    st.markdown("## 📡 EM Risk Dashboard")
    st.caption(
        f"Latin America · Turbulence & Fragility Monitor · "
        f"Last observation: **{last_date.strftime('%d %b %Y')}** · {mode_badge}"
    )

with col_regime:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<span class="regime-badge" style="{badge_css}">{current_regime}</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"Current regime — {current_pctile:.0f}th percentile")

st.markdown("---")


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "Turbulence Score",
    f"{current_score:.1f}",
    delta=f"{current_score - float(active_turb.turbulence.dropna().iloc[-2]):.2f}",
)
k2.metric(
    "Historical Percentile",
    f"{current_pctile:.0f}%",
)
ar_current = float(ar.absorption_ratio.dropna().iloc[-1])
ar_prev    = float(ar.absorption_ratio.dropna().iloc[-2])
k3.metric(
    "Absorption Ratio",
    f"{ar_current:.3f}",
    delta=f"{ar_current - ar_prev:+.4f}",
    delta_color="inverse",
)
delta_ar_current = float(ar.delta_ar.dropna().iloc[-1])
k4.metric(
    "ΔAbsorption (daily)",
    f"{delta_ar_current:+.4f}",
    delta_color="inverse",
)
stress_days = int((active_turb.regime.dropna() != "Calm").sum())
total_days  = int(active_turb.regime.dropna().shape[0])
k5.metric(
    "Stress Days (history)",
    f"{stress_days:,}",
    delta=f"{stress_days/total_days:.1%} of sample",
    delta_color="off",
)

st.markdown("---")


# ---------------------------------------------------------------------------
# Country scorecards
# ---------------------------------------------------------------------------

st.markdown("#### Country Risk Scorecards")
cc = st.columns(5)

countries = list(COUNTRY_NAMES.keys())
for i, ctry in enumerate(countries):
    ct     = turb["country_turb"][ctry]
    score  = ct.current_score()
    pctile = ct.current_pctile()
    regime = ct.current_regime()
    badge  = REGIME_BADGE_CSS.get(regime, "")

    recent_fx = fx_ret[ctry].dropna().iloc[-5:].mean() * 252
    fx_arrow  = "▲" if recent_fx > 0 else "▼"
    fx_color  = "#2ecc71" if recent_fx > 0 else "#e74c3c"

    with cc[i]:
        st.markdown(
            f"""
            <div class="country-card">
                <div style="font-size:1.3rem">{FLAG[ctry]} {COUNTRY_NAMES[ctry]}</div>
                <div style="margin:6px 0">
                    <span class="regime-badge" style="{badge}">{regime}</span>
                </div>
                <div class="section-label">Turbulence</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;font-weight:600">{score:.1f}</div>
                <div style="font-size:0.8rem;color:{MUTED}">{pctile:.0f}th pct</div>
                <div class="section-label" style="margin-top:8px">FX trend (5d ann.)</div>
                <div style="color:{fx_color};font-size:0.9rem;font-family:'JetBrains Mono',monospace">
                    {fx_arrow} {abs(recent_fx*100):.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")


# ---------------------------------------------------------------------------
# Main chart: Turbulence Index time series
# ---------------------------------------------------------------------------

st.markdown("#### Turbulence Index — " + signal_layer)

turb_series    = active_turb.turbulence.dropna()
regime_series  = active_turb.regime.reindex(turb_series.index)

fig_turb = go.Figure()

regime_colors_hex = {
    "Calm":      "rgba(46, 204, 113, 0.05)",
    "Elevated":  "rgba(243, 156, 18, 0.10)",
    "Turbulent": "rgba(230, 126, 34, 0.15)",
    "Crisis":    "rgba(231, 76, 60, 0.20)",
}

for label, color in regime_colors_hex.items():
    mask    = regime_series == label
    if not mask.any():
        continue
    in_block = False
    x0       = None
    for date, val in mask.items():
        if val and not in_block:
            x0       = date
            in_block = True
        elif not val and in_block:
            fig_turb.add_vrect(x0=x0, x1=date, fillcolor=color, line_width=0)
            in_block = False
    if in_block:
        fig_turb.add_vrect(x0=x0, x1=mask.index[-1], fillcolor=color, line_width=0)

for label, thresh_key in [("Elevated", "elevated"), ("Turbulent", "turbulent"), ("Crisis", "crisis")]:
    thresh_val = active_turb.thresholds[thresh_key]
    fig_turb.add_hline(
        y=thresh_val,
        line_dash="dash",
        line_color=REGIME_COLORS[label],
        line_width=0.8,
        opacity=0.6,
        annotation_text=label,
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color=REGIME_COLORS[label],
    )

fig_turb.add_trace(go.Scatter(
    x=turb_series.index,
    y=turb_series.values,
    mode="lines",
    name="Turbulence τ",
    line=dict(color=ACCENT, width=1.3),
    fill="tozeroy",
    fillcolor="rgba(0,212,170,0.06)",
))

if show_decomp:
    mag       = active_turb.magnitude_component.reindex(turb_series.index)
    corr_comp = active_turb.correlation_component.reindex(turb_series.index)
    fig_turb.add_trace(go.Scatter(
        x=mag.index, y=mag.values,
        mode="lines", name="Magnitude component",
        line=dict(color="#3498db", width=0.8, dash="dot"),
    ))
    fig_turb.add_trace(go.Scatter(
        x=corr_comp.index, y=corr_comp.values,
        mode="lines", name="Correlation component",
        line=dict(color="#9b59b6", width=0.8, dash="dot"),
    ))

fig_turb.update_layout(
    template=PLOTLY_TEMPLATE,
    height=340,
    margin=dict(l=40, r=80, t=20, b=40),
    legend=dict(orientation="h", y=1.02, x=0),
    xaxis=dict(showgrid=False),
    yaxis=dict(
        title="τ (Mahalanobis²)" + (" — vol-standardized" if vol_standardize else ""),
        gridcolor="#1e2a35", gridwidth=0.5,
    ),
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
)
st.plotly_chart(fig_turb, use_container_width=True)


# ---------------------------------------------------------------------------
# Spectral Fragility: Absorption Ratio + Chi-squared p-value
# ---------------------------------------------------------------------------

if show_ar:
    st.markdown("#### Spectral Fragility Monitor")
    col_ar, col_pval = st.columns([1, 1])

    with col_ar:
        ar_series  = ar.absorption_ratio.dropna()
        dar_series = ar.standardized_delta.reindex(ar_series.index)

        fig_ar = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.04,
        )

        fig_ar.add_trace(go.Scatter(
            x=ar_series.index, y=ar_series.values,
            mode="lines", name="Absorption Ratio",
            line=dict(color="#e67e22", width=1.5),
            fill="tozeroy", fillcolor="rgba(230,126,34,0.08)",
        ), row=1, col=1)

        fig_ar.add_hline(
            y=ar_series.mean(), line_dash="dash",
            line_color="#666", line_width=0.8,
            annotation_text="mean", annotation_font_size=9,
            row=1, col=1,
        )

        fig_ar.add_trace(go.Bar(
            x=dar_series.index, y=dar_series.values,
            name="Standardized ΔAR",
            marker_color=np.where(
                dar_series.values > 1.5, DANGER,
                np.where(dar_series.values > 0.5, WARN, ACCENT)
            ),
        ), row=2, col=1)

        fig_ar.update_layout(
            template=PLOTLY_TEMPLATE,
            height=320,
            margin=dict(l=40, r=40, t=10, b=40),
            showlegend=False,
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            yaxis=dict(title="AR", gridcolor="#1e2a35"),
            yaxis2=dict(title="σ(ΔAR)", gridcolor="#1e2a35"),
            xaxis2=dict(showgrid=False),
        )
        st.markdown(f"**Absorption Ratio** (EWMA λ={lam:.2f})")
        st.plotly_chart(fig_ar, use_container_width=True)
        st.caption(
            "AR = fraction of FX panel variance explained by top 20% of eigenvectors. "
            "Higher = more systemic. Standardized ΔAR > 1.5σ historically precedes stress episodes by 5–15 days. "
            f"EWMA λ={lam:.2f} — decrease λ for faster reaction."
        )

    with col_pval:
        pval     = active_turb.chi2_pvalue.dropna()
        log_pval = -np.log10(pval.clip(1e-10))

        fig_pval = go.Figure()
        fig_pval.add_trace(go.Scatter(
            x=log_pval.index, y=log_pval.values,
            mode="lines", name="-log₁₀(p)",
            line=dict(color="#9b59b6", width=1.2),
            fill="tozeroy", fillcolor="rgba(155,89,182,0.08)",
        ))
        for threshold, label, color in [
            (-np.log10(0.05), "p=0.05", "#f39c12"),
            (-np.log10(0.01), "p=0.01", "#e74c3c"),
        ]:
            fig_pval.add_hline(
                y=threshold, line_dash="dash",
                line_color=color, line_width=0.8,
                annotation_text=label,
                annotation_position="right",
                annotation_font_size=9,
                annotation_font_color=color,
            )
        fig_pval.update_layout(
            template=PLOTLY_TEMPLATE,
            height=320,
            margin=dict(l=40, r=60, t=10, b=40),
            showlegend=False,
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            yaxis=dict(title="-log₁₀(p-value)", gridcolor="#1e2a35"),
            xaxis=dict(showgrid=False),
        )
        st.markdown("**Statistical significance** — χ²(N) null")
        st.plotly_chart(fig_pval, use_container_width=True)
        st.caption(
            "Right-tail p-value under H₀: returns are multivariate normal. "
            "-log₁₀(p) > 1.3 rejects at 5%; > 2.0 rejects at 1%. "
            "Persistent rejection = structural break in joint distribution."
        )

    # --- Correlation Regime Shock: ||ΔR||_F ---
    if show_inorm:
        inorm = dyn.pca.innovation_norm.dropna()

        if len(inorm) > 0:
            inorm_mean = float(inorm.mean())
            inorm_std  = float(inorm.std())
            threshold  = inorm_mean + 2.0 * inorm_std

            spike_dates = inorm.index[inorm > threshold]

            fig_inorm = go.Figure()

            fig_inorm.add_trace(go.Scatter(
                x=inorm.index,
                y=inorm.values,
                mode="lines",
                name="||ΔR||_F",
                line=dict(color="#e74c3c", width=1.2),
                fill="tozeroy",
                fillcolor="rgba(231,76,60,0.07)",
            ))

            fig_inorm.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="#f39c12",
                line_width=1.0,
                annotation_text="mean + 2σ",
                annotation_position="right",
                annotation_font_size=10,
                annotation_font_color="#f39c12",
            )

            # Vertical dashed lines at spike dates (subsample to avoid clutter)
            for dt in spike_dates:
                fig_inorm.add_vline(
                    x=dt,
                    line_dash="dash",
                    line_color="rgba(243,156,18,0.35)",
                    line_width=0.8,
                )

            fig_inorm.update_layout(
                title=dict(
                    text="Correlation Regime Shock  ||ΔR||_F",
                    font=dict(size=13),
                    x=0.0,
                ),
                template=PLOTLY_TEMPLATE,
                height=280,
                margin=dict(l=40, r=80, t=40, b=40),
                showlegend=False,
                plot_bgcolor="#0d1117",
                paper_bgcolor="#0d1117",
                yaxis=dict(title="Frobenius norm", gridcolor="#1e2a35"),
                xaxis=dict(showgrid=False),
            )

            st.plotly_chart(fig_inorm, use_container_width=True)
            n_spikes = len(spike_dates)
            st.caption(
                f"Frobenius norm of one-step Kalman innovation on EWMA correlation matrix: "
                f"||R̂_t − R_{{t|t-1}}||_F. "
                f"Dashed lines mark {n_spikes} dates above mean + 2σ = {threshold:.3f}. "
                f"Spikes indicate rapid correlation regime transitions not captured by slow-moving estimators."
            )

st.markdown("---")


# ---------------------------------------------------------------------------
# Rolling correlation heatmap
# ---------------------------------------------------------------------------

if show_heatmap:
    st.markdown("#### Cross-Country Correlation — FX Returns")

    window_corr = st.slider(
        "Correlation window (days)", 30, 252, 90,
        key="corr_window",
        help="Short windows reveal crisis-period correlation spikes",
    )

    corr_now   = fx_ret.iloc[-window_corr:].corr()
    corr_full  = fx_ret.corr()
    corr_delta = corr_now - corr_full

    col_h1, col_h2, col_h3 = st.columns(3)

    def _heatmap(matrix: pd.DataFrame, title: str, colorscale: str, zmin: float, zmax: float):
        labels = [f"{FLAG.get(c,'')} {c}" for c in matrix.columns]
        fig = go.Figure(go.Heatmap(
            z=matrix.values,
            x=labels, y=labels,
            colorscale=colorscale,
            zmin=zmin, zmax=zmax,
            text=np.round(matrix.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=10, family="JetBrains Mono"),
            showscale=True,
            colorbar=dict(thickness=12, len=0.8),
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=12), x=0.02),
            template=PLOTLY_TEMPLATE,
            height=280,
            margin=dict(l=60, r=20, t=40, b=60),
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
        )
        return fig

    with col_h1:
        st.plotly_chart(
            _heatmap(corr_now, f"Recent ({window_corr}d)", "RdBu_r", -1, 1),
            use_container_width=True,
        )
    with col_h2:
        st.plotly_chart(
            _heatmap(corr_full, "Full sample", "RdBu_r", -1, 1),
            use_container_width=True,
        )
    with col_h3:
        st.plotly_chart(
            _heatmap(corr_delta, "Delta (recent - full)", "RdYlGn_r", -0.4, 0.4),
            use_container_width=True,
        )
    st.caption(
        "Delta heatmap: positive (red) = correlation has risen in recent window vs. full sample. "
        "Rising cross-country FX correlation is a key early warning of systemic stress."
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# FX and Equity price charts
# ---------------------------------------------------------------------------

st.markdown("#### Price Levels (rebased to 100)")

col_px1, col_px2 = st.columns(2)

with col_px1:
    prices_fx  = raw["prices_fx"]
    fig_fx     = go.Figure()
    colors_fx  = [ACCENT, "#e74c3c", "#f39c12", "#9b59b6", "#3498db"]
    for i, col in enumerate(prices_fx.columns):
        fig_fx.add_trace(go.Scatter(
            x=prices_fx.index, y=prices_fx[col],
            mode="lines", name=f"{FLAG.get(col,'')} {col}",
            line=dict(width=1.2, color=colors_fx[i % len(colors_fx)]),
        ))
    fig_fx.update_layout(
        title="EM FX (vs. USD, index=100)",
        template=PLOTLY_TEMPLATE, height=280,
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis=dict(gridcolor="#1e2a35"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    )
    st.plotly_chart(fig_fx, use_container_width=True)

with col_px2:
    prices_eq = raw["prices_eq"]
    fig_eq    = go.Figure()
    for i, col in enumerate(prices_eq.columns):
        fig_eq.add_trace(go.Scatter(
            x=prices_eq.index, y=prices_eq[col],
            mode="lines", name=col,
            line=dict(width=1.2, color=colors_fx[i % len(colors_fx)]),
        ))
    fig_eq.update_layout(
        title="EM Equity ETFs (index=100)",
        template=PLOTLY_TEMPLATE, height=280,
        margin=dict(l=40, r=20, t=40, b=40),
        yaxis=dict(gridcolor="#1e2a35"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    )
    st.plotly_chart(fig_eq, use_container_width=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# Crisis episode table
# ---------------------------------------------------------------------------

st.markdown("#### Historical Crisis Episodes")

for regime_label, min_dur in [("Crisis", 3), ("Turbulent", 5)]:
    eps = crisis_episodes(active_turb, regime=regime_label, min_duration_days=min_dur)
    if len(eps) == 0:
        continue
    eps["start"]    = pd.to_datetime(eps["start"]).dt.strftime("%Y-%m-%d")
    eps["end"]      = pd.to_datetime(eps["end"]).dt.strftime("%Y-%m-%d")
    eps["peak_tau"] = eps["peak_tau"].round(1)
    eps["mean_tau"] = eps["mean_tau"].round(1)
    eps.columns     = ["Start", "End", "Duration (days)", "Peak τ", "Mean τ"]
    st.markdown(f"**{regime_label} episodes** (≥ {min_dur} days)")
    st.dataframe(eps, use_container_width=True, hide_index=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# Regime return analysis
# ---------------------------------------------------------------------------

st.markdown("#### Returns Conditioned on Risk Regime")
st.caption("Average annualized FX return per regime")

regime_aligned = active_turb.regime.reindex(fx_ret.index).dropna()
fx_aligned     = fx_ret.reindex(regime_aligned.index)

regime_returns = {}
for label in ["Calm", "Elevated", "Turbulent", "Crisis"]:
    mask = regime_aligned == label
    if mask.sum() == 0:
        continue
    regime_returns[label] = fx_aligned[mask].mean() * 252 * 100

df_regime = pd.DataFrame(regime_returns).T
if not df_regime.empty:
    fig_reg = go.Figure()
    x_labels = [f"{FLAG.get(c,'')} {c}" for c in df_regime.columns]

    for regime_label in df_regime.index:
        color = REGIME_COLORS.get(regime_label, MUTED)
        fig_reg.add_trace(go.Bar(
            name=regime_label,
            x=x_labels,
            y=df_regime.loc[regime_label].values,
            marker_color=color,
        ))

    fig_reg.update_layout(
        barmode="group",
        template=PLOTLY_TEMPLATE,
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        yaxis=dict(title="Ann. FX return (%)", gridcolor="#1e2a35"),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.08),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    )
    st.plotly_chart(fig_reg, use_container_width=True)

st.markdown("---")
st.caption(
    "EM Risk Dashboard — Prototype v0.2 · "
    f"Data: {'Synthetic (calibrated to 2015–2024 EM history)' if actual_mode == 'Synthetic' else 'Live (yfinance + FRED)'} · "
    f"λ={lam:.2f} · vol-standardize={'on' if vol_standardize else 'off'} · "
    "Method: Kritzman & Li (2010) Turbulence + Absorption Ratio + KalmanCorrelation · "
    "Built with Python / Streamlit / Plotly"
)
