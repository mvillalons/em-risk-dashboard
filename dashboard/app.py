"""
dashboard/app.py
==============================
EM Risk Dashboard — Streamlit Application
==============================

Run with: streamlit run dashboard/app.py

Data mode: set DATA_MODE env var to 'live' (yfinance) or 'synthetic' (default).
  export DATA_MODE=live   # uses data/fetcher.py + yfinance
  export DATA_MODE=synthetic  # uses data/synthetic.py (no API needed)
"""

import os
import sys
from pathlib import Path

# Path setup — works whether launched from root or dashboard/
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
from data.synthetic import generate_em_universe, get_regime_series

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_MODE = os.environ.get("DATA_MODE", "synthetic")

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

# Inject custom CSS
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
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner="Loading market data...")
def load_data(mode: str, start: str, end: str, window: int):
    if mode == "synthetic":
        uni = generate_em_universe(start=start, end=end, seed=42)
        fx_ret = uni.fx
        eq_ret = uni.equity
        panel = uni.panel
        prices_fx = uni.prices_fx
        prices_eq = uni.prices_eq
    else:
        from data.fetcher import load_em_universe
        from core.returns import log_returns
        raw = load_em_universe(start=start)
        fx_ret = log_returns(raw["fx"])
        eq_ret = log_returns(raw["equity"])
        global_ret = log_returns(raw["global"])
        panel = pd.concat([fx_ret, eq_ret, global_ret], axis=1).dropna(how="all")
        prices_fx = raw["fx"]
        prices_eq = raw["equity"]

    # Turbulence: FX panel (primary signal)
    turb_fx = compute_turbulence_index(fx_ret, window=window, min_periods=60)

    # Turbulence: Equity panel
    turb_eq = compute_turbulence_index(eq_ret, window=window, min_periods=60)

    # Turbulence: Full panel
    turb_panel = compute_turbulence_index(panel, window=window, min_periods=60)

    # Absorption Ratio: FX
    ar_fx = compute_absorption_ratio(fx_ret, window=window, min_periods=60)

    # Per-country turbulence (single-asset, for country scorecards)
    country_turb = {}
    for col in fx_ret.columns:
        single = fx_ret[[col]].copy()
        # For single asset, turbulence = standardized squared return
        ct = compute_turbulence_index(
            pd.concat([single, eq_ret[[list(eq_ret.columns)[list(fx_ret.columns).index(col)]]]], axis=1),
            window=window, min_periods=60
        )
        country_turb[col] = ct

    return {
        "fx_ret": fx_ret,
        "eq_ret": eq_ret,
        "panel": panel,
        "prices_fx": prices_fx,
        "prices_eq": prices_eq,
        "turb_fx": turb_fx,
        "turb_eq": turb_eq,
        "turb_panel": turb_panel,
        "ar_fx": ar_fx,
        "country_turb": country_turb,
    }


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    st.markdown("---")

    data_mode_display = "🔴 Live (yfinance)" if DATA_MODE == "live" else "🟡 Synthetic"
    st.markdown(f"**Data mode:** {data_mode_display}")
    st.caption("Set `DATA_MODE=live` env var to connect live data.")
    st.markdown("---")

    start_date = st.date_input("Start date", value=pd.Timestamp("2015-01-01"))
    end_date   = st.date_input("End date",   value=pd.Timestamp("2024-12-31"))

    st.markdown("---")
    window = st.slider("Rolling window (trading days)", 60, 504, 252, step=21,
                        help="Window for estimating μ and Σ in the Mahalanobis distance")
    signal_layer = st.selectbox(
        "Primary signal layer",
        ["FX Returns", "Equity Returns", "Full Panel (FX + Equity + Global)"],
        index=0,
    )
    cov_method = st.selectbox(
        "Covariance estimator",
        ["ledoit_wolf", "oas", "sample", "ewm"],
        index=0,
    )
    st.markdown("---")
    show_decomp = st.checkbox("Show Mahalanobis decomposition", value=True)
    show_ar = st.checkbox("Show Absorption Ratio", value=True)
    show_heatmap = st.checkbox("Show correlation heatmap", value=True)

    st.markdown("---")
    st.caption("EM Risk Dashboard v0.1 · Prototype")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data = load_data(
    mode=DATA_MODE,
    start=str(start_date),
    end=str(end_date),
    window=window,
)

# Select active turbulence result
signal_map = {
    "FX Returns": "turb_fx",
    "Equity Returns": "turb_eq",
    "Full Panel (FX + Equity + Global)": "turb_panel",
}
active_turb = data[signal_map[signal_layer]]
ar = data["ar_fx"]
fx_ret = data["fx_ret"]
eq_ret = data["eq_ret"]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

current_regime = active_turb.current_regime()
current_score  = active_turb.current_score()
current_pctile = active_turb.current_pctile()
last_date      = active_turb.turbulence.dropna().index[-1]
badge_css      = REGIME_BADGE_CSS.get(current_regime, "")

col_title, col_regime, col_spacer = st.columns([3, 1.5, 2])

with col_title:
    st.markdown("## 📡 EM Risk Dashboard")
    st.caption(f"Latin America · Turbulence & Fragility Monitor · Last observation: **{last_date.strftime('%d %b %Y')}**")

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
    ct = data["country_turb"][ctry]
    score = ct.current_score()
    pctile = ct.current_pctile()
    regime = ct.current_regime()
    badge = REGIME_BADGE_CSS.get(regime, "")

    # FX performance (last 5 days annualized)
    recent_fx = fx_ret[ctry].dropna().iloc[-5:].mean() * 252
    fx_arrow = "▲" if recent_fx > 0 else "▼"
    fx_color = "#2ecc71" if recent_fx > 0 else "#e74c3c"

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

turb_series = active_turb.turbulence.dropna()
regime_series = active_turb.regime.reindex(turb_series.index)

fig_turb = go.Figure()

# Regime shading
regime_colors_hex = {
    "Calm":      "rgba(46, 204, 113, 0.05)",
    "Elevated":  "rgba(243, 156, 18, 0.10)",
    "Turbulent": "rgba(230, 126, 34, 0.15)",
    "Crisis":    "rgba(231, 76, 60, 0.20)",
}

for label, color in regime_colors_hex.items():
    mask = regime_series == label
    if not mask.any():
        continue
    # Find contiguous blocks
    in_block = False
    x0 = None
    for date, val in mask.items():
        if val and not in_block:
            x0 = date
            in_block = True
        elif not val and in_block:
            fig_turb.add_vrect(x0=x0, x1=date, fillcolor=color, line_width=0)
            in_block = False
    if in_block:
        fig_turb.add_vrect(x0=x0, x1=mask.index[-1], fillcolor=color, line_width=0)

# Threshold lines
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

# Main turbulence series
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
    mag = active_turb.magnitude_component.reindex(turb_series.index)
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
    yaxis=dict(title="τ (Mahalanobis²)", gridcolor="#1e2a35", gridwidth=0.5),
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
)
st.plotly_chart(fig_turb, use_container_width=True)


# ---------------------------------------------------------------------------
# Absorption Ratio + Chi-squared p-value
# ---------------------------------------------------------------------------

if show_ar:
    st.markdown("#### Fragility Monitor")
    col_ar, col_pval = st.columns([1, 1])

    with col_ar:
        ar_series = ar.absorption_ratio.dropna()
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
        st.markdown("**Absorption Ratio** (eigenvalue concentration)")
        st.plotly_chart(fig_ar, use_container_width=True)
        st.caption(
            "AR = fraction of FX panel variance explained by top 20% of eigenvectors. "
            "Higher = more systemic. Standardized ΔAR > 1.5σ historically precedes stress episodes by 5–15 days."
        )

    with col_pval:
        pval = active_turb.chi2_pvalue.dropna()
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

st.markdown("---")


# ---------------------------------------------------------------------------
# Rolling correlation heatmap
# ---------------------------------------------------------------------------

if show_heatmap:
    st.markdown("#### Cross-Country Correlation — FX Returns")

    window_corr = st.slider(
        "Correlation window (days)", 30, 252, 90,
        key="corr_window",
        help="Short windows reveal crisis-period correlation spikes"
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
    prices_fx = data["prices_fx"]
    fig_fx = go.Figure()
    colors_fx = [ACCENT, "#e74c3c", "#f39c12", "#9b59b6", "#3498db"]
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
    prices_eq = data["prices_eq"]
    fig_eq = go.Figure()
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
    eps["start"] = pd.to_datetime(eps["start"]).dt.strftime("%Y-%m-%d")
    eps["end"]   = pd.to_datetime(eps["end"]).dt.strftime("%Y-%m-%d")
    eps["peak_tau"]  = eps["peak_tau"].round(1)
    eps["mean_tau"]  = eps["mean_tau"].round(1)
    eps.columns = ["Start", "End", "Duration (days)", "Peak τ", "Mean τ"]
    st.markdown(f"**{regime_label} episodes** (≥ {min_dur} days)")
    st.dataframe(eps, use_container_width=True, hide_index=True)

st.markdown("---")


# ---------------------------------------------------------------------------
# Regime return analysis
# ---------------------------------------------------------------------------

st.markdown("#### Returns Conditioned on Risk Regime")
st.caption("Average annualized FX return per regime — separating regime impact from other factors")

regime_aligned = active_turb.regime.reindex(fx_ret.index).dropna()
fx_aligned = fx_ret.reindex(regime_aligned.index)

regime_returns = {}
for label in ["Calm", "Elevated", "Turbulent", "Crisis"]:
    mask = regime_aligned == label
    if mask.sum() == 0:
        continue
    ann_rets = fx_aligned[mask].mean() * 252 * 100  # annualized %
    regime_returns[label] = ann_rets

df_regime = pd.DataFrame(regime_returns).T
if not df_regime.empty:
    fig_reg = go.Figure()
    countries_plot = list(df_regime.columns)
    x_labels = [f"{FLAG.get(c,'')} {c}" for c in countries_plot]

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
    st.caption(
        "All series are synthetic calibrated data. In live mode, this table reveals the "
        "average carry/depreciation cost of EM FX exposure conditional on the risk regime, "
        "which directly informs U.S. portfolio EM allocation decisions."
    )

st.markdown("---")
st.caption(
    "EM Risk Dashboard — Prototype v0.1 · "
    f"Data: {'Synthetic (calibrated to 2015–2024 EM history)' if DATA_MODE == 'synthetic' else 'Live (yfinance + FRED)'} · "
    "Method: Kritzman & Li (2010) Turbulence Index + Absorption Ratio · "
    "Built with Python / Streamlit / Plotly"
)
