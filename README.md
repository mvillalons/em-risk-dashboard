# EM Risk Dashboard — Architecture & Specification

## Overview

A quantitative monitoring tool tracking systemic risk across LatAm economies
(Chile, Brazil, Mexico, Colombia, Peru) and their spillover effects on U.S. portfolios.

---

## Methodological Core

| Module | Method | Output |
|---|---|---|
| **Turbulence Index** | Mahalanobis distance on rolling return vector | Scalar per period [0, ∞) |
| **Absorption Ratio** | Eigenvalue concentration of correlation matrix | Scalar [0, 1] |
| **Dynamic Factor Structure** | PCA + Kalman filter on latent factors | Factor loadings + regime state |
| **Spillover Metrics** | DCC-GARCH / rolling beta vs. U.S. benchmarks | Cross-country contagion score |

---

## Project Structure

```
em_risk_dashboard/
│
├── config/
│   └── universe.yaml          # Asset universe, tickers, data sources
│
├── data/
│   ├── fetcher.py             # Unified data fetcher (FRED, yfinance, BIS)
│   ├── cache.py               # Local parquet cache with staleness check
│   └── universe.py            # Asset universe definitions
│
├── core/
│   ├── returns.py             # Return computation, cleaning, alignment
│   ├── covariance.py          # Ledoit-Wolf + rolling covariance estimators
│   └── regime.py              # Regime detection utilities
│
├── modules/
│   ├── turbulence.py          # *** PRIMARY MODULE *** Turbulence Index
│   ├── absorption.py          # Absorption Ratio (spectral fragility)
│   ├── pca_kalman.py          # Dynamic factor structure
│   └── spillover.py           # EM → U.S. contagion metrics
│
├── dashboard/
│   ├── app.py                 # Streamlit entrypoint
│   ├── components/            # Reusable chart components
│   └── intelligence.py        # Written narrative layer (Claude API)
│
├── tests/
│   ├── test_turbulence.py
│   ├── test_absorption.py
│   └── fixtures/              # Deterministic test data
│
├── notebooks/
│   └── 01_prototype.ipynb     # Exploration and validation
│
├── requirements.txt
└── README.md
```

---

## Data Sources

### FX Rates (primary signal layer)
| Series | Source | Ticker/Code | Frequency |
|---|---|---|---|
| USD/CLP | yfinance | `CLPUSD=X` (inverted) | daily |
| USD/BRL | yfinance | `BRLUSD=X` | daily |
| USD/MXN | yfinance | `MXNUSD=X` | daily |
| USD/COP | yfinance | `COPUSD=X` | daily |
| USD/PEN | yfinance | `PENUSD=X` | daily |
| DXY | yfinance | `DX-Y.NYB` | daily |

### Equity Indices
| Country | Ticker | Notes |
|---|---|---|
| Chile | `ECH` | iShares MSCI Chile ETF (USD) |
| Brazil | `EWZ` | iShares MSCI Brazil ETF |
| Mexico | `EWW` | iShares MSCI Mexico ETF |
| Colombia | `GXG` | Global X MSCI Colombia ETF |
| Peru | `EPU` | iShares MSCI All Peru ETF |
| U.S. benchmark | `SPY`, `EEM` | Spillover reference |

### Sovereign Spreads / Credit Risk
| Series | Source | Notes |
|---|---|---|
| EMBI spreads | FRED (`BAMLEMCBPIOAS`) | EM USD-denominated bond spread, aggregate |
| VIX | yfinance (`^VIX`) | Risk appetite proxy |
| U.S. HY spread | FRED (`BAMLH0A0HYM2`) | Global risk-off transmission |
| 10Y U.S. Treasury | FRED (`DGS10`) | Dollar funding cost |

### Macro Aggregates (lower frequency, contextual)
| Series | Source | Code |
|---|---|---|
| Chile CPI | FRED | `CPALTT01CLM659N` |
| Brazil CPI | FRED | `BRACPIALLMINMEI` |
| Mexico CPI | FRED | `CPALTT01MXM659N` |
| Commodity index | FRED | `PALLFNFINDEXM` |

---

## Turbulence Index — Mathematical Specification

### Definition (Kritzman & Li, 2010 / State Street)

For period t, define the return vector **r_t** = [r₁_t, r₂_t, ..., rₙ_t]' across N assets.

The turbulence score is:

```
τ_t = (r_t - μ)' Σ⁻¹ (r_t - μ)
```

Where:
- **μ** = sample mean vector (trailing window, typically 252 days)
- **Σ** = sample covariance matrix (trailing window)
- **Σ⁻¹** = inverse of covariance matrix (Moore-Penrose pseudoinverse for stability)

This is the squared Mahalanobis distance: it measures not just the magnitude of returns
but also whether the *cross-sectional correlation structure* is unusual.

### Key Properties
- τ_t ~ χ²(N) under Gaussian returns, so percentile interpretation is straightforward
- High τ when: large return magnitudes AND/OR unusual correlation patterns (decorrelation = panic)
- Stationary transformation: log(τ_t) is more normally distributed for z-scoring

### Implementation Choices
1. **Covariance estimation**: Ledoit-Wolf shrinkage (sklearn) > sample covariance for small N
2. **Rolling window**: 252 trading days default; parameterizable
3. **Turb vs. Corr decomposition**: split τ into magnitude component and correlation component
4. **State classification**: Low / Elevated / High / Extreme regimes via percentile thresholds

---

## Absorption Ratio — Mathematical Specification

For correlation matrix **R** (N×N), compute eigendecomposition:
```
R = V Λ Vᵀ
```

Absorption Ratio:
```
AR = Σᵢ₌₁ᵏ λᵢ / Σᵢ₌₁ᴺ λᵢ
```

Where k = fraction of eigenvectors explaining F% of variance (typically F=20%).

- AR → 1: one dominant factor (systemic, correlated)
- AR → 1/N: uniformly distributed factors (diversified)

Standardized shift (ΔAR) is the actionable signal — a spike in AR preceded most crisis episodes.

---

## Regime Classification

| Turbulence Percentile | Absorption Ratio | Label | Color |
|---|---|---|---|
| < 75th | < 75th | Calm | green |
| 75–90th | any | Elevated | amber |
| > 90th | any | Turbulent | orange |
| > 95th | > 90th | Crisis | red |

---

## Dashboard Layout (Streamlit)

```
┌─────────────────────────────────────────────────────────────┐
│  HEADER: Date, overall risk regime, last update              │
├──────────────┬──────────────────────┬────────────────────────┤
│  Country     │  Turbulence Index    │  Absorption Ratio      │
│  Scorecard   │  Time Series (chart) │  Time Series (chart)   │
│  (5 gauges)  │                      │                        │
├──────────────┴──────────────────────┴────────────────────────┤
│  Cross-Country Heatmap (correlation matrix, rolling)          │
├─────────────────────────────────────────────────────────────┐
│  Spillover to U.S. Portfolios (rolling beta / IR decay)      │
├─────────────────────────────────────────────────────────────┤
│  Intelligence Layer: Written risk narrative (Claude API)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Development Roadmap

### Phase 1 (current): Core engine
- [x] Project structure and spec
- [ ] Data fetcher and cache layer
- [x] Turbulence Index module (fully tested)
- [ ] Absorption Ratio module
- [ ] Basic Streamlit skeleton

### Phase 2: Dynamic factors
- [ ] PCA decomposition on return panel
- [ ] Kalman filter for time-varying loadings
- [ ] Factor interpretation layer

### Phase 3: Spillover and intelligence
- [ ] DCC-GARCH or rolling beta spillover
- [ ] Claude API intelligence narrative
- [ ] PDF/email report generation

### Phase 4: Deployment
- [ ] Streamlit Cloud or self-hosted
- [ ] Scheduled weekly refresh (GitHub Actions)
- [ ] Alert system (email/Slack on regime shift)
