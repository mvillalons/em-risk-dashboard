# EM Risk Dashboard — Claude Code Context

## What this project is

A quantitative systemic risk monitoring tool for LatAm economies (Chile, Brazil,
Mexico, Colombia, Peru) and their spillover effects on U.S. portfolios.
Built in Python. Dashboard in Streamlit. No external database — all state lives
in parquet cache files under data/_cache/.

The author is a financial economist with deep quant background (central banking,
ALM, asset management). Do not over-explain math. Use precise terminology.
When in doubt, ask rather than assume.

---

## Repo structure
```
em_risk_dashboard/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── config/
│   └── universe.yaml
├── core/
│   ├── __init__.py
│   └── returns.py
├── data/
│   ├── __init__.py
│   ├── fetcher.py
│   ├── synthetic.py
│   └── validate.py
├── modules/
│   ├── __init__.py
│   ├── turbulence.py
│   ├── absorption.py
│   └── pca_kalman.py
├── dashboard/
│   ├── __init__.py
│   └── app.py
└── tests/
    ├── __init__.py
    └── test_turbulence.py
```

---

## Architecture principles

Never change these without explicit instruction:
- All modules accept pd.DataFrame with DatetimeIndex as primary input
- All modules return typed dataclasses (TurbulenceResult, AbsorptionResult, etc.)
- Return computation lives in core/returns.py, not in modules
- Cache lives in data/_cache/ as parquet files
- Tests use synthetic deterministic data (seed=42), never live data

Covariance estimation — two-track design (NOT YET IMPLEMENTED):
- SLOW track (window=504, Ledoit-Wolf): feeds turbulence.py only
  Rationale: tau measures deviation from long-run structural normality.
- FAST track (EWMA, lambda~0.94): feeds absorption.py and pca_kalman.py
  Rationale: AR and PCA track current correlation architecture in real time.
- VolStandardizer (EWMA D_t): pre-whitens returns before turbulence computation.
core/covariance.py does not exist yet. This is Session 1 of the roadmap.

---

## Mathematical core

Turbulence: tau_t = (r_t - mu)' Sigma^{-1} (r_t - mu)
  Sigma = SLOW-track covariance (long-run structural reference)
  Vol-standardized target: r_tilde = D_t^{-1/2} r_t, tau on r_tilde

Absorption Ratio: AR_t = sum_{i=1}^{k} lambda_i / sum_{i=1}^{N} lambda_i
  Currently uses rolling sample correlation. Target: FAST-track EWMA R_t.

Dynamic Factors: Rolling PCA + Kalman filter
  Currently: Kalman smooths output factor scores
  Target: Kalman smooths input correlation matrix R_t
  Innovation norm ||R_hat_t - R_{t|t-1}||_F is the new leading indicator

---

## Data universe

FX (log returns): CLP, BRL, MXN, COP, PEN
Equity ETFs (log returns): ECH, EWZ, EWW, GXG, EPU
Global risk: VIX, DXY, SPY, EEM
FRED: EMBI spread, US HY spread, 10Y Treasury (needs FRED_API_KEY env var)

Synthetic data replicates realistic EM vols + regime switching using
t-distributed innovations. Stress episodes hardcoded: 2018H2,
COVID (2020-02-20 to 2020-04-30), 2022H1, 2023-03 SVB.

---

## How to run
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
python data/validate.py
pytest tests/ -v
```

---

## What is done vs not done

Done and tested:
- core/returns.py
- modules/turbulence.py (Mahalanobis tau, decomposition, regimes, TurbulenceResult)
- modules/absorption.py (AR, delta AR, AbsorptionResult)
- modules/pca_kalman.py (rolling PCA, Kalman on factor scores, DynamicFactorResult)
- data/synthetic.py, data/fetcher.py, data/validate.py
- dashboard/app.py (full Streamlit dashboard on synthetic data)
- tests/test_turbulence.py

NOT done — this is the active roadmap:
- core/covariance.py (does not exist)
- Two-track covariance wired into modules
- Vol-standardization in turbulence.py
- EWMA R_t in absorption.py
- Kalman on R_t (not factor scores) in pca_kalman.py
- Live data switch in dashboard
- DCC-GARCH (deferred)

---

## Roadmap — run sessions IN ORDER

### Session 1 — core/covariance.py
Prerequisite: none. Start here.

Create core/covariance.py with three classes:

SlowCovariance:
  Method: fit(returns: pd.DataFrame, window: int = 504, method: str = 'ledoit_wolf')
  Returns dataclass with fields: mu (ndarray), sigma (ndarray), sigma_inv (ndarray)
  Use sklearn LedoitWolf. Pseudoinverse via np.linalg.pinv(rcond=1e-10).
  Raise ValueError if n_samples < n_features (not enough data for LW).

FastCovariance:
  Method: fit(returns: pd.DataFrame, lam: float = 0.94)
  Compute EWMA covariance matrix sigma_t using exponential weights.
  Decompose: sigma_t = D_t @ R_t @ D_t
    D_t = diagonal matrix of conditional vols (sqrt of EWMA variance)
    R_t = correlation matrix (unit diagonal, off-diagonal in [-1,1])
  Enforce PSD on R_t: eigenvalue floor at 1e-6, reconstruct.
  Returns dataclass with fields: R_t (ndarray), D_t (ndarray), sigma_t (ndarray)

VolStandardizer:
  Method: fit_transform(returns: pd.DataFrame, lam: float = 0.94)
  At each date t, compute EWMA vol vector v_t (one scalar per asset).
  Divide each return r_t by v_t elementwise.
  Returns pd.DataFrame with same index and columns as input.
  First min_periods=30 rows returned as NaN.

Create tests/test_covariance.py:
  - SlowCovariance: all eigenvalues of sigma > -1e-10 (PSD)
  - SlowCovariance: sigma_inv @ sigma is close to identity (np.allclose, atol=1e-6)
  - FastCovariance: np.allclose(np.diag(R_t), np.ones(N))
  - FastCovariance: all off-diagonal elements of R_t in [-1, 1]
  - FastCovariance: R_t is PSD
  - VolStandardizer: output column stds are in [0.8, 1.2] on 500-obs synthetic data
  - All three: no crash on input with 5% NaN (filled before computation)

Success criterion: pytest tests/test_covariance.py -v all green.
Do not modify any existing module until this passes.

### Session 2 — turbulence vol-standardization
Prerequisite: Session 1 green.

Modify modules/turbulence.py:
  Import SlowCovariance, VolStandardizer from core.covariance.
  Add parameter vol_standardize: bool = True to compute_turbulence_index().
  Add parameter slow_window: int = 504 to compute_turbulence_index().
  If vol_standardize=True:
    Apply VolStandardizer.fit_transform(returns, lam=0.94) first.
    Use the standardized returns as input to Mahalanobis computation.
  Replace estimate_covariance() call with SlowCovariance.fit().
  All existing TurbulenceResult fields must remain unchanged.
  All existing tests must still pass.

Add to tests/test_turbulence.py — pure vol spike test:
  Generate 300 days of 5-asset normal returns (seed=42, daily vol=0.01).
  At index 250, multiply ALL asset returns by 10 (pure vol spike, no correlation change).
  Run compute_turbulence_index with vol_standardize=True.
  Assert regime at index 250 is NOT Crisis.
  Run again with vol_standardize=False.
  Assert regime at index 250 IS Crisis or Turbulent.
  This confirms vol-standardization is working.

Success criterion: pytest tests/ -v all green including new test.

### Session 3 — EWMA R_t into absorption
Prerequisite: Session 1 green. Session 2 not required.

Modify modules/absorption.py:
  Import FastCovariance from core.covariance.
  Add parameter lam: float = 0.94 to compute_absorption_ratio().
  Replace the rolling .corr() call with FastCovariance.fit(window_data, lam=lam).
  Use R_t from FastCovariance as input to eigendecomposition.
  AbsorptionResult interface unchanged.

After modifying, run this validation in the same session and print output:
  Load synthetic data (generate_em_universe, seed=42).
  Compute AR with old method (lam=None, use rolling window — keep as fallback param).
  Compute AR with new EWMA method (lam=0.94).
  Print a table: date, AR_rolling, AR_ewma for the 30 days around 2020-02-20.
  Confirm EWMA version rises earlier (target: 3-10 days lead at stress onset).

Success criterion: pytest tests/ -v green + empirical lead confirmed in printed output.

### Session 4 — Kalman filter on R_t
Prerequisite: Sessions 1 and 3 complete.

Add KalmanCorrelation class to core/covariance.py:
  Method: fit(R_hat_series: list[ndarray], q_scale: float = 0.01)
    R_hat_series: list of T correlation matrices (N x N each), from FastCovariance
    q_scale: controls how fast the latent R_t can move (Q = q_scale * Var(vech(R_hat)))
  State vector: vech(R_t) — lower triangle excluding diagonal, length N*(N-1)/2
  Q: diagonal matrix, q_scale * empirical variance of each vech element
  R_noise: diagonal matrix, rolling 63-day variance of vech innovations
  After each Kalman update, reconstruct full N x N matrix:
    Set diagonal to 1.0
    Clip off-diagonal to [-0.999, 0.999]
    Enforce PSD: eigenvalue decomposition, floor negative eigenvalues at 1e-6, reconstruct
  Returns: filtered_R (list of N x N arrays), innovations_norm (list of floats)
    innovations_norm[t] = Frobenius norm of (R_hat_t - R_{t|t-1})

Modify modules/pca_kalman.py:
  compute_rolling_pca() gains optional parameter R_t_series: list[ndarray] = None
  If R_t_series is provided, skip internal covariance estimation and use these directly.
  Add field innovation_norm: pd.Series to RollingPCAResult dataclass.

Add compute_dynamic_factors_v2() to pca_kalman.py (keep v1 intact):
  Uses FastCovariance to get R_hat_t at each date.
  Passes R_hat_series to KalmanCorrelation.fit() to get smoothed R_t series.
  Passes smoothed R_t to compute_rolling_pca().
  Returns DynamicFactorResult with innovation_norm populated.

Add tests:
  KalmanCorrelation output matrices all have diagonal = 1.0.
  KalmanCorrelation output matrices are all PSD.
  innovation_norm is a positive float series with no negatives.
  innovation_norm mean during synthetic stress episode > 2x mean during calm.

Success criterion: pytest tests/ -v green.

### Session 5 — Dashboard integration + live data
Prerequisite: Sessions 2, 3, 4 complete.

Modify dashboard/app.py:
  Sidebar additions:
    lam = st.slider("EWMA decay lambda", 0.90, 0.99, 0.94, step=0.01)
    vol_standardize = st.toggle("Vol-standardize turbulence", value=True)
    data_mode = st.radio("Data source", ["Synthetic", "Live"])
  Pass lam and vol_standardize through to all compute calls.
  Live mode: call load_em_universe() from data.fetcher.
    Wrap in try/except. On any failure: st.warning("Live fetch failed, using synthetic")
    then fall back to generate_em_universe().
  Add to Spectral Fragility section (after AR chart):
    innovation_norm time series chart from DynamicFactorResult.
    Title: "Correlation Regime Shock ||delta R||_F"
    Add horizontal line at mean + 2*std.
    Annotate dates above that threshold with vertical dashed lines.

Success criterion: streamlit run dashboard/app.py runs without error.
  Test both Synthetic and Live modes manually.
  Confirm lam slider changes AR responsiveness visibly in the chart.

---

## Style and conventions

- Type hints on all function signatures.
- Dataclasses for all module outputs. No bare tuples from public functions.
- Docstrings: one-line summary, Parameters, Returns, Notes with formula in ASCII.
- np.linalg.pinv over np.linalg.inv everywhere.
- np.random.default_rng(seed) — no global random state.
- No print() in modules. Use logging.getLogger(__name__).
- print() is acceptable in tests/ and data/validate.py only.

## Known gotchas

- yfinance COP=X has multi-day gaps. Always ffill(limit=3).
- CLPUSD=X: verify quote direction on first live fetch.
- LedoitWolf fails on rank-deficient input. Always check n_samples > n_features.
- np.linalg.cholesky fails near-singular. Use eigenvalue floor before Cholesky.
- st.cache_data won't invalidate on code change. Restart streamlit manually.
- SVD eigenvector sign is arbitrary. Do not interpret loading sign across time.
