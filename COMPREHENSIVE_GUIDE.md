# Whitmore Capital Partners — FIN 496 Comprehensive System Guide

> **Purpose**: This document is the authoritative reference for the entire Whitmore Capital Partners quantitative portfolio management system built for the Chapman University FIN 496 Foundation Project. It covers every architectural layer, signal, constraint, design decision, and empirical outcome. It is written for a technically rigorous academic audience.

---

## 1. Executive Summary & System Architecture

### 1.1 Mandate
The system manages a fictional **~USD 1.8 billion** family-office portfolio governed by the **Whitmore Investment Policy Statement (IPS)**. The portfolio is constructed in two layers:

1.  **Strategic Asset Allocation (SAA)**: A long-term policy portfolio rebalanced **annually** (last trading day of the calendar year).
2.  **Tactical Asset Allocation (TAA)**: A monthly overlay that tilts sleeves away from SAA targets based on a multi-signal ensemble, subject to IPS hard constraints.

### 1.2 High-Level Objective
- **Return Target**: 8.0% annualized over rolling 5-year periods.
- **Volatility Ceiling**: 15% annualized (hard).
- **Max Drawdown Tolerance**: 25% peak-to-trough (hard).

### 1.3 Architecture Flow
```
Raw Data (prices + FRED)
    |
    v
Data Audit (Point-in-Time cleaning, gaps preserved)
    |
    v
SAA Construction (Annual Rebalance: 7 methods compared)
    |
    v
Benchmark Construction (BM1: 60/40, BM2: Diversified Policy)
    |
    v
TAA Signal Ensemble (Regime HMM + Trend + Momentum + Macro + VIX)
    |
    v
Monthly Optimization (cvxpy / scipy SLSQP / CVaR / Nested Risk)
    |
    v
Walk-Forward Backtest (5-fold expanding OOS, 21-day embargo)
    |
    v
Attribution, Compliance Audit, Reporting, Figures
```

### 1.4 Design Philosophy
Every module in this codebase adheres to three principles:
1. **Point-in-Time (PIT) Safety**: No forward-looking data ever enters a decision.
2. **Constraint Supremacy**: IPS hard bounds are invariants; signals are suggestions.
3. **Auditability**: Every weight, every breach, every decision is logged to CSV.

---

## 2. The Investment Policy Statement (IPS) Framework

All logic in the codebase is ultimately subservient to the constraints defined in `IPS.md` and summarized in `Guidelines.md`. The config file `taa_project/config.py` is the **single source of truth**.

### 2.1 Asset Universe

| Asset | Ticker | Tier | SAA Min | SAA Target | SAA Max |
|-------|--------|------|---------|------------|---------|
| US Equity | SPXT | Core | 30% | 40% | 45% |
| UK Equity | FTSE100 | Core | 0% | 0% | 10% |
| US Treasury Bonds | LBUSTRUU | Core | 5% | 10% | 15% |
| US TIPS | BROAD_TIPS | Core | 0% | 5% | 10% |
| US REITs | B3REITT | Satellite | 5% | 10% | 20% |
| Gold | XAU | Satellite | 10% | 15% | 25% |
| Japan Equity | NIKKEI225 | Satellite | 0% | 5% | 15% |
| Silver | SILVER_FUT | Satellite | 0% | 5% | 15% |
| China A-shares | CSI300_CHINA | Satellite | 0% | 5% | 15% |
| Bitcoin | BITCOIN | Non-Traditional | 0% | 0% | 5% |
| Swiss Franc | CHF_FRANC | Non-Traditional | 0% | 5% | 10% |

### 2.2 Hard Constraints (Aggregate)
These apply at **all times** to both SAA and TAA weights:
- **Core Floor**: `sum(Core) >= 40%`
- **Satellite Cap**: `sum(Satellite) <= 45%`
- **Non-Traditional Cap**: `sum(NonTraditional) <= 20%` *(Amended from 15% via Resolution 2026-02)*
- **Single Sleeve Max**: `max(any asset) <= 45%`
- **Opportunistic Cap**: `sum(Opportunistic) <= 15%` aggregate, `<= 5%` per asset
- **Fully Invested**: `sum(weights) == 100%` (no cash drag)
- **Short Selling**: Not permitted (`w >= 0`)
- **Transaction Cost**: 5 bps round-trip on all turnover

### 2.3 TAA Bands
When a TAA signal is active, per-asset bounds widen. For example:
- SPXT: `[20%, 45%]` (vs SAA `[30%, 45%]`)
- LBUSTRUU: `[0%, 35%]` (vs SAA `[5%, 15%]`)
- Gold: `[0%, 30%]` (vs SAA `[10%, 25%]`)

These widened bands are defined in `taa_project/config.py` under `TAA_BANDS`.

### 2.4 Benchmarks
- **BM1 (Traditional 60/40)**: 60% SPXT / 40% LBUSTRUU.
- **BM2 (Diversified Policy Portfolio)**: SPXT 40%, NIKKEI225 5%, CSI300 5%, LBUSTRUU 10%, BROAD_TIPS 5%, B3REITT 10%, XAU 15%, SILVER_FUT 5%, CHF_FRANC 5%.

Both benchmarks are fixed-weight and rebalanced annually.

---

## 3. Data Architecture & Pipeline

### 3.1 Data Sources
The system consumes three authoritative files:
1.  `data/asset_data/whitmore_daily.csv`: Daily prices for all SAA and Opportunistic assets.
2.  `data/asset_data/data_key.csv`: Metadata mapping columns to asset names and currencies.
3.  `data/consolidated_csvs/fred/master/fred_data.csv`: Macro features (VIX, HY OAS, Yield Curve, NFCI, DFII10, etc.).

### 3.2 Point-in-Time (PIT) Safety Rules
The `tasks.md` file mandates strict causal data handling, implemented in `taa_project/data_audit.py` and `data_loader.py`:
- **No forward-fill**: Missing values are never imputed with future data.
- **No backward-fill**: Missing values are never imputed with past data.
- **Gaps preserved**: If an asset does not trade on a given day, it shows `NaN`.
- **Returns calculated only between consecutive observed prices**:
```python
# From data_audit.py -> compute_consecutive_log_returns
observed = series.dropna()
log_returns = np.log(observed / observed.shift(1))
return log_returns.reindex(series.index) # NaN preserved for gap days
```
- **FRED Lag**: All macro features are shifted forward by **1 business day** before use, ensuring that a feature observed on day `t` is only available for decisions on day `t+1`.

### 3.3 Expanding Universe
Assets are not required to exist from day one. The portfolio starts on **2000-01-03** with the subset of assets available at that time. Assets are introduced at the **next scheduled rebalance date** after their data begins.
- **Inception dates**:
  - `SPXT, FTSE100, LBUSTRUU, BROAD_TIPS, NIKKEI225, XAU, SILVER_FUT, CHF_FRANC`: 2000-01-03
  - `CSI300_CHINA`: 2002-01-04
  - `B3REITT`: 2003-03-31
  - `BITCOIN`: 2010-07-19

---

## 4. Strategic Asset Allocation (SAA)

The SAA layer is built in `taa_project/saa/build_saa.py` and `taa_project/saa/saa_comparison.py`. It produces a policy portfolio that drifts between annual rebalances.

### 4.1 Rebalance Schedule
Rebalances occur on the **last common investable date** of each calendar year, defined as the last date in the year where every asset currently in the expanding universe has an observed price.

### 4.2 Covariance Estimation
At each rebalance date, the covariance matrix is estimated using only data `<=` that date:
```python
# From build_saa.py -> estimate_covariance
history = returns.loc[:as_of_date, assets].tail(LOOKBACK_DAYS) # 756 days

covariance = history.cov(min_periods=MIN_COV_OBSERVATIONS) # 63 min obs
# Shrinkage: 75% sample + 25% diagonal
covariance = 0.75 * covariance + 0.25 * np.diag(np.diag(covariance))
# Annualize and force PSD
covariance_values = covariance.to_numpy(dtype=float) * 252.0
eigenvalues, eigenvectors = np.linalg.eigh(covariance_values)
eigenvalues = np.clip(eigenvalues, DIAGONAL_FLOOR, None)
psd_values = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
```

### 4.3 SAA Methods (All 7 Methods)
The system supports seven SAA construction methods, selectable via `--saa-method`. All share the same annual rebalance schedule, expanding universe, IPS hard constraints, and 5 bps round-trip turnover cost.

#### 1. Equal Weight (1/N)
```python
def solve_equal_weight(assets):
    raw = np.full(len(assets), 1.0 / len(assets))
    return _feasible_projection(raw, assets)
```
Every available asset receives identical weight, then projected into the IPS feasible set. This is the simplest baseline with no estimation error, but it ignores covariance structure entirely.

#### 2. Inverse Volatility
```python
def solve_inverse_vol(assets, covariance):
    vols = np.sqrt(np.diag(covariance.loc[assets, assets].values))
    raw = (1.0 / vols) / (1.0 / vols).sum()
    return _feasible_projection(raw, assets)
```
Weights are proportional to the inverse of each asset's realized volatility. This uses only diagonal covariance information and is robust to estimation error in correlations.

#### 3. Constrained Minimum Variance (default)
Solves:
```
minimize:  w'Σw
subject to:
  sum(w) = 1.0
  w >= 0
  w_i in [SAA_Lower_i, SAA_Upper_i]
  sum(Core) >= 40%
  sum(Satellite) <= 45%
  sum(NonTraditional) <= 20%
```
Using `scipy.optimize.minimize` (SLSQP). This method produced the highest SAA-only Sharpe (~0.56) because it uses covariance information without requiring expected return estimates.

#### 4. Constrained Risk Parity
Solves:
```
minimize:  Σ(RC_i - RB_i)²
where RC_i = w_i * (Σw)_i / (w'Σw)
      RB_i = target risk budget from IPS target weights
```
Using IPS targets as risk budgets (rather than 1/N) is the correct choice because: (a) the IPS targets encode the CIO's policy views on desired risk exposure, and (b) true 1/N ERC cannot be achieved under binding IPS minimum constraints on high-volatility assets (SPXT >= 30%, XAU >= 10%).

#### 5. Maximum Diversification
Maximizes the diversification ratio:
```
DR(w) = (w'σ) / sqrt(w'Σw)
```
where σ is the vector of asset volatilities. This seeks the portfolio where the weighted average volatility most exceeds the portfolio volatility, i.e., the portfolio that extracts the maximum correlation benefit.

#### 6. Hierarchical Risk Parity (HRP)
Lopez de Prado (2016) HRP algorithm:
```python
# 1. Correlation distance
d_ij = sqrt(0.5 * (1 - rho_ij))

# 2. Single-linkage hierarchical clustering
link = linkage(squareform(dist), method="single")

# 3. Quasi-diagonalization
sort_idx = _quasi_diagonal(link)

# 4. Recursive bisection using inverse-variance allocations
hrp_weights = _recursive_bisection(cov, sort_idx)
```
HRP does not invert the covariance matrix, making it robust to ill-conditioned estimates. It produced competitive risk-adjusted returns with lower max drawdown than mean-variance methods.

#### 7. Mean-Variance (Black-Litterman + Momentum)
Uses a blended expected return vector:
```
μ = (1 - α) * π_BL  +  α * mom_signal
```
Where:
- `π_BL = rf + λ * Σ * w_BM2` (Black-Litterman equilibrium implied returns using BM2 as the reference portfolio)
- `mom_signal` = 12-1 month cross-sectional momentum (Jegadeesh-Titman specification), winsorized at 10/90th percentile
- `α = 0.15` (momentum blend weight)

If the tangency portfolio exceeds the internal vol target (8%), the solver falls back to vol-constrained return maximization.

BM2 is the natural equilibrium because it is the IPS policy portfolio — it has 0% Bitcoin, preventing the sample-mean from extrapolating Bitcoin's bull-market history as a long-run expected return.

### 4.4 SAA Simulation
Between annual rebalances, the SAA portfolio is **not** traded. It drifts with market movements. If drift causes an IPS constraint breach, an **unscheduled compliance rebalance** is triggered, projecting drifted weights back into the feasible set.

---

## 5. TAA Signal Stack

The TAA overlay is driven by an ensemble of **five independent signals** combined into a single expected-return proxy `mu`. This is the core intellectual contribution of the project.

### 5.1 Signal 1: Regime HMM (`taa_project/signals/regime_hmm.py`)
**Economic Thesis**: Financial conditions stress and yield curve inversion precede equity drawdowns. A Hidden Markov Model (HMM) classifies the macro state into `risk_on`, `neutral`, or `stress`.

**Implementation**:
- **Features**: 4 stationary stress indicators: `VIXCLS`, `BAMLH0A0HYM2`, `T10Y3M`, `NFCI`.
- **Critical Design Choice**: DFII10 (10Y TIPS real yield) was **intentionally excluded** from the HMM. A previous version used 5 features, but DFII10's persistent decline from 2003-2021 (due to QE) caused the HMM to misclassify nearly the entire 2009-2021 bull market as `risk_on`, destroying the signal's defensive utility.
- **Model**: 3-state Gaussian HMM with full covariance, fit monthly on an **expanding window**.
- **State Interpretation**: States are ordered by a "stress score" derived from the fitted state means. The state with the highest mean VIX/HY-OAS/NFCI is labeled `stress`.

```python
# From regime_hmm.py -> _state_stress_scores
for state in range(model.n_components):
    means = model.means_[state]
    score = 0.0
    for stress_col in ["VIXCLS", "BAMLH0A0HYM2", "NFCI"]:
        score += means[feature_positions[stress_col]]
    for calm_col in ["T10Y3M"]:
        score -= means[feature_positions[calm_col]]
```

**Regime Tilt Vector**: Based on the classified regime, a static tilt vector is assigned to each asset. For example, in `stress`:
- SPXT: 25% (down from 35% neutral)
- LBUSTRUU: 25% (up from 15%)
- XAU: 20% (up from 15%)
- BITCOIN: 0% (down from 3%)

### 5.2 Signal 2: Faber Trend (`taa_project/signals/trend_faber.py`)
**Economic Thesis**: Long-horizon trend persistence (Faber, 2007). Assets above their moving average have materially higher risk-adjusted returns.

**Implementation**:
- 120-observation simple moving average per asset.
- Trend score is `tanh((price / SMA - 1) / sigma_60d)`.
- Uses **only observed prices**; gaps remain `NaN`.

```python
# From trend_faber.py -> trend_score
sma = _observed_rolling_mean(price_series, 120)
sigma_60d = _observed_return_volatility(price_series, 60)
normalized_distance = (price_series / sma - 1.0) / sigma_60d
scores = np.tanh(normalized_distance)
```

### 5.3 Signal 3: Accelerating Dual Momentum (ADM) (`taa_project/signals/momentum_adm.py`)
**Economic Thesis**: Cross-sectional momentum among equities and real assets; absolute momentum filter avoids investing during market-wide negative trends (Antonacci, 2012).

**Implementation**:
- Blended total return over 1/2/3/6-month windows (21, 42, 63, 120 trading days).
- Cross-sectionally ranked within buckets (Equity, Fixed Income, Real Assets, etc.).
- **Absolute Filter**: Assets with non-positive blended momentum are clipped to `0` on the positive side.

```python
# From momentum_adm.py -> adm_score
components = [period_return(prices, window) for window in (21, 42, 63, 120)]
blended = sum(components) / 4.0
# Cross-sectional rank within buckets
percentile_rank = bucket_scores.rank(axis=1, pct=True, method="average")
centered_rank = (percentile_rank - 0.5) * 2.0
# Absolute filter
bucket_output = bucket_output.where(bucket_scores > 0.0, bucket_output.clip(upper=0.0))
```

### 5.4 Signal 4: Asset-Specific Macro Factor (`taa_project/signals/macro_factor.py`)
**Economic Thesis**: Real yields, credit spreads, and crypto momentum help distinguish inflationary stress from deflationary stress.

**Implementation** (3 sub-signals):

**A. Real-Yield Tilt**:
- Driver: `DFII10` (10Y TIPS real yield).
- Hypothesis: Falling real yields reduce the opportunity cost of holding non-yielding real assets (Gold, Silver, TIPS).
- Signal: `z = rolling_zscore(DFII10, 63)`. Asset loading * `-z`.
- **Why 63 days?** A 252-day window produced a near-constant negative z-score from 2009-2021, creating a structural long-bias in real assets that inflated backtest returns by ~3-4%.

**B. Credit-Premium Tilt**:
- Driver: `BAMLH0A0HYM2 - BAMLC0A0CM` (HY - IG OAS spread).
- Hypothesis: Tightening spread = risk-on (favor equities, REITs); Widening = risk-off (favor bonds).
- Monthly correlation with SPXT: -0.60.

**C. Crypto-Momentum Tilt**:
- Driver: Bitcoin's own 3/6/12-month absolute momentum.
- Only `BITCOIN` receives a non-zero score. Capped at ±60% annualized to prevent blow-off-top leverage.

**Asset Loadings** (hand-specified based on economic theory, not regression):
- XAU real-yield loading: `0.18`
- SPXT credit-premium loading: `0.14`
- BITCOIN crypto loading: `0.60` (capped)

### 5.5 Signal 5: VIX / Yield-Curve Trip-Wire (`taa_project/signals/vix_yield_curve.py`)
**Economic Thesis**: A fast, auditable stress overlay. Reacts to VIX spikes faster than the monthly HMM.

**Implementation**:
- Uses lagged `VIXCLS` and `T10Y3M` only.
- Produces a defensive tilt when VIX is elevated or the yield curve is deeply inverted.
- Blended at **10%** into the final signal score.

### 5.6 Ensemble Scoring
The five signals are combined in `taa_project/optimizer/cvxpy_opt.py`:

```python
# From cvxpy_opt.py -> ensemble_score
score = (
    cfg.regime_weight * regime_tilt * cfg.regime_scale      # 0.20 * 0.10
    + cfg.trend_weight * trend_sig * cfg.trend_scale          # 0.30 * 0.06
    + cfg.momo_weight * momo_sig * cfg.momo_scale             # 0.30 * 0.06
    + cfg.macro_factor_weight * macro_mu * cfg.macro_scale    # 0.20 * 0.20
)
# Final blend with VIX trip-wire (10%)
mu = 0.90 * score + 0.10 * vix_tilt
```

**Why these weights?** This is the result of an extensive root-cause analysis documented in `DECISIONS.md`:
- `macro_factor_weight` was originally 0.40, then 0.05, then settled at **0.20**. Too high, and macro factor persistence crowded out protective regime signals. Too low, and the system lost the ability to distinguish inflationary vs deflationary stress.
- `regime_weight` was restored to **0.20** after being cut to 0.10, because the HMM is the primary risk governor.

---

## 6. Portfolio Optimization

### 6.1 Monthly TAA Objective
At each month-end decision date `t`, the optimizer solves:

```
maximize:  μ'w - λ·w'Σw - c·||w - w_prev||₁
subject to:
  sum(w) = 1.0
  w >= 0 (no shorts)
  w_i in [TAA_Lower_i, TAA_Upper_i] for all i
  sum(Core) >= 40%
  sum(Satellite) <= 45%
  sum(NonTraditional) <= 20%
  sum(Opportunistic) <= 15%
  max(w_i) <= 45%
  w'Σw <= vol_budget²  (soft, penalized slack)
```

Where:
- `μ`: Expected return proxy from the signal ensemble.
- `Σ`: Annualized covariance matrix (70% sample + 30% diagonal shrinkage).
- `λ`: Risk aversion (default 1.5).
- `c`: Turnover cost = 5 bps = `0.0005`.

### 6.2 Default Solver: SLSQP
The default mode uses **`scipy.optimize.minimize` with SLSQP**:

```python
# From cvxpy_opt.py -> _solve_vol_taa_scipy
def objective(weights):
    variance = weights @ sigma @ weights
    turnover = np.abs(weights - prev).sum()
    return -mu @ weights + risk_aversion * variance + turnover_cost * turnover

constraints = [sum_to_one, *inequalities]
constraints.append({"type": "ineq", "fun": lambda w: vol_budget**2 - w @ sigma @ w})

result = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
```

### 6.3 CVaR Optimization (Rockafellar-Uryasev)
An optional **CVaR mode** is available using `cvxpy` with the Rockafellar-Uryasev (2000) linear programming formulation:

```
minimize:   ξ + (1 / ((1-α) * S)) * Σ_s u_s
subject to:
  u_s >= -w'r_s - ξ          for all scenarios s = 1..S
  u_s >= 0
  sum(w) = 1.0
  w >= 0
  w_i in [Lower_i, Upper_i]
  aggregate IPS constraints
```

Where:
- `α` = confidence level (default 0.95)
- `S` = number of historical return scenarios (default 252-day lookback)
- `r_s` = vector of asset returns on scenario day `s`
- `ξ` = VaR auxiliary variable
- `u_s` = excess-loss auxiliary variables

The ECOS solver is used for its lighter memory footprint vs SCS. Scenarios are stored in `float32` to halve memory usage.

```python
# From cvxpy_opt.py -> _build_cvar_scenarios and _solve_cvar_taa
scenario_matrix = _build_cvar_scenarios(
    asset_log_returns, decision_date, universe, lookback_days=252
)
# scenario_matrix shape: (S, n_assets)
# each row is one day's asset log-return vector
```

### 6.4 Nested Sleeve Risk Budgeting
The `--nested-risk` mode solves three independent per-sleeve optimizations, then blends them at strategic sleeve weights.

**Algorithm**:
1.  **Sleeve Availability Check**: Determine which assets in each sleeve (Core, Satellite, Non-Traditional) have prices on the decision date.
2.  **Sleeve Weight Renormalization**: If a sleeve has no available assets, redistribute its strategic weight to active sleeves while preserving the Core floor (>= 40%). Log any renormalization to `nested_renormalizations.csv`.
3.  **Per-Sleeve Optimization**: For each active sleeve:
    - Normalize TAA bounds by the sleeve's strategic weight.
    - Solve either: (a) mean-variance with vol target, or (b) CVaR with sleeve-specific budget.
    - Vol targets: Core = 6%, Satellite = 10.5%, Non-Traditional = 15%.
4.  **Blend**: `w_total = Σ_sleeve (sleeve_weight * w_sleeve)`
5.  **Assert Outer Constraints**: Verify the blended weights satisfy all IPS hard constraints.

```python
# From nested_risk.py -> solve_nested_taa
active_sleeve_weights, changed = _active_sleeve_weights(config.sleeve_weights, sleeve_assets)
for sleeve_name, sleeve_weight in zip(SLEEVE_NAMES, active_sleeve_weights):
    assets = sleeve_assets[sleeve_name]
    if use_cvar:
        sleeve_solution = _solve_cvar_taa(...)
    else:
        sleeve_solution = _solve_vol_sleeve(
            assets, expected_returns, cov_matrix, previous_weights,
            sleeve_weight, vol_targets[sleeve_name]
        )
    blended.loc[assets] = sleeve_weight * sleeve_solution
_assert_outer_constraints(blended, available)
```

**Why nested risk?** It prevents a single high-volatility satellite asset from dominating the portfolio's risk budget. By capping each sleeve's ex-ante volatility independently, the optimizer cannot arbitrage the aggregate constraint by hiding risk inside a loosely bounded sleeve.

### 6.5 Black-Litterman Stress Views
The `--bl-stress-views` mode modifies the mean-variance expected return vector when the HMM detects a `stress` regime:

```
π_stress = π_equilibrium - σ_shock * σ_asset   for all equity assets
```

Where `σ_shock` defaults to 1.0 annualized standard deviations. This pessimistic prior shifts the optimal portfolio away from equities during stress without requiring a separate optimization objective.

### 6.6 Fallback Mechanism
If the optimizer fails (infeasible or solver error), the system:
1.  Logs an incident to `outputs/breaches.log`.
2.  Falls back to the **previously held weights** projected into the current feasible region.
3.  If that also fails, falls back to **BM2 target weights** projected into the feasible region.

### 6.7 Opportunistic Alpha Sleeve
The TAA overlay may allocate to Appendix A opportunistic assets (e.g., Copper, Ethereum, Israeli Shekel). A signal-ranked opportunistic sleeve is carved out:
- Max **5%** per asset / **8%** aggregate *(tighter than the IPS 15% cap; walk-forward showed the full 15% increased short-window vol without improving risk-adjusted returns)*.
- Assets are ranked by a 60/40 blend of trend and momentum scores.
- Only assets with a score >= 0.15 are eligible.

### 6.8 Optional Risk Overlays
- **Drawdown Guardrail**: If trailing 6-month drawdown exceeds a threshold, the active vol budget is halved until recovery.
- **Daily Risk Governor**: Disabled by default. If enabled, enters a defensive TAA target (heavily weighted to bonds and CHF) when realized 21d/63d vol exceeds 12% or drawdown exceeds -3%.

---

## 7. Walk-Forward Backtest

### 7.1 Fold Construction
The OOS period (default 2003-01-01 to 2025-12-31) is split into **5 contiguous expanding folds** with a **21-business-day embargo** between the training end and the first test decision.

```python
# From walkforward.py -> build_walkforward_folds
split_arrays = np.array_split(decision_dates, folds)
for fold_id, chunk in enumerate(split_arrays, start=1):
    test_start = chunk[0]
    train_end = test_start - pd.offsets.BDay(21) # 21-day embargo
```

The embargo prevents information leakage from the HMM training window into the first test decision.

### 7.2 Decision Dates
Monthly rebalance dates are the **last common investable date** in each month. This means the last date in the month where every asset in the *current* expanding universe has an observed price. This prevents scheduling rebalances on crypto-only weekends.

### 7.3 Simulation Loop
For each decision date:
1.  **Refit HMM** on expanding window data up to the decision date (if it's the first date of a new fold, fit only up to `train_end`; otherwise up to decision date).
2.  **Build Signal Bundle**: Compute regime, trend, momentum, macro, and VIX signals using only data `<= t`.
3.  **Estimate Covariance**: 252-day trailing window, annualized, shrunk.
4.  **Solve Optimizer**: Get target weights.
5.  **Simulate Daily Returns**: Between decision date `t` and next decision date `t+1`:
    - Apply daily asset returns to holdings.
    - Charge 5 bps turnover cost on the rebalance day.
    - If drifted weights breach IPS constraints, trigger a compliance rebalance (project back to feasible set) and charge additional turnover costs.

### 7.4 Daily Mechanics & Compliance Rebalances
Between scheduled rebalances, the portfolio drifts. The simulation checks every trading day:

```python
# From walkforward.py simulation logic
gross = 1.0 + daily_asset_returns
post_move = current_weights * gross
current_weights = post_move / post_move.sum()

# Check for IPS breach
if breached:
    projected = project_weights_to_feasible_set(
        current_weights, lower_bounds, upper_bounds, active_assets
    )
    turnover = np.abs(projected - current_weights).sum()
    cost = turnover * COST_PER_TURNOVER
    current_weights = projected
```

This ensures the portfolio never holds an IPS-infeasible position for more than one trading day.

### 7.5 Output Artifacts
The backtest produces:
- `oos_returns.csv`: Daily portfolio returns, gross return, turnover, turnover cost.
- `oos_weights.csv`: Target weights at each decision date.
- `oos_holdings.csv`: Start-of-day holdings for every day.
- `oos_regimes.csv`: Regime label and probabilities at each decision date.
- `guardrail_switches.csv`: Log of drawdown guardrail tighten/release events.
- `walkforward_folds.csv`: Metadata for each fold (train/test dates, embargo).

---

## 8. Attribution & Diagnostics

### 8.1 Performance Attribution
Implemented in `taa_project/analysis/attribution.py`.

**SAA vs BM2**:
Computes active return from the difference between SAA daily holdings and BM2 daily holdings, multiplied by realized asset returns.

**TAA vs SAA / BM1 / BM2**:
Similarly computes the incremental return contributed by the TAA overlay's active weights vs the underlying benchmarks.

### 8.2 Signal Ablations (Leave-One-Out)
To isolate the marginal contribution of each signal layer, the system reruns the full walk-forward backtest with one signal weight set to zero:
- `no_regime`: `regime_weight = 0`
- `no_trend`: `trend_weight = 0`
- `no_momo`: `momo_weight = 0`
- `no_macro`: `macro_factor_weight = 0`

The difference in OOS Sharpe, return, and drawdown between the baseline and the ablated run quantifies that signal's value.

### 8.3 Deflated Sharpe Ratio (DSR)
The `TRIAL_LEDGER.csv` records every configuration run. The **Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014) adjusts the observed Sharpe ratio for the number of trials conducted, penalizing data-mining bias.

```python
# From analysis/common.py -> deflated_sharpe_ratio
daily_rf = (1.0 + RISK_FREE_RATE) ** (1.0 / 252) - 1.0
excess = returns - daily_rf
sr = sample_sharpe_ratio(returns)

skew = stats.skew(excess, bias=False)
kurt = stats.kurtosis(excess, fisher=False, bias=False)
denominator_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
sr_variance = denominator_term / (len(excess) - 1)
sr_std = sqrt(sr_variance)

if n_trials > 1:
    q1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * e))
    expected_max_sr = sr_std * ((1.0 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2)
else:
    expected_max_sr = 0.0

z_score = (sr - expected_max_sr) * sqrt(len(excess) - 1) / sqrt(denominator_term)
dsr = stats.norm.cdf(z_score)
```

This produces a probability in `[0, 1]` that the observed Sharpe is statistically significant after accounting for multiple trials.

---

## 9. Reporting & Compliance

### 9.1 Metrics Table
`taa_project/analysis/reporting.py` computes a comprehensive metrics table for BM1, BM2, SAA, and SAA+TAA:
- Annualized Return, Volatility, Sharpe, Sortino, Calmar
- Max Drawdown, Historical VaR (95%)
- Hit Rate, Turnover per year, Cost drag per year
- Average weights by tier (Core, Satellite, Non-Trad, Opportunistic)

### 9.2 IPS Compliance Audit
The system performs a **daily audit** of realized holdings against every IPS hard constraint. It checks:
- Sum to 100%, no shorts
- Core floor, Satellite/Non-Trad/Opportunistic caps
- Single-sleeve max
- Per-asset SAA and TAA band bounds
- Rolling realized vol (21d, 63d, 252d) vs 15% ceiling
- Drawdown vs 25% tolerance

```python
# From compliance.py -> audit_ips_compliance
for date, row in aligned.iterrows():
    core_weight = float(row.reindex(CORE).sum())
    satellite_weight = float(row.reindex(SATELLITE).sum())
    nontrad_weight = float(row.reindex(NONTRAD).sum())
    opportunistic_weight = float(row.reindex(OPPORTUNISTIC).sum())

    checks = [
        ("core_floor", core_weight >= CORE_FLOOR - 1e-8),
        ("satellite_cap", satellite_weight <= SATELLITE_CAP + 1e-8),
        ("nontrad_cap", nontrad_weight <= NONTRAD_CAP + 1e-8),
        ("opportunistic_cap", opportunistic_weight <= opportunistic_cap + 1e-8),
    ]

rolling_vol = rolling_21d_volatility(returns)
for date, value in rolling_vol.loc[rolling_vol > VOL_CEILING + 1e-8].items():
    rows.append({"date": date, "rule": "rolling_21d_vol_ceiling", "value": float(value)})

drawdown = drawdown_series(returns)
for date, value in drawdown.loc[drawdown < -MAX_DD - 1e-8].items():
    rows.append({"date": date, "rule": "max_drawdown", "value": float(value)})
```

Violations are logged in `ips_compliance.csv`. Note: realized vol and drawdown breaches are treated as **soft** violations (market-driven), while weight-cap breaches are **hard** failures that trigger an exception.

### 9.3 Compliance Rebalance Logging
When a drift-induced compliance rebalance occurs, the system logs structured rows:

```python
# From compliance.py -> compliance_breach_rows
checks = [
    ("sum_to_one", abs(total_weight - 1.0) > tolerance),
    ("no_short", min_weight < -tolerance),
    ("core_floor", core < CORE_FLOOR - tolerance),
    ("satellite_cap", satellite > SATELLITE_CAP + tolerance),
    ("nontrad_cap", nontrad > NONTRAD_CAP + tolerance),
    ("opportunistic_cap", opportunistic > OPPO_CAP + tolerance),
    ("single_sleeve_cap", max_weight > SINGLE_SLEEVE_MAX + tolerance),
    ("lower_bound", weight < lower - tolerance),
    ("upper_bound", weight > upper + tolerance),
]
```

### 9.4 Figures
The reporting module generates 18+ high-DPI figures:
1.  `fig01_cumgrowth.png`: Cumulative growth of all 4 portfolios
2.  `fig02_drawdown.png`: Underwater chart
3.  `fig03_rolling_vol.png`: Rolling realized volatility
4.  `fig04_taa_weights_stacked.png`: Stacked area chart of TAA weights
5.  `fig05_regime_shading.png`: Portfolio growth with regime shading
6.  `fig06_oos_folds.png`: OOS fold boundaries
7.  `fig07_attribution_bar.png`: TAA attribution bar chart
8.  `fig08_per_fold_oos.png`: Per-fold OOS performance
9.  `fig09_signal_history.png`: Signal history over time
10. `fig10_contribution.png`: Asset contribution chart
11. `fig11_rolling_alpha.png`: Rolling alpha vs BM2
12. `fig12_regime_forward_returns.png`: Forward returns by regime
13. `fig13_annual_returns.png`: Annual return bar chart
14. `fig14_risk_return_scatter.png`: Risk/return scatter
15. `fig15_monthly_heatmap.png`: Monthly return heatmap
16. `fig16_annual_costs.png`: Annual transaction cost chart
17. `fig17_correlation_heatmap.png`: Asset correlation heatmap
18. `fig18_cumulative_alpha.png`: Cumulative alpha vs BM2

### 9.5 Reports
- **PDF Report**: Generated by `report/build_report.py` (Markdown -> PDF via ReportLab).
- **PowerPoint Deck**: Generated by `report/build_pptx.py`.

---

## 10. Configuration Runs & Empirical Results

### 10.1 Canonical Sweep
The `Makefile` and `taa_project/scripts/run_sweep.py` run a canonical 13-configuration sweep to test robustness:

| Run ID | Description | Key Parameters |
|--------|-------------|----------------|
| `baseline` | No TimesFM, 10% vol budget | `--no-timesfm --vol-budget 0.10` |
| `timesfm_vb10` | TimesFM enabled, 10% vol | `--timesfm --vol-budget 0.10` |
| `timesfm_vb08` | TimesFM enabled, 8% vol | `--timesfm --vol-budget 0.08` |
| `timesfm_vb07` | TimesFM enabled, 7% vol | `--timesfm --vol-budget 0.07` |
| `timesfm_regime_vb` | Regime-conditional vol budgets | `{"risk_on":0.10,"neutral":0.08,"stress":0.05}` |
| `timesfm_regime_dd` | Regime vol + drawdown guardrail | `--dd-guardrail` |
| `cvar95_vb_2_5` | CVaR optimizer (alpha=0.95, budget=2.5%) | `--optimizer-mode cvar` |
| `cvar99_vb_4_0` | CVaR optimizer (alpha=0.99, budget=4.0%) | `--optimizer-mode cvar` |
| `nested_risk_default` | Nested sleeve risk budgeting | `--nested-risk` |
| `nested_risk_cvar` | Nested sleeve + CVaR | `--nested-risk --optimizer-mode cvar` |
| `hrp_saa` | HRP for SAA instead of min-var | `--saa-method hrp` |
| `bl_stress_full` | Black-Litterman stress views | `--bl-stress-views` |
| `kitchen_sink` | All optional features enabled | Combined flags |

### 10.2 Final Submission
The final submission configuration is the **default non-TimesFM pipeline** (`--no-timesfm`) with the internal volatility budget set to the default `TARGET_VOL = 0.12` (12%). TimesFM was explicitly excluded from the final submission.

**Why?**
- The non-TimesFM configuration achieved the **best balance of return, risk, and constraint compliance**.
- It is **the only tested configuration that satisfied all IPS hard constraints** simultaneously:
  - Return objective: **8.39%** (target: 8.0%)
  - Volatility ceiling: **7.25%** (limit: 15.0%)
  - Max drawdown: **-21.92%** (tolerance: -25.0%)
- It beat both benchmarks on risk-adjusted metrics.
- It avoids the dependency risk and memory overhead of the TimesFM optional layer.

### 10.3 Empirical Performance (from main outputs)
Final submission run (non-TimesFM, 5 folds, 2003-01-30 to 2026-04-15):
- **Annualized Return**: **8.39%**
- **Annualized Volatility**: **7.25%**
- **Max Drawdown**: **-21.92%**
- **Sharpe Ratio**: **0.88**
- **Sortino Ratio**: **1.24**
- **Calmar Ratio**: **0.38**
- **Average Core Weight**: ~57.7%
- **Average Satellite Weight**: ~30.7%
- **Average Non-Traditional Weight**: ~8.1%
- **Average Opportunistic Weight**: ~3.5%

Benchmark comparison over the same period:
- **BM1 (60/40)**: 5.24% return, 9.59% vol, -33.9% MDD, Sharpe 0.34
- **BM2 (Diversified)**: 7.84% return, 9.61% vol, -35.2% MDD, Sharpe 0.61

SAA-only comparison (from SAA method comparison):
- **Minimum Variance SAA**: Sharpe ~0.56
- **Mean-Variance SAA**: Sharpe ~0.51
- **Risk Parity SAA**: Sharpe ~0.47

This demonstrates that the **TAA overlay adds substantial risk-adjusted value** over the SAA baseline while maintaining full IPS compliance.

### 10.4 Why This Configuration Works
- **Conservative signal ensemble**: The 0.12 internal vol budget, combined with the quadratic risk penalty (`λ = 1.5`) and 70/30 covariance shrinkage, naturally produces realized volatility well below the 15% ceiling.
- **Defensive regime detection**: The 4-feature HMM (without the DFII10 bias) correctly identifies stress periods, tilting the portfolio toward bonds and gold.
- **Macro factor refinement**: The 63-day z-score window prevents structural long-bias while still capturing genuine macro signals.
- **Opportunistic cap discipline**: The 8% aggregate opportunistic cap prevents excessive short-window volatility from speculative positions.

---

## 11. Code Architecture, Testing & Build Process

### 11.1 Module Map
```
FIN496-Foundation-Project/
|
|-- IPS.md                          # Full Investment Policy Statement
|-- Guidelines.md                   # Condensed rule set
|-- tasks.md                        # Data handling constraints
|-- DECISIONS.md                    # Engineering decision log
|-- TRIAL_LEDGER.csv                # Audit trail of all runs
|-- requirements.txt                # Python dependencies
|-- Makefile                        # Build orchestration (test, pipeline, zip)
|
|-- data/
|   |-- asset_data/
|   |   |-- whitmore_daily.csv      # Authoritative price panel
|   |   |-- data_key.csv            # Asset metadata
|   |-- consolidated_csvs/fred/     # Macro data
|
|-- taa_project/                    # Main Python package
|   |
|   |-- main.py                     # PIPELINE ORCHESTRATOR
|   |   |-- run_pipeline()          # Executes Tasks 1-11 in sequence
|   |   |-- CLI entrypoint          # Accepts --vol-budget, --timesfm, etc.
|   |
|   |-- config.py                   # SINGLE SOURCE OF TRUTH
|   |   |-- Asset lists (CORE, SATELLITE, NONTRAD, OPPORTUNISTIC)
|   |   |-- SAA_TARGETS, SAA_BANDS, TAA_BANDS
|   |   |-- Hard constraints (CORE_FLOOR, SATELLITE_CAP, etc.)
|   |   |-- BM1_WEIGHTS, BM2_WEIGHTS
|   |
|   |-- data_loader.py              # Audited data accessors
|   |-- data_audit.py               # Data validation, return construction
|   |-- benchmarks.py               # BM1 & BM2 construction
|   |-- compliance.py               # IPS breach logging utilities
|   |-- memory.py                   # RAM guards for subprocesses
|   |-- pandas_utils.py             # PIT-safe helpers
|   |
|   |-- signals/                    # TAA SIGNAL MODULES
|   |   |-- regime_hmm.py           # Signal 1: 3-state Gaussian HMM
|   |   |-- trend_faber.py          # Signal 2: 120-day SMA trend
|   |   |-- momentum_adm.py         # Signal 3: ADM cross-sectional momentum
|   |   |-- macro_factor.py         # Signal 4: Real yield, credit, crypto
|   |   |-- vix_yield_curve.py      # Signal 5: Fast stress trip-wire
|   |   |-- dd_guardrail.py         # Drawdown clipping overlay
|   |
|   |-- optimizer/                  # PORTFOLIO SOLVERS
|   |   |-- cvxpy_opt.py            # Main TAA optimizer (SLSQP + CVaR)
|   |   |-- nested_risk.py          # Sequential sleeve risk budgeting
|   |
|   |-- saa/                        # STRATEGIC ALLOCATION
|   |   |-- build_saa.py            # Annual SAA builder (min-var / RP / HRP)
|   |   |-- saa_comparison.py       # All 7 SAA methods + BL + HRP
|   |
|   |-- backtest/                   # BACKTEST ENGINE
|   |   |-- walkforward.py          # 5-fold expanding OOS backtest
|   |   |-- run_backtest.py         # Standalone backtest runner
|   |   |-- sweep_vol_budgets.py    # Volatility budget sweep
|   |
|   |-- analysis/                   # ANALYTICS & REPORTING
|   |   |-- attribution.py          # SAA/TAA performance attribution
|   |   |-- reporting.py            # Metrics, figures, IPS audit
|   |   |-- common.py               # Shared metric functions (Sharpe, DSR, etc.)
|   |   |-- bridge_comparison.py    # Cross-run comparison logic
|   |   |-- config_comparison.py    # Sweep result consolidation
|   |
|   |-- report/                     # OUTPUT GENERATION
|   |   |-- build_report.py         # Markdown -> PDF report
|   |   |-- build_pptx.py           # PowerPoint slide deck
|   |   |-- build_deck.py           # Deck orchestration
|   |
|   |-- scripts/                    # ORCHESTRATION SCRIPTS
|   |   |-- run_sweep.py            # Canonical 13-run sweep
|   |   |-- build_submission.py     # Submission bundler
|   |   |-- build_comparison.py     # Compare run outputs
|   |   |-- run_bridge_sweep.py     # Bridge run orchestration
|   |
|   |-- tests/                      # 20+ PYTEST MODULES
|   |   |-- test_main.py
|   |   |-- test_signals.py
|   |   |-- test_optimizer.py
|   |   |-- test_walkforward.py
|   |   |-- ... (and 16 more)
|   |
|   |-- notebooks/                  # DIAGNOSTICS
|   |   |-- build_diagnostics.py    # Generates Jupyter diagnostics notebook
|   |
|   |-- outputs/                    # GENERATED ARTIFACTS
|       |-- *.csv                   # Weights, returns, regimes, compliance
|       |-- figures/*.png           # 18+ chart images
|       |-- reports/*.md            # Markdown reports
|       |-- runs/*/                 # Per-configuration outputs
```

### 11.2 Testing Infrastructure
The project contains **20+ pytest modules** covering:
- **Data audit**: PIT safety, gap preservation, FRED lag
- **Signals**: HMM state ordering, trend score monotonicity, momentum absolute filter, macro factor sign consistency
- **Optimizer**: Constraint satisfaction, fallback mechanics, CVaR scenario construction
- **Walk-forward**: Fold non-overlap, embargo length, expanding window correctness
- **Compliance**: Breach detection, rebalance logging, tier aggregation
- **Attribution**: Active return arithmetic, benchmark replication

Run all tests:
```bash
make test
# or
python -m pytest -q
```

### 11.3 Build Process (Makefile)
```makefile
PYTHON := python3
PROJECT_ROOT := /Users/scottthomasswitzer/Desktop/FIN496FP/FIN496-Foundation-Project
ZIP_PATH := $(PROJECT_ROOT)/whitmore_taa_submission.zip
VERIFY_DIR := /tmp/whitmore_taa_verify

.PHONY: test pipeline zip verify-zip

test:
	$(PYTHON) -m pytest -q

pipeline:
	PYTHONPATH=. $(PYTHON) taa_project/main.py --start 2003-01-01 --end 2025-12-31 --folds 5

zip: pipeline
	cd $(PROJECT_ROOT) && zip -r $(ZIP_PATH) \
		taa_project \
		requirements.txt \
		.python-version \
		DECISIONS.md \
		TRIAL_LEDGER.csv \
		IPS.md \
		Guidelines.md \
		tasks.md \
		data/asset_data/whitmore_daily.csv \
		data/asset_data/data_key.csv \
		data/consolidated_csvs/fred/master/fred_data.csv \
		-x 'taa_project/**/__pycache__/*' \
		-x 'taa_project/**/*.pyc' \
		-x 'taa_project/outputs/*.csv' \
		-x 'taa_project/outputs/*.md' \
		-x 'taa_project/outputs/*.log' \
		-x 'taa_project/outputs/ablations/*' \
		-x 'taa_project/outputs/.mplconfig/*'

verify-zip: zip
	rm -rf $(VERIFY_DIR)
	mkdir -p $(VERIFY_DIR)
	unzip -q $(ZIP_PATH) -d $(VERIFY_DIR)
	cd $(VERIFY_DIR) && $(PYTHON) taa_project/main.py --start 2003-01-01 --end 2025-12-31 --folds 5 --no-timesfm
```

**Targets**:
- `make test`: Runs the full pytest suite.
- `make pipeline`: Executes the main pipeline with default parameters.
- `make zip`: Runs the pipeline, then bundles source code + data into a zip, excluding generated outputs.
- `make verify-zip`: Extracts the zip to `/tmp` and reruns the pipeline to verify reproducibility.

### 11.4 Memory Management
To prevent the 13-run sweep from exhausting 8GB of RAM:
- **Subprocess-per-run**: `run_sweep.py` launches each configuration in a separate Python process, guaranteeing memory is released by the OS.
- **Float32 CVaR scenarios**: Halves memory vs float64.
- **ECOS solver**: Lighter memory footprint than SCS for CVaR problems.
- **TimesFM cache**: Forecasts are cached in a shared Parquet file to avoid recomputing the model.
- **Guard Process Memory**: `memory.py` monitors RSS and triggers `gc.collect()` before heavy operations.

### 11.5 Known Issues
- **OpenMP / Joblib Warning**: On macOS, the HMM fitting step may emit a libomp.dylib duplicate-loading warning. This is harmless and occurs because scikit-learn's HMM uses joblib, which loads OpenMP independently from numpy's BLAS. It does not affect results.

---

## 12. Key Design Decisions & Root-Cause Analysis

This section covers the most critical and defensible architectural choices.

### 12.1 The DFII10 HMM Bias (The Most Important Fix)
**Problem**: A 5-feature HMM (including DFII10) persistently labeled 2009-2021 as `risk_on` because DFII10 declined monotonically due to QE. This suppressed defensive tilts during the longest bull market in history, inflating backtest returns by ~3-4%.

**Fix**: Split the FRED features into two panels:
- `hmm_features` (4 features: VIX, HY OAS, Yield Curve, NFCI) for HMM training.
- `fred_features` (5 features, including DFII10) for the macro_factor signal only.

**Why this is correct**: The HMM needs stationary, mean-reverting stress indicators. DFII10 is a secular trend. The macro_factor signal can handle trends because its z-score uses a short 63-day window and its loadings are small.

### 12.2 Macro Factor Weight Calibration
**Problem**: When macro_factor_weight was 0.40, the optimizer saw macro as the dominant signal and took aggressive positions in high-loading assets (BTC, XAU), worsening max drawdown from -25.5% to -33.5%.

**Fix**: Reduced `macro_factor_weight` to 0.20 and added `macro_scale=0.20`.
- Max macro contribution per asset (BTC at cap): `0.20 * 0.60 * 0.20 = 0.024`
- Max regime contribution per asset: `0.20 * 1.0 * 0.10 = 0.020`
This keeps macro as an **additive refinement** rather than a primary driver.

### 12.3 Opportunistic Sleeve Cap
**Problem**: The IPS allows 15% aggregate opportunistic allocation. Walk-forward tests showed that using the full 15% increased short-window realized volatility without improving risk-adjusted returns.

**Fix**: Capped the opportunistic alpha sleeve at **8% aggregate** and **3 names max**. This is a deliberate tightening of the IPS bounds based on empirical evidence.

### 12.4 Why Minimum Variance for SAA (Default)
The original design used constrained risk parity because it uses covariance information without fragile expected-return estimates. After testing all 7 methods, the default SAA method was changed to **constrained minimum variance** (`min_variance`) because it produced the highest Sharpe ratio among the SAA-only methods (0.56 vs 0.47 for risk parity).

### 12.5 Why 63-Day Z-Score Window
The 252-day window for macro factor z-scoring produced a near-constant signal for 12 years. The 63-day window ensures the signal reflects **quarterly changes** in macro conditions, aligning with the rebalance frequency and preventing structural bias.

### 12.6 Why Nested Risk Budgeting
Without nested risk, the optimizer could concentrate risk in a single high-volatility satellite asset (e.g., Bitcoin in a bull market) while still satisfying aggregate constraints. Nested risk enforces **per-sleeve vol targets**, preventing this arbitrage.

### 12.7 Why Black-Litterman for Mean-Variance SAA
Using sample means for mean-variance optimization is notoriously unstable (Michaud, 1989). The BL equilibrium prior anchors expected returns to the IPS policy portfolio (BM2), preventing extreme allocations driven by short-sample mean estimates. Momentum is blended at only 15% to avoid overfitting to recent performance.

---

## 13. How to Reproduce

1.  **Install**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the full pipeline**:
    ```bash
    python -m taa_project.main --start 2003-01-01 --end 2025-12-31 --folds 5
    ```
3.  **Run the final submission configuration** (non-TimesFM):
    ```bash
    python -m taa_project.main --start 2003-01-01 --end 2025-12-31 --folds 5 --no-timesfm
    ```
4.  **Run the canonical sweep**:
    ```bash
    python -m taa_project.scripts.run_sweep
    ```
5.  **Run tests**:
    ```bash
    make test
    ```
6.  **Build submission bundle**:
    ```bash
    make zip
    ```

---

## 14. Glossary of Key Terms

| Term | Definition |
|------|------------|
| **SAA** | Strategic Asset Allocation: Long-term policy weights, rebalanced annually. |
| **TAA** | Tactical Asset Allocation: Short-term deviations from SAA based on signals. |
| **PIT** | Point-in-Time: A rule ensuring no future data is used in a decision. |
| **HMM** | Hidden Markov Model: A statistical model used for regime detection. |
| **ADM** | Accelerating Dual Momentum: A blended lookback momentum strategy. |
| **CVaR** | Conditional Value-at-Risk: Expected loss in the tail (e.g., worst 5%). |
| **SLSQP** | Sequential Least Squares Programming: A constrained optimization algorithm. |
| **DSR** | Deflated Sharpe Ratio: Sharpe ratio adjusted for multiple trials. |
| **BM1/BM2** | Benchmark 1 (60/40) and Benchmark 2 (Diversified Policy Portfolio). |
| **OOS** | Out-of-Sample: Performance on data not used for training/fitting. |
| **HRP** | Hierarchical Risk Parity: López de Prado (2016) clustering-based allocation. |
| **BL** | Black-Litterman: Bayesian framework blending equilibrium views with investor views. |

---

*End of Comprehensive Guide*
