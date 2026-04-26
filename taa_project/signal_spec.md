# TAA Signal Specification — Whitmore Capital Partners

**Filed under IPS §6.3 (required before deployment).**

## Hypothesis

The portfolio's realized Sharpe and drawdown profile can be improved over
Benchmark 2 by tilting sleeves toward/away from their TAA bands based on the
**joint output of four independent signals** drawn from distinct information
sources (macro, price-trend, cross-sectional momentum, machine-learning
forecast). The four signals are designed to be **imperfectly correlated**, so
that an ensemble is more robust than any single signal.

## Economic rationale

| Signal | Economic thesis | Reference |
|---|---|---|
| **1. Regime HMM (NFCI, HY OAS, 10Y-3M, VIX)** | Financial-conditions stress and curve inversion historically precede equity drawdowns and favor defensive tilts (bonds, gold, CHF). | Hamilton 1989; QuantStart HMM |
| **2. Faber 10-month / 200-day SMA** | Long-horizon trend persistence across asset classes; above-MA regimes have materially higher risk-adjusted returns than below-MA regimes. | Faber 2007 JoWM |
| **3. Accelerating Dual Momentum (1/3/6/12M blend)** | Cross-sectional momentum among equities and real assets; absolute momentum filter avoids full-invested risk during market-wide negative trends. | Antonacci 2012; EngineeredPortfolio |
| **4. TimesFM 2.5 quantile forecast** | Foundation model pretrained on 100B time points captures non-parametric distributional properties (vol shape, tail risk) that parametric GARCH underestimates. Used as vol-forecast and lightly-weighted directional vote. | Das et al. 2024 (Google Research) |

## Decision rule (exactly what the optimizer sees)

For each rebalance date t (month-end, or an intra-month date on HMM regime flip):

1. Build macro feature matrix `X_t = {VIXCLS, BAMLH0A0HYM2, T10Y3M, NFCI, +derived}` using data ≤ t.
2. Refit 3-state Gaussian HMM on expanding-window `X_{:t}` → regime label ∈ {risk_on, neutral, stress}.
3. For each SAA sleeve and Appendix A opportunistic asset, compute `trend_t` (tanh-scaled distance from 200-day SMA normalized by 60-day vol).
4. For each bucket (equity, fixed-income, real, non-trad, opportunistic sub-sleeves), compute `momo_t` = cross-sectional rank of ADM score.
5. (Optional) Call TimesFM 2.5 on each sleeve's log-return series with context 1024, horizon 21 → `mu_fcst`, `sigma_fcst`.
6. Ensemble: `mu_asset = 0.20·regime_tilt·10% + 0.30·trend·6% + 0.30·momo·6% + 0.15·mu_fcst + 0.05·macro_factor·20%`. If TimesFM is disabled or unavailable, redistribute its 15% weight equally to trend and momentum.
7. Covariance = 0.7·sample(252-day) + 0.3·diag, with TimesFM variances overriding the diagonal when available.
8. Solve cvxpy problem: `max  μ'w − 3·w'Σw − 5bps·‖w−w_prev‖₁`  s.t. all IPS §6-7 constraints.
9. If Appendix A assets have positive point-in-time trend/momentum scores, carve out a capped opportunistic alpha sleeve from the monthly TAA target: max 5% aggregate and max 5% per name, then re-project the full target through all IPS aggregate caps. This is deliberately tighter than the 15% IPS maximum because walk-forward diagnostics showed the larger sleeve increased short-window realized-volatility pressure without improving risk-adjusted return.
10. Optional daily realized-risk governor: disabled in the filed default configuration. If explicitly enabled for an incident test, after each close and using only realized strategy returns known at that time, enter defensive TAA mode when current drawdown is ≤ -3% or trailing 21d/63d annualized realized volatility is ≥ 12%. Defensive mode targets SPXT 20%, LBUSTRUU 26%, BROAD_TIPS 24.5%, CHF 15%, and three liquid Appendix A bond sleeves at 4.833% each, then re-projects through all IPS §6-7 caps. Exit only when drawdown is ≥ -1%, trailing 21d/63d realized volatility is ≤ 9%, and the HMM regime is not `stress`. All governor turnover is charged at 5 bps.

## Walk-forward validation plan

- **Period split:** 2005-01-01 → 2025-12-31 using only data ≤ t at each step.
- **Refit cadence:** HMM monthly (expanding window). TimesFM is zero-shot (no refit).
- **Benchmarks:** BM1 (60/40) and BM2 (Diversified Policy Portfolio). Target excess over BM2.
- **Required metrics:** annualized return, vol, Sharpe, Calmar, max DD, turnover, per-regime hit-rate, signal-layer attribution.
- **Failure conditions to audit:**
  - Realized ann vol ever exceeds 15%
  - Peak-to-trough DD exceeds 25%
  - Any aggregate cap breached (Core, Satellite, NT, single-sleeve, Oppo)
  - Turnover implies < 5 bps net drag is actually exceeded

## Kill-switches (documented per IPS §10.2)

- If trailing 3-month realized vol > 13%, scale the signal weight down by 50% until it recovers below 10%.
- If the optional daily realized-risk governor is explicitly enabled and trailing 21-day or 63-day realized vol rises above 12%, force the documented defensive TAA governor target until release conditions are met.
- If the optimizer returns infeasible for 3 consecutive rebalance dates, revert to Benchmark 2 target weights and file a §10.2 incident report.
