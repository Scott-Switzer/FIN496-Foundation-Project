# Whitmore Capital Partners SAA/TAA Report

## Executive Summary
- The final implementation uses the configured SAA allocator for this run together with a monthly cvxpy TAA overlay driven by HMM regime, Faber trend, Antonacci-style ADM, and an optional TimesFM layer. This report reflects run mode `--timesfm`. [Sources: `taa_project/saa/build_saa.py`, `taa_project/backtest/walkforward.py`, `taa_project/analysis/reporting.py`]
- Net annualized return is 8.41% for `SAA+TAA` versus 6.55% for `SAA` and 7.86% for `BM2`. [Source: `taa_project/outputs/portfolio_metrics.csv`]
- Net Sharpe improves by 0.28 versus `BM2`, while the Deflated Sharpe Ratio is 0.941 across 691 disclosed trials in `TRIAL_LEDGER.csv`. [Sources: `taa_project/outputs/portfolio_metrics.csv`, `taa_project/outputs/dsr_summary.csv`, `TRIAL_LEDGER.csv`]
- Daily IPS audit produced 1057 hard-constraint exception rows requiring CIO escalation and written documentation before live use. [Source: `taa_project/outputs/ips_compliance.csv`]

## SAA Construction and IPS Compliance
- The strategic allocator is chosen from the SAA method comparison table, balancing IPS compliance and risk/return trade-offs across inverse volatility, minimum variance, risk parity, maximum diversification, mean-variance, and the optional HRP variant. [Sources: `taa_project/saa/build_saa.py`, `taa_project/outputs/saa_method_comparison.csv`]
- The amended Non-Traditional cap of 20% from Resolution 2026-02 is applied as binding policy throughout the pipeline. [Sources: `IPS.md`, `Guidelines.md`, `taa_project/config.py`]

## TAA Signal Design
- The regime layer is a 3-state Gaussian HMM on lagged `VIXCLS`, `BAMLH0A0HYM2`, `T10Y3M`, and `NFCI`, refit monthly on an expanding window. [Sources: `taa_project/signals/regime_hmm.py`, Hamilton (1989), QuantStart HMM tutorial]
- Trend uses the Faber 200-day SMA with a smooth tanh score, ADM uses 1/3/6/12M blended momentum within sleeve buckets, and TimesFM remains optional so the pipeline still runs end-to-end on machines without that dependency. [Sources: `taa_project/signals/trend_faber.py`, `taa_project/signals/momentum_adm.py`, `taa_project/signals/vol_timesfm.py`, Faber (2007), Allocate Smartly ADM write-up]
- The optimizer does not hard-code a safe-haven switch. Instead it resolves a fresh monthly portfolio inside the TAA bands with the current signal ensemble as `mu`. [Source: `taa_project/optimizer/cvxpy_opt.py`]

## Risk Overlays & Portfolio-Construction Sweep
- Thirteen canonical portfolio-construction configurations were tested, spanning vol budgets, regime overlays, CVaR constraints, nested sleeve risk budgeting, HRP SAA, BL stress views, and a combined kitchen-sink stack. [Source: `taa_project/outputs/config_comparison.csv`]
- The lowest drawdown among the tested configurations was -16.59% in `TimesFM Regime VB`. [Source: `taa_project/outputs/config_comparison.csv`]
- The comparison scatter is color-coded by lever family, so the risk/return dispersion is visible by construction choice rather than by arbitrary run order. [Source: `taa_project/outputs/config_comparison.png`]
- The sweep materially changed the realized risk/return path across the tested overlay variants. [Source: `taa_project/outputs/config_comparison.csv`]

## Walk-Forward Validation
- The OOS period is split into five contiguous expanding folds with a 21-business-day embargo before each fold's first test decision. [Sources: `taa_project/backtest/walkforward.py`, `taa_project/outputs/walkforward_folds.csv`]
- All macro inputs are lagged by one business day before signal use, and no asset-price gaps are forward-filled or backward-filled. [Sources: `taa_project/data_audit.py`, `taa_project/outputs/data_audit_report.md`]

## Performance Results
- `SAA+TAA` delivers 8.41% annualized return, 7.22% annualized volatility, and -21.92% max drawdown. [Source: `taa_project/outputs/portfolio_metrics.csv`]
- Relative to `SAA`, the TAA overlay changes annualized return by 1.86% and cost drag by 0.24% per year. [Source: `taa_project/outputs/portfolio_metrics.csv`]

## Contribution Analysis
- Active-return decomposition is reported separately for `SAA vs BM2`, `TAA vs SAA`, `TAA vs BM1`, and `TAA vs BM2`. [Sources: `taa_project/outputs/attribution_saa_vs_bm2.csv`, `taa_project/outputs/attribution_taa_vs_saa.csv`]
- Signal-layer marginal value is measured with leave-one-out OOS reruns rather than by reading optimizer coefficients off one fitted run. [Source: `taa_project/outputs/attribution_per_signal.csv`]

## Risk Limit Compliance
- Over the 2000–2025 backtest window, no portfolio respecting the IPS minimum-allocation constraints, the no-short rule, and the -25% max-drawdown limit was found under our signal stack. `BM2` itself registered -35.23% drawdown, and `BM1` -33.91%. [Sources: `taa_project/outputs/config_comparison.csv`, `taa_project/outputs/submission_selection.json`]
- The submitted configuration (`timesfm_regime_vb`) achieved -16.59% maximum drawdown, the lowest across 13 tested configurations, representing 1864 bps improvement over `BM2` and 4520 bps improvement over the IPS-target `SAA`. [Source: `taa_project/outputs/submission_selection.json`]
- The residual drawdown breach versus the IPS tolerance is 0 bps. Because the -25% drawdown tolerance is a hard IPS constraint, any selected configuration with a residual breach requires formal exception approval and incident documentation before live deployment.
- Decision tree outcome: `rule_2_mdd_only` selected `timesfm_regime_vb` because it delivered the best available drawdown profile while still considering DSR and return ordering. [Source: `taa_project/outputs/submission_selection.json`]

## Portfolio Construction Lever Analysis
- CVaR-aware optimizer: `CVaR 95 2.5%` worsened max drawdown by -1486 bps relative to the `TimesFM VB10` anchor, landing at -31.48% max drawdown. [Source: `taa_project/outputs/config_comparison.csv`]
- Nested sleeve budgeting: `Nested Risk` worsened max drawdown by -735 bps relative to the `TimesFM VB10` anchor, landing at -23.97% max drawdown. [Source: `taa_project/outputs/config_comparison.csv`]
- Hierarchical Risk Parity: `HRP SAA` left max drawdown unchanged relative to the `TimesFM VB10` anchor, landing at -16.63% max drawdown. [Source: `taa_project/outputs/config_comparison.csv`]
- BL stress views: `BL Stress Views` improved max drawdown by 1 bps relative to the `TimesFM VB10` anchor, landing at -16.61% max drawdown. [Source: `taa_project/outputs/config_comparison.csv`]

## Limitations and Failure Modes
- The HMM is still a retrospective statistical classifier and state meanings can drift over time. [Source: `taa_project/signals/regime_hmm.py`]
- TimesFM is not finance-native. Even though the live runtime path now works end-to-end, the best tested result still misses the -25% IPS drawdown tolerance, so the model should be treated as an incremental signal layer rather than as a standalone risk solution. [Sources: `taa_project/signals/vol_timesfm.py`, `taa_project/outputs/config_comparison.csv`, `taa_project/outputs/submission_selection.json`]
- The optimizer uses a shrinkage-style covariance stabilization and a soft volatility ceiling, so results depend on those engineering choices even under walk-forward discipline. [Source: `taa_project/backtest/walkforward.py`]

## Recommendation
- Do not treat `timesfm_regime_vb` as live-deployable without formal IPS exception approval; use it only as the best tested research candidate while hard-constraint exceptions remain open. [Sources: `taa_project/outputs/portfolio_metrics.csv`, `taa_project/outputs/submission_selection.json`, `TRIAL_LEDGER.csv`]
- Monitor the regime layer and turnover drag closely in stressed periods; the contribution tables show whether the overlay is being paid for by genuine alpha or by benchmark timing luck. [Source: `taa_project/outputs/attribution_per_signal.csv`]

## Appendix
- Appendix tables in the PDF include the SAA method comparison, per-fold OOS metrics, the portfolio metrics table, and the leading rows of the trial ledger and IPS compliance log. [Sources: `taa_project/outputs/saa_method_comparison.csv`, `taa_project/outputs/per_fold_metrics.csv`, `TRIAL_LEDGER.csv`, `taa_project/outputs/ips_compliance.csv`]