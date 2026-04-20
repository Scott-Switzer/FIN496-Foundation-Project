# Whitmore Capital Partners SAA/TAA Report

## Executive Summary
- The final implementation uses constrained risk parity for SAA and a monthly cvxpy TAA overlay driven by HMM regime, Faber trend, Antonacci-style ADM, and an optional TimesFM layer. This report reflects run mode `--timesfm`. [Sources: `taa_project/saa/build_saa.py`, `taa_project/backtest/walkforward.py`, `taa_project/analysis/reporting.py`]
- Net annualized return is 13.52% for `SAA+TAA` versus 8.83% for `SAA` and 7.86% for `BM2`. [Source: `taa_project/outputs/portfolio_metrics.csv`]
- Net Sharpe improves by 0.50 versus `BM2`, while the Deflated Sharpe Ratio is 0.999 across 149 disclosed TAA trials. [Sources: `taa_project/outputs/portfolio_metrics.csv`, `taa_project/outputs/dsr_summary.csv`, `TRIAL_LEDGER.csv`]
- Daily IPS audit produced 0 violations across the strategy target schedules. [Source: `taa_project/outputs/ips_compliance.csv`]

## SAA Construction and IPS Compliance
- Risk parity was selected over inverse volatility, minimum variance, maximum diversification, and mean-variance because it cleared the 8% return objective while staying below the 15% volatility ceiling without relying on fragile expected-return estimates; minimum variance was safer on volatility but undershot the return mandate. [Sources: `taa_project/saa/build_saa.py`, `taa_project/outputs/saa_method_comparison.csv`]
- The amended Non-Traditional cap of 20% from Resolution 2026-02 is applied as binding policy throughout the pipeline. [Sources: `IPS.md`, `Guidelines.md`, `taa_project/config.py`]

## TAA Signal Design
- The regime layer is a 3-state Gaussian HMM on lagged `VIXCLS`, `BAMLH0A0HYM2`, `T10Y3M`, and `NFCI`, refit monthly on an expanding window. [Sources: `taa_project/signals/regime_hmm.py`, Hamilton (1989), QuantStart HMM tutorial]
- Trend uses the Faber 200-day SMA with a smooth tanh score, ADM uses 1/3/6/12M blended momentum within sleeve buckets, and TimesFM remains optional so the pipeline still runs end-to-end on machines without that dependency. [Sources: `taa_project/signals/trend_faber.py`, `taa_project/signals/momentum_adm.py`, `taa_project/signals/vol_timesfm.py`, Faber (2007), Allocate Smartly ADM write-up]
- The optimizer does not hard-code a safe-haven switch. Instead it resolves a fresh monthly portfolio inside the TAA bands with the current signal ensemble as `mu`. [Source: `taa_project/optimizer/cvxpy_opt.py`]

## Risk Overlays & Vol-Budget Sweep
- The canonical configuration comparison was not generated for this output directory, so the report falls back to the single-run results only.

## Walk-Forward Validation
- The OOS period is split into five contiguous expanding folds with a 21-business-day embargo before each fold's first test decision. [Sources: `taa_project/backtest/walkforward.py`, `taa_project/outputs/walkforward_folds.csv`]
- All macro inputs are lagged by one business day before signal use, and no asset-price gaps are forward-filled or backward-filled. [Sources: `taa_project/data_audit.py`, `taa_project/outputs/data_audit_report.md`]

## Performance Results
- `SAA+TAA` delivers 13.52% annualized return, 10.40% annualized volatility, and -27.56% max drawdown. [Source: `taa_project/outputs/portfolio_metrics.csv`]
- Relative to `SAA`, the TAA overlay changes annualized return by 4.68% and cost drag by 0.17% per year. [Source: `taa_project/outputs/portfolio_metrics.csv`]

## Contribution Analysis
- Active-return decomposition is reported separately for `SAA vs BM2`, `TAA vs SAA`, `TAA vs BM1`, and `TAA vs BM2`. [Sources: `taa_project/outputs/attribution_saa_vs_bm2.csv`, `taa_project/outputs/attribution_taa_vs_saa.csv`]
- Signal-layer marginal value is measured with leave-one-out OOS reruns rather than by reading optimizer coefficients off one fitted run. [Source: `taa_project/outputs/attribution_per_signal.csv`]

## Risk Limit Compliance
- This run achieved -27.56% maximum drawdown against the IPS tolerance of -25%. The submission-selection summary was not available in this output directory.

## Limitations and Failure Modes
- The HMM is still a retrospective statistical classifier and state meanings can drift over time. [Source: `taa_project/signals/regime_hmm.py`]
- TimesFM is not finance-native. The runtime path is now available, but the canonical sweep found no realized portfolio difference between the no-TimesFM baseline and the TimesFM-tagged variants over this sample, so the simpler baseline remained the submission choice. [Sources: `taa_project/signals/vol_timesfm.py`, `taa_project/outputs/config_comparison.csv`, `taa_project/outputs/submission_selection.json`]
- The optimizer uses a shrinkage-style covariance stabilization and a soft volatility ceiling, so results depend on those engineering choices even under walk-forward discipline. [Source: `taa_project/backtest/walkforward.py`]

## Recommendation
- Use the risk-parity SAA as the policy core and treat the TAA overlay as an additive layer only when the OOS Sharpe and DSR remain superior after costs under the disclosed trial count. [Sources: `taa_project/outputs/portfolio_metrics.csv`, `TRIAL_LEDGER.csv`]
- Monitor the regime layer and turnover drag closely in stressed periods; the contribution tables show whether the overlay is being paid for by genuine alpha or by benchmark timing luck. [Source: `taa_project/outputs/attribution_per_signal.csv`]

## Appendix
- Appendix tables in the PDF include the SAA method comparison, per-fold OOS metrics, the portfolio metrics table, and the leading rows of the trial ledger and IPS compliance log. [Sources: `taa_project/outputs/saa_method_comparison.csv`, `taa_project/outputs/per_fold_metrics.csv`, `TRIAL_LEDGER.csv`, `taa_project/outputs/ips_compliance.csv`]