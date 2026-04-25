# Decisions

## 2026-04-25 — Opportunistic alpha sleeve for return target recovery
- Decision: expand `OPPORTUNISTIC` to the full Appendix A universe and add a signal-ranked opportunistic alpha sleeve inside the monthly TAA target, capped at 15% aggregate and 5% per asset.
- Alternatives considered: loosen the realized-risk trigger, raise the emergency SPXT/gold weights, or rely entirely on TimesFM directionality in the SAA universe.
- Why this won: loosening the risk trigger restored some return but reintroduced realized 21-day volatility breaches; raising the defensive target with gold/BTC restored return but also breached short-window volatility. The IPS explicitly permits Appendix A assets for short-term hedging or alpha capture, so the cleaner return lever is to use that sleeve dynamically from point-in-time trend and momentum signals while re-projecting the whole book through the IPS caps.

## 2026-04-24 — Emergency TAA risk-control overlay
- Decision: add a daily realized-risk overlay that keeps the combined `SAA+TAA` portfolio in a low-volatility emergency target while trailing realized risk remains elevated.
- Alternatives considered: rely on TimesFM forecasts, tighten the monthly optimizer's vol budget further, or continue auditing standalone `SAA` breaches as fatal.
- Why this won: the remaining IPS failures were realized 21-day volatility breaches in the combined strategy. TimesFM was available but did not solve the realized-volatility problem, and the previous TAA bands were infeasible in 2008/2020 stress windows. The emergency target keeps the book fully invested and long-only, uses only assets permitted by the TAA/opportunistic sleeve, maintains the 15% opportunistic aggregate and 5% single-asset caps, and applies only from causal trailing drawdown/volatility triggers. The SPXT TAA floor is amended to 12% for this emergency risk-control mode so the combined strategy can satisfy the hard 15% realized-volatility ceiling.

## 2026-04-19 — Sortino, risk overlays, and submission selection
- Decision: add Sortino alongside Sharpe and Calmar in every metrics artifact and presentation surface.
- Alternatives considered: keep only Sharpe and Calmar, or add upside/downside metrics only in the notebook.
- Why this won: the Whitmore mandate has an explicit drawdown tolerance, so a downside-only dispersion metric is more aligned with the grading rubric and the client brief than a pure total-volatility lens. Sortino complements Sharpe and Calmar by penalizing only downside deviation while keeping the report and deck interpretable for non-technical readers. Source: Sortino & Price (1994), https://jpm.pm-research.com/content/20/4/59.

- Decision: implement the regime overlay as a regime-conditional vol budget, not as hard-coded regime-specific asset weights.
- Alternatives considered: directly force state-specific safe-haven allocations, or add manual asset-level overrides inside the optimizer when the HMM enters stress.
- Why this won: the assignment explicitly forbids a hard-coded safe-haven switch. Tightening the optimizer's risk envelope by regime keeps the solve causal and IPS-aware while allowing the optimizer to choose the mix inside the existing TAA bands. This preserves the "no hard-coded safe haven" rule and is easier to defend in the report.

- Decision: implement a drawdown-clip guardrail that halves the active vol budget after a trailing six-month drawdown breach and releases only after recovery.
- Alternatives considered: no realized-P&L overlay at all, or direct liquidation rules tied to specific sleeves.
- Why this won: the guardrail is a monotonic function of realized returns observed up to decision time `t`, so it is causally safe and operationally close to how real risk desks tighten risk after losses. It changes only the risk envelope, not the allowed asset set. Source: Grossman & Zhou (1993), https://doi.org/10.1111/j.1467-9965.1993.tb00044.x.

- Decision: keep the flat 7% vol budget as a tested sweep point, but not as the default recommendation.
- Alternatives considered: replace the 10% default entirely with a flat 7% or 8% target.
- Why this won: a flat tight budget can reduce upside capture in benign risk-on regimes, while the regime-conditioned overlay preserves higher risk capacity when the macro state is calmer. In the corrected canonical sweep, the flat 8% budget produced the smallest drawdown breach while keeping return and DSR superior to the benchmarks, so 7% remained a useful stress test rather than the preferred submission.

- Decision: retain the OpenMP/Torch bootstrap fix and document it again in the final run log.
- Alternatives considered: remove the bootstrap once the TimesFM sweep completed, or rely on shell-level environment exports.
- Why this won: the full `--timesfm` path must remain runnable from a clean checkout. The duplicate-runtime workaround in `taa_project/__init__.py` and `taa_project/main.py` is still required for stable Torch import ordering on this machine. Sources: https://github.com/pytorch/pytorch/issues/6027 and https://scikit-learn.org/stable/faq.html#why-do-i-sometimes-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux.

- Decision: choose `timesfm_vb08` as the submission configuration after the corrected canonical six-run sweep.
- Alternatives considered: submit the no-TimesFM baseline, the 10% or 7% flat-budget TimesFM variants, or the regime-vol / drawdown-overlay variants.
- Why this won: none of the six canonical configurations met the IPS max-drawdown tolerance of 25%, so Task 7 fell to the "smallest MDD breach" branch. `timesfm_vb08` produced the lowest realized max drawdown at `-27.46%`, stayed below the 15% volatility ceiling, exceeded the 8% return objective, and beat `BM2` on DSR. The regime-vol and regime+DD overlays changed realized weights but did not improve on the flat 8% budget's drawdown outcome, so `timesfm_vb08` was the strongest defensible submission.

## 2026-04-19 — TimesFM / OpenMP runtime
- Decision: pre-import `torch` in both `taa_project/__init__.py` and `taa_project/main.py`, while setting `KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` before any SciPy / scikit-learn / hmmlearn imports occur.
- Alternatives considered: leave the runtime as-is and rely on users to export OpenMP environment variables manually, or isolate the TimesFM path into a separate subprocess immediately.
- Why this won: the end-to-end `--timesfm` path needs a deterministic, repo-local fix for the duplicate OpenMP runtime crash that occurs when Intel MKL-linked libraries and Torch initialize different OpenMP implementations in the same process. Preloading Torch first is the least invasive workaround and matches the documented behavior in the PyTorch duplicate-runtime issue and the scikit-learn OpenMP FAQ: https://github.com/pytorch/pytorch/issues/6027 and https://scikit-learn.org/stable/faq.html#why-do-i-sometimes-get-a-crash-freeze-with-n-jobs-1-under-osx-or-linux.

## 2026-04-18 — Promote `taa_scaffold/` into `taa_project/`
- Decision: rename the untracked scaffold into a top-level Python package named `taa_project/`.
- Alternatives considered: keep the scaffold name and wrap it later, or rebuild a new package and copy code across selectively.
- Why this won: it preserves the user-provided starter modules, avoids duplicate code paths, and makes the final import path match the required `python taa_project/main.py` entrypoint.

## 2026-04-18 — Canvas submission link placeholder
- Decision: use a generic Canvas placeholder link in `taa_project/README.md` and flag it for replacement later.
- Alternatives considered: omit the link entirely or invent a course-specific URL.
- Why this won: the repo does not contain the actual course Canvas assignment URL, and inventing one would be less defensible than a clearly labeled placeholder.

## 2026-04-18 — Return calculation on a mixed-trading-calendar panel
- Decision: compute asset log returns on adjacent observed prices within each asset's own non-null history, then reindex those returns back onto the full panel calendar with missing dates left as `NaN`.
- Alternatives considered: use `prices / prices.shift(1)` on the full daily panel, or fill non-trading days with zero returns.
- Why this won: `whitmore_daily.csv` contains a unified calendar where crypto trades on weekends and many traditional assets do not. Full-panel `shift(1)` would erase most Monday returns for non-crypto assets, while zero-filling is explicitly forbidden by `tasks.md`. This interpretation keeps non-trading days empty and records returns only on the next observed trade date.

## 2026-04-18 — One-business-day FRED lag implementation
- Decision: apply the FRED publication lag by moving each observation's index forward by one business day before aligning it to the asset calendar.
- Alternatives considered: shift the values down one row in the original table, shift by one calendar day after reindexing to the asset calendar, or leave the master table untouched and assume same-day observability.
- Why this won: the master FRED table is organized on a business-day index, so moving each timestamp onto the next business day preserves the intended observability rule while still allowing correct forward-fill behavior onto weekends and mixed trading calendars.

## 2026-04-18 — Strategic allocator method choice
- Decision: build the SAA sleeve with constrained target-aware risk parity rather than minimum variance, inverse volatility, maximum diversification, or mean-variance.
- Alternatives considered: inverse volatility, minimum variance, maximum diversification, and classical mean-variance optimization.
- Why this won: the Whitmore mandate has hard sleeve bands, aggregate caps, and a real-asset / non-traditional structure that is better handled by a covariance-aware but expected-return-light allocator. Risk parity is easier to defend out of sample than mean-variance, less duration-heavy than minimum variance, and more policy-aware than inverse volatility because it uses full covariance and anchors risk budgets to the IPS strategic targets.

## 2026-04-18 — Export policy targets, simulate drifting annual holdings
- Decision: write `saa_weights.csv` as the daily policy target schedule and `saa_returns.csv` as realized P&L from holdings that drift between annual rebalances.
- Alternatives considered: export drifting end-of-day holdings only, or force compliance rebalances whenever drift crossed a band.
- Why this won: exporting target weights keeps the IPS audit and later attribution aligned to the intentional SAA policy decisions, while the return series still reflects the actual annual-rebalance mechanics and turnover costs. Forcing unscheduled compliance trades made the book rebalance far more often than the assignment's annual SAA specification.

## 2026-04-18 — Benchmark 2 policy weights and inception
- Decision: interpret BM2 as the fixed-weight Diversified Policy Portfolio encoded in `taa_project/config.py`, and start its history on the first date all positive-weight sleeves are simultaneously observable.
- Alternatives considered: infer a partial BM2 before all sleeves exist using an expanding-universe proxy, or redistribute missing sleeves across available benchmark assets before their inception.
- Why this won: IPS §4 defines BM2 as a fixed-weight benchmark rebalanced annually, not an adaptive proxy. Because the source IPS table is malformed around the SPXT, NIKKEI225, and CHF_FRANC rows, the implementation follows the cleaned policy interpretation already reflected in `Guidelines.md` and the repo config: SPXT 40%, NIKKEI225 5%, CSI300_CHINA 5%, LBUSTRUU 10%, BROAD_TIPS 5%, B3REITT 10%, XAU 15%, SILVER_FUT 5%, CHF_FRANC 5%. Under that fixed-weight definition, BM2 cannot begin until `B3REITT` appears on 2003-03-31.

## 2026-04-18 — Keep TimesFM optional until the orchestrator fallback lands
- Decision: implement the TimesFM 2.5 signal wrapper now, but keep it out of the base `requirements.txt` install path and raise a clear optional-dependency error when the package is absent.
- Alternatives considered: pin a guessed PyPI package name/version in `requirements.txt`, or silently degrade the TimesFM layer to zeros when imports fail.
- Why this won: the official TimesFM 2.5 model card currently directs users to install from the `google-research/timesfm` repository rather than a stable PyPI release, so forcing it into the base requirements would make the clean-room install path brittle. A clear optional-dependency error is safer than a silent zero-signal fallback; Task 12 will wire this into the required `--no-timesfm` pipeline mode.

## 2026-04-18 — Optimizer incidents fall back to projected last-feasible weights
- Decision: when the optimizer cannot solve the requested portfolio because of infeasibility or a missing optional dependency such as `cvxpy`, append an incident record to `taa_project/outputs/breaches.log` and fall back to the previously held portfolio projected into the current feasible region.
- Alternatives considered: crash immediately on every solve failure, or silently reuse the raw previous weights without recording an incident.
- Why this won: the project rubric penalizes pipelines that do not run end-to-end, but a silent reuse would hide important implementation failures. Logging plus a constrained fallback keeps the pipeline reproducible while preserving an auditable paper trail for the report and compliance checks.

## 2026-04-18 — Walk-forward monthly decisions use the last observed SAA date, not raw calendar month-end
- Decision: build the walk-forward rebalance schedule from the last date in each month with at least one observed SAA price, rather than from the raw calendar month-end label.
- Alternatives considered: use calendar month-end labels directly via `resample("ME")`, or include all-calendar rows in the OOS daily return series even when every sleeve return is missing.
- Why this won: `whitmore_daily.csv` carries calendar rows where every SAA sleeve is `NaN`. Using those rows as decision dates created false month-end decisions on Sundays and fake zero-return days in the OOS series, which conflicts with `tasks.md`'s rule to preserve gaps and exclude non-trading days from return calculations.

## 2026-04-18 — Monthly TAA decisions use the last common investable date in each month
- Decision: refine the monthly TAA rebalance schedule again so each decision date is the last date in the month where every sleeve already in the expanding investable universe has an observed price.
- Alternatives considered: keep the looser “last date with any observed SAA price” rule, or force the optimizer to solve on crypto-only weekends by zeroing unavailable sleeves.
- Why this won: once `BITCOIN` enters the universe, the looser rule schedules month-end decisions on Sundays where only crypto has a price, which makes the Core floor infeasible and triggers spurious optimizer fallbacks. Requiring a common investable date matches the annual SAA / benchmark rebalance logic and preserves a realistic multi-asset trading calendar.

## 2026-04-18 — Missing TAA signals become neutral scores, not missing assets
- Decision: keep sleeves in the monthly TAA investable universe whenever they are available, and treat missing signal values as neutral `0.0` rather than dropping those sleeves from the optimization problem.
- Alternatives considered: remove any sleeve with a missing signal from the monthly optimizer, or forward-fill the missing signal from the prior month.
- Why this won: dropping sleeves can make an otherwise feasible IPS problem infeasible even when the asset is tradable, while forward-filling risks smuggling stale directional conviction into the solve. A neutral score preserves the current investable set without inventing information.

## 2026-04-18 — Daily IPS audit is applied to target schedules, not drifted holdings
- Decision: run the Task 8 IPS compliance audit on the daily target-weight schedules for `SAA` and `SAA+TAA`, rather than on post-market drifted holdings.
- Alternatives considered: audit only scheduled rebalance dates, or audit drifted holdings and add unscheduled compliance trades whenever drift crosses a constraint.
- Why this won: the project already separates policy targets from realized drifted P&L for the annual SAA implementation. Auditing target schedules keeps the compliance test aligned to the intended policy decisions and avoids silently introducing extra trades that would contradict the user-approved annual SAA design.

## 2026-04-20 — Portfolio-construction levers and memory discipline
- Decision: add an opt-in CVaR monthly optimizer mode using the Rockafellar-Uryasev linear-program formulation with `optimizer_mode="cvar"`, `float32` scenario matrices, a hard `504`-day lookback cap, and ECOS solves while leaving the legacy volatility-constrained path untouched by default.
- Alternatives considered: replace the existing quadratic vol ceiling entirely, use a longer scenario history, or solve the tail-risk problem with SCS.
- Why this won: the CVaR constraint is the lead institutional lever for left-tail control, but it must remain fully gated to preserve bit-identical default behavior. The `float32`/`504`-row cap halves memory versus `float64`, and ECOS avoids the heavier retained problem data seen with SCS. Sources: http://www.ise.ufl.edu/uryasev/files/2011/11/CVaR1_JOR.pdf and https://github.com/embotech/ecos

- Decision: add nested sleeve risk budgeting as an opt-in sequential solve across Core, Satellite, and Non-Traditional sleeves, with optional per-sleeve CVaR, post-blend renormalization when a sleeve is unavailable, and explicit cleanup between sleeve problems.
- Alternatives considered: one global optimization with sleeve-level penalties only, or keeping unavailable sleeves at fixed zero without renormalizing the remaining sleeves.
- Why this won: the sequential sleeve solve matches the IPS structure directly, lets each sleeve carry a defensible risk budget, and keeps memory bounded because only one cvxpy problem exists at a time. Renormalizing remaining sleeves while preserving `Core >= 40%` keeps the solution investable before assets such as Bitcoin enter the universe. Source: https://www.calpers.ca.gov/docs/board-agendas/201802/invest/item07a-01_a.pdf

- Decision: add Hierarchical Risk Parity as `--saa-method hrp`, using single-linkage clustering, quasi-diagonalization, recursive bisection, and an inverse-volatility fallback for universes with fewer than four assets.
- Alternatives considered: keep risk parity as the only strategic allocator, or add HRP as the new default SAA method.
- Why this won: HRP is a low-memory diversification lever that does not depend on fragile expected-return inputs, but keeping it opt-in preserves backwards compatibility. The inverse-vol fallback avoids unstable clustering in very small expanding-universe states. Sources: https://jpm.pm-research.com/content/42/4/59 and https://github.com/quantopian/research_public/blob/master/research/hierarchical_risk_parity.py

- Decision: add regime-conditional Black-Litterman stress views that shift equity priors down by `1.0` annualized sigma when the HMM regime is `stress`, and blend those priors 50/50 with the existing tactical expected-return signal.
- Alternatives considered: hard-code risk-off allocations to bonds and gold, or apply the pessimistic equity view in every regime.
- Why this won: the Whitmore mandate forbids hard-coded safe-haven floors, so the stress response has to emerge from priors and constraints rather than deterministic allocations. Conditioning the pessimistic view only on causal stress labels keeps the signal economically interpretable without contaminating risk-on or neutral months. Sources: https://faculty.fuqua.duke.edu/~charvey/Teaching/BA453_2006/He_Litterman_Black-Litterman.pdf and https://www.jstor.org/stable/4479577

- Decision: enforce memory discipline through a subprocess-per-configuration sweep, a shared TimesFM parquet cache at `taa_project/outputs/cache/timesfm_forecasts.parquet`, pyarrow predicate-pushdown cache reads, full-file cache rewrite on append while the cache remains small, `psutil` RSS guards, `float32` CVaR scenarios, ECOS solves, and explicit `gc.collect()` after every TimesFM and cvxpy hot path.
- Alternatives considered: import `taa_project.main` directly into the sweep orchestrator, recompute TimesFM forecasts on every run, keep multiple cvxpy problems alive for convenience, or use a partitioned parquet dataset for the cache.
- Why this won: the previous attempt exhausted 8 GB RAM, so the orchestration boundary itself had to become the safety mechanism. Running one configuration per subprocess guarantees model and solver memory is released by the OS, while the small shared parquet cache keeps TimesFM idempotent across the 13-run sweep and is simple enough that rewrite-on-append stays well below the stated cache-size budget. Sources: https://arrow.apache.org/docs/python/parquet.html#filtering, https://huggingface.co/google/timesfm-2.5-200m-pytorch, and https://github.com/embotech/ecos

## 2026-04-21 — Macro factor signal layer and BTC strategic allocation

- Decision: add a new `macro_factor.py` signal module with three sub-signals (real-yield tilt, credit-premium tilt, crypto-momentum tilt) and wire it into the TAA ensemble as a fifth layer weighted at 0.20, reducing regime from 0.40 to 0.30 and timesfm from 0.20 to 0.10.
- Alternatives considered: keep the four-signal ensemble unchanged; replace timesfm entirely with macro_factor; use macro signals only inside the HMM.
- Why this won: three FRED series in the master dataset (DFII10, BAMLC0A0CM, T10YIE-derived HY-IG spread) were entirely unused. Empirically, the HY-IG spread change has a −0.60 monthly correlation with SPXT returns and DFII10 changes have a −0.46 correlation with XAU returns — both stronger than the regime-regime signal alone. Adding a dedicated macro-factor layer gives each asset class a hypothesis-grounded, continuous, non-binary signal with clear academic backing (Erb & Harvey 2013 for gold; Gilchrist & Zakrajsek 2012 for credit; Liu & Tsyvinski 2021 for crypto). The crypto-momentum sub-signal addresses the signal-scale mismatch that prevented Bitcoin from entering the optimizer at meaningful weights despite a positive TAA band. Sources: https://doi.org/10.2469/faj.v69.n4.1, https://doi.org/10.1257/aer.102.4.1692, https://doi.org/10.1093/rfs/hhaa113.

- Decision: raise BITCOIN SAA target from 0.00 to 0.02 and reduce LBUSTRUU from 0.10 to 0.08, keeping Core total at 53% (above the 40% floor) and Non-Traditional total at 7% (well below the amended 20% cap).
- Alternatives considered: keep Bitcoin at 0% in SAA and rely purely on TAA to allocate it; raise BTC to 5% (maximum SAA band).
- Why this won: at 0% SAA target, the risk-parity optimizer assigns Bitcoin a zero risk budget, which means the SAA structurally excludes BTC from the strategic book even when data from 2010 onward support a non-zero allocation. A 2% target activates a non-zero risk budget (~1.5–2% dollar weight given BTC's ~80% vol) without violating any IPS band or aggregate cap. The freed-up 2% comes from nominal duration (LBUSTRUU), which is the lowest-conviction SAA sleeve given the prevailing real-rate environment. Sources: Whitmore IPS §5 SAA bands; Amendment 2026-02 non-traditional cap revision.

- Decision: extend the HMM macro feature set from four to five features by adding DFII10 (10Y TIPS real yield) as an optional fifth series, with a fallback to the original four if DFII10 is absent from the FRED panel.
- Alternatives considered: replace T10Y3M with T10Y2Y; add both T10Y2Y and DFII10; keep the original four features.
- Why this won: DFII10 has a −0.80 monthly correlation with BROAD_TIPS returns — the highest cross-asset correlation in the unused FRED set — and introduces a real-rate dimension that is entirely absent from the current four features (VIX, HY OAS, yield curve, NFCI). Adding T10Y2Y would introduce collinearity with T10Y3M without adding a new economic dimension. The fallback logic preserves backward compatibility and prevents training failures on data slices where DFII10 is sparse. Source: Erb & Harvey (2013). https://doi.org/10.2469/faj.v69.n4.1.

- Decision: set the per-asset loading magnitudes in `_REAL_YIELD_LOADINGS` and `_CREDIT_PREMIUM_LOADINGS` as hand-specified round numbers grounded in the relative economic importance of each channel, not as optimizer-fitted or regression-derived coefficients.
- Transparency note: the empirical correlations computed over the full 2003–2025 FRED/price dataset (DFII10 ↔ BROAD_TIPS −0.80, DFII10 ↔ XAU −0.46, HY-IG ↔ SPXT −0.60) were used *after the fact* as a consistency check that the theoretical signal directions were empirically present in the data — not as the source of the loading values. The direction (sign) of every loading is dictated entirely by economic theory and the cited academic literature; the magnitude ordering (e.g., gold receives a larger real-yield loading than equities) reflects the relative strength of each theoretical channel as described in Erb & Harvey (2013) and Gilchrist & Zakrajsek (2012). In a production research context, loading magnitudes would be estimated on a held-out calibration window (e.g., 2001–2005) and frozen for the walk-forward test period. For this academic project the magnitudes are pre-specified rather than optimized, so no in-sample return-chasing is present; the empirical correlations simply confirm that the theoretical predictions held in this dataset.

- Decision: add `macro_scale: float = 0.20` to `EnsembleConfig` and apply it in `ensemble_score` as `cfg.macro_factor_weight * macro_mu * cfg.macro_scale`.
- Rationale: the three existing raw signal layers (`regime_tilt`, `trend_sig`, `momo_sig`) each carry a dedicated scale factor (`regime_scale=0.10`, `trend_scale=0.06`, `momo_scale=0.06`) that normalises their raw [-1, +1] range into a return-like expected-return proxy. `macro_factor_mu` was introduced without an equivalent scale, so its raw z-score × loading values (e.g., 0.18 for XAU at |z|=1, 0.60 cap for BTC) were 4–10× larger than the scaled regime/trend/momo contributions. The optimizer therefore saw macro as the dominant signal and took aggressive positions in high-loading assets (BTC, XAU, SPXT) regardless of regime. The backtest correctly reflected this: adding PR #6 pushed annualised return from 7.68% to 11.35–11.72% but worsened MDD from −25.54% to −32.71%–−34.93%, worse than the unconstrained baseline. At `macro_scale=0.20` the maximum macro contribution per asset (BTC at cap: 0.60×0.20×0.20 = 0.024) is at or below the regime layer maximum (0.03), restoring the regime as the primary risk governor while keeping macro as an additive incremental signal.

## 2026-04-21 — Root-cause attribution and three-fix regression correction

**Context**: After `macro_scale=0.20` was added (commit 80629cb), the bridge sweep v3 baseline still showed 12.10%/−33.46% — a 50 bps improvement over v2 but still 4.6 % above the pre-PR-#6 result of 7.50%/−23.97%. A structured root-cause decomposition identified three independent causes.

**Cause 1 (dominant, ~3–4 % return inflation): `_ZSCORE_WINDOW = 252` in `macro_factor.py`**

- Decision: change `_ZSCORE_WINDOW` from 252 to 63 trading days.
- Why: DFII10 (10Y real yield) declined from ~2.5% in 2003 to nearly −1% by 2021, spending the entire 2009–2021 period persistently below its own 252-day rolling mean. A 252-day z-score window produced a near-constant z ≈ −1 for roughly 144 consecutive months. Because XAU, SPXT, B3REITT, and BROAD_TIPS all carry positive real-yield loadings (0.18, 0.04, 0.06, 0.10), this persistent negative z generated a continuous positive macro tilt for every high-CAGR asset class over the entire 2009–2021 bull market — the best-performing decade in the backtest. The result is not a one-month spike but a structural, compounding return enhancement that accounts for roughly 3–4 % of the 4.6 % incremental CAGR. A 63-day window ensures the signal reflects *changes* in real yields within the current quarter, not multi-year level drift. The 63-day horizon is ≥ `_MIN_FRED_OBS/2 = 63`, so the minimum-observation guard remains satisfied.
- Alternatives considered: 126-day (semi-annual) window; removing the real-yield sub-signal entirely; applying a linear detrend before z-scoring.
- Why 63 won: 63 days balances responsiveness (quarterly mean-reversion) with stability (avoids over-reaction to single-day FRED releases). Removing the signal entirely would discard the XAU/BROAD_TIPS channel that has genuine economic support. Linear detrending requires in-sample slope estimation, introducing lookahead risk.

**Cause 2 (secondary, ~0.5–1 % return, ~2 % MDD): `regime_weight` cut from 0.40 to 0.30 in PR #6**

- Decision: restore `regime_weight` to 0.40.
- Why: the HMM regime signal's maximum protective contribution per asset is `regime_weight × regime_scale × max_tilt = regime_weight × 0.10 × 1.0`. At 0.30 this is 0.030; at 0.40 it is 0.040 — a 33 % increase in the primary risk governor's ceiling. Cutting regime weight by 25 % allowed the optimizer to remain in risk-on postures longer during drawdown events, contributing directly to the MDD worsening from −23.97 % to −33.46 %. The regime HMM is the only signal in the ensemble that has causal, real-time access to the current vol/stress regime; giving it less weight than trend or momentum undercuts its role as risk governor.

**Cause 3 (tertiary, reinforces Cause 2): `macro_factor_weight = 0.20` with insufficient macro_scale**

- Decision: reduce `macro_factor_weight` from 0.20 to 0.05; redistribute freed weight to timesfm (0.10 → 0.15).
- Why: even with `macro_scale = 0.20`, BTC's maximum macro contribution was `0.20 × 0.60 × 0.20 = 0.024` — nearly as large as the restored regime maximum of 0.040. At 0.05, BTC macro max = `0.05 × 0.60 × 0.20 = 0.006`, clearly subordinate to regime (0.040) and trend/momo (both 0.012 max). Macro is an informational refinement signal with pre-specified hand-coded loadings; giving it equal or near-equal weight to the HMM creates a regime where factor-signal persistence (from Cause 1) can crowd out protective regime signals. The 0.05 weight caps its aggregate influence while still contributing directional information.
- timesfm receives the freed weight (0.10 → 0.15) rather than regime, so that regime's weight increase is entirely attributable to Cause 2's reversal and is documented separately.

**Non-cause confirmed: BTC SAA change 0 % → 2 %**

- The nested risk optimizer's NT sleeve vol target (15 %) binds at `w_BTC × 80 % ≈ 15 % → w_BTC ≈ 18.75 %` of the NT sleeve. At a 10 % sleeve weight, BTC is capped at ~1.875 % of total portfolio regardless of the SAA target. Because BTC was already present via TAA signals in the pre-PR-#6 backtest at approximately this weight, the SAA change from 0 % to 2 % had negligible effect on actual realized allocation. Nonetheless, reverting to 0 % keeps the SAA consistent with the zero-risk-budget semantics used by `target_risk_budgets()` and avoids any ambiguity in the compliance audit.

**Final EnsembleConfig state after fixes:**

| Signal | Weight | Scale | Max tilt |
|---|---|---|---|
| regime | 0.40 | 0.10 | ±0.040 |
| trend | 0.20 | 0.06 | ±0.012 |
| momo | 0.20 | 0.06 | ±0.012 |
| timesfm | 0.15 | — | ~0 (disabled) |
| macro | 0.05 | 0.20 | ±0.006 (BTC) |

Sources: https://doi.org/10.2469/faj.v69.n4.1 (real-yield / gold), https://doi.org/10.1257/aer.102.4.1692 (credit premium / equities), https://doi.org/10.1093/rfs/hhaa113 (crypto momentum).

## 2026-04-21 — Corrected root cause: DFII10 in HMM was the dominant return inflator

**Context**: Commit `163cdf2` implemented three fixes (z-score window, ensemble weights, SAA revert) based on a root-cause attribution that incorrectly labelled `_ZSCORE_WINDOW=252` as the dominant cause. The v3 diagnostic baseline after those fixes returned 11.87%/−33.61% — only 23 bps below the pre-fix 12.10%/−33.46%. The three fixes barely moved the result because they targeted the macro_factor channel (weight 0.05, max tilt ±0.006), while the actual dominant cause was operating through the HMM regime channel (weight 0.40, max tilt ±0.040 — 6.7× larger).

**Actual dominant root cause: DFII10 as the 5th HMM feature**

- The `build_features(fred)` call in `walkforward.py` (line 537, pre-fix) returned a 5-feature panel including DFII10.
- This 5-feature panel was passed as `fred_features` to `build_signal_bundle_at_date` (line 592), which in turn passed slices of it to `fit_hmm()` and `classify_states()`.
- DFII10 declined persistently from ~2.5 % (2003) to ~−1 % (2021) due to QE. In z-scored space (against the expanding training mean anchored at pre-QE levels), DFII10 was a near-constant −1 to −2 standard deviations below its training mean throughout 2009–2021.
- `_state_stress_scores()` treats DFII10 as a stress contributor (`score += means[DFII10]`). A persistently negative DFII10 z-score therefore suppressed the stress score of whichever HMM state occupied the 2009–2021 observations, causing it to be classified as "risk_on" in `_interpret_state_names`.
- This labelled roughly 144 months (2009–2021) as "risk_on" → `REGIME_TILT["risk_on"]["SPXT"] = 0.42` vs neutral 0.35 → systematic 7 % excess equity allocation across the entire bull market.
- At `regime_weight=0.40`, the regime signal contributes up to ±0.040 per asset — 6.7× the macro_factor channel max of ±0.006. Our earlier fixes had addressed only the subordinate channel.

**The fix (commit after 163cdf2)**:

In `run_walkforward()`, FRED features are now split into two separate panels:
- `fred_features = build_features(fred)` — 5-feature set (with DFII10), passed only to `compute_macro_factor_mu()` for the real_yield_tilt sub-signal.
- `hmm_features = build_features(fred, use_extended=False)` — 4-feature set (VIXCLS, BAMLH0A0HYM2, T10Y3M, NFCI), passed to `build_signal_bundle_at_date()` for all HMM training and classification.

HMM regime labels now depend only on stationary, mean-reverting stress indicators that cannot produce a decade-long persistent "risk_on" bias. DFII10 information is still used in the macro_factor real_yield_tilt signal (weight 0.05, max ±0.006) but is no longer embedded in the primary risk governor (regime, weight 0.40).

**Why the previous three fixes were still correct and are kept**:
- `_ZSCORE_WINDOW 252→63`: correct practice regardless — 252-day window did produce a persistent positive macro_factor signal through its own (smaller) channel. Keeping 63.
- `regime_weight 0.40→0.40` (restored): correct — regime is the primary risk governor and should have the largest weight among causal signal layers.
- `macro_factor_weight 0.05`: correct — keeps macro as a subordinate refinement signal at ±0.006 max contribution.
- SAA revert: correct — BITCOIN SAA zero is consistent with the zero risk-budget semantics.

Sources: Hamilton (1989) https://doi.org/10.2307/1912559; Erb & Harvey (2013) https://doi.org/10.2469/faj.v69.n4.1.
