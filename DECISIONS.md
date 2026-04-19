# Decisions

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
