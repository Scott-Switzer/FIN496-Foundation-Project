# Whitmore FIN 496 Project

This package contains the reproducible research code for the Whitmore Capital
Partners SAA/TAA mandate in the FIN 496 Foundation Project. The codebase is
being built to satisfy the IPS in [IPS.md](/Users/scottthomasswitzer/Desktop/FIN496FP/FIN496-Foundation-Project/IPS.md),
the concise rule set in [Guidelines.md](/Users/scottthomasswitzer/Desktop/FIN496FP/FIN496-Foundation-Project/Guidelines.md),
and the data-handling constraints in [tasks.md](/Users/scottthomasswitzer/Desktop/FIN496FP/FIN496-Foundation-Project/tasks.md).

## Scope

- Build an IPS-compliant SAA layer on the authoritative Whitmore asset data.
- Add a causal TAA overlay driven by regime, trend, momentum, and optional
  TimesFM forecasting layers.
- Validate the combined process with expanding-window walk-forward backtests.
- Generate charts, attribution tables, a report PDF, and a presentation deck.

## File Map

- `config.py`: IPS constraints, universe definitions, and project paths.
- `data_loader.py`: audited data-access layer for prices and macro inputs.
- `data_audit.py`: data audit, return construction, and FRED lag handling.
- `main.py`: end-to-end pipeline entrypoint.
- `signals/`: regime, trend, momentum, and TimesFM signal modules.
- `optimizer/`: cvxpy portfolio solvers.
- `backtest/`: walk-forward backtesting code.
- `analysis/`: attribution, metrics, figures, IPS audit, and trial ledger logic.
- `report/`: PDF report and slide-deck builders.
- `signal_spec.md`: evolving written spec for the TAA signal stack.
- `outputs/`: generated CSV, PNG, PDF, and log artifacts. Kept out of git.
- `notebooks/`: diagnostics notebook generator and notebook outputs.

## Reproducibility

1. Use Python `3.11` as pinned in `.python-version`.
2. Install the base environment from the repo root:
   `pip install -r requirements.txt`
3. Install the official TimesFM runtime if you plan to use `--timesfm` or run
   the full canonical sweep. The sweep entrypoint expects a Python environment
   where `timesfm`, `torch`, `pyarrow`, `cvxpy`, and `psutil` are all
   available.
4. Run modules from the repository root so package imports resolve as
   `taa_project.*`.
5. Reproduce the full canonical 13-configuration sweep with:
   `python -m taa_project.scripts.run_sweep`
6. Run one configuration directly with `python -m taa_project.main ...` when
   you only need a single backtest.
7. Generated CSV/PNG/PDF artifacts belong under `taa_project/outputs/`.
8. The submission bundle is produced by `make zip`.

## Memory Discipline

- `python -m taa_project.scripts.run_sweep` launches one configuration per
  subprocess and never imports `taa_project.main` into the orchestrator.
- Peak RAM is designed to stay below `4 GB` per subprocess, with the
  orchestrator held below `200 MB`.
- TimesFM forecasts are cached on disk at
  `taa_project/outputs/cache/timesfm_forecasts.parquet` so repeated sweep runs
  reuse prior forecasts instead of rebuilding the model for cache hits.

## CLI Flags

- `--timesfm` / `--no-timesfm`: enable or disable the optional TimesFM layer.
- `--vol-budget`: flat internal ex-ante vol budget for the TAA optimizer.
- `--optimizer-mode {vol,cvar}`: choose the legacy volatility ceiling or the
  opt-in CVaR constraint family.
- `--cvar-alpha`, `--cvar-budget`, `--cvar-lookback`: configure the CVaR tail
  probability, budget, and causal trailing scenario window.
- `--regime-vol-budgets`: JSON map for regime-conditional risk envelopes, for example `{"risk_on":0.10,"neutral":0.08,"stress":0.05}`.
- `--dd-guardrail` / `--no-dd-guardrail`: enable or disable the realized drawdown clip overlay.
- `--nested-risk` / `--no-nested-risk`: enable or disable sequential
  Core/Satellite/Non-Traditional sleeve risk budgeting.
- `--nested-core-vol`, `--nested-sat-vol`, `--nested-nt-vol`: per-sleeve
  annualized risk targets for nested risk budgeting.
- `--nested-sleeve-weights`: comma-separated sleeve blend, for example
  `0.55,0.35,0.10`.
- `--saa-method {risk_parity,hrp}`: choose the strategic allocator used in the
  annual SAA book.
- `--bl-stress-views` / `--no-bl-stress-views`: enable or disable
  regime-conditional pessimistic Black-Litterman equity views.
- `--bl-stress-shock`: size of the stress-regime equity shock in annualized
  sigmas.
- `--output-dir`, `--figure-dir`, `--report-dir`, `--notebook-dir`: override artifact destinations.

## Canonical Run Sweep

- The sweep entrypoint is `python -m taa_project.scripts.run_sweep`.
- It warms the shared TimesFM cache first, then runs the following 13
  configurations sequentially in isolated subprocesses:
- `baseline`
- `timesfm_vb10`
- `timesfm_vb08`
- `timesfm_vb07`
- `timesfm_regime_vb`
- `timesfm_regime_dd`
- `cvar95_vb_2_5`
- `cvar99_vb_4_0`
- `nested_risk_default`
- `nested_risk_cvar`
- `hrp_saa`
- `bl_stress_full`
- `kitchen_sink`
- The consolidated comparison is written to
  `taa_project/outputs/config_comparison.csv`, and the selected submission
  configuration is recorded in
  `taa_project/outputs/submission_selection.json`.

## Submission

- Canvas submission requirements placeholder:
  [course Canvas assignment page](https://canvas.instructure.com/)
- Replace that generic link with the actual course assignment URL before final
  submission packaging.

## Known Limitations

- Over the 2003-01-01 to 2025-12-31 evaluation window, no tested long-only configuration satisfying the IPS allocation constraints achieved the IPS max-drawdown tolerance of `-25%`. The selected `timesfm_vb08` submission is the closest tested variant by drawdown at `-27.46%` while still beating both benchmarks on return and staying below the 15% volatility ceiling.
