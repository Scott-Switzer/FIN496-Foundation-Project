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
3. Run modules from the repository root so package imports resolve as
   `taa_project.*`.
4. Chosen submission configuration:
   `python -m taa_project.main --start 2003-01-01 --end 2025-12-31 --folds 5 --no-timesfm --vol-budget 0.10`
5. Generated CSV/PNG/PDF artifacts belong under `taa_project/outputs/`.
6. The submission bundle is produced by `make zip`.

## CLI Flags

- `--timesfm` / `--no-timesfm`: enable or disable the optional TimesFM layer.
- `--vol-budget`: flat internal ex-ante vol budget for the TAA optimizer.
- `--regime-vol-budgets`: JSON map for regime-conditional risk envelopes, for example `{"risk_on":0.10,"neutral":0.08,"stress":0.05}`.
- `--dd-guardrail` / `--no-dd-guardrail`: enable or disable the realized drawdown clip overlay.
- `--output-dir`, `--figure-dir`, `--report-dir`, `--notebook-dir`: override artifact destinations.

## Canonical Run Sweep

- `baseline`: `python -m taa_project.main --no-timesfm --vol-budget 0.10`
- `timesfm_vb10`: `python -m taa_project.main --timesfm --vol-budget 0.10`
- `timesfm_vb08`: `python -m taa_project.main --timesfm --vol-budget 0.08`
- `timesfm_vb07`: `python -m taa_project.main --timesfm --vol-budget 0.07`
- `timesfm_regime_vb`: `python -m taa_project.main --timesfm --regime-vol-budgets '{"risk_on":0.10,"neutral":0.08,"stress":0.05}'`
- `timesfm_regime_dd`: `python -m taa_project.main --timesfm --regime-vol-budgets '{"risk_on":0.10,"neutral":0.08,"stress":0.05}' --dd-guardrail`

The consolidated comparison is written to `taa_project/outputs/config_comparison.csv`, and the selected submission configuration is recorded in `taa_project/outputs/submission_selection.json`.

## Submission

- Canvas submission requirements placeholder:
  [course Canvas assignment page](https://canvas.instructure.com/)
- Replace that generic link with the actual course assignment URL before final
  submission packaging.

## Known Limitations

- Over the 2003-01-01 to 2025-12-31 evaluation window, no tested long-only configuration satisfying the IPS allocation constraints achieved the IPS max-drawdown tolerance of `-25%`. The selected `timesfm_vb08` submission is the closest tested variant by drawdown at `-27.46%` while still beating both benchmarks on return and staying below the 15% volatility ceiling.
