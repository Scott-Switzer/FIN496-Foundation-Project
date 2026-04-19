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
4. Base end-to-end run without the optional TimesFM dependency:
   `python taa_project/main.py --start 2003-01-01 --end 2025-12-31 --folds 5 --no-timesfm`
5. If the official TimesFM stack is installed separately, enable it explicitly:
   `python taa_project/main.py --start 2003-01-01 --end 2025-12-31 --folds 5 --timesfm`
6. Generated CSV/PNG/PDF artifacts belong under `taa_project/outputs/`.
7. The submission bundle is produced by `make zip`.

## Submission

- Canvas submission requirements placeholder:
  [course Canvas assignment page](https://canvas.instructure.com/)
- Replace that generic link with the actual course assignment URL before final
  submission packaging.
