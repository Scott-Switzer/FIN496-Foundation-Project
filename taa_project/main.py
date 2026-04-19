# Addresses the reproducibility requirement and Tasks 12-13 by providing a
# single deterministic entrypoint for the full Whitmore pipeline.
"""Whitmore end-to-end pipeline orchestrator.

References:
- Whitmore project brief and `tasks.md`.
- Bailey & López de Prado (2014):
  https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

Point-in-time safety:
- Safe. This orchestrator only composes the underlying point-in-time-safe
  modules in the required task order.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import torch as _torch  # noqa: F401  # Pin the OpenMP runtime before SciPy / sklearn imports.
except ImportError:
    _torch = None

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from taa_project.analysis.attribution import build_attribution
from taa_project.analysis.reporting import build_reporting
from taa_project.backtest.walkforward import run_walkforward
from taa_project.benchmarks import build_benchmarks
from taa_project.config import (
    DEFAULT_RANDOM_SEED,
    FIGURES_DIR,
    MPLCONFIG_DIR,
    NOTEBOOK_DIR,
    OUTPUT_DIR,
    REPORT_DIR,
    TARGET_VOL,
    TRIAL_LEDGER_CSV,
    VOL_CEILING,
)
from taa_project.data_audit import run_data_audit
from taa_project.notebooks.build_diagnostics import build_diagnostics_notebook
from taa_project.optimizer.cvxpy_opt import EnsembleConfig
from taa_project.report.build_deck import build_deck
from taa_project.report.build_report import build_report
from taa_project.saa.build_saa import build_saa_portfolio
from taa_project.signals.vol_timesfm import timesfm_is_available


RUN_TRIAL_LEDGER_COLUMNS = [
    "trial_id",
    "timestamp_utc",
    "use_timesfm",
    "vol_budget",
    "folds",
    "start",
    "end",
    "ann_return",
    "ann_vol",
    "max_dd",
    "sharpe",
    "sortino",
    "calmar",
    "notes",
]


def _timestamp() -> str:
    """Return the current local timestamp used in pipeline logs.

    Inputs:
    - None.

    Outputs:
    - ISO-formatted timestamp string.

    Citation:
    - Internal logging helper.

    Point-in-time safety:
    - Operational logging only.
    """

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_step(message: str) -> None:
    """Print one timestamped pipeline log line.

    Inputs:
    - `message`: human-readable log message.

    Outputs:
    - Writes one line to stdout.

    Citation:
    - Whitmore Task 12 logging requirement.

    Point-in-time safety:
    - Operational logging only.
    """

    print(f"[{_timestamp()}] {message}")


def _validate_vol_budget(vol_budget: float) -> float:
    """Validate the internal TAA volatility budget.

    Inputs:
    - `vol_budget`: requested internal ex-ante annualized volatility target.

    Outputs:
    - The validated `vol_budget`.

    Citation:
    - Whitmore Task 3 vol-budget requirement.

    Point-in-time safety:
    - Safe. This is static runtime validation only.
    """

    if vol_budget > VOL_CEILING:
        raise ValueError(
            f"vol_budget={vol_budget:.4f} exceeds VOL_CEILING={VOL_CEILING:.4f}. "
            "Use an internal target at or below the IPS volatility ceiling."
        )
    if vol_budget < 0.02:
        raise ValueError(
            f"vol_budget={vol_budget:.4f} is below 0.0200. "
            "This is likely a typo; the pipeline refuses to run with unrealistically tight budgets."
        )
    return vol_budget


def _append_pipeline_trial_row(
    *,
    start: str,
    end: str,
    folds: int,
    use_timesfm: bool,
    vol_budget: float,
    metrics: pd.DataFrame,
) -> None:
    """Append one pipeline-run disclosure row to `TRIAL_LEDGER.csv`.

    Inputs:
    - `start`, `end`, `folds`: run configuration.
    - `use_timesfm`: whether TimesFM was enabled.
    - `vol_budget`: internal optimizer volatility target for this run.
    - `metrics`: Task 8 portfolio-metrics dataframe.

    Outputs:
    - Appends one row to `TRIAL_LEDGER.csv` while preserving existing history.

    Citation:
    - Whitmore Task 3 trial-disclosure requirement.

    Point-in-time safety:
    - Safe. This is ex-post run logging only.
    """

    strategy_metrics = metrics.loc[metrics["portfolio"] == "SAA+TAA"]
    row_source = strategy_metrics.iloc[0] if not strategy_metrics.empty else pd.Series(dtype=float)
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    trial_id = (
        f"pipeline_{timestamp_utc.replace(':', '').replace('-', '').replace('+00:00', 'Z')}"
        f"_{'timesfm' if use_timesfm else 'no_timesfm'}_vb{int(round(vol_budget * 1000)):03d}"
    )
    new_row = pd.DataFrame(
        [
            {
                "trial_id": trial_id,
                "timestamp_utc": timestamp_utc,
                "use_timesfm": int(use_timesfm),
                "vol_budget": vol_budget,
                "folds": folds,
                "start": start,
                "end": end,
                "ann_return": row_source.get("annualized_return", np.nan),
                "ann_vol": row_source.get("annualized_volatility", np.nan),
                "max_dd": row_source.get("max_drawdown", np.nan),
                "sharpe": row_source.get("sharpe_rf_2pct", np.nan),
                "sortino": row_source.get("sortino_rf_2pct", np.nan),
                "calmar": row_source.get("calmar", np.nan),
                "notes": "Pipeline run summary row for the SAA+TAA portfolio.",
            }
        ]
    )
    if TRIAL_LEDGER_CSV.exists():
        existing = pd.read_csv(TRIAL_LEDGER_CSV)
    else:
        existing = pd.DataFrame(columns=RUN_TRIAL_LEDGER_COLUMNS)
    all_columns = list(dict.fromkeys(existing.columns.tolist() + RUN_TRIAL_LEDGER_COLUMNS + new_row.columns.tolist()))
    frames = [frame for frame in (existing, new_row) if not frame.empty]
    combined = (
        pd.concat([frame.reindex(columns=all_columns) for frame in frames], ignore_index=True)
        if frames
        else pd.DataFrame(columns=all_columns)
    )
    combined.to_csv(TRIAL_LEDGER_CSV, index=False)


def _parse_regime_vol_budgets(raw_json: str | None) -> dict[str, float] | None:
    """Parse optional regime-specific volatility budgets from JSON.

    Inputs:
    - `raw_json`: JSON string such as `{"risk_on": 0.10, "neutral": 0.08, "stress": 0.05}`.

    Outputs:
    - Parsed mapping from regime label to validated vol budget, or `None`.

    Citation:
    - Whitmore Task 4 regime-vol overlay requirement.

    Point-in-time safety:
    - Safe. This is static runtime configuration only.
    """

    if raw_json is None:
        return None
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse --regime-vol-budgets JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("--regime-vol-budgets must decode to a JSON object.")
    budgets: dict[str, float] = {}
    for regime_label, value in parsed.items():
        budgets[str(regime_label)] = _validate_vol_budget(float(value))
    return budgets


def seed_everything(seed: int = DEFAULT_RANDOM_SEED, seed_torch: bool = False) -> None:
    """Seed Python, NumPy, and Torch RNGs for deterministic runs.

    Inputs:
    - `seed`: integer random seed.
    - `seed_torch`: whether to attempt Torch seeding as well.

    Outputs:
    - Seeds the relevant RNGs in-place.

    Citation:
    - Whitmore reproducibility requirement.

    Point-in-time safety:
    - Safe. This affects only computational determinism, not data visibility.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def _history_start(start: str) -> str:
    """Return the earliest date used for SAA and benchmark history builds.

    Inputs:
    - `start`: requested OOS walk-forward start date.

    Outputs:
    - Earliest history start date for non-OOS modules.

    Citation:
    - Whitmore Tasks 2 and 3 full-history requirements.

    Point-in-time safety:
    - Safe. This is a deterministic date helper only.
    """

    return str(min(pd.Timestamp("2000-01-01"), pd.Timestamp(start)).date())


def run_pipeline(
    start: str = "2003-01-01",
    end: str = "2025-12-31",
    folds: int = 5,
    use_timesfm: bool = False,
    vol_budget: float = TARGET_VOL,
    regime_vol_budgets: dict[str, float] | None = None,
    use_dd_guardrail: bool = False,
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURES_DIR,
    report_dir: Path = REPORT_DIR,
    notebook_dir: Path = NOTEBOOK_DIR,
) -> dict[str, object]:
    """Run the full Whitmore pipeline from audit through PDFs.

    Inputs:
    - `start`, `end`, `folds`: walk-forward settings.
    - `use_timesfm`: whether to enable the optional TimesFM signal layer.
    - `vol_budget`: internal ex-ante vol target passed into the TAA optimizer.
    - `regime_vol_budgets`: optional regime-specific vol targets that override
      the flat monthly budget by inferred HMM state.
    - `use_dd_guardrail`: whether to enable the drawdown-clip overlay.
    - `output_dir`: destination directory for generated CSV artifacts.
    - `figure_dir`: destination directory for figure PNGs.
    - `report_dir`: destination directory for report/deck PDFs.
    - `notebook_dir`: destination directory for the diagnostics notebook.

    Outputs:
    - Dictionary of artifact paths and in-memory summary tables.

    Citation:
    - Whitmore Tasks 0-13.

    Point-in-time safety:
    - Safe. This function composes only the project modules that already
      enforce the point-in-time data rules.
    """

    if use_timesfm and not timesfm_is_available():
        raise RuntimeError(
            "TimesFM was requested with --timesfm, but the dependency is not installed. "
            "Rerun with --no-timesfm or install the official google-research/timesfm stack."
        )
    vol_budget = _validate_vol_budget(vol_budget)
    if regime_vol_budgets is not None:
        for regime_label, regime_budget in regime_vol_budgets.items():
            regime_vol_budgets[regime_label] = _validate_vol_budget(regime_budget)
    ensemble_config = EnsembleConfig(vol_budget_by_regime=regime_vol_budgets, use_dd_guardrail=use_dd_guardrail)

    seed_everything(DEFAULT_RANDOM_SEED, seed_torch=use_timesfm)
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    notebook_dir.mkdir(parents=True, exist_ok=True)

    history_start = _history_start(start)
    breaches_log = output_dir / "breaches.log"
    breaches_log.write_text("", encoding="utf-8")

    log_step("Task 1: data audit")
    audit_artifacts = run_data_audit(output_dir=output_dir)

    log_step("Task 2: SAA portfolio")
    build_saa_portfolio(start_date=history_start, end_date=end, output_dir=output_dir)

    log_step("Task 3: benchmarks")
    build_benchmarks(start_date=history_start, end_date=end, output_dir=output_dir)

    log_step("Task 6: walk-forward backtest")
    walkforward_artifacts = run_walkforward(
        start=start,
        end=end,
        folds=folds,
        use_timesfm=use_timesfm,
        vol_budget=vol_budget,
        ensemble_config=ensemble_config,
        output_dir=output_dir,
    )

    log_step("Task 7: attribution")
    attribution_artifacts = build_attribution(
        start=start,
        end=end,
        folds=folds,
        use_timesfm=use_timesfm,
        vol_budget=vol_budget,
        ensemble_config=ensemble_config,
        output_dir=output_dir,
    )

    log_step("Task 8: metrics, figures, IPS audit, trial ledger")
    reporting_artifacts = build_reporting(
        start=start,
        end=end,
        folds=folds,
        use_timesfm=use_timesfm,
        vol_budget=vol_budget,
        ensemble_config=ensemble_config,
        output_dir=output_dir,
        figure_dir=figure_dir,
    )
    _append_pipeline_trial_row(
        start=start,
        end=end,
        folds=folds,
        use_timesfm=use_timesfm,
        vol_budget=vol_budget,
        metrics=reporting_artifacts["metrics"],
    )

    ips_compliance = reporting_artifacts["ips_compliance"]
    if not ips_compliance.empty:
        raise RuntimeError(
            f"IPS compliance audit failed with {len(ips_compliance)} violation rows. "
            f"Inspect {output_dir / 'ips_compliance.csv'}."
        )

    log_step("Task 9: diagnostics notebook")
    notebook_path = build_diagnostics_notebook(output_dir=output_dir, notebook_dir=notebook_dir)

    log_step("Task 10: report PDF")
    report_markdown_path, report_pdf_path = build_report(
        output_dir=output_dir,
        figure_dir=figure_dir,
        report_dir=report_dir,
    )

    log_step("Task 11: presentation deck PDF")
    deck_pdf_path = build_deck(
        output_dir=output_dir,
        figure_dir=figure_dir,
        report_dir=report_dir,
    )

    log_step("Pipeline complete")
    return {
        "audit": audit_artifacts,
        "walkforward": walkforward_artifacts,
        "attribution": attribution_artifacts,
        "reporting": reporting_artifacts,
        "notebook_path": notebook_path,
        "report_markdown_path": report_markdown_path,
        "report_pdf_path": report_pdf_path,
        "deck_pdf_path": deck_pdf_path,
    }


def main() -> None:
    """CLI entrypoint for the full Whitmore pipeline.

    Inputs:
    - `--start`, `--end`, `--folds`: walk-forward settings.
    - `--timesfm` / `--no-timesfm`: enable or disable the optional TimesFM
      layer.
    - `--vol-budget`: internal ex-ante vol target passed into the TAA
      optimizer.
    - `--regime-vol-budgets`: optional JSON mapping from regime label to
      regime-specific vol budget.
    - `--dd-guardrail` / `--no-dd-guardrail`: enable or disable the
      drawdown-clip overlay.
    - `--output-dir`, `--figure-dir`, `--report-dir`, `--notebook-dir`:
      artifact destinations.

    Outputs:
    - Regenerates the full Whitmore artifact set.

    Citation:
    - Whitmore Task 12 single-entrypoint requirement.

    Point-in-time safety:
    - Safe. The CLI only orchestrates the point-in-time-safe pipeline above.
    """

    parser = argparse.ArgumentParser(description="Run the full Whitmore SAA/TAA pipeline.")
    parser.add_argument("--start", default="2003-01-01", help="First OOS date for the walk-forward backtest.")
    parser.add_argument("--end", default="2025-12-31", help="Last date for the generated outputs.")
    parser.add_argument("--folds", type=int, default=5, help="Number of contiguous OOS folds.")
    timesfm_group = parser.add_mutually_exclusive_group()
    timesfm_group.add_argument("--timesfm", dest="use_timesfm", action="store_true", help="Enable the optional TimesFM layer.")
    timesfm_group.add_argument("--no-timesfm", dest="use_timesfm", action="store_false", help="Disable the TimesFM layer.")
    parser.set_defaults(use_timesfm=False)
    parser.add_argument(
        "--vol-budget",
        dest="vol_budget",
        type=float,
        default=TARGET_VOL,
        help="Internal ex-ante vol target used by the TAA optimizer (default 0.10 = IPS internal target).",
    )
    parser.add_argument(
        "--regime-vol-budgets",
        dest="regime_vol_budgets",
        default=None,
        help='Optional JSON mapping such as {"risk_on":0.10,"neutral":0.08,"stress":0.05}.',
    )
    guardrail_group = parser.add_mutually_exclusive_group()
    guardrail_group.add_argument("--dd-guardrail", dest="use_dd_guardrail", action="store_true", help="Enable the drawdown-clip overlay.")
    guardrail_group.add_argument("--no-dd-guardrail", dest="use_dd_guardrail", action="store_false", help="Disable the drawdown-clip overlay.")
    parser.set_defaults(use_dd_guardrail=False)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Destination directory for CSV outputs.")
    parser.add_argument("--figure-dir", default=str(FIGURES_DIR), help="Destination directory for figure PNGs.")
    parser.add_argument("--report-dir", default=str(REPORT_DIR), help="Destination directory for report/deck PDFs.")
    parser.add_argument("--notebook-dir", default=str(NOTEBOOK_DIR), help="Destination directory for the diagnostics notebook.")
    args = parser.parse_args()

    run_pipeline(
        start=args.start,
        end=args.end,
        folds=args.folds,
        use_timesfm=args.use_timesfm,
        vol_budget=args.vol_budget,
        regime_vol_budgets=_parse_regime_vol_budgets(args.regime_vol_budgets),
        use_dd_guardrail=args.use_dd_guardrail,
        output_dir=Path(args.output_dir),
        figure_dir=Path(args.figure_dir),
        report_dir=Path(args.report_dir),
        notebook_dir=Path(args.notebook_dir),
    )


if __name__ == "__main__":
    main()
