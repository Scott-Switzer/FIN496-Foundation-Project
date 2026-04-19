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
import random
import sys
from datetime import datetime
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
)
from taa_project.data_audit import run_data_audit
from taa_project.notebooks.build_diagnostics import build_diagnostics_notebook
from taa_project.report.build_deck import build_deck
from taa_project.report.build_report import build_report
from taa_project.saa.build_saa import build_saa_portfolio
from taa_project.signals.vol_timesfm import timesfm_is_available


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
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURES_DIR,
    report_dir: Path = REPORT_DIR,
    notebook_dir: Path = NOTEBOOK_DIR,
) -> dict[str, object]:
    """Run the full Whitmore pipeline from audit through PDFs.

    Inputs:
    - `start`, `end`, `folds`: walk-forward settings.
    - `use_timesfm`: whether to enable the optional TimesFM signal layer.
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
        output_dir=output_dir,
    )

    log_step("Task 7: attribution")
    attribution_artifacts = build_attribution(
        start=start,
        end=end,
        folds=folds,
        use_timesfm=use_timesfm,
        output_dir=output_dir,
    )

    log_step("Task 8: metrics, figures, IPS audit, trial ledger")
    reporting_artifacts = build_reporting(
        start=start,
        end=end,
        folds=folds,
        use_timesfm=use_timesfm,
        output_dir=output_dir,
        figure_dir=figure_dir,
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
        output_dir=Path(args.output_dir),
        figure_dir=Path(args.figure_dir),
        report_dir=Path(args.report_dir),
        notebook_dir=Path(args.notebook_dir),
    )


if __name__ == "__main__":
    main()
