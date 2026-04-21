"""Select the winning canonical run and rebuild the final top-level artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from taa_project.analysis.config_comparison import (
    CONFIG_COMPARISON_FILENAME,
    SUBMISSION_SELECTION_FILENAME,
    select_submission_configuration,
    write_submission_selection,
)
from taa_project.config import FIGURES_DIR, NOTEBOOK_DIR, OUTPUT_DIR, REPORT_DIR


COMPARISON_DTYPE = {
    "Run": "string",
    "Ann. Return": "float32",
    "Ann. Vol": "float32",
    "Max DD": "float32",
    "Sharpe": "float32",
    "Sortino": "float32",
    "Calmar": "float32",
    "Deflated Sharpe": "float32",
    "Pass MDD": "string",
    "Pass Vol": "string",
    "Pass Return": "string",
}


def _copy_tree_contents(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        destination = target_dir / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def _run_builder(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the submission run and rebuild the final artifacts.")
    parser.add_argument("--run-root", default=str(OUTPUT_DIR / "runs"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(FIGURES_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    parser.add_argument("--notebook-dir", default=str(NOTEBOOK_DIR))
    args = parser.parse_args()

    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    report_dir = Path(args.report_dir)
    notebook_dir = Path(args.notebook_dir)

    comparison = pd.read_csv(output_dir / CONFIG_COMPARISON_FILENAME, dtype=COMPARISON_DTYPE)
    selection = select_submission_configuration(comparison, run_root=run_root)
    selection_path = write_submission_selection(selection, output_dir=output_dir)

    selected_run_root = run_root / selection["run_id"]
    _copy_tree_contents(selected_run_root / "outputs", output_dir)
    _copy_tree_contents(selected_run_root / "figures", figure_dir)

    _run_builder(
        [
            sys.executable,
            "-m",
            "taa_project.notebooks.build_diagnostics",
            "--output-dir",
            str(output_dir),
            "--notebook-dir",
            str(notebook_dir),
        ]
    )
    _run_builder(
        [
            sys.executable,
            "-m",
            "taa_project.report.build_report",
            "--output-dir",
            str(output_dir),
            "--figure-dir",
            str(figure_dir),
            "--report-dir",
            str(report_dir),
        ]
    )
    _run_builder(
        [
            sys.executable,
            "-m",
            "taa_project.report.build_deck",
            "--output-dir",
            str(output_dir),
            "--figure-dir",
            str(figure_dir),
            "--report-dir",
            str(report_dir),
        ]
    )

    print(f"Selected {selection['run_id']} and wrote {selection_path.name} to {output_dir}.")
    print(f"Final report: {report_dir / 'whitmore_report.pdf'}")
    print(f"Final deck: {report_dir / 'whitmore_deck.pdf'}")
    print(f"Final notebook: {notebook_dir / 'diagnostics.ipynb'}")


if __name__ == "__main__":
    main()
