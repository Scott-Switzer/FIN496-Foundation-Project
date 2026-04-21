"""Build the canonical configuration comparison table and scatter plot."""

from __future__ import annotations

import argparse
from pathlib import Path

from taa_project.analysis.config_comparison import (
    CONFIG_COMPARISON_FILENAME,
    CONFIG_COMPARISON_FIGURE,
    build_config_comparison,
)
from taa_project.config import FIGURES_DIR, OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Whitmore canonical configuration comparison.")
    parser.add_argument("--run-root", default=str(OUTPUT_DIR / "runs"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(FIGURES_DIR))
    args = parser.parse_args()

    comparison = build_config_comparison(
        run_root=Path(args.run_root),
        output_dir=Path(args.output_dir),
        figure_dir=Path(args.figure_dir),
    )
    print(
        f"Wrote {Path(args.output_dir) / CONFIG_COMPARISON_FILENAME} and "
        f"{Path(args.figure_dir) / CONFIG_COMPARISON_FIGURE} with {len(comparison)} rows."
    )


if __name__ == "__main__":
    main()
