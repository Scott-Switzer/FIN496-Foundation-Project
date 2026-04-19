# Addresses rubric criteria 2-5 and Task 6 by consolidating the canonical
# run sweep into a comparison table and risk-return figure.
"""Canonical configuration comparison utilities for Whitmore.

References:
- Whitmore Task 6 configuration-sweep requirement.

Point-in-time safety:
- Safe. This module reads already-generated ex-post run artifacts only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from taa_project.analysis.reporting import DSR_SUMMARY_FILENAME, PORTFOLIO_METRICS_FILENAME
from taa_project.config import FIGURES_DIR, MAX_DD, OUTPUT_DIR, VOL_CEILING


RUNS_ROOT_DIRNAME = "runs"
CONFIG_COMPARISON_FILENAME = "config_comparison.csv"
CONFIG_COMPARISON_FIGURE = "config_comparison.png"
CANONICAL_RUN_ORDER = [
    "baseline",
    "timesfm_vb10",
    "timesfm_vb08",
    "timesfm_vb07",
    "timesfm_regime_vb",
    "timesfm_regime_dd",
]
RETURN_TARGET = 0.08


def _run_label(run_id: str) -> str:
    return {
        "baseline": "Baseline",
        "timesfm_vb10": "TimesFM VB10",
        "timesfm_vb08": "TimesFM VB08",
        "timesfm_vb07": "TimesFM VB07",
        "timesfm_regime_vb": "TimesFM Regime VB",
        "timesfm_regime_dd": "TimesFM Regime + DD",
    }.get(run_id, run_id)


def _overlay_bucket(run_id: str) -> str:
    if run_id in {"BM1", "BM2"}:
        return "Benchmark"
    if run_id == "baseline":
        return "No TimesFM"
    if run_id == "timesfm_regime_dd":
        return "Regime + DD"
    if run_id == "timesfm_regime_vb":
        return "Regime Vol"
    return "Flat Vol + TimesFM"


def _pass_fail(flag: bool) -> str:
    return "Y" if flag else "N"


def build_config_comparison(
    run_root: Path = OUTPUT_DIR / RUNS_ROOT_DIRNAME,
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """Build the canonical configuration comparison table and scatter figure.

    Inputs:
    - `run_root`: directory containing per-run subdirectories.
    - `output_dir`: destination for `config_comparison.csv`.
    - `figure_dir`: destination for `config_comparison.png`.

    Outputs:
    - Comparison dataframe covering the six canonical runs plus BM1 and BM2.

    Citation:
    - Whitmore Task 6 configuration-sweep requirement.

    Point-in-time safety:
    - Safe. This module reads only already-generated run artifacts.
    """

    missing = [run_id for run_id in CANONICAL_RUN_ORDER if not (run_root / run_id / "outputs" / PORTFOLIO_METRICS_FILENAME).exists()]
    if missing:
        raise FileNotFoundError(f"Missing canonical run outputs for: {missing}")

    rows: list[dict[str, object]] = []
    benchmark_added = False

    for run_id in CANONICAL_RUN_ORDER:
        run_dir = run_root / run_id
        metrics = pd.read_csv(run_dir / "outputs" / PORTFOLIO_METRICS_FILENAME)
        dsr_summary = pd.read_csv(run_dir / "outputs" / DSR_SUMMARY_FILENAME).iloc[0]

        if not benchmark_added:
            for benchmark in ("BM1", "BM2"):
                benchmark_row = metrics.loc[metrics["portfolio"] == benchmark].iloc[0]
                rows.append(
                    {
                        "Run": benchmark,
                        "Ann. Return": benchmark_row["annualized_return"],
                        "Ann. Vol": benchmark_row["annualized_volatility"],
                        "Max DD": benchmark_row["max_drawdown"],
                        "Sharpe": benchmark_row["sharpe_rf_2pct"],
                        "Sortino": benchmark_row["sortino_rf_2pct"],
                        "Calmar": benchmark_row["calmar"],
                        "Deflated Sharpe": np.nan,
                        "Pass MDD": _pass_fail(float(benchmark_row["max_drawdown"]) >= -MAX_DD),
                        "Pass Vol": _pass_fail(float(benchmark_row["annualized_volatility"]) <= VOL_CEILING),
                        "Pass Return": _pass_fail(float(benchmark_row["annualized_return"]) >= RETURN_TARGET),
                        "overlay_bucket": _overlay_bucket(benchmark),
                    }
                )
            benchmark_added = True

        strategy_row = metrics.loc[metrics["portfolio"] == "SAA+TAA"].iloc[0]
        rows.append(
            {
                "Run": _run_label(run_id),
                "Ann. Return": strategy_row["annualized_return"],
                "Ann. Vol": strategy_row["annualized_volatility"],
                "Max DD": strategy_row["max_drawdown"],
                "Sharpe": strategy_row["sharpe_rf_2pct"],
                "Sortino": strategy_row["sortino_rf_2pct"],
                "Calmar": strategy_row["calmar"],
                "Deflated Sharpe": dsr_summary["baseline_dsr"],
                "Pass MDD": _pass_fail(float(strategy_row["max_drawdown"]) >= -MAX_DD),
                "Pass Vol": _pass_fail(float(strategy_row["annualized_volatility"]) <= VOL_CEILING),
                "Pass Return": _pass_fail(float(strategy_row["annualized_return"]) >= RETURN_TARGET),
                "overlay_bucket": _overlay_bucket(run_id),
            }
        )

    comparison = pd.DataFrame(rows)
    ordered_runs = ["BM1", "BM2"] + [_run_label(run_id) for run_id in CANONICAL_RUN_ORDER]
    comparison["Run"] = pd.Categorical(comparison["Run"], categories=ordered_runs, ordered=True)
    comparison = comparison.sort_values("Run").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    comparison.drop(columns=["overlay_bucket"]).to_csv(output_dir / CONFIG_COMPARISON_FILENAME, index=False)

    colors = {
        "Benchmark": "#475569",
        "No TimesFM": "#94a3b8",
        "Flat Vol + TimesFM": "#2563eb",
        "Regime Vol": "#059669",
        "Regime + DD": "#dc2626",
    }
    plt.figure(figsize=(10, 6), dpi=200)
    for _, row in comparison.iterrows():
        bucket = row["overlay_bucket"]
        plt.scatter(
            100.0 * float(row["Ann. Vol"]),
            100.0 * float(row["Ann. Return"]),
            s=80,
            color=colors.get(bucket, "#111827"),
            alpha=0.9,
        )
        plt.annotate(str(row["Run"]), (100.0 * float(row["Ann. Vol"]), 100.0 * float(row["Ann. Return"])), xytext=(6, 4), textcoords="offset points", fontsize=8)

    plt.axvline(100.0 * VOL_CEILING, color="#0f172a", linestyle="--", linewidth=1.0, label="15% vol ceiling")
    plt.axhline(100.0 * RETURN_TARGET, color="#7c3aed", linestyle="--", linewidth=1.0, label="8% return target")
    plt.xlabel("Annualized Volatility (%)")
    plt.ylabel("Annualized Return (%)")
    plt.title("Canonical Configuration Comparison")
    plt.grid(alpha=0.25)
    legend_labels = []
    legend_handles = []
    for bucket, color in colors.items():
        if bucket in comparison["overlay_bucket"].values:
            legend_labels.append(bucket)
            legend_handles.append(plt.Line2D([], [], color=color, marker="o", linestyle="", markersize=7))
    legend_handles.append(plt.Line2D([], [], color="#0f172a", linestyle="--"))
    legend_labels.append("15% vol ceiling")
    legend_handles.append(plt.Line2D([], [], color="#7c3aed", linestyle="--"))
    legend_labels.append("8% return target")
    plt.legend(legend_handles, legend_labels, fontsize=8, frameon=True)
    plt.tight_layout()
    plt.savefig(figure_dir / CONFIG_COMPARISON_FIGURE, bbox_inches="tight")
    plt.close()

    return comparison.drop(columns=["overlay_bucket"])


def main() -> None:
    """CLI entrypoint for Task 6 configuration-comparison artifacts.

    Inputs:
    - `--run-root`: directory containing per-run subdirectories.
    - `--output-dir`: destination for the comparison CSV.
    - `--figure-dir`: destination for the comparison PNG.

    Outputs:
    - Writes `config_comparison.csv` and `config_comparison.png`.

    Citation:
    - Whitmore Task 6 configuration-sweep requirement.

    Point-in-time safety:
    - Ex-post comparison only.
    """

    parser = argparse.ArgumentParser(description="Build the Whitmore canonical configuration comparison.")
    parser.add_argument("--run-root", default=str(OUTPUT_DIR / RUNS_ROOT_DIRNAME), help="Directory containing per-run outputs.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Destination for the comparison CSV.")
    parser.add_argument("--figure-dir", default=str(FIGURES_DIR), help="Destination for the comparison PNG.")
    args = parser.parse_args()

    comparison = build_config_comparison(run_root=Path(args.run_root), output_dir=Path(args.output_dir), figure_dir=Path(args.figure_dir))
    print(
        f"Wrote {Path(args.output_dir) / CONFIG_COMPARISON_FILENAME} and "
        f"{Path(args.figure_dir) / CONFIG_COMPARISON_FIGURE}. Rows: {len(comparison)}"
    )


if __name__ == "__main__":
    main()
