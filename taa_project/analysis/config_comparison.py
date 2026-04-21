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
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from taa_project.analysis.common import deflated_sharpe_ratio, disclosed_trial_count
from taa_project.analysis.reporting import PORTFOLIO_METRICS_FILENAME
from taa_project.config import FIGURES_DIR, MAX_DD, OUTPUT_DIR, TRIAL_LEDGER_CSV, VOL_CEILING


RUNS_ROOT_DIRNAME = "runs"
CONFIG_COMPARISON_FILENAME = "config_comparison.csv"
CONFIG_COMPARISON_FIGURE = "config_comparison.png"
SUBMISSION_SELECTION_FILENAME = "submission_selection.json"
CANONICAL_RUN_ORDER = [
    "baseline",
    "timesfm_vb10",
    "timesfm_vb08",
    "timesfm_vb07",
    "timesfm_regime_vb",
    "timesfm_regime_dd",
    "cvar95_vb_2_5",
    "cvar99_vb_4_0",
    "nested_risk_default",
    "nested_risk_cvar",
    "hrp_saa",
    "bl_stress_full",
    "kitchen_sink",
]
RETURN_TARGET = 0.08

RUN_LABELS = {
    "baseline": "Baseline",
    "timesfm_vb10": "TimesFM VB10",
    "timesfm_vb08": "TimesFM VB08",
    "timesfm_vb07": "TimesFM VB07",
    "timesfm_regime_vb": "TimesFM Regime VB",
    "timesfm_regime_dd": "TimesFM Regime + DD",
    "cvar95_vb_2_5": "CVaR 95 2.5%",
    "cvar99_vb_4_0": "CVaR 99 4.0%",
    "nested_risk_default": "Nested Risk",
    "nested_risk_cvar": "Nested Risk + CVaR",
    "hrp_saa": "HRP SAA",
    "bl_stress_full": "BL Stress Views",
    "kitchen_sink": "Kitchen Sink",
}
RUN_IDS_BY_LABEL = {label: run_id for run_id, label in RUN_LABELS.items()}
PORTFOLIO_METRICS_DTYPE = {
    "portfolio": "string",
    "annualized_return": "float32",
    "annualized_volatility": "float32",
    "max_drawdown": "float32",
    "sharpe_rf_2pct": "float32",
    "sortino_rf_2pct": "float32",
    "calmar": "float32",
}
RETURNS_DTYPE = {
    "portfolio_return": "float32",
}


def _run_label(run_id: str) -> str:
    return RUN_LABELS.get(run_id, run_id)


def _overlay_bucket(run_id: str) -> str:
    if run_id in {"BM1", "BM2"}:
        return "Benchmark"
    if run_id == "baseline":
        return "No TimesFM"
    if run_id in {"timesfm_vb10", "timesfm_vb08", "timesfm_vb07", "timesfm_regime_vb", "timesfm_regime_dd"}:
        return "Vol-only"
    if run_id in {"cvar95_vb_2_5", "cvar99_vb_4_0"}:
        return "CVaR"
    if run_id in {"nested_risk_default", "nested_risk_cvar"}:
        return "Nested"
    if run_id == "hrp_saa":
        return "HRP"
    if run_id == "bl_stress_full":
        return "BL-stress"
    if run_id == "kitchen_sink":
        return "Kitchen-sink"
    return "Other"


def _pass_fail(flag: bool) -> str:
    return "Y" if flag else "N"


def _benchmark_dsr(run_dir: Path, benchmark: str, n_trials: int) -> float:
    """Compute the benchmark DSR using the full disclosed trial count.

    Inputs:
    - `run_dir`: canonical run directory containing benchmark return CSVs.
    - `benchmark`: benchmark label, either `BM1` or `BM2`.

    Outputs:
    - Benchmark DSR float.

    Citation:
    - Bailey & López de Prado (2014), Deflated Sharpe Ratio:
      https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

    Point-in-time safety:
    - Safe. This is an ex-post evaluation metric computed on realized returns.
    """

    returns_name = "bm1_returns.csv" if benchmark == "BM1" else "bm2_returns.csv"
    returns = pd.read_csv(run_dir / "outputs" / returns_name, dtype=RETURNS_DTYPE)["portfolio_return"]
    return float(deflated_sharpe_ratio(returns, n_trials=n_trials))


def _strategy_dsr(run_dir: Path, n_trials: int) -> float:
    """Compute the strategy DSR from the realized OOS return stream.

    Inputs:
    - `run_dir`: canonical run directory containing `oos_returns.csv`.
    - `n_trials`: full disclosed trial count from `TRIAL_LEDGER.csv`.

    Outputs:
    - Strategy DSR float.

    Citation:
    - Bailey & López de Prado (2014), Deflated Sharpe Ratio:
      https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

    Point-in-time safety:
    - Safe. This is an ex-post evaluation metric computed on realized returns.
    """

    returns = pd.read_csv(run_dir / "outputs" / "oos_returns.csv", dtype=RETURNS_DTYPE)["portfolio_return"]
    return float(deflated_sharpe_ratio(returns, n_trials=n_trials))


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
    n_trials = disclosed_trial_count(ledger_path=TRIAL_LEDGER_CSV)

    rows: list[dict[str, object]] = []
    benchmark_added = False

    for run_id in CANONICAL_RUN_ORDER:
        run_dir = run_root / run_id
        metrics = pd.read_csv(run_dir / "outputs" / PORTFOLIO_METRICS_FILENAME, dtype=PORTFOLIO_METRICS_DTYPE)

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
                        "Deflated Sharpe": _benchmark_dsr(run_dir, benchmark, n_trials=n_trials),
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
                "Deflated Sharpe": _strategy_dsr(run_dir, n_trials=n_trials),
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
        "Vol-only": "#2563eb",
        "CVaR": "#dc2626",
        "Nested": "#059669",
        "HRP": "#7c3aed",
        "BL-stress": "#d97706",
        "Kitchen-sink": "#111827",
        "Other": "#64748b",
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


def select_submission_configuration(
    comparison: pd.DataFrame,
    run_root: Path = OUTPUT_DIR / RUNS_ROOT_DIRNAME,
) -> dict[str, object]:
    """Select the submission configuration using the Task 7 decision tree.

    Inputs:
    - `comparison`: canonical configuration comparison dataframe.
    - `run_root`: canonical run-output root for loading SAA metrics.

    Outputs:
    - Dictionary summarizing the chosen configuration and the decision rule used.

    Citation:
    - Whitmore Task 7 submission-selection rule set.

    Point-in-time safety:
    - Safe. This is an ex-post selection summary using already-generated runs.
    """

    benchmarks = comparison.loc[comparison["Run"].isin(["BM1", "BM2"])].set_index("Run")
    strategies = comparison.loc[~comparison["Run"].isin(["BM1", "BM2"])].copy()
    if strategies.empty:
        raise ValueError("No strategy rows found in config comparison.")

    run_rank = {_run_label(run_id): rank for rank, run_id in enumerate(CANONICAL_RUN_ORDER)}
    strategies["run_rank"] = strategies["Run"].map(run_rank).fillna(len(run_rank)).astype(int)
    strategies["pass_all"] = (
        strategies["Pass MDD"].eq("Y") & strategies["Pass Vol"].eq("Y") & strategies["Pass Return"].eq("Y")
    )
    strategies["pass_mdd"] = strategies["Pass MDD"].eq("Y")
    bm2_dsr = float(benchmarks.loc["BM2", "Deflated Sharpe"])

    selection_rule = ""
    pool = strategies.loc[strategies["pass_all"]]
    if not pool.empty:
        selection_rule = "rule_1_all_constraints"
    else:
        pool = strategies.loc[strategies["pass_mdd"]]
        if not pool.empty:
            selection_rule = "rule_2_mdd_only"
        else:
            pool = strategies.loc[strategies["Deflated Sharpe"] > bm2_dsr]
            if pool.empty:
                pool = strategies.copy()
            selection_rule = "rule_3_smallest_mdd_breach"

    chosen = pool.sort_values(
        by=["Max DD", "Deflated Sharpe", "Ann. Return", "run_rank"],
        ascending=[False, False, False, True],
    ).iloc[0]
    run_label = str(chosen["Run"])
    run_id = RUN_IDS_BY_LABEL[run_label]
    run_metrics = pd.read_csv(
        run_root / run_id / "outputs" / PORTFOLIO_METRICS_FILENAME,
        dtype=PORTFOLIO_METRICS_DTYPE,
    ).set_index("portfolio")
    saa_dd = float(run_metrics.loc["SAA", "max_drawdown"])
    target_dd = -MAX_DD
    chosen_dd = float(chosen["Max DD"])
    breach_bps = max(0.0, target_dd - chosen_dd) * 10000.0

    return {
        "run_id": run_id,
        "display_name": run_label,
        "decision_rule": selection_rule,
        "n_tested_configurations": int(len(strategies)),
        "ann_return": float(chosen["Ann. Return"]),
        "ann_vol": float(chosen["Ann. Vol"]),
        "max_dd": chosen_dd,
        "sharpe": float(chosen["Sharpe"]),
        "sortino": float(chosen["Sortino"]),
        "calmar": float(chosen["Calmar"]),
        "deflated_sharpe": float(chosen["Deflated Sharpe"]),
        "pass_mdd": str(chosen["Pass MDD"]) == "Y",
        "pass_vol": str(chosen["Pass Vol"]) == "Y",
        "pass_return": str(chosen["Pass Return"]) == "Y",
        "all_constraints_passed": bool(
            (str(chosen["Pass MDD"]) == "Y")
            and (str(chosen["Pass Vol"]) == "Y")
            and (str(chosen["Pass Return"]) == "Y")
        ),
        "bm1_max_dd": float(benchmarks.loc["BM1", "Max DD"]),
        "bm2_max_dd": float(benchmarks.loc["BM2", "Max DD"]),
        "saa_max_dd": saa_dd,
        "bm2_dsr": bm2_dsr,
        "beats_bm2_on_dsr": bool(float(chosen["Deflated Sharpe"]) > bm2_dsr),
        "mdd_breach_bps": breach_bps,
        "mdd_improvement_bps_vs_bm2": (chosen_dd - float(benchmarks.loc["BM2", "Max DD"])) * 10000.0,
        "mdd_improvement_bps_vs_saa": (chosen_dd - saa_dd) * 10000.0,
    }


def write_submission_selection(selection: dict[str, object], output_dir: Path = OUTPUT_DIR) -> Path:
    """Write the submission-selection summary to JSON.

    Inputs:
    - `selection`: selection-summary dictionary.
    - `output_dir`: destination directory for the JSON file.

    Outputs:
    - Path to the written JSON file.

    Citation:
    - Whitmore Task 7 submission-selection documentation requirement.

    Point-in-time safety:
    - Safe. This writes an ex-post summary only.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SUBMISSION_SELECTION_FILENAME
    path.write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
    return path


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
    selection = select_submission_configuration(comparison, run_root=Path(args.run_root))
    selection_path = write_submission_selection(selection, output_dir=Path(args.output_dir))
    print(
        f"Wrote {Path(args.output_dir) / CONFIG_COMPARISON_FILENAME} and "
        f"{Path(args.figure_dir) / CONFIG_COMPARISON_FIGURE}. Rows: {len(comparison)}. "
        f"Selection: {selection['run_id']} -> {selection_path}"
    )


if __name__ == "__main__":
    main()
