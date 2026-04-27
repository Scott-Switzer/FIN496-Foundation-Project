"""Focused bridge-sweep comparison and ranking utilities for Whitmore.

References:
- Whitmore IPS return and drawdown targets in the repo.
- Bailey & López de Prado (2014), Deflated Sharpe Ratio:
  https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

Point-in-time safety:
- Safe. This module reads only already-generated out-of-sample artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from taa_project.analysis.common import deflated_sharpe_ratio, disclosed_trial_count
from taa_project.analysis.reporting import PORTFOLIO_METRICS_FILENAME
from taa_project.config import MAX_DD, OUTPUT_DIR, TRIAL_LEDGER_CSV, VOL_CEILING


BRIDGE_RUNS_ROOT_DIRNAME = "bridge_runs"
BRIDGE_OUTPUT_DIRNAME = "bridge"
BRIDGE_COMPARISON_FILENAME = "bridge_comparison.csv"
BRIDGE_RANKING_FILENAME = "bridge_ranking.csv"
BRIDGE_SELECTION_FILENAME = "bridge_selection.json"
RETURN_TARGET = 0.08

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


def _load_manifest_entries(manifest_path: Path | None, run_root: Path) -> list[dict[str, str]]:
    if manifest_path is not None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [
            {
                "run_id": str(entry["run_id"]),
                "label": str(entry.get("label", entry["run_id"])),
                "family": str(entry.get("family", "Bridge")),
            }
            for entry in manifest
        ]

    return [
        {"run_id": path.name, "label": path.name, "family": "Bridge"}
        for path in sorted(run_root.iterdir())
        if path.is_dir()
    ]


def _pass_fail(flag: bool) -> str:
    return "Y" if flag else "N"


def _strategy_dsr(run_dir: Path, n_trials: int) -> float:
    returns = pd.read_csv(run_dir / "outputs" / "oos_returns.csv", dtype=RETURNS_DTYPE)["portfolio_return"]
    return float(deflated_sharpe_ratio(returns, n_trials=n_trials))


def _ips_gap_bps(
    ann_return: float,
    ann_vol: float,
    max_dd: float,
) -> tuple[float, float, float, float, float, float]:
    mdd_headroom_bps = (max_dd + MAX_DD) * 10000.0
    return_headroom_bps = (ann_return - RETURN_TARGET) * 10000.0
    vol_headroom_bps = (VOL_CEILING - ann_vol) * 10000.0
    mdd_breach_bps = max(0.0, -mdd_headroom_bps)
    return_shortfall_bps = max(0.0, -return_headroom_bps)
    vol_excess_bps = max(0.0, -vol_headroom_bps)
    return (
        mdd_headroom_bps,
        return_headroom_bps,
        vol_headroom_bps,
        mdd_breach_bps,
        return_shortfall_bps,
        mdd_breach_bps + return_shortfall_bps + vol_excess_bps,
    )


def build_bridge_comparison(
    run_root: Path = OUTPUT_DIR / BRIDGE_RUNS_ROOT_DIRNAME,
    output_dir: Path = OUTPUT_DIR / BRIDGE_OUTPUT_DIRNAME,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    """Build the focused bridge-run comparison table from on-disk artifacts.

    Inputs:
    - `run_root`: directory containing per-bridge-run subdirectories.
    - `output_dir`: destination for `bridge_comparison.csv`.
    - `manifest_path`: optional JSON manifest that preserves labels and order.

    Outputs:
    - DataFrame with one row per bridge candidate.

    Citation:
    - Bailey & López de Prado (2014), Deflated Sharpe Ratio:
      https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

    Point-in-time safety:
    - Safe. This is an ex-post summary over already-generated OOS outputs.
    """

    entries = _load_manifest_entries(manifest_path=manifest_path, run_root=run_root)
    missing = [entry["run_id"] for entry in entries if not (run_root / entry["run_id"] / "outputs" / PORTFOLIO_METRICS_FILENAME).exists()]
    if missing:
        raise FileNotFoundError(f"Missing bridge-run outputs for: {missing}")

    n_trials = disclosed_trial_count(ledger_path=TRIAL_LEDGER_CSV)
    rows: list[dict[str, object]] = []
    for order, entry in enumerate(entries):
        run_dir = run_root / entry["run_id"]
        metrics = pd.read_csv(run_dir / "outputs" / PORTFOLIO_METRICS_FILENAME, dtype=PORTFOLIO_METRICS_DTYPE)
        strategy_row = metrics.loc[metrics["portfolio"] == "SAA+TAA"].iloc[0]

        ann_return = float(strategy_row["annualized_return"])
        ann_vol = float(strategy_row["annualized_volatility"])
        max_dd = float(strategy_row["max_drawdown"])
        (
            mdd_headroom_bps,
            return_headroom_bps,
            vol_headroom_bps,
            mdd_breach_bps,
            return_shortfall_bps,
            ips_gap_bps,
        ) = _ips_gap_bps(ann_return=ann_return, ann_vol=ann_vol, max_dd=max_dd)

        pass_mdd = max_dd >= -MAX_DD
        pass_vol = ann_vol <= VOL_CEILING
        pass_return = ann_return >= RETURN_TARGET

        rows.append(
            {
                "run_order": int(order),
                "run_id": entry["run_id"],
                "Run": entry["label"],
                "Family": entry["family"],
                "Ann. Return": ann_return,
                "Ann. Vol": ann_vol,
                "Max DD": max_dd,
                "Sharpe": float(strategy_row["sharpe_rf_2pct"]),
                "Sortino": float(strategy_row["sortino_rf_2pct"]),
                "Calmar": float(strategy_row["calmar"]),
                "Deflated Sharpe": _strategy_dsr(run_dir, n_trials=n_trials),
                "Pass MDD": _pass_fail(pass_mdd),
                "Pass Vol": _pass_fail(pass_vol),
                "Pass Return": _pass_fail(pass_return),
                "Feasible": _pass_fail(pass_mdd and pass_vol and pass_return),
                "MDD Headroom (bps)": mdd_headroom_bps,
                "Return Gap (bps)": return_headroom_bps,
                "Vol Headroom (bps)": vol_headroom_bps,
                "MDD Breach (bps)": mdd_breach_bps,
                "Return Shortfall (bps)": return_shortfall_bps,
                "IPS Gap (bps)": ips_gap_bps,
            }
        )

    comparison = pd.DataFrame(rows).sort_values("run_order").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.drop(columns=["run_order"]).to_csv(output_dir / BRIDGE_COMPARISON_FILENAME, index=False)
    return comparison.drop(columns=["run_order"])


def rank_bridge_candidates(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rank bridge candidates by IPS feasibility, gap, and DSR.

    Ranking logic:
    1. Fully feasible runs first.
    2. Then preserve the requested pass/fail ordering on MDD and return.
    3. Then minimize aggregate IPS gap in basis points.
    4. Among feasible runs, `IPS Gap (bps) == 0`, so Deflated Sharpe breaks ties.
    """

    ranked = comparison.copy()
    ranked["_feasible"] = ranked["Feasible"].eq("Y")
    ranked["_pass_mdd"] = ranked["Pass MDD"].eq("Y")
    ranked["_pass_return"] = ranked["Pass Return"].eq("Y")
    ranked["_pass_vol"] = ranked["Pass Vol"].eq("Y")
    ranked = ranked.sort_values(
        by=[
            "_feasible",
            "_pass_mdd",
            "_pass_return",
            "_pass_vol",
            "IPS Gap (bps)",
            "Deflated Sharpe",
            "Ann. Return",
            "Max DD",
            "run_id",
        ],
        ascending=[False, False, False, False, True, False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "Bridge Rank", np.arange(1, len(ranked) + 1, dtype=np.int32))
    return ranked.drop(columns=["_feasible", "_pass_mdd", "_pass_return", "_pass_vol"])


def select_bridge_candidate(ranked: pd.DataFrame) -> dict[str, object]:
    if ranked.empty:
        raise ValueError("No bridge candidates available for selection.")

    top = ranked.iloc[0]
    feasible_pool = ranked.loc[ranked["Feasible"] == "Y"]
    selection_rule = "feasible_highest_dsr" if not feasible_pool.empty else "closest_to_ips"

    return {
        "run_id": str(top["run_id"]),
        "display_name": str(top["Run"]),
        "family": str(top["Family"]),
        "selection_rule": selection_rule,
        "n_tested_configurations": int(len(ranked)),
        "n_feasible_configurations": int(len(feasible_pool)),
        "bridge_rank": int(top["Bridge Rank"]),
        "ann_return": float(top["Ann. Return"]),
        "ann_vol": float(top["Ann. Vol"]),
        "max_dd": float(top["Max DD"]),
        "deflated_sharpe": float(top["Deflated Sharpe"]),
        "pass_mdd": str(top["Pass MDD"]) == "Y",
        "pass_vol": str(top["Pass Vol"]) == "Y",
        "pass_return": str(top["Pass Return"]) == "Y",
        "feasible": str(top["Feasible"]) == "Y",
        "mdd_headroom_bps": float(top["MDD Headroom (bps)"]),
        "return_gap_bps": float(top["Return Gap (bps)"]),
        "ips_gap_bps": float(top["IPS Gap (bps)"]),
    }


def write_bridge_selection(selection: dict[str, object], output_dir: Path = OUTPUT_DIR / BRIDGE_OUTPUT_DIRNAME) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / BRIDGE_SELECTION_FILENAME
    path.write_text(json.dumps(selection, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and rank the focused bridge sweep.")
    parser.add_argument("--run-root", default=str(OUTPUT_DIR / BRIDGE_RUNS_ROOT_DIRNAME))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / BRIDGE_OUTPUT_DIRNAME))
    parser.add_argument("--manifest-path", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path) if args.manifest_path else None
    comparison = build_bridge_comparison(
        run_root=Path(args.run_root),
        output_dir=Path(args.output_dir),
        manifest_path=manifest_path,
    )
    ranked = rank_bridge_candidates(comparison)
    ranking_path = Path(args.output_dir) / BRIDGE_RANKING_FILENAME
    ranked.to_csv(ranking_path, index=False)
    selection_path = write_bridge_selection(select_bridge_candidate(ranked), output_dir=Path(args.output_dir))
    print(
        f"Wrote {Path(args.output_dir) / BRIDGE_COMPARISON_FILENAME}, "
        f"{ranking_path}, and {selection_path}."
    )


if __name__ == "__main__":
    main()
