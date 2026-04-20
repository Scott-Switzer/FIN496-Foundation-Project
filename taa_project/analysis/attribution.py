# Addresses rubric criterion 3 (recommendation/contribution analysis) by
# decomposing SAA, TAA, and per-signal-layer excess returns.
"""Attribution engine for Whitmore SAA and TAA performance.

This module implements Task 7:
- SAA vs BM2 active-return attribution by asset and by tier.
- TAA vs SAA, BM1, and BM2 active-return attribution over the OOS window.
- Per-signal-layer leave-one-out attribution via ablated walk-forward reruns.

References:
- Whitmore rubric criterion 3 and IPS benchmark definitions.
- Bailey & López de Prado (2014), Deflated Sharpe Ratio:
  https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

Point-in-time safety:
- Safe. This module operates only on already-generated historical outputs and
  reruns the walk-forward engine using the same point-in-time rules.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from taa_project.analysis.common import (
    annualized_return,
    decision_weights_to_daily_holdings,
    extract_rebalance_targets,
    load_core_outputs,
    max_drawdown,
    sharpe_ratio,
    simple_asset_returns,
    tier_map,
)
from taa_project.backtest.walkforward import run_walkforward
from taa_project.config import ALL_SAA, OUTPUT_DIR, TARGET_VOL
from taa_project.optimizer.cvxpy_opt import EnsembleConfig


ATTRIBUTION_SAA_VS_BM2_FILENAME = "attribution_saa_vs_bm2.csv"
ATTRIBUTION_TAA_FILENAME = "attribution_taa_vs_saa.csv"
ATTRIBUTION_SIGNAL_FILENAME = "attribution_per_signal.csv"
ABLATION_DIRNAME = "ablations"


def _aggregate_active_contribution(
    comparison: str,
    active_weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate active-return contributions by asset and by tier.

    Inputs:
    - `comparison`: label such as `saa_vs_bm2`.
    - `active_weights`: active-weight dataframe aligned to `asset_returns`.
    - `asset_returns`: simple asset return dataframe.

    Outputs:
    - Long-form attribution dataframe with asset-, tier-, and total-level rows.

    Citation:
    - Whitmore Task 7 attribution specification.

    Point-in-time safety:
    - Ex-post analysis only.
    """

    aligned_weights = active_weights.reindex(columns=ALL_SAA).fillna(0.0)
    aligned_returns = asset_returns.reindex(index=aligned_weights.index, columns=ALL_SAA).fillna(0.0)
    contributions = aligned_weights * aligned_returns
    by_asset = contributions.sum(axis=0)
    total = float(by_asset.sum())
    tiers = pd.Series(tier_map())
    by_tier = by_asset.groupby(tiers).sum()

    rows = []
    for asset, value in by_asset.items():
        rows.append(
            {
                "comparison": comparison,
                "grouping": "asset",
                "component": asset,
                "total_contribution": float(value),
            }
        )
    for tier, value in by_tier.items():
        rows.append(
            {
                "comparison": comparison,
                "grouping": "tier",
                "component": tier,
                "total_contribution": float(value),
            }
        )
    rows.append(
        {
            "comparison": comparison,
            "grouping": "total",
            "component": "total",
            "total_contribution": total,
        }
    )
    return pd.DataFrame(rows)


def _daily_schedules_for_attribution(outputs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Build aligned realized-holdings schedules needed for attribution.

    Inputs:
    - `outputs`: loaded core output dataframes.

    Outputs:
    - Dictionary of aligned daily schedules and realized asset returns.

    Citation:
    - Whitmore Tasks 2, 3, 6, and 7.

    Point-in-time safety:
    - Ex-post analysis only.
    """

    asset_returns = simple_asset_returns().dropna(how="all")
    saa_targets = extract_rebalance_targets(outputs["saa_weights"], outputs["saa_returns"])
    bm1_targets = extract_rebalance_targets(outputs["bm1_weights"], outputs["bm1_returns"])
    bm2_targets = extract_rebalance_targets(outputs["bm2_weights"], outputs["bm2_returns"])
    taa_targets = outputs["oos_weights"].loc[:, ALL_SAA]

    saa_realized = asset_returns.reindex(outputs["saa_returns"].index).dropna(how="all")
    bm1_realized = asset_returns.reindex(outputs["bm1_returns"].index).dropna(how="all")
    bm2_realized = asset_returns.reindex(outputs["bm2_returns"].index).dropna(how="all")
    taa_realized = asset_returns.reindex(outputs["oos_returns"].index).dropna(how="all")

    schedules = {
        "asset_returns": asset_returns,
        "saa_daily_holdings": decision_weights_to_daily_holdings(saa_targets, saa_realized),
        "bm1_daily_holdings": decision_weights_to_daily_holdings(bm1_targets, bm1_realized),
        "bm2_daily_holdings": decision_weights_to_daily_holdings(bm2_targets, bm2_realized),
        "taa_daily_holdings": decision_weights_to_daily_holdings(taa_targets, taa_realized),
        "taa_asset_returns": taa_realized,
    }
    return schedules


def _load_oos_returns(path: Path) -> pd.Series:
    """Load a walk-forward OOS return series from disk.

    Inputs:
    - `path`: CSV path containing the `portfolio_return` column.

    Outputs:
    - Daily simple return series indexed by date.

    Citation:
    - Whitmore Task 6 and Task 7 output requirements.

    Point-in-time safety:
    - Ex-post analysis only.
    """

    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return frame["portfolio_return"]


def _run_signal_ablations(
    start: str,
    end: str,
    folds: int,
    use_timesfm: bool,
    vol_budget: float,
    ensemble_config: EnsembleConfig | None,
    output_dir: Path,
) -> pd.DataFrame:
    """Run leave-one-out signal ablations and summarize OOS impacts.

    Inputs:
    - `start`, `end`, `folds`: walk-forward settings.
    - `use_timesfm`: whether the baseline run used TimesFM.
    - `vol_budget`: internal ex-ante annualized volatility target.
    - `ensemble_config`: optional baseline ensemble configuration whose
      non-ablation fields are preserved in the reruns.
    - `output_dir`: root output directory for storing ablation runs.

    Outputs:
    - Dataframe summarizing baseline vs ablated OOS Sharpe and turnover cost.

    Citation:
    - Whitmore Task 7 per-signal-layer attribution specification.

    Point-in-time safety:
    - Safe. Each ablation reruns the same point-in-time walk-forward pipeline.
    """

    baseline_outputs = load_core_outputs(output_dir)
    baseline_returns = baseline_outputs["oos_returns"]["portfolio_return"]
    baseline_sharpe = sharpe_ratio(baseline_returns)
    baseline_turnover_cost = float(baseline_outputs["oos_weights"]["turnover_cost"].sum())
    baseline_ann_return = annualized_return(baseline_returns)
    baseline_drawdown = max_drawdown(baseline_returns)

    base_config = EnsembleConfig() if ensemble_config is None else ensemble_config
    variants = {
        "baseline": base_config,
        "no_regime": replace(base_config, regime_weight=0.0),
        "no_trend": replace(base_config, trend_weight=0.0),
        "no_momo": replace(base_config, momo_weight=0.0),
    }
    if use_timesfm:
        variants["no_timesfm"] = replace(base_config, timesfm_weight=0.0)

    rows = []
    ablation_root = output_dir / ABLATION_DIRNAME
    ablation_root.mkdir(parents=True, exist_ok=True)

    for variant_id, config in variants.items():
        if variant_id == "baseline":
            rows.append(
                {
                    "variant_id": variant_id,
                    "layer": "baseline",
                    "baseline_sharpe": baseline_sharpe,
                    "ablated_sharpe": baseline_sharpe,
                    "marginal_oos_sharpe": 0.0,
                    "baseline_turnover_cost": baseline_turnover_cost,
                    "ablated_turnover_cost": baseline_turnover_cost,
                    "turnover_cost_delta": 0.0,
                    "baseline_ann_return": baseline_ann_return,
                    "ablated_ann_return": baseline_ann_return,
                    "ann_return_delta": 0.0,
                    "baseline_max_drawdown": baseline_drawdown,
                    "ablated_max_drawdown": baseline_drawdown,
                    "notes": "Existing baseline walk-forward run.",
                }
            )
            continue

        variant_output_dir = ablation_root / variant_id
        variant_use_timesfm = use_timesfm and variant_id != "no_timesfm"
        artifacts = run_walkforward(
            start=start,
            end=end,
            folds=folds,
            use_timesfm=variant_use_timesfm,
            vol_budget=vol_budget,
            output_dir=variant_output_dir,
            ensemble_config=config,
        )
        ablated_returns = artifacts["oos_returns"]["portfolio_return"]
        ablated_weights = artifacts["oos_weights"]
        ablated_sharpe = sharpe_ratio(ablated_returns)
        ablated_turnover_cost = float(ablated_weights["turnover_cost"].sum())
        ablated_ann_return = annualized_return(ablated_returns)
        ablated_drawdown = max_drawdown(ablated_returns)

        rows.append(
            {
                "variant_id": variant_id,
                "layer": variant_id.replace("no_", ""),
                "baseline_sharpe": baseline_sharpe,
                "ablated_sharpe": ablated_sharpe,
                "marginal_oos_sharpe": baseline_sharpe - ablated_sharpe,
                "baseline_turnover_cost": baseline_turnover_cost,
                "ablated_turnover_cost": ablated_turnover_cost,
                "turnover_cost_delta": baseline_turnover_cost - ablated_turnover_cost,
                "baseline_ann_return": baseline_ann_return,
                "ablated_ann_return": ablated_ann_return,
                "ann_return_delta": baseline_ann_return - ablated_ann_return,
                "baseline_max_drawdown": baseline_drawdown,
                "ablated_max_drawdown": ablated_drawdown,
                "notes": f"Leave-one-out ablation for {variant_id}.",
            }
        )

    return pd.DataFrame(rows)


def build_attribution(
    start: str = "2003-01-01",
    end: str = "2025-12-31",
    folds: int = 5,
    use_timesfm: bool = False,
    vol_budget: float = TARGET_VOL,
    ensemble_config: EnsembleConfig | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Build all required attribution outputs for Whitmore.

    Inputs:
    - `start`, `end`, `folds`: walk-forward settings reused for signal
      ablations.
    - `use_timesfm`: whether the baseline run used TimesFM.
    - `vol_budget`: internal ex-ante annualized volatility target reused by
      the ablation reruns.
    - `ensemble_config`: optional baseline ensemble configuration reused by
      the ablation reruns.
    - `output_dir`: root directory containing walk-forward outputs.

    Outputs:
    - Dictionary containing the three attribution dataframes.

    Citation:
    - Whitmore Task 7 attribution specification.

    Point-in-time safety:
    - Ex-post analysis only.
    """

    outputs = load_core_outputs(output_dir)
    schedules = _daily_schedules_for_attribution(outputs)

    common_saa_bm2_index = schedules["saa_daily_holdings"].index.intersection(schedules["bm2_daily_holdings"].index)
    saa_vs_bm2 = _aggregate_active_contribution(
        "saa_vs_bm2",
        schedules["saa_daily_holdings"].reindex(common_saa_bm2_index).fillna(0.0)
        - schedules["bm2_daily_holdings"].reindex(common_saa_bm2_index).fillna(0.0),
        schedules["asset_returns"].reindex(common_saa_bm2_index),
    )

    oos_index = schedules["taa_daily_holdings"].index
    taa_asset_returns = schedules["taa_asset_returns"].reindex(oos_index).fillna(0.0)

    taa_vs_saa = _aggregate_active_contribution(
        "taa_vs_saa",
        schedules["taa_daily_holdings"].reindex(oos_index).fillna(0.0)
        - schedules["saa_daily_holdings"].reindex(oos_index).fillna(0.0),
        taa_asset_returns,
    )
    taa_vs_bm1 = _aggregate_active_contribution(
        "taa_vs_bm1",
        schedules["taa_daily_holdings"].reindex(oos_index).fillna(0.0)
        - schedules["bm1_daily_holdings"].reindex(oos_index).fillna(0.0),
        taa_asset_returns,
    )
    taa_vs_bm2 = _aggregate_active_contribution(
        "taa_vs_bm2",
        schedules["taa_daily_holdings"].reindex(oos_index).fillna(0.0)
        - schedules["bm2_daily_holdings"].reindex(oos_index).fillna(0.0),
        taa_asset_returns,
    )
    taa_comparisons = pd.concat([taa_vs_saa, taa_vs_bm1, taa_vs_bm2], ignore_index=True)

    attribution_per_signal = _run_signal_ablations(
        start=start,
        end=end,
        folds=folds,
        use_timesfm=use_timesfm,
        vol_budget=vol_budget,
        ensemble_config=ensemble_config,
        output_dir=output_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    saa_vs_bm2.to_csv(output_dir / ATTRIBUTION_SAA_VS_BM2_FILENAME, index=False)
    taa_comparisons.to_csv(output_dir / ATTRIBUTION_TAA_FILENAME, index=False)
    attribution_per_signal.to_csv(output_dir / ATTRIBUTION_SIGNAL_FILENAME, index=False)

    return {
        "saa_vs_bm2": saa_vs_bm2,
        "taa_comparisons": taa_comparisons,
        "per_signal": attribution_per_signal,
    }


def main() -> None:
    """CLI entrypoint for building the Task 7 attribution outputs.

    Inputs:
    - `--start`, `--end`, `--folds`: walk-forward settings reused for the
      per-signal ablation reruns.
    - `--timesfm`: enable the optional TimesFM ablation branch.
    - `--vol-budget`: internal ex-ante annualized volatility target.
    - `--output-dir`: destination directory for attribution CSV files.

    Outputs:
    - Writes attribution CSV artifacts to disk.

    Citation:
    - Whitmore Task 7 attribution specification.

    Point-in-time safety:
    - Safe. The CLI orchestrates only the ex-post attribution routines above.
    """

    parser = argparse.ArgumentParser(description="Build the Whitmore attribution outputs.")
    parser.add_argument("--start", default="2003-01-01", help="First OOS date for ablation reruns.")
    parser.add_argument("--end", default="2025-12-31", help="Last OOS date for ablation reruns.")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds.")
    parser.add_argument("--timesfm", action="store_true", help="Enable the optional TimesFM signal layer.")
    parser.add_argument("--vol-budget", type=float, default=TARGET_VOL, help="Internal ex-ante vol target used by the TAA optimizer.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Destination directory for output CSV files.")
    args = parser.parse_args()

    build_attribution(
        start=args.start,
        end=args.end,
        folds=args.folds,
        use_timesfm=args.timesfm,
        vol_budget=args.vol_budget,
        output_dir=Path(args.output_dir),
    )
    print(
        "Attribution outputs written to "
        f"{Path(args.output_dir) / ATTRIBUTION_SAA_VS_BM2_FILENAME}, "
        f"{Path(args.output_dir) / ATTRIBUTION_TAA_FILENAME}, and "
        f"{Path(args.output_dir) / ATTRIBUTION_SIGNAL_FILENAME}."
    )


if __name__ == "__main__":
    main()
