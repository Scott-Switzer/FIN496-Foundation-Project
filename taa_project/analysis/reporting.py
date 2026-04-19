# Addresses rubric criteria 1-4 and Tasks 8-9 by generating metrics, figures,
# IPS audits, trial disclosure artifacts, and diagnostics-ready summary tables.
"""Reporting and diagnostics builders for the Whitmore project.

This module implements Task 8 directly and provides shared artifacts for
Tasks 9-13:
- Metrics for BM1, BM2, SAA, and SAA+TAA.
- IPS compliance audit for the strategy target schedules.
- High-DPI figures used in the report PDF and slide deck.
- SAA method comparison across the five permitted construction methods.
- Trial ledger and Deflated Sharpe Ratio summary.

References:
- Whitmore IPS / Guidelines in the repo.
- Faber (2007): https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf
- Antonacci / ADM summary:
  https://allocatesmartly.com/taa-strategy-accelerating-dual-momentum/
- Bailey & López de Prado (2014):
  https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

Point-in-time safety:
- Safe. This module operates only on already-generated causal outputs or on
  expanding-window SAA comparisons estimated strictly from information known at
  each annual rebalance date.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.optimize import Bounds, minimize

from taa_project.analysis.attribution import build_attribution
from taa_project.analysis.common import (
    RISK_FREE_RATE,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cost_drag_per_year,
    cumulative_growth_index,
    decision_weights_to_daily_holdings,
    decision_weights_to_daily_target,
    deflated_sharpe_ratio,
    drawdown_curve,
    extract_rebalance_targets,
    hit_rate,
    historical_var_95,
    load_core_outputs,
    max_drawdown,
    rolling_annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    simple_asset_returns,
    tier_map,
    tier_weight_frame,
    turnover_per_year,
)
from taa_project.config import (
    ALL_SAA,
    CORE,
    CORE_FLOOR,
    DEFAULT_RANDOM_SEED,
    FIGURES_DIR,
    MPLCONFIG_DIR,
    NONTRAD,
    NONTRAD_CAP,
    OUTPUT_DIR,
    REPORT_DIR,
    SATELLITE,
    SATELLITE_CAP,
    SINGLE_SLEEVE_MAX,
    TARGET_VOL,
    TRIAL_LEDGER_CSV,
    VOL_CEILING,
)
from taa_project.data_loader import log_returns
from taa_project.optimizer.cvxpy_opt import EnsembleConfig
from taa_project.saa.build_saa import (
    DIAGONAL_FLOOR as SAA_DIAGONAL_FLOOR,
    DEFAULT_END as SAA_DEFAULT_END,
    DEFAULT_START as SAA_DEFAULT_START,
    SAAOptimizationInputs,
    available_assets_on,
    bounds_for_assets,
    build_linear_constraints,
    build_rebalance_schedule as build_saa_rebalance_schedule,
    compute_target_weights,
    estimate_covariance,
    first_valid_dates,
    load_saa_prices,
    project_policy_targets_to_feasible_set,
    project_weights_to_feasible_set,
    simulate_portfolio,
    solve_target_risk_parity,
    target_risk_budgets,
)


PORTFOLIO_METRICS_FILENAME = "portfolio_metrics.csv"
REGIME_ALLOCATION_FILENAME = "regime_allocation_summary.csv"
IPS_COMPLIANCE_FILENAME = "ips_compliance.csv"
DSR_SUMMARY_FILENAME = "dsr_summary.csv"
SAA_METHOD_COMPARISON_FILENAME = "saa_method_comparison.csv"
PER_FOLD_METRICS_FILENAME = "per_fold_metrics.csv"

FIGURE_FILENAMES = {
    "cumgrowth": "fig01_cumgrowth.png",
    "drawdown": "fig02_drawdown.png",
    "rolling_vol": "fig03_rolling_vol.png",
    "taa_weights": "fig04_taa_weights_stacked.png",
    "regime_shading": "fig05_regime_shading.png",
    "oos_folds": "fig06_oos_folds.png",
    "attribution": "fig07_attribution_bar.png",
}

SAA_METHODS = (
    "inverse_vol",
    "minimum_variance",
    "risk_parity",
    "maximum_diversification",
    "mean_variance",
)

TAA_BASELINE_VARIANT_ID = "taa_baseline"


def _annualized_expected_returns(
    returns: pd.DataFrame,
    as_of_date: pd.Timestamp,
    assets: list[str],
    lookback_days: int = 252,
) -> pd.Series:
    """Estimate annualized expected returns for mean-variance comparison.

    Inputs:
    - `returns`: audited log-return dataframe.
    - `as_of_date`: annual SAA rebalance date.
    - `assets`: eligible SAA sleeves.
    - `lookback_days`: trailing history length for the mean estimate.

    Outputs:
    - Annualized expected-return vector indexed by `assets`.

    Citation:
    - Whitmore Task 9 method-comparison requirement.

    Point-in-time safety:
    - Safe. The estimate uses only returns dated on or before `as_of_date`.
    """

    history = returns.loc[:as_of_date, assets].tail(lookback_days)
    return history.mean(skipna=True).fillna(0.0) * 252.0


def _solve_saa_objective(
    covariance: pd.DataFrame,
    assets: list[str],
    objective,
) -> pd.Series:
    """Solve a constrained annual SAA portfolio under a custom objective.

    Inputs:
    - `covariance`: annualized covariance matrix on the eligible universe.
    - `assets`: eligible SAA sleeves.
    - `objective`: callable from weight vector to scalar objective value.

    Outputs:
    - Feasible weight vector indexed by `assets`.

    Citation:
    - SciPy SLSQP documentation:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Point-in-time safety:
    - Safe. The caller is responsible for passing a covariance matrix built
      strictly from information available at the rebalance date.
    """

    lower_bounds, upper_bounds = bounds_for_assets(assets)
    initial = project_policy_targets_to_feasible_set(lower_bounds, upper_bounds, assets)
    bounds = Bounds(lower_bounds.values, upper_bounds.values)
    sum_to_one, inequalities = build_linear_constraints(assets)

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_to_one, *inequalities],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    solution = result.x if result.success else initial
    projected = project_weights_to_feasible_set(solution, lower_bounds, upper_bounds, assets)
    return pd.Series(projected, index=assets, dtype=float)


def _inverse_vol_weights(covariance: pd.DataFrame, assets: list[str]) -> pd.Series:
    """Build an IPS-feasible inverse-volatility SAA portfolio.

    Inputs:
    - `covariance`: annualized covariance matrix on the eligible universe.
    - `assets`: eligible SAA sleeves.

    Outputs:
    - Feasible inverse-volatility target weights.

    Citation:
    - Whitmore Task 9 SAA method comparison list.

    Point-in-time safety:
    - Safe. The method uses only covariance estimated at the rebalance date.
    """

    sigma = np.sqrt(np.maximum(np.diag(covariance.to_numpy(dtype=float)), SAA_DIAGONAL_FLOOR))
    raw = pd.Series(1.0 / sigma, index=assets, dtype=float)
    lower_bounds, upper_bounds = bounds_for_assets(assets)
    projected = project_weights_to_feasible_set(raw.values, lower_bounds, upper_bounds, assets)
    return pd.Series(projected, index=assets, dtype=float)


def _minimum_variance_weights(covariance: pd.DataFrame, assets: list[str]) -> pd.Series:
    """Build an IPS-feasible minimum-variance SAA portfolio.

    Inputs:
    - `covariance`: annualized covariance matrix on the eligible universe.
    - `assets`: eligible SAA sleeves.

    Outputs:
    - Feasible minimum-variance target weights.

    Citation:
    - Whitmore Task 9 SAA method comparison list.

    Point-in-time safety:
    - Safe. The method uses only covariance estimated at the rebalance date.
    """

    matrix = covariance.loc[assets, assets].to_numpy(dtype=float)
    return _solve_saa_objective(covariance, assets, lambda weights: float(weights @ matrix @ weights))


def _maximum_diversification_weights(covariance: pd.DataFrame, assets: list[str]) -> pd.Series:
    """Build an IPS-feasible maximum-diversification SAA portfolio.

    Inputs:
    - `covariance`: annualized covariance matrix on the eligible universe.
    - `assets`: eligible SAA sleeves.

    Outputs:
    - Feasible maximum-diversification target weights.

    Citation:
    - Whitmore Task 9 SAA method comparison list.

    Point-in-time safety:
    - Safe. The method uses only covariance estimated at the rebalance date.
    """

    matrix = covariance.loc[assets, assets].to_numpy(dtype=float)
    sigma = np.sqrt(np.maximum(np.diag(matrix), SAA_DIAGONAL_FLOOR))

    def objective(weights: np.ndarray) -> float:
        numerator = float(weights @ sigma)
        denominator = float(np.sqrt(max(weights @ matrix @ weights, SAA_DIAGONAL_FLOOR)))
        return -numerator / denominator

    return _solve_saa_objective(covariance, assets, objective)


def _mean_variance_weights(
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    assets: list[str],
    risk_aversion: float = 4.0,
) -> pd.Series:
    """Build an IPS-feasible mean-variance SAA portfolio.

    Inputs:
    - `covariance`: annualized covariance matrix on the eligible universe.
    - `expected_returns`: annualized expected-return vector.
    - `assets`: eligible SAA sleeves.
    - `risk_aversion`: quadratic risk penalty.

    Outputs:
    - Feasible mean-variance target weights.

    Citation:
    - Whitmore Task 9 SAA method comparison list.

    Point-in-time safety:
    - Safe. The expected-return estimate is supplied by the caller from data
      dated on or before the rebalance date.
    """

    matrix = covariance.loc[assets, assets].to_numpy(dtype=float)
    mu = expected_returns.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    return _solve_saa_objective(
        covariance,
        assets,
        lambda weights: float(0.5 * risk_aversion * (weights @ matrix @ weights) - weights @ mu),
    )


def _candidate_saa_weights(
    method: str,
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    assets: list[str],
) -> pd.Series:
    """Compute one candidate SAA method's target weights.

    Inputs:
    - `method`: one of the five permitted SAA comparison methods.
    - `covariance`: annualized covariance matrix on the eligible universe.
    - `expected_returns`: annualized expected-return vector for mean-variance.
    - `assets`: eligible SAA sleeves.

    Outputs:
    - Feasible target weights indexed by `assets`.

    Citation:
    - Whitmore Task 2 and Task 9 SAA method requirements.

    Point-in-time safety:
    - Safe. The method uses only inputs estimated from data available at the
      rebalance date.
    """

    if method == "inverse_vol":
        return _inverse_vol_weights(covariance, assets)
    if method == "minimum_variance":
        return _minimum_variance_weights(covariance, assets)
    if method == "risk_parity":
        return solve_target_risk_parity(
            SAAOptimizationInputs(
                covariance=covariance,
                lower_bounds=bounds_for_assets(assets)[0],
                upper_bounds=bounds_for_assets(assets)[1],
                risk_budgets=target_risk_budgets(assets),
                assets=assets,
            )
        )
    if method == "maximum_diversification":
        return _maximum_diversification_weights(covariance, assets)
    if method == "mean_variance":
        return _mean_variance_weights(covariance, expected_returns, assets)
    raise ValueError(f"Unsupported SAA method: {method}")


def build_saa_method_comparison(
    start_date: str = SAA_DEFAULT_START,
    end_date: str = SAA_DEFAULT_END,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compare the five permitted SAA methods under the Whitmore IPS.

    Inputs:
    - `start_date`: earliest allowed SAA start date.
    - `end_date`: last date to include.
    - `output_dir`: destination directory for the comparison CSV.

    Outputs:
    - Tuple `(comparison_df, returns_by_method)` where `comparison_df`
      summarizes the metrics and `returns_by_method` stores each method's daily
      return series for diagnostics.

    Citation:
    - Whitmore Task 9 diagnostics requirement.

    Point-in-time safety:
    - Safe. Every rebalance date uses only covariance and expected-return
      estimates based on data observed on or before that date.
    """

    prices = load_saa_prices()
    returns = log_returns(prices)
    inception_dates = first_valid_dates(prices)
    schedule = build_saa_rebalance_schedule(prices, start_date, end_date)

    method_returns: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []

    for method in SAA_METHODS:
        rebalance_targets: dict[pd.Timestamp, pd.Series] = {}
        for rebalance_date in schedule:
            eligible_assets = available_assets_on(rebalance_date, inception_dates)
            observed_assets = [asset for asset in eligible_assets if pd.notna(prices.loc[rebalance_date, asset])]
            covariance = estimate_covariance(returns, rebalance_date, observed_assets)
            expected_returns = _annualized_expected_returns(returns, rebalance_date, observed_assets)
            target = _candidate_saa_weights(method, covariance, expected_returns, observed_assets)

            full_weights = pd.Series(0.0, index=ALL_SAA, dtype=float)
            full_weights.loc[observed_assets] = target.reindex(observed_assets).fillna(0.0)
            rebalance_targets[rebalance_date] = full_weights

        weights_df, returns_df = simulate_portfolio(
            returns=returns,
            rebalance_targets=rebalance_targets,
            start_date=schedule[0],
            end_date=pd.Timestamp(end_date),
        )
        method_returns[method] = returns_df
        rows.append(
            {
                "method": method,
                "annualized_return": annualized_return(returns_df["portfolio_return"]),
                "annualized_volatility": annualized_volatility(returns_df["portfolio_return"]),
                "max_drawdown": max_drawdown(returns_df["portfolio_return"]),
                "sharpe": sharpe_ratio(returns_df["portfolio_return"]),
                "sortino": sortino_ratio(returns_df["portfolio_return"], mar=RISK_FREE_RATE),
                "calmar": calmar_ratio(returns_df["portfolio_return"]),
                "turnover_pa": turnover_per_year(float(returns_df["turnover"].sum()), returns_df.index),
                "cost_drag_pa": cost_drag_per_year(float(returns_df["turnover_cost"].sum()), returns_df.index),
                "start_date": str(returns_df.index.min().date()),
                "end_date": str(returns_df.index.max().date()),
            }
        )

    comparison_df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_dir / SAA_METHOD_COMPARISON_FILENAME, index=False)
    return comparison_df, method_returns


def _load_attribution_outputs(
    start: str,
    end: str,
    folds: int,
    use_timesfm: bool,
    vol_budget: float,
    ensemble_config: EnsembleConfig | None,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load attribution outputs, building them first when missing.

    Inputs:
    - `start`, `end`, `folds`, `use_timesfm`: attribution rerun settings.
    - `vol_budget`: internal ex-ante annualized volatility target reused by
      attribution reruns when outputs are missing.
    - `ensemble_config`: optional baseline ensemble configuration reused by
      attribution reruns when outputs are missing.
    - `output_dir`: directory containing attribution artifacts.

    Outputs:
    - Dictionary of attribution dataframes.

    Citation:
    - Whitmore Tasks 7 and 8 output dependencies.

    Point-in-time safety:
    - Ex-post orchestration only.
    """

    required_paths = {
        "saa_vs_bm2": output_dir / "attribution_saa_vs_bm2.csv",
        "taa_comparisons": output_dir / "attribution_taa_vs_saa.csv",
        "per_signal": output_dir / "attribution_per_signal.csv",
    }
    if not all(path.exists() for path in required_paths.values()):
        build_attribution(
            start=start,
            end=end,
            folds=folds,
            use_timesfm=use_timesfm,
            vol_budget=vol_budget,
            ensemble_config=ensemble_config,
            output_dir=output_dir,
        )

    return {
        "saa_vs_bm2": pd.read_csv(required_paths["saa_vs_bm2"]),
        "taa_comparisons": pd.read_csv(required_paths["taa_comparisons"]),
        "per_signal": pd.read_csv(required_paths["per_signal"]),
    }


def _build_strategy_panels(
    outputs: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, object]:
    """Reconstruct daily schedules and return panels used in reporting.

    Inputs:
    - `outputs`: loaded core output dataframes.
    - `output_dir`: directory containing the fold metadata CSV.

    Outputs:
    - Dictionary of daily returns, target schedules, holdings, and common dates.

    Citation:
    - Whitmore Tasks 2, 3, 6, and 8.

    Point-in-time safety:
    - Ex-post analysis only.
    """

    asset_returns = simple_asset_returns().dropna(how="all")

    saa_targets = extract_rebalance_targets(outputs["saa_weights"], outputs["saa_returns"])
    bm1_targets = extract_rebalance_targets(outputs["bm1_weights"], outputs["bm1_returns"])
    bm2_targets = extract_rebalance_targets(outputs["bm2_weights"], outputs["bm2_returns"])
    taa_targets = outputs["oos_weights"].loc[:, ALL_SAA]

    saa_realized_asset = asset_returns.reindex(outputs["saa_returns"].index).dropna(how="all")
    bm1_realized_asset = asset_returns.reindex(outputs["bm1_returns"].index).dropna(how="all")
    bm2_realized_asset = asset_returns.reindex(outputs["bm2_returns"].index).dropna(how="all")
    taa_realized_asset = asset_returns.reindex(outputs["oos_returns"].index).dropna(how="all")

    saa_target_daily = outputs["saa_weights"].loc[:, ALL_SAA].astype(float)
    bm1_target_daily = outputs["bm1_weights"].loc[:, ALL_SAA].astype(float)
    bm2_target_daily = outputs["bm2_weights"].loc[:, ALL_SAA].astype(float)
    taa_target_daily = decision_weights_to_daily_target(taa_targets, taa_realized_asset.index)

    panels = {
        "asset_returns": asset_returns,
        "returns": {
            "BM1": outputs["bm1_returns"]["portfolio_return"],
            "BM2": outputs["bm2_returns"]["portfolio_return"],
            "SAA": outputs["saa_returns"]["portfolio_return"],
            "SAA+TAA": outputs["oos_returns"]["portfolio_return"],
        },
        "gross_returns": {
            "BM1": outputs["bm1_returns"]["gross_return"],
            "BM2": outputs["bm2_returns"]["gross_return"],
            "SAA": outputs["saa_returns"]["gross_return"],
            "SAA+TAA": outputs["oos_returns"]["gross_return"],
        },
        "turnover": {
            "BM1": outputs["bm1_returns"]["turnover"],
            "BM2": outputs["bm2_returns"]["turnover"],
            "SAA": outputs["saa_returns"]["turnover"],
            "SAA+TAA": outputs["oos_weights"]["turnover"],
        },
        "costs": {
            "BM1": outputs["bm1_returns"]["turnover_cost"],
            "BM2": outputs["bm2_returns"]["turnover_cost"],
            "SAA": outputs["saa_returns"]["turnover_cost"],
            "SAA+TAA": outputs["oos_returns"]["turnover_cost"],
        },
        "target_weights": {
            "SAA": saa_target_daily,
            "BM1": bm1_target_daily,
            "BM2": bm2_target_daily,
            "SAA+TAA": taa_target_daily,
        },
        "holdings": {
            "SAA": decision_weights_to_daily_holdings(saa_targets, saa_realized_asset),
            "BM1": decision_weights_to_daily_holdings(bm1_targets, bm1_realized_asset),
            "BM2": decision_weights_to_daily_holdings(bm2_targets, bm2_realized_asset),
            "SAA+TAA": decision_weights_to_daily_holdings(taa_targets, taa_realized_asset),
        },
        "regimes_daily": outputs["oos_returns"]["regime"].copy(),
        "regimes_decision": outputs["oos_regimes"].copy(),
        "folds": pd.read_csv(
            output_dir / "walkforward_folds.csv",
            parse_dates=["train_start", "train_end", "embargo_start", "embargo_end", "test_start", "test_end"],
        ),
    }
    return panels


def _portfolio_metrics_table(
    panels: dict[str, object],
    trial_ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Build the metrics table covering BM1, BM2, SAA, and SAA+TAA.

    Inputs:
    - `panels`: reconstructed reporting panels.
    - `trial_ledger`: full trial ledger dataframe.

    Outputs:
    - Portfolio metrics dataframe.

    Citation:
    - Whitmore Task 8 metrics specification.

    Point-in-time safety:
    - Ex-post analysis only.
    """

    rows = []
    target_weights = panels["target_weights"]
    tier_totals = {name: tier_weight_frame(weights) for name, weights in target_weights.items()}
    taa_dsr = pd.to_numeric(
        trial_ledger.loc[trial_ledger["variant_id"] == TAA_BASELINE_VARIANT_ID, "DSR"],
        errors="coerce",
    )
    taa_dsr_value = float(taa_dsr.iloc[0]) if not taa_dsr.empty else float("nan")

    for portfolio, returns in panels["returns"].items():
        gross_returns = panels["gross_returns"][portfolio]
        turnover_series = panels["turnover"][portfolio]
        cost_series = panels["costs"][portfolio]
        tiers = tier_totals[portfolio]

        rows.append(
            {
                "portfolio": portfolio,
                "start_date": str(returns.index.min().date()),
                "end_date": str(returns.index.max().date()),
                "annualized_return": annualized_return(returns),
                "annualized_volatility": annualized_volatility(returns),
                "max_drawdown": max_drawdown(returns),
                "var_95_historical": historical_var_95(returns),
                "sharpe_rf_2pct": sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE),
                "sortino_rf_2pct": sortino_ratio(returns, mar=RISK_FREE_RATE),
                "calmar": calmar_ratio(returns),
                "turnover_pa": turnover_per_year(float(turnover_series.sum()), returns.index),
                "hit_rate": hit_rate(returns),
                "cost_drag_pa": cost_drag_per_year(float(cost_series.sum()), returns.index),
                "avg_core_weight": float(tiers["Core"].mean()),
                "avg_satellite_weight": float(tiers["Satellite"].mean()),
                "avg_nontrad_weight": float(tiers["Non-Traditional"].mean()),
                "dsr": taa_dsr_value if portfolio == "SAA+TAA" else float("nan"),
                "gross_minus_net_ann_return": annualized_return(gross_returns) - annualized_return(returns),
            }
        )

    return pd.DataFrame(rows)


def _per_fold_metrics(oos_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute per-fold OOS metrics for the walk-forward study.

    Inputs:
    - `oos_returns`: Task 6 OOS return dataframe with `fold_id`.

    Outputs:
    - Per-fold metrics dataframe.

    Citation:
    - Whitmore Task 6 and Task 10 walk-forward validation requirements.

    Point-in-time safety:
    - Ex-post analysis only.
    """

    rows = []
    for fold_id, block in oos_returns.groupby("fold_id"):
        returns = block["portfolio_return"]
        rows.append(
            {
                "fold_id": int(fold_id),
                "start_date": str(block.index.min().date()),
                "end_date": str(block.index.max().date()),
                "days": int(len(block)),
                "annualized_return": annualized_return(returns),
                "annualized_volatility": annualized_volatility(returns),
                "sharpe": sharpe_ratio(returns),
                "sortino": sortino_ratio(returns, mar=RISK_FREE_RATE),
                "max_drawdown": max_drawdown(returns),
                "turnover_cost": float(block["turnover_cost"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _regime_allocation_summary(panels: dict[str, object]) -> pd.DataFrame:
    """Summarize average TAA allocations by inferred regime.

    Inputs:
    - `panels`: reconstructed reporting panels.

    Outputs:
    - Long-form dataframe of average TAA allocations by regime and component.

    Citation:
    - Whitmore Task 8 average-regime-allocation requirement.

    Point-in-time safety:
    - Ex-post analysis only.
    """

    taa_holdings = panels["holdings"]["SAA+TAA"].reindex(panels["regimes_daily"].index).fillna(0.0)
    regime_series = panels["regimes_daily"].reindex(taa_holdings.index)
    tiered = tier_weight_frame(taa_holdings)

    rows = []
    for regime, index in regime_series.groupby(regime_series).groups.items():
        if pd.isna(regime):
            continue
        block = taa_holdings.loc[index]
        tier_block = tiered.loc[index]
        for asset in ALL_SAA:
            rows.append(
                {
                    "regime": regime,
                    "grouping": "asset",
                    "component": asset,
                    "avg_weight": float(block[asset].mean()),
                    "days": int(len(block)),
                }
            )
        for tier in ["Core", "Satellite", "Non-Traditional"]:
            rows.append(
                {
                    "regime": regime,
                    "grouping": "tier",
                    "component": tier,
                    "avg_weight": float(tier_block[tier].mean()),
                    "days": int(len(tier_block)),
                }
            )
    return pd.DataFrame(rows)


def _ips_compliance_rows(portfolio: str, weights: pd.DataFrame) -> list[dict[str, object]]:
    """Audit one strategy's daily target schedule against the IPS constraints.

    Inputs:
    - `portfolio`: strategy label.
    - `weights`: daily target-weight dataframe containing `ALL_SAA` columns.

    Outputs:
    - List of violation rows; empty when the schedule is fully compliant.

    Citation:
    - Whitmore IPS `Guidelines.md` hard constraints.

    Point-in-time safety:
    - Ex-post audit only.
    """

    rows: list[dict[str, object]] = []
    aligned = weights.reindex(columns=ALL_SAA).fillna(0.0)
    tiered = tier_weight_frame(aligned)

    for date, row in aligned.iterrows():
        total_weight = float(row.sum())
        min_weight = float(row.min())
        max_weight = float(row.max())
        core_weight = float(tiered.loc[date, "Core"])
        satellite_weight = float(tiered.loc[date, "Satellite"])
        nontrad_weight = float(tiered.loc[date, "Non-Traditional"])

        checks = [
            ("sum_to_one", abs(total_weight - 1.0) <= 1e-8, total_weight, 1.0),
            ("no_shorts", min_weight >= -1e-8, min_weight, 0.0),
            ("core_floor", core_weight >= CORE_FLOOR - 1e-8, core_weight, CORE_FLOOR),
            ("satellite_cap", satellite_weight <= SATELLITE_CAP + 1e-8, satellite_weight, SATELLITE_CAP),
            ("nontrad_cap", nontrad_weight <= NONTRAD_CAP + 1e-8, nontrad_weight, NONTRAD_CAP),
            ("single_sleeve_cap", max_weight <= SINGLE_SLEEVE_MAX + 1e-8, max_weight, SINGLE_SLEEVE_MAX),
        ]
        for rule, passed, value, bound in checks:
            if not passed:
                rows.append(
                    {
                        "portfolio": portfolio,
                        "date": date,
                        "rule": rule,
                        "value": value,
                        "bound": bound,
                    }
                )
    return rows


def _build_trial_ledger(
    output_dir: Path,
    use_timesfm: bool,
    folds: int,
    per_signal: pd.DataFrame,
    saa_method_comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the required trial ledger and DSR summary tables.

    Inputs:
    - `output_dir`: output directory containing baseline and ablation runs.
    - `use_timesfm`: whether the baseline run used TimesFM.
    - `folds`: number of walk-forward folds.
    - `per_signal`: Task 7 per-signal attribution dataframe.
    - `saa_method_comparison`: SAA method comparison dataframe.

    Outputs:
    - Tuple `(trial_ledger_df, dsr_summary_df)`.

    Citation:
    - Bailey & López de Prado (2014):
      https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
    - Whitmore documentation requirement for `TRIAL_LEDGER.csv`.

    Point-in-time safety:
    - Ex-post disclosure only.
    """

    timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    taa_returns: dict[str, pd.Series] = {}

    taa_variants = [
        ("baseline", output_dir / "oos_returns.csv", "Baseline signal ensemble."),
        ("no_regime", output_dir / "ablations" / "no_regime" / "oos_returns.csv", "Leave-one-out regime ablation."),
        ("no_trend", output_dir / "ablations" / "no_trend" / "oos_returns.csv", "Leave-one-out trend ablation."),
        ("no_momo", output_dir / "ablations" / "no_momo" / "oos_returns.csv", "Leave-one-out ADM ablation."),
    ]
    if use_timesfm:
        taa_variants.append(
            (
                "no_timesfm",
                output_dir / "ablations" / "no_timesfm" / "oos_returns.csv",
                "Leave-one-out TimesFM ablation.",
            )
        )

    for variant_id, path, notes in taa_variants:
        returns = pd.read_csv(path, parse_dates=["date"]).set_index("date")["portfolio_return"]
        ledger_variant_id = TAA_BASELINE_VARIANT_ID if variant_id == "baseline" else f"taa_{variant_id}"
        taa_returns[ledger_variant_id] = returns
        rows.append(
            {
                "timestamp": timestamp,
                "variant_id": ledger_variant_id,
                "hmm_states": 3,
                "trend_window": 200,
                "momo_windows": "1|3|6|12",
                "signal_weights": json.dumps(
                    {
                        "regime_scale": 0.10,
                        "trend_scale": 0.06,
                        "momo_scale": 0.06,
                        "timesfm_scale": "mu_ann",
                    },
                    sort_keys=True,
                ),
                "ensemble_weights": json.dumps(
                    {
                        "regime": 0.40 if variant_id != "no_regime" else 0.0,
                        "trend": 0.20 if variant_id != "no_trend" else 0.0,
                        "momo": 0.20 if variant_id != "no_momo" else 0.0,
                        "timesfm": 0.20 if variant_id != "no_timesfm" else 0.0,
                    },
                    sort_keys=True,
                ),
                "cov_shrinkage": "0.70_sample_0.30_diag",
                "cv_folds": folds,
                "IS_sharpe": np.nan,
                "OOS_sharpe": sharpe_ratio(returns),
                "DSR": np.nan,
                "notes": notes if use_timesfm or variant_id != "baseline" else f"{notes} Baseline run used --no-timesfm.",
            }
        )

    n_trials = len(taa_returns)
    for row in rows:
        if str(row["variant_id"]).startswith("taa_"):
            row["DSR"] = deflated_sharpe_ratio(taa_returns[row["variant_id"]], n_trials)

    for _, row in saa_method_comparison.iterrows():
        rows.append(
            {
                "timestamp": timestamp,
                "variant_id": f"saa_{row['method']}",
                "hmm_states": np.nan,
                "trend_window": np.nan,
                "momo_windows": "",
                "signal_weights": "",
                "ensemble_weights": "",
                "cov_shrinkage": "",
                "cv_folds": np.nan,
                "IS_sharpe": np.nan,
                "OOS_sharpe": row["sharpe"],
                "DSR": np.nan,
                "notes": f"SAA method comparison row for {row['method']}.",
            }
        )

    trial_ledger = pd.DataFrame(rows)
    dsr_summary = pd.DataFrame(
        [
            {
                "timestamp": timestamp,
                "n_taa_trials": n_trials,
                "baseline_variant_id": TAA_BASELINE_VARIANT_ID,
                "baseline_sharpe": float(trial_ledger.loc[trial_ledger["variant_id"] == TAA_BASELINE_VARIANT_ID, "OOS_sharpe"].iloc[0]),
                "baseline_dsr": float(trial_ledger.loc[trial_ledger["variant_id"] == TAA_BASELINE_VARIANT_ID, "DSR"].iloc[0]),
                "timesfm_enabled": int(use_timesfm),
            }
        ]
    )
    return trial_ledger, dsr_summary


def _common_plot_window(panels: dict[str, object]) -> pd.Timestamp:
    """Return the latest common start date across all plotted portfolios.

    Inputs:
    - `panels`: reconstructed reporting panels.

    Outputs:
    - Common start date used by the multi-portfolio figures.

    Citation:
    - Whitmore Task 8 chart requirement.

    Point-in-time safety:
    - Ex-post plotting helper only.
    """

    starts = [series.dropna().index.min() for series in panels["returns"].values()]
    return max(pd.Timestamp(value) for value in starts)


def _save_cumgrowth_figure(panels: dict[str, object], figure_dir: Path) -> Path:
    """Save the cumulative-growth chart covering all four portfolios.

    Inputs:
    - `panels`: reconstructed reporting panels.
    - `figure_dir`: destination figure directory.

    Outputs:
    - Path to the saved PNG.

    Citation:
    - Whitmore Task 8 `fig01_cumgrowth.png` requirement.

    Point-in-time safety:
    - Ex-post plotting only.
    """

    common_start = _common_plot_window(panels)
    plt.figure(figsize=(10.5, 6.0))
    for label, returns in panels["returns"].items():
        growth = cumulative_growth_index(returns.loc[common_start:])
        plt.plot(growth.index, growth.values, label=label, linewidth=2.0)
    plt.title("Cumulative Growth (Index 100)")
    plt.ylabel("Index Level")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.25)
    path = figure_dir / FIGURE_FILENAMES["cumgrowth"]
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _save_drawdown_figure(panels: dict[str, object], figure_dir: Path) -> Path:
    """Save the underwater chart covering all four portfolios.

    Inputs:
    - `panels`: reconstructed reporting panels.
    - `figure_dir`: destination figure directory.

    Outputs:
    - Path to the saved PNG.

    Citation:
    - Whitmore Task 8 `fig02_drawdown.png` requirement.

    Point-in-time safety:
    - Ex-post plotting only.
    """

    common_start = _common_plot_window(panels)
    plt.figure(figsize=(10.5, 6.0))
    for label, returns in panels["returns"].items():
        drawdown = drawdown_curve(returns.loc[common_start:])
        plt.plot(drawdown.index, drawdown.values, label=label, linewidth=2.0)
    plt.title("Underwater Curves")
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.25)
    path = figure_dir / FIGURE_FILENAMES["drawdown"]
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _save_rolling_vol_figure(panels: dict[str, object], figure_dir: Path) -> Path:
    """Save the 12-month rolling annualized volatility chart.

    Inputs:
    - `panels`: reconstructed reporting panels.
    - `figure_dir`: destination figure directory.

    Outputs:
    - Path to the saved PNG.

    Citation:
    - Whitmore Task 8 `fig03_rolling_vol.png` requirement.

    Point-in-time safety:
    - Ex-post plotting only.
    """

    common_start = _common_plot_window(panels)
    plt.figure(figsize=(10.5, 6.0))
    for label, returns in panels["returns"].items():
        rolling_vol = rolling_annualized_volatility(returns.loc[common_start:])
        plt.plot(rolling_vol.index, rolling_vol.values, label=label, linewidth=2.0)
    plt.axhline(VOL_CEILING, color="black", linestyle="--", linewidth=1.5, label="15% IPS Ceiling")
    plt.axhline(TARGET_VOL, color="gray", linestyle=":", linewidth=1.5, label="10% Internal Buffer")
    plt.title("12M Rolling Annualized Volatility")
    plt.ylabel("Annualized Volatility")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(alpha=0.25)
    path = figure_dir / FIGURE_FILENAMES["rolling_vol"]
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _save_taa_weights_figure(panels: dict[str, object], figure_dir: Path) -> Path:
    """Save the stacked TAA weights chart colored by tier.

    Inputs:
    - `panels`: reconstructed reporting panels.
    - `figure_dir`: destination figure directory.

    Outputs:
    - Path to the saved PNG.

    Citation:
    - Whitmore Task 8 `fig04_taa_weights_stacked.png` requirement.

    Point-in-time safety:
    - Ex-post plotting only.
    """

    weights = panels["target_weights"]["SAA+TAA"].copy()
    color_map = {
        "SPXT": "#164f86",
        "FTSE100": "#2b6cb0",
        "LBUSTRUU": "#3a7ca5",
        "BROAD_TIPS": "#5fa8d3",
        "B3REITT": "#8f5c2c",
        "XAU": "#c49a00",
        "SILVER_FUT": "#9e9e9e",
        "NIKKEI225": "#c8553d",
        "CSI300_CHINA": "#e07a5f",
        "BITCOIN": "#ff8c00",
        "CHF_FRANC": "#4a7c59",
    }
    legend_patches = [
        Patch(facecolor="#164f86", label="Core"),
        Patch(facecolor="#c49a00", label="Satellite"),
        Patch(facecolor="#ff8c00", label="Non-Traditional"),
    ]

    plt.figure(figsize=(11.0, 6.2))
    plt.stackplot(
        weights.index,
        [weights[column].values for column in ALL_SAA],
        labels=ALL_SAA,
        colors=[color_map[column] for column in ALL_SAA],
        alpha=0.9,
    )
    plt.title("TAA Target Weights")
    plt.ylabel("Weight")
    plt.xlabel("Date")
    plt.ylim(0.0, 1.0)
    plt.legend(handles=legend_patches, loc="upper left")
    plt.grid(alpha=0.20)
    path = figure_dir / FIGURE_FILENAMES["taa_weights"]
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _save_regime_shading_figure(panels: dict[str, object], figure_dir: Path) -> Path:
    """Save the BM2 growth chart with HMM-regime shading.

    Inputs:
    - `panels`: reconstructed reporting panels.
    - `figure_dir`: destination figure directory.

    Outputs:
    - Path to the saved PNG.

    Citation:
    - Whitmore Task 8 `fig05_regime_shading.png` requirement.

    Point-in-time safety:
    - Ex-post plotting only.
    """

    bm2_returns = panels["returns"]["BM2"].reindex(panels["regimes_daily"].index).dropna()
    regime_series = panels["regimes_daily"].reindex(bm2_returns.index).ffill()
    growth = cumulative_growth_index(bm2_returns)

    regime_colors = {"risk_on": "#d8f3dc", "neutral": "#fef3c7", "stress": "#fecaca"}
    plt.figure(figsize=(10.5, 6.0))
    plt.plot(growth.index, growth.values, color="#1f2937", linewidth=2.0, label="BM2")

    current_regime = None
    block_start = None
    for date, regime in regime_series.items():
        if regime != current_regime:
            if current_regime is not None and block_start is not None:
                plt.axvspan(block_start, date, color=regime_colors.get(current_regime, "#e5e7eb"), alpha=0.45)
            current_regime = regime
            block_start = date
    if current_regime is not None and block_start is not None:
        plt.axvspan(block_start, regime_series.index.max(), color=regime_colors.get(current_regime, "#e5e7eb"), alpha=0.45)

    plt.title("BM2 Growth with HMM Regime Shading")
    plt.ylabel("Index Level")
    plt.xlabel("Date")
    plt.grid(alpha=0.25)
    plt.legend()
    path = figure_dir / FIGURE_FILENAMES["regime_shading"]
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _save_fold_figure(panels: dict[str, object], figure_dir: Path) -> Path:
    """Save the walk-forward fold-structure visualization.

    Inputs:
    - `panels`: reconstructed reporting panels.
    - `figure_dir`: destination figure directory.

    Outputs:
    - Path to the saved PNG.

    Citation:
    - Whitmore Task 8 `fig06_oos_folds.png` requirement.

    Point-in-time safety:
    - Ex-post plotting only.
    """

    folds = panels["folds"].copy()
    plt.figure(figsize=(10.5, 4.8))
    for idx, row in folds.iterrows():
        y = idx + 1
        plt.plot([row["train_start"], row["train_end"]], [y, y], color="#94a3b8", linewidth=6, solid_capstyle="butt")
        plt.plot([row["embargo_start"], row["embargo_end"]], [y, y], color="#f59e0b", linewidth=6, solid_capstyle="butt")
        plt.plot([row["test_start"], row["test_end"]], [y, y], color="#2563eb", linewidth=6, solid_capstyle="butt")
    plt.title("Walk-Forward Fold Structure")
    plt.xlabel("Date")
    plt.yticks(range(1, len(folds) + 1), [f"Fold {i}" for i in folds["fold_id"]])
    plt.grid(alpha=0.25)
    path = figure_dir / FIGURE_FILENAMES["oos_folds"]
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def _save_attribution_figure(per_signal: pd.DataFrame, figure_dir: Path) -> Path:
    """Save the per-signal contribution bar chart.

    Inputs:
    - `per_signal`: per-signal attribution dataframe from Task 7.
    - `figure_dir`: destination figure directory.

    Outputs:
    - Path to the saved PNG.

    Citation:
    - Whitmore Task 8 `fig07_attribution_bar.png` requirement.

    Point-in-time safety:
    - Ex-post plotting only.
    """

    plot_df = per_signal.loc[per_signal["layer"] != "baseline"].copy()
    plt.figure(figsize=(9.5, 5.6))
    plt.bar(plot_df["layer"], plot_df["marginal_oos_sharpe"], color="#2563eb")
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title("Per-Signal Marginal OOS Sharpe Contribution")
    plt.ylabel("Baseline Sharpe Minus Ablated Sharpe")
    plt.xlabel("Signal Layer")
    plt.grid(axis="y", alpha=0.25)
    path = figure_dir / FIGURE_FILENAMES["attribution"]
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def build_reporting(
    start: str = "2003-01-01",
    end: str = "2025-12-31",
    folds: int = 5,
    use_timesfm: bool = False,
    vol_budget: float = TARGET_VOL,
    ensemble_config: EnsembleConfig | None = None,
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURES_DIR,
) -> dict[str, object]:
    """Build metrics, figures, IPS audits, and trial-disclosure artifacts.

    Inputs:
    - `start`, `end`, `folds`: walk-forward settings reused if attribution
      outputs are missing.
    - `use_timesfm`: whether the baseline run used TimesFM.
    - `vol_budget`: internal ex-ante annualized volatility target reused by
      attribution reruns when needed.
    - `ensemble_config`: optional baseline ensemble configuration reused by
      attribution reruns when needed.
    - `output_dir`: destination directory for CSV artifacts.
    - `figure_dir`: destination directory for figure PNG files.

    Outputs:
    - Dictionary containing the main reporting artifacts and figure paths.

    Citation:
    - Whitmore Tasks 8 and 9.

    Point-in-time safety:
    - Safe. This module consumes only already-generated causal outputs and
      point-in-time-safe SAA comparison routines.
    """

    np.random.seed(DEFAULT_RANDOM_SEED)
    MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = load_core_outputs(output_dir)
    attribution = _load_attribution_outputs(
        start=start,
        end=end,
        folds=folds,
        use_timesfm=use_timesfm,
        vol_budget=vol_budget,
        ensemble_config=ensemble_config,
        output_dir=output_dir,
    )
    panels = _build_strategy_panels(outputs, output_dir=output_dir)
    saa_method_comparison, _ = build_saa_method_comparison(output_dir=output_dir)
    trial_ledger, dsr_summary = _build_trial_ledger(
        output_dir=output_dir,
        use_timesfm=use_timesfm,
        folds=folds,
        per_signal=attribution["per_signal"],
        saa_method_comparison=saa_method_comparison,
    )
    metrics = _portfolio_metrics_table(panels, trial_ledger)
    per_fold_metrics = _per_fold_metrics(outputs["oos_returns"])
    regime_allocations = _regime_allocation_summary(panels)

    compliance_rows = []
    compliance_rows.extend(_ips_compliance_rows("SAA", panels["target_weights"]["SAA"]))
    compliance_rows.extend(_ips_compliance_rows("SAA+TAA", panels["target_weights"]["SAA+TAA"]))
    compliance = pd.DataFrame(compliance_rows, columns=["portfolio", "date", "rule", "value", "bound"])

    metrics.to_csv(output_dir / PORTFOLIO_METRICS_FILENAME, index=False)
    per_fold_metrics.to_csv(output_dir / PER_FOLD_METRICS_FILENAME, index=False)
    regime_allocations.to_csv(output_dir / REGIME_ALLOCATION_FILENAME, index=False)
    compliance.to_csv(output_dir / IPS_COMPLIANCE_FILENAME, index=False)
    dsr_summary.to_csv(output_dir / DSR_SUMMARY_FILENAME, index=False)
    if TRIAL_LEDGER_CSV.exists():
        existing_trial_ledger = pd.read_csv(TRIAL_LEDGER_CSV)
    else:
        existing_trial_ledger = pd.DataFrame()
    trial_ledger_columns = list(dict.fromkeys(existing_trial_ledger.columns.tolist() + trial_ledger.columns.tolist()))
    trial_ledger_frames = [frame for frame in (existing_trial_ledger, trial_ledger) if not frame.empty]
    merged_trial_ledger = (
        pd.concat([frame.reindex(columns=trial_ledger_columns) for frame in trial_ledger_frames], ignore_index=True)
        if trial_ledger_frames
        else pd.DataFrame(columns=trial_ledger_columns)
    )
    merged_trial_ledger.to_csv(TRIAL_LEDGER_CSV, index=False)

    figures = {
        "cumgrowth": _save_cumgrowth_figure(panels, figure_dir),
        "drawdown": _save_drawdown_figure(panels, figure_dir),
        "rolling_vol": _save_rolling_vol_figure(panels, figure_dir),
        "taa_weights": _save_taa_weights_figure(panels, figure_dir),
        "regime_shading": _save_regime_shading_figure(panels, figure_dir),
        "oos_folds": _save_fold_figure(panels, figure_dir),
        "attribution": _save_attribution_figure(attribution["per_signal"], figure_dir),
    }

    return {
        "metrics": metrics,
        "regime_allocations": regime_allocations,
        "per_fold_metrics": per_fold_metrics,
        "ips_compliance": compliance,
        "trial_ledger": trial_ledger,
        "dsr_summary": dsr_summary,
        "saa_method_comparison": saa_method_comparison,
        "figures": figures,
    }


def main() -> None:
    """CLI entrypoint for Task 8 reporting artifacts.

    Inputs:
    - `--start`, `--end`, `--folds`: walk-forward settings reused when
      attribution outputs are missing.
    - `--timesfm`: whether the baseline run used TimesFM.
    - `--vol-budget`: internal ex-ante annualized volatility target reused by
      attribution dependency rebuilds.
    - `--output-dir`: destination directory for CSV artifacts.
    - `--figure-dir`: destination directory for figure PNG files.

    Outputs:
    - Writes Task 8 reporting artifacts to disk.

    Citation:
    - Whitmore Tasks 8 and 9.

    Point-in-time safety:
    - Safe. The CLI orchestrates the ex-post reporting routines above.
    """

    parser = argparse.ArgumentParser(description="Build Whitmore reporting artifacts.")
    parser.add_argument("--start", default="2003-01-01", help="First OOS date for attribution dependency rebuilds.")
    parser.add_argument("--end", default="2025-12-31", help="Last OOS date for attribution dependency rebuilds.")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds.")
    parser.add_argument("--timesfm", action="store_true", help="Enable the optional TimesFM layer in dependency rebuilds.")
    parser.add_argument("--vol-budget", type=float, default=TARGET_VOL, help="Internal ex-ante vol target used by the TAA optimizer.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Destination directory for CSV outputs.")
    parser.add_argument("--figure-dir", default=str(FIGURES_DIR), help="Destination directory for figure PNG files.")
    args = parser.parse_args()

    build_reporting(
        start=args.start,
        end=args.end,
        folds=args.folds,
        use_timesfm=args.timesfm,
        vol_budget=args.vol_budget,
        output_dir=Path(args.output_dir),
        figure_dir=Path(args.figure_dir),
    )
    print(
        "Reporting outputs written to "
        f"{Path(args.output_dir) / PORTFOLIO_METRICS_FILENAME}, "
        f"{Path(args.output_dir) / IPS_COMPLIANCE_FILENAME}, "
        f"{Path(args.output_dir) / DSR_SUMMARY_FILENAME}, and "
        f"{Path(args.figure_dir)}."
    )


if __name__ == "__main__":
    main()
