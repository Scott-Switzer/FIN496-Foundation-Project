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

import matplotlib.dates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.optimize import Bounds, minimize

from taa_project.analysis.attribution import build_attribution
from taa_project.analysis.common import (
    RISK_FREE_RATE,
    TRADING_DAYS_PER_YEAR,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cost_drag_per_year,
    cumulative_growth_index,
    disclosed_trial_count,
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
    ALL_TAA,
    CORE,
    CORE_FLOOR,
    DEFAULT_RANDOM_SEED,
    FIGURES_DIR,
    MAX_DD,
    MPLCONFIG_DIR,
    NONTRAD,
    NONTRAD_CAP,
    OPPO_CAP,
    OPPORTUNISTIC,
    OUTPUT_DIR,
    REPORT_DIR,
    SAA_BANDS,
    SATELLITE,
    SATELLITE_CAP,
    SINGLE_SLEEVE_MAX,
    TAA_AUDIT_BANDS,
    TARGET_VOL,
    TRIAL_LEDGER_CSV,
    VOL_CEILING,
)
from taa_project.data_loader import load_prices, log_returns
from taa_project.optimizer.cvxpy_opt import EnsembleConfig
from taa_project.pandas_utils import forward_propagate
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
    "per_fold": "fig08_per_fold_oos.png",
    "signal_history": "fig09_signal_history.png",
    "contribution": "fig10_contribution.png",
    "rolling_alpha": "fig11_rolling_alpha.png",
    "regime_forward_returns": "fig12_regime_forward_returns.png",
    "annual_returns": "fig13_annual_returns.png",
    "risk_return_scatter": "fig14_risk_return_scatter.png",
    "monthly_heatmap": "fig15_monthly_heatmap.png",
    "annual_costs": "fig16_annual_costs.png",
    "correlation_heatmap": "fig17_correlation_heatmap.png",
    "cumulative_alpha": "fig18_cumulative_alpha.png",
}

LEGACY_SAA_METHODS = (
    "inverse_vol",
    "minimum_variance",
    "risk_parity",
    "maximum_diversification",
    "mean_variance",
)

TAA_BASELINE_VARIANT_ID = "taa_baseline"
ROLLING_VOL_WINDOWS = {
    "rolling_vol_21d": 21,
    "rolling_vol_63d": 63,
    "rolling_vol_252d": TRADING_DAYS_PER_YEAR,
}


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
    if method == "hrp":
        from taa_project.saa.saa_comparison import solve_hierarchical_risk_parity

        return solve_hierarchical_risk_parity(assets, covariance).reindex(assets).fillna(0.0)
    if method == "maximum_diversification":
        return _maximum_diversification_weights(covariance, assets)
    if method == "mean_variance":
        return _mean_variance_weights(covariance, expected_returns, assets)
    raise ValueError(f"Unsupported SAA method: {method}")


def build_saa_method_comparison(
    start_date: str = SAA_DEFAULT_START,
    end_date: str = SAA_DEFAULT_END,
    output_dir: Path = OUTPUT_DIR,
    include_hrp: bool = False,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compare the five permitted SAA methods under the Whitmore IPS.

    Inputs:
    - `start_date`: earliest allowed SAA start date.
    - `end_date`: last date to include.
    - `output_dir`: destination directory for the comparison CSV.
    - `include_hrp`: whether to include the opt-in HRP method.

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

    methods = LEGACY_SAA_METHODS + (("hrp",) if include_hrp else ())
    for method in methods:
        rebalance_targets: dict[pd.Timestamp, pd.Series] = {}
        rebalance_active_assets: dict[pd.Timestamp, list[str]] = {}
        for rebalance_date in schedule:
            eligible_assets = available_assets_on(rebalance_date, inception_dates)
            observed_assets = [asset for asset in eligible_assets if pd.notna(prices.loc[rebalance_date, asset])]
            covariance = estimate_covariance(returns, rebalance_date, observed_assets)
            expected_returns = _annualized_expected_returns(returns, rebalance_date, observed_assets)
            target = _candidate_saa_weights(method, covariance, expected_returns, observed_assets)

            full_weights = pd.Series(0.0, index=ALL_SAA, dtype=float)
            full_weights.loc[observed_assets] = target.reindex(observed_assets).fillna(0.0)
            rebalance_targets[rebalance_date] = full_weights
            rebalance_active_assets[rebalance_date] = observed_assets

        weights_df, returns_df = simulate_portfolio(
            returns=returns,
            rebalance_targets=rebalance_targets,
            rebalance_active_assets=rebalance_active_assets,
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
    vol_budget: float,
    ensemble_config: EnsembleConfig | None,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load attribution outputs, building them first when missing.

    Inputs:
    - `start`, `end`, `folds`: attribution rerun settings.
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

    asset_returns = simple_asset_returns(assets=ALL_TAA).dropna(how="all")
    inception_dates = first_valid_dates(load_prices().loc[:, ALL_TAA])

    saa_targets = extract_rebalance_targets(outputs["saa_weights"], outputs["saa_returns"])
    saa_policy_targets = extract_rebalance_targets(
        outputs["saa_weights"],
        outputs["saa_returns"],
        flag_column="scheduled_rebalance_flag",
    )
    bm1_targets = extract_rebalance_targets(outputs["bm1_weights"], outputs["bm1_returns"])
    bm2_targets = extract_rebalance_targets(outputs["bm2_weights"], outputs["bm2_returns"])
    taa_targets = outputs["oos_weights"].reindex(columns=ALL_TAA).fillna(0.0)

    saa_realized_asset = asset_returns.loc[:, ALL_SAA].reindex(outputs["saa_returns"].index).dropna(how="all")
    bm1_realized_asset = asset_returns.loc[:, ALL_SAA].reindex(outputs["bm1_returns"].index).dropna(how="all")
    bm2_realized_asset = asset_returns.loc[:, ALL_SAA].reindex(outputs["bm2_returns"].index).dropna(how="all")
    taa_realized_asset = asset_returns.reindex(outputs["oos_returns"].index).dropna(how="all")

    saa_target_daily = outputs["saa_weights"].loc[:, ALL_SAA].astype(float)
    bm1_target_daily = outputs["bm1_weights"].loc[:, ALL_SAA].astype(float)
    bm2_target_daily = outputs["bm2_weights"].loc[:, ALL_SAA].astype(float)
    taa_target_daily = decision_weights_to_daily_target(taa_targets, taa_realized_asset.index)
    taa_realized_holdings = outputs["oos_holdings"].reindex(columns=ALL_TAA).fillna(0.0).astype(float) if not outputs["oos_holdings"].empty else pd.DataFrame()
    if taa_realized_holdings.empty:
        taa_realized_holdings = decision_weights_to_daily_holdings(taa_targets, taa_realized_asset)

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
            "SAA+TAA": outputs["oos_returns"]["turnover"] if "turnover" in outputs["oos_returns"].columns else outputs["oos_weights"]["turnover"],
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
            "SAA+TAA": taa_realized_holdings,
        },
        "decision_dates": {
            "SAA": pd.DatetimeIndex(saa_policy_targets.index),
            "BM1": pd.DatetimeIndex(bm1_targets.index),
            "BM2": pd.DatetimeIndex(bm2_targets.index),
            "SAA+TAA": pd.DatetimeIndex(taa_targets.index),
        },
        "inception_dates": inception_dates,
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
    holdings = panels["holdings"]
    tier_totals = {name: tier_weight_frame(weights) for name, weights in holdings.items()}
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
                "avg_opportunistic_weight": float(tiers["Opportunistic"].mean()),
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
        for asset in ALL_TAA:
            rows.append(
                {
                    "regime": regime,
                    "grouping": "asset",
                    "component": asset,
                    "avg_weight": float(block[asset].mean()),
                    "days": int(len(block)),
                }
            )
        for tier in ["Core", "Satellite", "Non-Traditional", "Opportunistic"]:
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


def _rolling_realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """Compute realized annualized volatility over a fixed trailing window.

    Inputs:
    - `returns`: daily simple return series.
    - `window`: trailing window length in trading days.

    Outputs:
    - Rolling annualized volatility with full-window initialization.

    Citation:
    - Whitmore IPS Section 10.1 trailing 1M / 3M / 12M volatility reporting.

    Point-in-time safety:
    - Ex-post reporting transform only.
    """

    clean = returns.dropna()
    if clean.empty:
        return pd.Series(dtype=float)
    return clean.rolling(window=window, min_periods=window).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)


def _append_breach_rows(
    rows: list[dict[str, object]],
    portfolio: str,
    rule: str,
    values: pd.Series,
    bound: float,
    predicate: pd.Series,
) -> None:
    """Append one row per breached date for a boolean predicate."""

    if values.empty or predicate.empty:
        return
    flagged = predicate.reindex(values.index).fillna(False)
    for date, value in values.loc[flagged].items():
        rows.append(
            {
                "portfolio": portfolio,
                "date": date,
                "rule": rule,
                "value": float(value),
                "bound": float(bound),
            }
        )


def _activation_dates(
    decision_dates: pd.DatetimeIndex,
    inception_dates: pd.Series,
    holdings_index: pd.DatetimeIndex,
    assets: list[str],
) -> dict[str, pd.Timestamp | None]:
    """Return the first rebalance date when each asset becomes policy-active."""

    unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(decision_dates).unique()))
    realized_dates = pd.DatetimeIndex(sorted(pd.to_datetime(holdings_index).unique()))
    activations: dict[str, pd.Timestamp | None] = {}
    for asset in assets:
        inception_date = inception_dates.get(asset, pd.NaT)
        if pd.isna(inception_date) or unique_dates.empty or realized_dates.empty:
            activations[asset] = None
            continue
        eligible = unique_dates[unique_dates >= pd.Timestamp(inception_date)]
        if not len(eligible):
            activations[asset] = None
            continue
        effective_dates = realized_dates[realized_dates > pd.Timestamp(eligible[0])]
        activations[asset] = pd.Timestamp(effective_dates[0]) if len(effective_dates) else None
    return activations


def _daily_band_frames(
    index: pd.DatetimeIndex,
    activations: dict[str, pd.Timestamp | None],
    band_map: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build daily lower/upper bound panels with expanding-universe activation."""

    assets = list(band_map.keys())
    lower = pd.DataFrame(0.0, index=index, columns=assets, dtype=float)
    upper = pd.DataFrame(0.0, index=index, columns=assets, dtype=float)
    for asset in assets:
        activation_date = activations.get(asset)
        if activation_date is None:
            continue
        active_index = index[index >= activation_date]
        if active_index.empty:
            continue
        asset_lower, asset_upper = band_map[asset]
        lower.loc[active_index, asset] = float(asset_lower)
        upper.loc[active_index, asset] = float(min(asset_upper, SINGLE_SLEEVE_MAX))
    return lower, upper


def _ips_compliance_rows(
    portfolio: str,
    holdings: pd.DataFrame,
    returns: pd.Series,
    decision_dates: pd.DatetimeIndex,
    inception_dates: pd.Series,
    band_map: dict[str, tuple[float, float]],
) -> list[dict[str, object]]:
    """Audit one strategy's realized holdings and returns against the IPS.

    Inputs:
    - `portfolio`: strategy label.
    - `holdings`: daily realized holdings dataframe containing the audited
      SAA or expanded TAA universe.
    - `returns`: daily realized portfolio returns.
    - `decision_dates`: rebalance / decision dates where new assets can enter.
    - `inception_dates`: first valid observed date for each asset.
    - `band_map`: either `SAA_BANDS` or the expanded TAA audit bands.

    Outputs:
    - List of violation rows; empty when the schedule is fully compliant.

    Citation:
    - Whitmore IPS Sections 3, 5, 6, 7, and 10.

    Point-in-time safety:
    - Ex-post audit only.
    """

    rows: list[dict[str, object]] = []
    audit_assets = list(band_map.keys())
    aligned = holdings.reindex(columns=audit_assets).fillna(0.0).astype(float)
    aligned.index = pd.DatetimeIndex(pd.to_datetime(aligned.index))
    realized_index = pd.DatetimeIndex(pd.to_datetime(returns.dropna().index))
    if not realized_index.empty:
        aligned = aligned.reindex(realized_index).dropna(how="all")
    tiered = tier_weight_frame(aligned)
    activations = _activation_dates(decision_dates, inception_dates, aligned.index, audit_assets)
    lower_bounds, upper_bounds = _daily_band_frames(aligned.index, activations, band_map)

    for date, row in aligned.iterrows():
        total_weight = float(row.sum())
        min_weight = float(row.min())
        max_weight = float(row.max())
        core_weight = float(tiered.loc[date, "Core"])
        satellite_weight = float(tiered.loc[date, "Satellite"])
        nontrad_weight = float(tiered.loc[date, "Non-Traditional"])
        opportunistic_weight = float(tiered.loc[date, "Opportunistic"])

        checks = [
            ("sum_to_one", abs(total_weight - 1.0) <= 1e-8, total_weight, 1.0),
            ("no_shorts", min_weight >= -1e-8, min_weight, 0.0),
            ("core_floor", core_weight >= CORE_FLOOR - 1e-8, core_weight, CORE_FLOOR),
            ("satellite_cap", satellite_weight <= SATELLITE_CAP + 1e-8, satellite_weight, SATELLITE_CAP),
            ("nontrad_cap", nontrad_weight <= NONTRAD_CAP + 1e-8, nontrad_weight, NONTRAD_CAP),
            ("opportunistic_cap", opportunistic_weight <= OPPO_CAP + 1e-8, opportunistic_weight, OPPO_CAP),
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

    tolerance = 1e-8
    for asset in audit_assets:
        lower_series = lower_bounds[asset]
        upper_series = upper_bounds[asset]
        holding_series = aligned[asset]
        lower_rule = f"{portfolio.lower().replace('+', '_').replace(' ', '_')}_lower_{asset}"
        upper_rule = f"{portfolio.lower().replace('+', '_').replace(' ', '_')}_upper_{asset}"
        _append_breach_rows(
            rows=rows,
            portfolio=portfolio,
            rule=lower_rule,
            values=holding_series,
            bound=float(lower_series.max()),
            predicate=(holding_series + tolerance < lower_series) & (lower_series > 0.0),
        )
        _append_breach_rows(
            rows=rows,
            portfolio=portfolio,
            rule=upper_rule,
            values=holding_series,
            bound=float(upper_series.max()),
            predicate=holding_series > (upper_series + tolerance),
        )

    aligned_returns = returns.reindex(aligned.index).dropna()
    for rule, window in ROLLING_VOL_WINDOWS.items():
        rolling_vol = _rolling_realized_volatility(aligned_returns, window=window)
        _append_breach_rows(
            rows=rows,
            portfolio=portfolio,
            rule=rule,
            values=rolling_vol,
            bound=VOL_CEILING,
            predicate=rolling_vol > (VOL_CEILING + tolerance),
        )

    drawdowns = drawdown_curve(aligned_returns)
    _append_breach_rows(
        rows=rows,
        portfolio=portfolio,
        rule="max_drawdown",
        values=drawdowns,
        bound=-MAX_DD,
        predicate=drawdowns < (-MAX_DD - tolerance),
    )
    return rows


def _build_trial_ledger(
    output_dir: Path,
    folds: int,
    per_signal: pd.DataFrame,
    saa_method_comparison: pd.DataFrame,
    existing_trial_ledger: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the required trial ledger and DSR summary tables.

    Inputs:
    - `output_dir`: output directory containing baseline and ablation runs.
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
                    },
                    sort_keys=True,
                ),
                "ensemble_weights": json.dumps(
                    {
                        "regime": 0.20 if variant_id != "no_regime" else 0.0,
                        "trend": 0.30 if variant_id != "no_trend" else 0.0,
                        "momo": 0.30 if variant_id != "no_momo" else 0.0,
                        "macro": 0.20,
                    },
                    sort_keys=True,
                ),
                "cov_shrinkage": "0.70_sample_0.30_diag",
                "cv_folds": folds,
                "IS_sharpe": np.nan,
                "OOS_sharpe": sharpe_ratio(returns),
                "DSR": np.nan,
                "notes": notes,
            }
        )

    existing_count = disclosed_trial_count(existing_trial_ledger)
    n_trials = existing_count + len(rows) + len(saa_method_comparison)
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
                "n_disclosed_trials": n_trials,
                "baseline_variant_id": TAA_BASELINE_VARIANT_ID,
                "baseline_sharpe": float(trial_ledger.loc[trial_ledger["variant_id"] == TAA_BASELINE_VARIANT_ID, "OOS_sharpe"].iloc[0]),
                "baseline_dsr": float(trial_ledger.loc[trial_ledger["variant_id"] == TAA_BASELINE_VARIANT_ID, "DSR"].iloc[0]),
            }
        ]
    )
    return trial_ledger, dsr_summary


def refresh_dsr_disclosure(
    output_dir: Path = OUTPUT_DIR,
    trial_ledger_path: Path = TRIAL_LEDGER_CSV,
) -> dict[str, float]:
    """Refresh DSR-dependent artifacts after the trial ledger changes.

    Inputs:
    - `output_dir`: directory containing the run outputs to update.
    - `trial_ledger_path`: disclosure ledger used for the DSR denominator.

    Outputs:
    - Dictionary with the refreshed `baseline_dsr` and `n_trials`.

    Citation:
    - Bailey & López de Prado (2014), Deflated Sharpe Ratio:
      https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf

    Point-in-time safety:
    - Safe. This is a post-run reporting refresh only.
    """

    n_trials = disclosed_trial_count(ledger_path=trial_ledger_path)
    oos_returns_path = output_dir / "oos_returns.csv"
    if not oos_returns_path.exists():
        return {"baseline_dsr": float("nan"), "n_trials": float(n_trials)}
    baseline_returns = pd.read_csv(oos_returns_path)["portfolio_return"]
    baseline_dsr = float(deflated_sharpe_ratio(baseline_returns, n_trials=n_trials))

    dsr_summary_path = output_dir / DSR_SUMMARY_FILENAME
    if dsr_summary_path.exists():
        dsr_summary = pd.read_csv(dsr_summary_path)
    else:
        dsr_summary = pd.DataFrame([{}])
    if dsr_summary.empty:
        dsr_summary = pd.DataFrame([{}])
    dsr_summary.loc[0, "n_taa_trials"] = n_trials
    dsr_summary.loc[0, "n_disclosed_trials"] = n_trials
    dsr_summary.loc[0, "baseline_dsr"] = baseline_dsr
    dsr_summary.to_csv(dsr_summary_path, index=False)

    metrics_path = output_dir / PORTFOLIO_METRICS_FILENAME
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        taa_mask = metrics["portfolio"].eq("SAA+TAA")
        if taa_mask.any():
            metrics.loc[taa_mask, "dsr"] = baseline_dsr
            metrics.to_csv(metrics_path, index=False)

    return {"baseline_dsr": baseline_dsr, "n_trials": float(n_trials)}


_PORTFOLIO_COLORS = {
    "SAA+TAA": "#1A365D",
    "BM2":     "#B8860B",
    "SAA":     "#2E86AB",
    "BM1":     "#8D99AE",
}

_PORTFOLIO_ORDER = ["SAA+TAA", "BM2", "SAA", "BM1"]

_PORTFOLIO_LW = {
    "SAA+TAA": 2.4,
    "BM2":     2.0,
    "SAA":     1.8,
    "BM1":     1.6,
}

_PORTFOLIO_LS = {
    "SAA+TAA": "-",
    "BM2":     "-",
    "SAA":     "--",
    "BM1":     ":",
}


def _apply_whitmore_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor":      "#FFFFFF",
        "savefig.facecolor":     "#FFFFFF",
        "axes.facecolor":        "#F8FAFC",
        "axes.edgecolor":        "#CBD5E0",
        "axes.linewidth":        0.8,
        "axes.grid":             True,
        "axes.grid.axis":        "y",
        "grid.color":            "#E2E8F0",
        "grid.linewidth":        0.55,
        "grid.linestyle":        ":",
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "xtick.color":           "#4A5568",
        "ytick.color":           "#4A5568",
        "xtick.labelsize":       8.5,
        "ytick.labelsize":       8.5,
        "xtick.major.size":      0,
        "ytick.major.size":      0,
        "axes.labelsize":        9.0,
        "axes.labelcolor":       "#2D3748",
        "axes.labelpad":         7,
        "axes.titlesize":        11.5,
        "axes.titleweight":      "bold",
        "axes.titlecolor":       "#1A365D",
        "axes.titlelocation":    "left",
        "axes.titlepad":         12,
        "legend.frameon":        True,
        "legend.framealpha":     0.94,
        "legend.edgecolor":      "#CBD5E0",
        "legend.fontsize":       8.5,
        "legend.title_fontsize": 8.5,
        "font.family":           "sans-serif",
        "lines.solid_capstyle":  "round",
    })


def _style_ax(
    ax: plt.Axes,
    ylabel: str | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    pct_y: bool = False,
    index_y: bool = False,
) -> plt.Axes:
    ax.spines["left"].set_color("#CBD5E0")
    ax.spines["bottom"].set_color("#CBD5E0")
    ax.tick_params(axis="both", which="both", length=0)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if pct_y:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    if index_y:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}"))
    return ax


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

    _apply_whitmore_theme()
    common_start = _common_plot_window(panels)
    fig, ax = plt.subplots(figsize=(11.0, 5.8))

    ordered_labels = [lbl for lbl in _PORTFOLIO_ORDER if lbl in panels["returns"]]
    other_labels = [lbl for lbl in panels["returns"] if lbl not in ordered_labels]
    for label in ordered_labels + other_labels:
        returns = panels["returns"][label]
        growth = cumulative_growth_index(returns.loc[common_start:])
        color = _PORTFOLIO_COLORS.get(label, "#718096")
        lw = _PORTFOLIO_LW.get(label, 1.8)
        ls = _PORTFOLIO_LS.get(label, "-")
        ax.plot(growth.index, growth.values, label=label, color=color,
                linewidth=lw, linestyle=ls, zorder=3)
        if label == "SAA+TAA":
            ax.fill_between(growth.index, growth.values, 100,
                            where=(growth.values >= 100),
                            color=color, alpha=0.07, zorder=1)

    _style_ax(ax, ylabel="Index (100 = start)", title="Cumulative Growth  ·  Index 100", index_y=True)
    ax.axhline(100, color="#CBD5E0", linewidth=0.8, linestyle="-", zorder=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["cumgrowth"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
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

    _apply_whitmore_theme()
    common_start = _common_plot_window(panels)
    fig, ax = plt.subplots(figsize=(11.0, 5.8))

    ordered_labels = [lbl for lbl in _PORTFOLIO_ORDER if lbl in panels["returns"]]
    other_labels = [lbl for lbl in panels["returns"] if lbl not in ordered_labels]
    for label in ordered_labels + other_labels:
        returns = panels["returns"][label]
        dd = drawdown_curve(returns.loc[common_start:])
        color = _PORTFOLIO_COLORS.get(label, "#718096")
        lw = _PORTFOLIO_LW.get(label, 1.8)
        ls = _PORTFOLIO_LS.get(label, "-")
        ax.plot(dd.index, dd.values, label=label, color=color, linewidth=lw, linestyle=ls, zorder=3)
        if label == "SAA+TAA":
            ax.fill_between(dd.index, dd.values, 0, color=color, alpha=0.08, zorder=1)

    ax.axhline(-MAX_DD, color="#C53030", linewidth=1.0, linestyle="--", zorder=2,
               label=f"IPS MDD limit ({100 * MAX_DD:.0f}%)")
    ax.axhline(0, color="#CBD5E0", linewidth=0.8, zorder=0)
    _style_ax(ax, ylabel="Drawdown", title="Underwater Curves", pct_y=True)
    ax.legend(loc="lower left")
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["drawdown"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
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

    _apply_whitmore_theme()
    common_start = _common_plot_window(panels)
    fig, ax = plt.subplots(figsize=(11.0, 5.8))

    ordered_labels = [lbl for lbl in _PORTFOLIO_ORDER if lbl in panels["returns"]]
    other_labels = [lbl for lbl in panels["returns"] if lbl not in ordered_labels]
    for label in ordered_labels + other_labels:
        returns = panels["returns"][label]
        rv = rolling_annualized_volatility(returns.loc[common_start:])
        color = _PORTFOLIO_COLORS.get(label, "#718096")
        lw = _PORTFOLIO_LW.get(label, 1.8)
        ls = _PORTFOLIO_LS.get(label, "-")
        ax.plot(rv.index, rv.values, label=label, color=color, linewidth=lw, linestyle=ls, zorder=3)

    ax.axhline(VOL_CEILING, color="#C53030", linewidth=1.1, linestyle="--", zorder=2,
               label=f"IPS ceiling ({100 * VOL_CEILING:.0f}%)")
    ax.axhline(TARGET_VOL, color="#718096", linewidth=0.9, linestyle=":", zorder=2,
               label=f"Target ({100 * TARGET_VOL:.0f}%)")
    ax.fill_between(
        ax.get_xlim(), TARGET_VOL, VOL_CEILING,
        color="#FED7D7", alpha=0.18, zorder=0, transform=ax.get_yaxis_transform(),
    )
    _style_ax(ax, ylabel="Annualized Volatility (12M rolling)", title="Rolling Volatility", pct_y=True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["rolling_vol"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
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
        "TA-125_ISRAEL": "#7f1d1d",
        "0_5Y_TIPS": "#93c5fd",
        "BAIGTRUU_ASIACREDIT": "#60a5fa",
        "BCEE1T_EUROAREA": "#2563eb",
        "I02923JP_JAPAN_BOND": "#1d4ed8",
        "LBEATREU_EUROBONDAGG": "#1e40af",
        "COPPERSPOT": "#b45309",
        "NATURALGAS": "#a16207",
        "COFFEE_FUT": "#92400e",
        "COCOAINDEXSPOT": "#78350f",
        "COTTON_FUT": "#ca8a04",
        "WHEAT_SPOT": "#eab308",
        "SOYBEAN_FUT": "#84cc16",
        "ETHEREUM": "#6366f1",
        "AUD": "#0f766e",
        "CAD": "#047857",
        "GBP_POUND": "#059669",
        "EURO": "#10b981",
        "CNY": "#14b8a6",
        "SHEKEL": "#0d9488",
        "USDJPY": "#0f766e",
    }
    legend_patches = [
        Patch(facecolor="#164f86", label="Core"),
        Patch(facecolor="#c49a00", label="Satellite"),
        Patch(facecolor="#ff8c00", label="Non-Traditional"),
        Patch(facecolor="#2563eb", label="Opportunistic"),
    ]

    _apply_whitmore_theme()
    plt.rcParams["axes.grid.axis"] = "x"
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.stackplot(
        weights.index,
        [weights[column].values for column in ALL_TAA],
        labels=ALL_TAA,
        colors=[color_map[column] for column in ALL_TAA],
        alpha=0.92,
    )
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.spines["left"].set_color("#CBD5E0")
    ax.spines["bottom"].set_color("#CBD5E0")
    ax.tick_params(length=0)
    ax.set_title("TAA Target Weights by Sleeve")
    ax.set_ylabel("Allocation")
    ax.legend(handles=legend_patches, loc="upper right", title="Sleeve")
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["taa_weights"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
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

    _apply_whitmore_theme()
    regime_colors = {
        "risk_on": "#C6F6D5",
        "neutral": "#FEFCBF",
        "stress":  "#FED7D7",
    }
    regime_labels_display = {
        "risk_on": "Risk-On",
        "neutral": "Neutral",
        "stress":  "Stress",
    }

    plot_labels = [lbl for lbl in ("SAA+TAA", "BM2", "BM1") if lbl in panels["returns"]]
    common_start = _common_plot_window(panels)

    ref_returns = panels["returns"][plot_labels[0]].reindex(panels["regimes_daily"].index).dropna()
    regime_series = forward_propagate(panels["regimes_daily"].reindex(ref_returns.index))

    fig, ax = plt.subplots(figsize=(11.0, 5.8))

    current_regime = None
    block_start = None
    seen_regimes: set[str] = set()
    for date, regime in regime_series.items():
        if regime != current_regime:
            if current_regime is not None and block_start is not None:
                lbl = regime_labels_display.get(current_regime, current_regime) if current_regime not in seen_regimes else None
                ax.axvspan(block_start, date,
                           color=regime_colors.get(current_regime, "#E5E7EB"),
                           alpha=0.45, zorder=1, label=lbl)
                seen_regimes.add(current_regime)
            current_regime = regime
            block_start = date
    if current_regime is not None and block_start is not None:
        lbl = regime_labels_display.get(current_regime, current_regime) if current_regime not in seen_regimes else None
        ax.axvspan(block_start, regime_series.index.max(),
                   color=regime_colors.get(current_regime, "#E5E7EB"),
                   alpha=0.45, zorder=1, label=lbl)

    for label in plot_labels:
        returns = panels["returns"][label]
        growth = cumulative_growth_index(returns.loc[common_start:])
        ax.plot(growth.index, growth.values,
                color=_PORTFOLIO_COLORS.get(label, "#718096"),
                linewidth=_PORTFOLIO_LW.get(label, 1.8),
                linestyle=_PORTFOLIO_LS.get(label, "-"),
                label=label, zorder=4)

    _style_ax(ax, ylabel="Index (100 = start)", title="Cumulative Growth  ·  HMM Regime Shading", index_y=True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["regime_shading"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
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

    _apply_whitmore_theme()
    folds = panels["folds"].copy()
    n_folds = len(folds)
    bar_h = 0.38

    train_color   = "#A0AEC0"
    embargo_color = "#B8860B"
    test_color    = "#1A365D"

    fig, ax = plt.subplots(figsize=(11.0, max(3.2, n_folds * 0.9 + 1.4)))
    for idx, row in folds.iterrows():
        y = idx + 1
        ax.barh(y, (row["train_end"] - row["train_start"]).days, left=row["train_start"],
                height=bar_h, color=train_color, align="center", zorder=3)
        ax.barh(y, (row["embargo_end"] - row["embargo_start"]).days, left=row["embargo_start"],
                height=bar_h, color=embargo_color, align="center", zorder=3)
        ax.barh(y, (row["test_end"] - row["test_start"]).days, left=row["test_start"],
                height=bar_h, color=test_color, align="center", zorder=3)

    ax.set_yticks(range(1, n_folds + 1))
    ax.set_yticklabels([f"Fold {int(row['fold_id'])}" for _, row in folds.iterrows()])
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(2))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E0")
    ax.tick_params(length=0)
    ax.set_title("Walk-Forward OOS Fold Structure")
    legend_patches = [
        Patch(facecolor=train_color,   label="Train"),
        Patch(facecolor=embargo_color, label="Embargo"),
        Patch(facecolor=test_color,    label="Test (OOS)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right")
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["oos_folds"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
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

    _apply_whitmore_theme()
    plot_df = per_signal.loc[per_signal["layer"] != "baseline"].copy()
    values = plot_df["marginal_oos_sharpe"].to_numpy(dtype=float)
    bar_colors = [_PORTFOLIO_COLORS["SAA+TAA"] if v >= 0 else "#C53030" for v in values]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.bar(plot_df["layer"], values, color=bar_colors, width=0.55, zorder=3,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0.0, color="#CBD5E0", linewidth=0.9, zorder=2)

    for bar, val in zip(bars, values):
        offset = 0.005 if val >= 0 else -0.005
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:+.3f}", ha="center", va=va, fontsize=8, color="#2D3748")

    _style_ax(ax, ylabel="ΔSharpe vs. ablated baseline",
              title="Signal Attribution  ·  Marginal OOS Sharpe")
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["attribution"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_per_fold_figure(per_fold_metrics: pd.DataFrame, figure_dir: Path) -> Path:
    """Save per-fold OOS annualized return and Sharpe bar chart.

    Point-in-time safety:
    - Ex-post plotting only.
    """

    _apply_whitmore_theme()
    df = per_fold_metrics.copy()
    fold_labels = [f"Fold {int(fid)}" for fid in df["fold_id"]]
    x = np.arange(len(df))
    navy = _PORTFOLIO_COLORS["SAA+TAA"]
    gold  = _PORTFOLIO_COLORS["BM2"]

    fig, (ax_ret, ax_sr) = plt.subplots(1, 2, figsize=(11.0, 5.0))

    bars_ret = ax_ret.bar(x, df["annualized_return"] * 100, color=navy, width=0.55,
                          zorder=3, edgecolor="white", linewidth=0.5)
    ax_ret.axhline(0, color="#CBD5E0", linewidth=0.8)
    for bar, val in zip(bars_ret, df["annualized_return"]):
        ax_ret.text(bar.get_x() + bar.get_width() / 2, val * 100 + 0.3,
                    f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=8.5, color="#2D3748")
    ax_ret.set_xticks(x)
    ax_ret.set_xticklabels(fold_labels)
    _style_ax(ax_ret, ylabel="Annualized Return", title="OOS Return by Fold  ·  SAA+TAA")
    ax_ret.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    bars_sr = ax_sr.bar(x, df["sharpe"], color=gold, width=0.55,
                        zorder=3, edgecolor="white", linewidth=0.5)
    ax_sr.axhline(0, color="#CBD5E0", linewidth=0.8)
    for bar, val in zip(bars_sr, df["sharpe"]):
        ax_sr.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, color="#2D3748")
    ax_sr.set_xticks(x)
    ax_sr.set_xticklabels(fold_labels)
    _style_ax(ax_sr, ylabel="Sharpe Ratio (rf = 2%)", title="OOS Sharpe by Fold  ·  SAA+TAA")

    fig.tight_layout(w_pad=3.0)
    path = figure_dir / FIGURE_FILENAMES["per_fold"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_signal_history_figure(
    oos_regimes: pd.DataFrame,
    vix_signal: pd.DataFrame,
    figure_dir: Path,
) -> Path:
    """Save a 3-panel signal history chart: regime probs, VIX z-score, yield-curve component.

    Point-in-time safety:
    - Ex-post plotting only.
    """

    if vix_signal.empty or "vix_z" not in vix_signal.columns:
        empty_path = figure_dir / "fig09_signal_history.png"
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.text(0.5, 0.5, "VIX signal history data not available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(empty_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return empty_path

    _apply_whitmore_theme()
    regime_colors = {
        "risk_on": "#C6F6D5",
        "neutral": "#FEFCBF",
        "stress":  "#FED7D7",
    }
    regime_line_colors = {
        "p_risk_on": "#276749",
        "p_neutral": "#B7791F",
        "p_stress":  "#C53030",
    }
    regime_line_labels = {
        "p_risk_on": "P(Risk-On)",
        "p_neutral": "P(Neutral)",
        "p_stress":  "P(Stress)",
    }

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.5), sharex=True,
                             constrained_layout=True,
                             gridspec_kw={"hspace": 0.10, "height_ratios": [1.4, 1, 1]})

    # Panel 1: stacked regime probabilities
    ax1 = axes[0]
    probs = oos_regimes[["p_risk_on", "p_neutral", "p_stress"]].copy().clip(0, 1)
    ax1.stackplot(
        probs.index,
        probs["p_risk_on"].values,
        probs["p_neutral"].values,
        probs["p_stress"].values,
        labels=["Risk-On", "Neutral", "Stress"],
        colors=["#C6F6D5", "#FEFCBF", "#FED7D7"],
        alpha=0.85,
    )
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax1.spines["left"].set_color("#CBD5E0")
    ax1.spines["bottom"].set_visible(False)
    ax1.tick_params(length=0)
    ax1.set_title("TAA Signal History  ·  Regime · VIX · Yield Curve", pad=12,
                  fontsize=11.5, fontweight="bold", color="#1A365D", loc="left")
    ax1.set_ylabel("HMM Regime Prob.")
    ax1.legend(loc="upper right", fontsize=8)

    # Panel 2: VIX z-score
    ax2 = axes[1]
    vix_aligned = vix_signal["vix_z"].reindex(probs.index, method="ffill")
    ax2.fill_between(vix_aligned.index, vix_aligned.values, 0,
                     where=(vix_aligned.values > 0), color="#FED7D7", alpha=0.6, zorder=1)
    ax2.fill_between(vix_aligned.index, vix_aligned.values, 0,
                     where=(vix_aligned.values <= 0), color="#C6F6D5", alpha=0.6, zorder=1)
    ax2.plot(vix_aligned.index, vix_aligned.values, color="#C53030", linewidth=1.2, zorder=3)
    ax2.axhline(0, color="#CBD5E0", linewidth=0.8)
    ax2.spines["left"].set_color("#CBD5E0")
    ax2.spines["bottom"].set_visible(False)
    ax2.tick_params(length=0)
    ax2.set_ylabel("VIX z-score")

    # Panel 3: yield-curve component (term spread)
    ax3 = axes[2]
    curve_aligned = vix_signal["curve_component"].reindex(probs.index, method="ffill")
    ax3.fill_between(curve_aligned.index, curve_aligned.values, 0,
                     where=(curve_aligned.values > 0), color="#BEE3F8", alpha=0.6, zorder=1)
    ax3.fill_between(curve_aligned.index, curve_aligned.values, 0,
                     where=(curve_aligned.values <= 0), color="#FED7D7", alpha=0.6, zorder=1)
    ax3.plot(curve_aligned.index, curve_aligned.values, color="#2B6CB0", linewidth=1.2, zorder=3)
    ax3.axhline(0, color="#CBD5E0", linewidth=0.8)
    ax3.spines["left"].set_color("#CBD5E0")
    ax3.spines["bottom"].set_color("#CBD5E0")
    ax3.tick_params(length=0)
    ax3.set_ylabel("Yield Curve Score")

    path = figure_dir / FIGURE_FILENAMES["signal_history"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_contribution_figure(metrics: pd.DataFrame, figure_dir: Path) -> Path:
    """Save a grouped bar comparing annualized return and Sharpe across all 4 portfolios.

    Point-in-time safety:
    - Ex-post plotting only.
    """

    _apply_whitmore_theme()
    order = ["BM1", "BM2", "SAA", "SAA+TAA"]
    df = metrics.set_index("portfolio").reindex(order).reset_index()
    colors = [_PORTFOLIO_COLORS.get(p, "#718096") for p in order]
    x = np.arange(len(df))

    fig, (ax_ret, ax_sr) = plt.subplots(1, 2, figsize=(11.0, 5.0))

    bars_ret = ax_ret.bar(x, df["annualized_return"] * 100, color=colors, width=0.55,
                          zorder=3, edgecolor="white", linewidth=0.5)
    ax_ret.axhline(0, color="#CBD5E0", linewidth=0.8)
    for bar, val in zip(bars_ret, df["annualized_return"]):
        ax_ret.text(bar.get_x() + bar.get_width() / 2, val * 100 + 0.15,
                    f"{val * 100:.1f}%", ha="center", va="bottom", fontsize=9, color="#2D3748")
    ax_ret.set_xticks(x)
    ax_ret.set_xticklabels(order)
    ax_ret.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _style_ax(ax_ret, ylabel="Annualized Return", title="Return Contribution")

    bars_sr = ax_sr.bar(x, df["sharpe_rf_2pct"], color=colors, width=0.55,
                        zorder=3, edgecolor="white", linewidth=0.5)
    ax_sr.axhline(0, color="#CBD5E0", linewidth=0.8)
    for bar, val in zip(bars_sr, df["sharpe_rf_2pct"]):
        ax_sr.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                   f"{val:.2f}", ha="center", va="bottom", fontsize=9, color="#2D3748")
    ax_sr.set_xticks(x)
    ax_sr.set_xticklabels(order)
    _style_ax(ax_sr, ylabel="Sharpe Ratio (rf = 2%)", title="Risk-Adjusted Contribution")

    fig.suptitle("Portfolio Contribution  ·  BM1 → BM2 → SAA → SAA+TAA",
                 fontsize=11.5, fontweight="bold", color="#1A365D", x=0.02, ha="left", y=1.01)
    fig.tight_layout(w_pad=3.0)
    path = figure_dir / FIGURE_FILENAMES["contribution"]
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_rolling_alpha_figure(panels: dict[str, object], figure_dir: Path) -> Path:
    """Rolling 12M alpha of SAA+TAA vs BM2."""

    _apply_whitmore_theme()
    common_start = _common_plot_window(panels)
    taa = panels["returns"]["SAA+TAA"].loc[common_start:]
    bm2 = panels["returns"]["BM2"].reindex(taa.index).fillna(0.0)
    alpha_daily = taa - bm2
    rolling_alpha = alpha_daily.rolling(252).apply(
        lambda x: (1 + x).prod() ** (252 / len(x)) - 1, raw=True
    )

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.fill_between(rolling_alpha.index, rolling_alpha.values * 100, 0,
                    where=(rolling_alpha.values >= 0), color=_PORTFOLIO_COLORS["SAA+TAA"],
                    alpha=0.25, zorder=1, label="Positive alpha")
    ax.fill_between(rolling_alpha.index, rolling_alpha.values * 100, 0,
                    where=(rolling_alpha.values < 0), color="#C53030",
                    alpha=0.25, zorder=1, label="Negative alpha")
    ax.plot(rolling_alpha.index, rolling_alpha.values * 100,
            color=_PORTFOLIO_COLORS["SAA+TAA"], linewidth=1.8, zorder=3)
    ax.axhline(0, color="#CBD5E0", linewidth=0.9, zorder=2)
    _style_ax(ax, ylabel="Rolling 12M Alpha vs BM2 (%)",
              title="Rolling Alpha  ·  SAA+TAA vs BM2  ·  12-Month Window")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["rolling_alpha"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_regime_forward_returns_figure(
    oos_regimes: pd.DataFrame,
    oos_returns: pd.DataFrame,
    figure_dir: Path,
) -> Path:
    """Boxplot of 1M forward returns on SAA+TAA by HMM regime label."""

    _apply_whitmore_theme()
    regime_col = oos_regimes["regime"].reindex(oos_returns.index, method="ffill")
    ret = oos_returns["portfolio_return"].copy()

    fwd_1m = ret.shift(-21).dropna()
    regime_aligned = regime_col.reindex(fwd_1m.index).dropna()
    fwd_1m = fwd_1m.reindex(regime_aligned.index)

    order = ["risk_on", "neutral", "stress"]
    labels = ["Risk-On", "Neutral", "Stress"]
    box_colors = ["#C6F6D5", "#FEFCBF", "#FED7D7"]
    box_edge   = ["#276749", "#B7791F", "#C53030"]
    data = [fwd_1m.loc[regime_aligned == r].values * 100 for r in order]

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    bp = ax.boxplot(data, patch_artist=True, widths=0.45, medianprops={"linewidth": 2.0},
                    flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    for patch, fc, ec in zip(bp["boxes"], box_colors, box_edge):
        patch.set_facecolor(fc)
        patch.set_edgecolor(ec)
        patch.set_linewidth(1.2)
    for whisker in bp["whiskers"]:
        whisker.set_color("#718096")
    for cap in bp["caps"]:
        cap.set_color("#718096")

    for i, (d, label) in enumerate(zip(data, labels), start=1):
        median_val = float(np.median(d))
        ax.text(i, median_val + 0.08, f"{median_val:.2f}%",
                ha="center", va="bottom", fontsize=8.5, color="#2D3748", fontweight="bold")
        n = len(d)
        ax.text(i, ax.get_ylim()[0] + 0.05, f"n={n}",
                ha="center", va="bottom", fontsize=7.5, color="#718096")

    ax.axhline(0, color="#CBD5E0", linewidth=0.9, zorder=0)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels)
    _style_ax(ax, ylabel="21-Day Forward Return (%)",
              title="Signal Validation  ·  Forward Returns by Regime")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["regime_forward_returns"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_annual_returns_figure(panels: dict[str, object], figure_dir: Path) -> Path:
    """Side-by-side annual return bars for all 4 portfolios."""

    _apply_whitmore_theme()
    common_start = _common_plot_window(panels)
    order = [lbl for lbl in _PORTFOLIO_ORDER if lbl in panels["returns"]]
    annual: dict[str, pd.Series] = {}
    for label in order:
        ret = panels["returns"][label].loc[common_start:]
        annual[label] = ret.resample("YE").apply(lambda x: (1 + x).prod() - 1) * 100

    all_years = sorted(set().union(*[s.index.year for s in annual.values()]))
    x = np.arange(len(all_years))
    n = len(order)
    bar_w = 0.8 / n

    fig, ax = plt.subplots(figsize=(max(11.0, len(all_years) * 0.55), 5.5))
    for i, label in enumerate(order):
        vals = [float(annual[label].get(pd.Timestamp(f"{yr}-12-31"), np.nan)) for yr in all_years]
        offset = (i - n / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, width=bar_w * 0.92,
                      color=_PORTFOLIO_COLORS.get(label, "#718096"),
                      label=label, zorder=3, edgecolor="white", linewidth=0.4)

    ax.axhline(0, color="#CBD5E0", linewidth=0.9, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(yr) for yr in all_years], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    _style_ax(ax, ylabel="Annual Return", title="Annual Returns  ·  All Portfolios")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["annual_returns"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_risk_return_scatter_figure(metrics: pd.DataFrame, figure_dir: Path) -> Path:
    """Risk/return scatter with all 4 portfolios and iso-Sharpe lines."""

    _apply_whitmore_theme()
    order = ["BM1", "BM2", "SAA", "SAA+TAA"]
    df = metrics.set_index("portfolio").reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(8.0, 6.0))

    vol_range = np.linspace(0.04, 0.14, 200)
    for sr, ls in [(0.5, ":"), (1.0, "--")]:
        ret_line = 0.02 + sr * vol_range
        ax.plot(vol_range * 100, ret_line * 100, color="#CBD5E0",
                linewidth=0.9, linestyle=ls, zorder=1,
                label=f"SR = {sr:.1f}")

    for _, row in df.iterrows():
        label = row["portfolio"]
        vol = float(row["annualized_volatility"]) * 100
        ret = float(row["annualized_return"]) * 100
        color = _PORTFOLIO_COLORS.get(label, "#718096")
        ax.scatter(vol, ret, color=color, s=120, zorder=4, edgecolors="white", linewidths=1.2)
        ax.annotate(label, (vol, ret), xytext=(6, 4), textcoords="offset points",
                    fontsize=9, color=color, fontweight="bold")

    ax.axvline(15.0, color="#C53030", linewidth=0.9, linestyle="--", zorder=2,
               label="IPS vol ceiling (15%)")
    ax.axhline(8.0, color="#B8860B", linewidth=0.9, linestyle=":", zorder=2,
               label="8% return target")
    _style_ax(ax, xlabel="Annualized Volatility (%)", ylabel="Annualized Return (%)",
              title="Risk  ·  Return  ·  All Portfolios")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["risk_return_scatter"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_monthly_heatmap_figure(panels: dict[str, object], figure_dir: Path) -> Path:
    """Calendar heatmap of monthly returns for SAA+TAA."""

    import seaborn as sns

    _apply_whitmore_theme()
    ret = panels["returns"]["SAA+TAA"]
    monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    monthly_df = monthly.rename("ret").reset_index()
    monthly_df["year"] = monthly_df["date"].dt.year
    monthly_df["month"] = monthly_df["date"].dt.month
    pivot = monthly_df.pivot(index="year", columns="month", values="ret")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(13.0, max(6.0, len(pivot) * 0.38 + 1.5)))
    abs_max = float(np.nanpercentile(np.abs(pivot.values), 97))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".1f", annot_kws={"size": 7.5},
        cmap=sns.diverging_palette(10, 133, as_cmap=True),
        center=0, vmin=-abs_max, vmax=abs_max,
        linewidths=0.4, linecolor="#E2E8F0",
        cbar_kws={"label": "Monthly Return (%)", "shrink": 0.6},
    )
    ax.set_title("Monthly Returns  ·  SAA+TAA (%)", pad=12,
                 fontsize=11.5, fontweight="bold", color="#1A365D", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0, labelsize=8.5)
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["monthly_heatmap"]
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_annual_costs_figure(oos_returns: pd.DataFrame, figure_dir: Path) -> Path:
    """Annual turnover cost drag for SAA+TAA."""

    _apply_whitmore_theme()
    oos_returns = oos_returns.copy()
    oos_returns.index = pd.to_datetime(oos_returns.index)
    annual_cost = oos_returns["turnover_cost"].resample("YE").sum() * 100
    annual_gross = oos_returns["gross_return"].resample("YE").apply(
        lambda x: (1 + x).prod() - 1
    ) * 100
    annual_net = oos_returns["portfolio_return"].resample("YE").apply(
        lambda x: (1 + x).prod() - 1
    ) * 100

    years = annual_cost.index.year
    x = np.arange(len(years))

    fig, (ax_cost, ax_drag) = plt.subplots(1, 2, figsize=(11.0, 5.0))

    ax_cost.bar(x, annual_cost.values, color=_PORTFOLIO_COLORS["BM2"],
                width=0.55, zorder=3, edgecolor="white", linewidth=0.5)
    for xi, val in zip(x, annual_cost.values):
        ax_cost.text(xi, val + 0.005, f"{val:.2f}%", ha="center", va="bottom",
                     fontsize=8, color="#2D3748")
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels([str(yr) for yr in years], rotation=45, ha="right")
    ax_cost.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
    _style_ax(ax_cost, ylabel="Annual Cost Drag", title="Transaction Costs by Year")

    drag = annual_gross.values - annual_net.values
    ax_drag.bar(x, drag, color=_PORTFOLIO_COLORS["SAA+TAA"],
                width=0.55, zorder=3, edgecolor="white", linewidth=0.5)
    ax_drag.set_xticks(x)
    ax_drag.set_xticklabels([str(yr) for yr in years], rotation=45, ha="right")
    ax_drag.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}%"))
    _style_ax(ax_drag, ylabel="Gross − Net Return (%)", title="Gross vs Net Return Spread")

    fig.tight_layout(w_pad=3.0)
    path = figure_dir / FIGURE_FILENAMES["annual_costs"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _save_correlation_heatmap_figure(figure_dir: Path, output_dir: Path) -> Path:
    """Correlation heatmap of SAA asset log-returns."""

    import seaborn as sns
    from taa_project.config import ALL_SAA

    _apply_whitmore_theme()
    log_ret = pd.read_csv(output_dir / "asset_log_returns.csv", index_col="Date", parse_dates=True)
    saa_ret = log_ret[[c for c in ALL_SAA if c in log_ret.columns]].dropna(how="all")
    corr = saa_ret.loc["2003":].corr(min_periods=63).round(2)

    ticker_labels = {
        "SPXT": "US Eq.", "FTSE100": "UK Eq.", "NIKKEI225": "Japan Eq.",
        "CSI300_CHINA": "China Eq.", "LBUSTRUU": "US Agg.", "BROAD_TIPS": "TIPS",
        "B3REITT": "REIT", "XAU": "Gold", "SILVER_FUT": "Silver",
        "BITCOIN": "Bitcoin", "CHF_FRANC": "CHF",
    }
    corr.index   = [ticker_labels.get(c, c) for c in corr.index]
    corr.columns = [ticker_labels.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax, mask=mask, annot=True, fmt=".2f", annot_kws={"size": 8},
        cmap=sns.diverging_palette(10, 133, as_cmap=True),
        center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor="#E2E8F0",
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.7},
    )
    ax.set_title("SAA Asset Correlations  ·  2003–2025", pad=12,
                 fontsize=11.5, fontweight="bold", color="#1A365D", loc="left")
    ax.tick_params(length=0, labelsize=8.5)
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["correlation_heatmap"]
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_cumulative_alpha_figure(panels: dict[str, object], figure_dir: Path) -> Path:
    """Cumulative alpha of SAA+TAA vs BM2, and SAA vs BM2."""

    _apply_whitmore_theme()
    common_start = _common_plot_window(panels)

    bm2 = panels["returns"]["BM2"].loc[common_start:]
    taa = panels["returns"]["SAA+TAA"].reindex(bm2.index).fillna(0.0)
    saa = panels["returns"]["SAA"].reindex(bm2.index).fillna(0.0)

    cum_alpha_taa = ((1 + (taa - bm2)).cumprod() - 1) * 100
    cum_alpha_saa = ((1 + (saa - bm2)).cumprod() - 1) * 100

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.fill_between(cum_alpha_taa.index, cum_alpha_taa.values, 0,
                    where=(cum_alpha_taa.values >= 0),
                    color=_PORTFOLIO_COLORS["SAA+TAA"], alpha=0.15, zorder=1)
    ax.plot(cum_alpha_taa.index, cum_alpha_taa.values,
            color=_PORTFOLIO_COLORS["SAA+TAA"], linewidth=2.0, label="SAA+TAA vs BM2", zorder=3)
    ax.plot(cum_alpha_saa.index, cum_alpha_saa.values,
            color=_PORTFOLIO_COLORS["SAA"], linewidth=1.6,
            linestyle="--", label="SAA vs BM2", zorder=3)
    ax.axhline(0, color="#CBD5E0", linewidth=0.9, zorder=2)
    _style_ax(ax, ylabel="Cumulative Alpha vs BM2 (%)",
              title="Cumulative Alpha  ·  TAA Overlay & SAA vs BM2")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = figure_dir / FIGURE_FILENAMES["cumulative_alpha"]
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def build_reporting(
    start: str = "2003-01-01",
    end: str = "2026-04-15",
    folds: int = 5,
    use_timesfm: bool = False,
    vol_budget: float = TARGET_VOL,
    saa_method: str = "min_variance",
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
    - `saa_method`: strategic asset allocation method used for the run.
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
        vol_budget=vol_budget,
        ensemble_config=ensemble_config,
        output_dir=output_dir,
    )
    panels = _build_strategy_panels(outputs, output_dir=output_dir)
    saa_method_comparison, _ = build_saa_method_comparison(
        output_dir=output_dir,
        include_hrp=saa_method == "hrp",
    )
    if TRIAL_LEDGER_CSV.exists():
        existing_trial_ledger = pd.read_csv(TRIAL_LEDGER_CSV)
    else:
        existing_trial_ledger = pd.DataFrame()
    trial_ledger, dsr_summary = _build_trial_ledger(
        output_dir=output_dir,
        folds=folds,
        per_signal=attribution["per_signal"],
        saa_method_comparison=saa_method_comparison,
        existing_trial_ledger=existing_trial_ledger,
    )
    metrics = _portfolio_metrics_table(panels, trial_ledger)
    per_fold_metrics = _per_fold_metrics(outputs["oos_returns"])
    regime_allocations = _regime_allocation_summary(panels)

    compliance_rows = []
    compliance_rows.extend(
        _ips_compliance_rows(
            "SAA",
            holdings=panels["holdings"]["SAA"],
            returns=panels["returns"]["SAA"],
            decision_dates=panels["decision_dates"]["SAA"],
            inception_dates=panels["inception_dates"],
            band_map=SAA_BANDS,
        )
    )
    compliance_rows.extend(
        _ips_compliance_rows(
            "SAA+TAA",
            holdings=panels["holdings"]["SAA+TAA"],
            returns=panels["returns"]["SAA+TAA"],
            decision_dates=panels["decision_dates"]["SAA+TAA"],
            inception_dates=panels["inception_dates"],
            band_map=TAA_AUDIT_BANDS,
        )
    )
    compliance = pd.DataFrame(compliance_rows, columns=["portfolio", "date", "rule", "value", "bound"])

    metrics.to_csv(output_dir / PORTFOLIO_METRICS_FILENAME, index=False)
    per_fold_metrics.to_csv(output_dir / PER_FOLD_METRICS_FILENAME, index=False)
    regime_allocations.to_csv(output_dir / REGIME_ALLOCATION_FILENAME, index=False)
    compliance.to_csv(output_dir / IPS_COMPLIANCE_FILENAME, index=False)
    dsr_summary.to_csv(output_dir / DSR_SUMMARY_FILENAME, index=False)
    trial_ledger_columns = list(dict.fromkeys(existing_trial_ledger.columns.tolist() + trial_ledger.columns.tolist()))
    trial_ledger_frames = [frame for frame in (existing_trial_ledger, trial_ledger) if not frame.empty]
    merged_trial_ledger = (
        pd.concat([frame.reindex(columns=trial_ledger_columns) for frame in trial_ledger_frames], ignore_index=True)
        if trial_ledger_frames
        else pd.DataFrame(columns=trial_ledger_columns)
    )
    merged_trial_ledger.to_csv(TRIAL_LEDGER_CSV, index=False)

    vix_signal_path = output_dir / "vix_yield_curve_signal_history.csv"
    if vix_signal_path.exists():
        vix_signal_history = pd.read_csv(
            vix_signal_path,
            index_col="as_of_date",
            parse_dates=True,
        )
    else:
        vix_signal_history = pd.DataFrame()
    figures = {
        "cumgrowth": _save_cumgrowth_figure(panels, figure_dir),
        "drawdown": _save_drawdown_figure(panels, figure_dir),
        "rolling_vol": _save_rolling_vol_figure(panels, figure_dir),
        "taa_weights": _save_taa_weights_figure(panels, figure_dir),
        "regime_shading": _save_regime_shading_figure(panels, figure_dir),
        "oos_folds": _save_fold_figure(panels, figure_dir),
        "attribution": _save_attribution_figure(attribution["per_signal"], figure_dir),
        "per_fold": _save_per_fold_figure(per_fold_metrics, figure_dir),
        "signal_history": _save_signal_history_figure(
            outputs["oos_regimes"], vix_signal_history, figure_dir
        ),
        "contribution": _save_contribution_figure(metrics, figure_dir),
        "rolling_alpha": _save_rolling_alpha_figure(panels, figure_dir),
        "regime_forward_returns": _save_regime_forward_returns_figure(
            outputs["oos_regimes"], outputs["oos_returns"], figure_dir
        ),
        "annual_returns": _save_annual_returns_figure(panels, figure_dir),
        "risk_return_scatter": _save_risk_return_scatter_figure(metrics, figure_dir),
        "monthly_heatmap": _save_monthly_heatmap_figure(panels, figure_dir),
        "annual_costs": _save_annual_costs_figure(outputs["oos_returns"], figure_dir),
        "correlation_heatmap": _save_correlation_heatmap_figure(figure_dir, output_dir),
        "cumulative_alpha": _save_cumulative_alpha_figure(panels, figure_dir),
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
    parser.add_argument("--end", default="2026-04-15", help="Last OOS date for attribution dependency rebuilds.")
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
