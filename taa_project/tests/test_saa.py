"""Smoke tests for the Task 2 SAA builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa_project.config import CORE, CORE_FLOOR, NONTRAD, NONTRAD_CAP, SATELLITE, SATELLITE_CAP
from taa_project.saa.build_saa import (
    ALL_SAA,
    bounds_for_assets,
    build_rebalance_schedule,
    compute_target_weights,
    first_valid_dates,
    load_saa_prices,
    project_policy_targets_to_feasible_set,
    simulate_portfolio,
    target_risk_budgets,
    violates_saa_constraints,
)
from taa_project.data_loader import log_returns


def test_policy_target_projection_respects_ips_constraints() -> None:
    assets = ["SPXT", "LBUSTRUU", "BROAD_TIPS", "XAU", "NIKKEI225", "SILVER_FUT", "CHF_FRANC"]
    lower_bounds, upper_bounds = bounds_for_assets(assets)

    weights = pd.Series(project_policy_targets_to_feasible_set(lower_bounds, upper_bounds, assets), index=assets)

    assert np.isclose(weights.sum(), 1.0)
    assert bool((weights >= lower_bounds - 1e-10).all())
    assert bool((weights <= upper_bounds + 1e-10).all())
    assert weights.reindex(CORE).fillna(0.0).sum() >= CORE_FLOOR - 1e-10
    assert weights.reindex(SATELLITE).fillna(0.0).sum() <= SATELLITE_CAP + 1e-10
    assert weights.reindex(NONTRAD).fillna(0.0).sum() <= NONTRAD_CAP + 1e-10


def test_zero_target_assets_receive_zero_risk_budget() -> None:
    budgets = target_risk_budgets(["SPXT", "FTSE100", "BITCOIN", "XAU"])

    assert np.isclose(budgets.sum(), 1.0)
    # FTSE100 has a 0% SAA target — should have zero risk budget.
    assert budgets["FTSE100"] == 0.0
    # BITCOIN reverted to 0% SAA target (Amendment 2026-03 rolled back 2026-04)
    # — should also have zero risk budget.
    assert budgets["BITCOIN"] == 0.0
    assert budgets["SPXT"] > 0.0


def test_compute_target_weights_supports_min_variance() -> None:
    prices = load_saa_prices().loc[: "2006-12-31"]
    returns = log_returns(prices)
    inception_dates = first_valid_dates(prices)
    rebalance_date = build_rebalance_schedule(prices, "2000-01-01", "2006-12-31")[0]

    weights = compute_target_weights(
        prices=prices,
        returns=returns,
        rebalance_date=rebalance_date,
        inception_dates=inception_dates,
        method="min_variance",
    )

    assert np.isclose(float(weights.sum()), 1.0)
    assert bool((weights >= -1e-10).all())


def test_simulate_portfolio_triggers_compliance_rebalance() -> None:
    start = pd.Timestamp("2000-01-03")
    next_day = pd.Timestamp("2000-01-04")
    active_assets = ["SPXT", "LBUSTRUU", "BROAD_TIPS", "B3REITT", "XAU", "CHF_FRANC"]
    lower_bounds, upper_bounds = bounds_for_assets(active_assets)
    initial = pd.Series(
        project_policy_targets_to_feasible_set(lower_bounds, upper_bounds, active_assets),
        index=active_assets,
    )
    target = pd.Series(0.0, index=ALL_SAA, dtype=float)
    target.loc[active_assets] = initial

    returns = pd.DataFrame(0.0, index=[start, next_day], columns=ALL_SAA)
    returns.loc[next_day, "XAU"] = np.log(3.0)

    weights_df, returns_df = simulate_portfolio(
        returns=returns,
        rebalance_targets={start: target},
        rebalance_active_assets={start: active_assets},
        start_date=start,
        end_date=next_day,
    )

    assert int(returns_df.loc[next_day, "compliance_rebalance_flag"]) == 1
    assert not violates_saa_constraints(weights_df.loc[next_day], active_assets)
