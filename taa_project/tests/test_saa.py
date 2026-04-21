"""Smoke tests for the Task 2 SAA builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa_project.config import CORE, CORE_FLOOR, NONTRAD, NONTRAD_CAP, SATELLITE, SATELLITE_CAP
from taa_project.saa.build_saa import (
    bounds_for_assets,
    project_policy_targets_to_feasible_set,
    target_risk_budgets,
)


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
    # FTSE100 still has a 0% SAA target — should have zero risk budget.
    assert budgets["FTSE100"] == 0.0
    # BITCOIN now has a 2% SAA target (Amendment 2026-03), so its risk
    # budget should be positive.
    assert budgets["BITCOIN"] > 0.0
    assert budgets["SPXT"] > 0.0
