"""Smoke tests for the Task 7 attribution layer."""

from __future__ import annotations

import pandas as pd

from taa_project.analysis.attribution import _aggregate_active_contribution


def test_aggregate_active_contribution_sums_assets_tiers_and_total() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    active_weights = pd.DataFrame(
        {
            "SPXT": [0.10, 0.10],
            "LBUSTRUU": [-0.10, -0.10],
        },
        index=dates,
    )
    asset_returns = pd.DataFrame(
        {
            "SPXT": [0.01, 0.02],
            "LBUSTRUU": [0.005, -0.005],
        },
        index=dates,
    )

    result = _aggregate_active_contribution("test_case", active_weights, asset_returns)

    total_row = result.loc[result["grouping"] == "total", "total_contribution"].iloc[0]
    asset_sum = result.loc[result["grouping"] == "asset", "total_contribution"].sum()
    assert abs(total_row - asset_sum) < 1e-12
    assert set(result["grouping"]) == {"asset", "tier", "total"}
