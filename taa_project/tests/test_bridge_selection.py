"""Tests for the focused bridge-sweep ranking logic."""

from __future__ import annotations

import pandas as pd

from taa_project.analysis.bridge_comparison import rank_bridge_candidates, select_bridge_candidate


def _comparison_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_rank_bridge_candidates_prefers_feasible_high_dsr() -> None:
    comparison = _comparison_frame(
        [
            {
                "run_id": "feasible_low",
                "Run": "Feasible Low",
                "Family": "Nested",
                "Ann. Return": 0.081,
                "Ann. Vol": 0.070,
                "Max DD": -0.240,
                "Deflated Sharpe": 0.82,
                "Pass MDD": "Y",
                "Pass Vol": "Y",
                "Pass Return": "Y",
                "Feasible": "Y",
                "MDD Headroom (bps)": 100.0,
                "Return Gap (bps)": 10.0,
                "IPS Gap (bps)": 0.0,
            },
            {
                "run_id": "feasible_high",
                "Run": "Feasible High",
                "Family": "Nested + BL",
                "Ann. Return": 0.082,
                "Ann. Vol": 0.071,
                "Max DD": -0.241,
                "Deflated Sharpe": 0.90,
                "Pass MDD": "Y",
                "Pass Vol": "Y",
                "Pass Return": "Y",
                "Feasible": "Y",
                "MDD Headroom (bps)": 90.0,
                "Return Gap (bps)": 20.0,
                "IPS Gap (bps)": 0.0,
            },
            {
                "run_id": "mdd_only",
                "Run": "MDD Only",
                "Family": "Nested",
                "Ann. Return": 0.078,
                "Ann. Vol": 0.068,
                "Max DD": -0.238,
                "Deflated Sharpe": 0.95,
                "Pass MDD": "Y",
                "Pass Vol": "Y",
                "Pass Return": "N",
                "Feasible": "N",
                "MDD Headroom (bps)": 120.0,
                "Return Gap (bps)": -20.0,
                "IPS Gap (bps)": 20.0,
            },
        ]
    )

    ranked = rank_bridge_candidates(comparison)
    selection = select_bridge_candidate(ranked)

    assert ranked.iloc[0]["run_id"] == "feasible_high"
    assert selection["run_id"] == "feasible_high"
    assert selection["selection_rule"] == "feasible_highest_dsr"


def test_rank_bridge_candidates_prefers_smaller_gap_when_none_feasible() -> None:
    comparison = _comparison_frame(
        [
            {
                "run_id": "closer",
                "Run": "Closer",
                "Family": "Nested",
                "Ann. Return": 0.0795,
                "Ann. Vol": 0.069,
                "Max DD": -0.244,
                "Deflated Sharpe": 0.70,
                "Pass MDD": "Y",
                "Pass Vol": "Y",
                "Pass Return": "N",
                "Feasible": "N",
                "MDD Headroom (bps)": 60.0,
                "Return Gap (bps)": -5.0,
                "IPS Gap (bps)": 5.0,
            },
            {
                "run_id": "farther",
                "Run": "Farther",
                "Family": "Nested + BL",
                "Ann. Return": 0.076,
                "Ann. Vol": 0.068,
                "Max DD": -0.239,
                "Deflated Sharpe": 0.85,
                "Pass MDD": "Y",
                "Pass Vol": "Y",
                "Pass Return": "N",
                "Feasible": "N",
                "MDD Headroom (bps)": 110.0,
                "Return Gap (bps)": -40.0,
                "IPS Gap (bps)": 40.0,
            },
        ]
    )

    ranked = rank_bridge_candidates(comparison)
    selection = select_bridge_candidate(ranked)

    assert ranked.iloc[0]["run_id"] == "closer"
    assert selection["run_id"] == "closer"
    assert selection["selection_rule"] == "closest_to_ips"
