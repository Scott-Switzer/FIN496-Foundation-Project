"""Smoke tests for the Task 8 reporting layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from taa_project.analysis.common import deflated_sharpe_ratio, disclosed_trial_count, sortino_ratio
from taa_project.analysis.reporting import _ips_compliance_rows
from taa_project.config import ALL_SAA


def _weight_row(**weights: float) -> pd.DataFrame:
    row = pd.Series(0.0, index=ALL_SAA, dtype=float)
    for key, value in weights.items():
        row[key] = value
    return pd.DataFrame([row], index=pd.to_datetime(["2024-01-31"]))


def test_ips_compliance_rows_accepts_compliant_schedule() -> None:
    weights = _weight_row(
        SPXT=0.40,
        LBUSTRUU=0.10,
        BROAD_TIPS=0.05,
        B3REITT=0.10,
        XAU=0.15,
        SILVER_FUT=0.05,
        NIKKEI225=0.05,
        CSI300_CHINA=0.05,
        CHF_FRANC=0.05,
    )

    violations = _ips_compliance_rows("strategy", weights)

    assert violations == []


def test_ips_compliance_rows_flags_single_sleeve_breach() -> None:
    weights = _weight_row(SPXT=0.60, LBUSTRUU=0.40)

    violations = _ips_compliance_rows("strategy", weights)

    assert any(row["rule"] == "single_sleeve_cap" for row in violations)


def test_sortino_ratio_is_finite_for_mixed_return_series() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.008], index=pd.date_range("2024-01-01", periods=5))

    value = sortino_ratio(returns, mar=0.02)

    assert pd.notna(value)


def test_dsr_is_proper_probability() -> None:
    repo_root = Path("/Users/scottthomasswitzer/Desktop/FIN496FP/FIN496-Foundation-Project")
    trial_ledger = pd.read_csv(repo_root / "TRIAL_LEDGER.csv")
    n_trials = disclosed_trial_count(trial_ledger)
    bm1_returns = pd.read_csv(repo_root / "taa_project/outputs/runs/baseline/outputs/bm1_returns.csv")["portfolio_return"]
    baseline_returns = pd.read_csv(repo_root / "taa_project/outputs/runs/baseline/outputs/oos_returns.csv")["portfolio_return"]

    bm1_dsr = deflated_sharpe_ratio(bm1_returns, n_trials=n_trials)
    baseline_dsr = deflated_sharpe_ratio(baseline_returns, n_trials=n_trials)

    assert 0.0 < bm1_dsr < 1.0
    assert 0.0 < baseline_dsr < 1.0
    assert bm1_dsr < baseline_dsr - 0.05
