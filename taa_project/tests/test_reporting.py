"""Smoke tests for the Task 8 reporting layer."""

from __future__ import annotations

import pandas as pd

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
