"""Smoke tests for the Task 8 reporting layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from taa_project.analysis.common import deflated_sharpe_ratio, disclosed_trial_count, sortino_ratio
from taa_project.analysis.reporting import _ips_compliance_rows
from taa_project.config import ALL_TAA, OPPORTUNISTIC, SAA_BANDS, TAA_AUDIT_BANDS, TAA_BANDS


def _holdings_frame(index: pd.DatetimeIndex, **weights: float) -> pd.DataFrame:
    row = pd.Series(0.0, index=ALL_TAA, dtype=float)
    for key, value in weights.items():
        row[key] = value
    return pd.DataFrame([row] * len(index), index=index)


def _returns(index: pd.DatetimeIndex, value: float = 0.0005) -> pd.Series:
    return pd.Series(value, index=index, dtype=float)


def _inception_dates(default: str = "2000-01-01", **overrides: str) -> pd.Series:
    dates = {asset: pd.Timestamp(default) for asset in ALL_TAA}
    dates.update({asset: pd.Timestamp(value) for asset, value in overrides.items()})
    return pd.Series(dates)


def test_ips_compliance_rows_accepts_compliant_schedule() -> None:
    index = pd.bdate_range("2024-02-01", periods=252)
    holdings = _holdings_frame(
        index,
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

    violations = _ips_compliance_rows(
        "SAA",
        holdings=holdings,
        returns=_returns(index),
        decision_dates=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
        inception_dates=_inception_dates(),
        band_map=SAA_BANDS,
    )

    assert violations == []


def test_ips_compliance_rows_flags_single_sleeve_breach() -> None:
    index = pd.bdate_range("2024-02-01", periods=30)
    holdings = _holdings_frame(index, SPXT=0.60, LBUSTRUU=0.40)

    violations = _ips_compliance_rows(
        "SAA",
        holdings=holdings,
        returns=_returns(index),
        decision_dates=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
        inception_dates=_inception_dates(),
        band_map=SAA_BANDS,
    )

    assert any(row["rule"] == "single_sleeve_cap" for row in violations)


def test_ips_compliance_rows_allows_inactive_asset_before_next_saa_rebalance() -> None:
    index = pd.bdate_range("2003-04-01", periods=40)
    holdings = _holdings_frame(
        index,
        SPXT=0.40,
        LBUSTRUU=0.10,
        BROAD_TIPS=0.05,
        XAU=0.20,
        SILVER_FUT=0.10,
        NIKKEI225=0.05,
        CSI300_CHINA=0.05,
        CHF_FRANC=0.05,
    )

    violations = _ips_compliance_rows(
        "SAA",
        holdings=holdings,
        returns=_returns(index),
        decision_dates=pd.DatetimeIndex([pd.Timestamp("2002-12-31"), pd.Timestamp("2003-12-31")]),
        inception_dates=_inception_dates(B3REITT="2003-03-31"),
        band_map=SAA_BANDS,
    )

    assert violations == []


def test_ips_compliance_rows_flags_per_asset_taa_band_breach() -> None:
    index = pd.bdate_range("2024-02-01", periods=30)
    holdings = _holdings_frame(
        index,
        SPXT=0.20,
        LBUSTRUU=0.10,
        BROAD_TIPS=0.05,
        XAU=0.35,
        B3REITT=0.10,
        NIKKEI225=0.05,
        CSI300_CHINA=0.05,
        CHF_FRANC=0.10,
    )

    violations = _ips_compliance_rows(
        "SAA+TAA",
        holdings=holdings,
        returns=_returns(index),
        decision_dates=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
        inception_dates=_inception_dates(),
        band_map=TAA_BANDS,
    )

    assert any(row["rule"] == "saa_taa_upper_XAU" for row in violations)


def test_ips_compliance_rows_flags_rolling_vol_breach() -> None:
    index = pd.bdate_range("2024-02-01", periods=63)
    holdings = _holdings_frame(
        index,
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
    returns = pd.Series(np.tile([0.02, -0.02, 0.018, -0.018, 0.021, -0.021, 0.019], 9)[: len(index)], index=index)

    violations = _ips_compliance_rows(
        "SAA+TAA",
        holdings=holdings,
        returns=returns,
        decision_dates=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
        inception_dates=_inception_dates(),
        band_map=TAA_BANDS,
    )

    assert any(row["rule"] == "rolling_vol_21d" for row in violations)


def test_ips_compliance_rows_accepts_opportunistic_hedge_sleeve() -> None:
    index = pd.bdate_range("2024-02-01", periods=63)
    holdings = _holdings_frame(
        index,
        SPXT=0.20,
        LBUSTRUU=0.28,
        BROAD_TIPS=0.24,
        CHF_FRANC=0.145,
        BCEE1T_EUROAREA=0.045,
        I02923JP_JAPAN_BOND=0.045,
        LBEATREU_EUROBONDAGG=0.045,
    )

    violations = _ips_compliance_rows(
        "SAA+TAA",
        holdings=holdings,
        returns=_returns(index),
        decision_dates=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
        inception_dates=_inception_dates(),
        band_map=TAA_AUDIT_BANDS,
    )

    assert violations == []
    assert holdings.loc[:, OPPORTUNISTIC].sum(axis=1).max() <= 0.15


def test_ips_compliance_rows_flags_drawdown_breach() -> None:
    index = pd.bdate_range("2024-02-01", periods=30)
    holdings = _holdings_frame(
        index,
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
    returns = pd.Series([-0.04] * 10 + [0.0] * 20, index=index, dtype=float)

    violations = _ips_compliance_rows(
        "SAA+TAA",
        holdings=holdings,
        returns=returns,
        decision_dates=pd.DatetimeIndex([pd.Timestamp("2024-01-31")]),
        inception_dates=_inception_dates(),
        band_map=TAA_BANDS,
    )

    assert any(row["rule"] == "max_drawdown" for row in violations)


def test_sortino_ratio_is_finite_for_mixed_return_series() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, -0.005, 0.008], index=pd.date_range("2024-01-01", periods=5))

    value = sortino_ratio(returns, mar=0.02)

    assert pd.notna(value)


def test_dsr_is_proper_probability() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    trial_ledger = pd.read_csv(repo_root / "TRIAL_LEDGER.csv")
    n_trials = disclosed_trial_count(trial_ledger)
    low_sharpe_returns = pd.Series([0.0001, -0.0002, 0.0000, 0.0002] * 64, dtype=float)
    high_sharpe_returns = pd.Series([0.0004, 0.0001, 0.0002, -0.0002] * 64, dtype=float)

    low_sharpe_dsr = deflated_sharpe_ratio(low_sharpe_returns, n_trials=n_trials)
    high_sharpe_dsr = deflated_sharpe_ratio(high_sharpe_returns, n_trials=n_trials)

    assert 0.0 < low_sharpe_dsr < 1.0
    assert 0.0 < high_sharpe_dsr < 1.0
    assert low_sharpe_dsr < high_sharpe_dsr - 0.05
