"""Unit tests for the Task 5 drawdown guardrail."""

from __future__ import annotations

import pandas as pd

from taa_project.signals.dd_guardrail import DrawdownGuardrailConfig, dd_guardrail_multiplier


def test_dd_guardrail_tightens_then_releases_once() -> None:
    returns = pd.Series(
        [0.0] * 5 + [-0.20] + [0.03] * 10 + [0.0] * 8,
        index=pd.date_range("2024-01-01", periods=24, freq="B"),
        dtype=float,
    )
    config = DrawdownGuardrailConfig(
        trigger_dd=-0.15,
        lookback_days=5,
        tightening_factor=0.5,
        release_dd=-0.05,
        min_days_between_switches=2,
    )

    multiplier = dd_guardrail_multiplier(returns, config)
    switches = multiplier.diff().fillna(0.0)

    assert (switches < 0).sum() == 1
    assert (switches > 0).sum() == 1
    assert multiplier.min() == 0.5
    assert multiplier.iloc[-1] == 1.0


def test_dd_guardrail_is_causal_under_truncation() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.0, -0.01, 0.015, -0.03, 0.02, 0.01, -0.005, 0.0],
        index=pd.date_range("2024-02-01", periods=10, freq="B"),
        dtype=float,
    )
    config = DrawdownGuardrailConfig(
        trigger_dd=-0.04,
        lookback_days=4,
        tightening_factor=0.5,
        release_dd=-0.01,
        min_days_between_switches=2,
    )

    full = dd_guardrail_multiplier(returns, config)
    for date in returns.index:
        truncated = dd_guardrail_multiplier(returns.loc[:date], config)
        assert truncated.iloc[-1] == full.loc[date]
