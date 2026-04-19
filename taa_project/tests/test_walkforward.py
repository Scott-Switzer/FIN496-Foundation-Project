"""Smoke tests for the Task 6 walk-forward backtester."""

from __future__ import annotations

import pandas as pd

from taa_project.backtest.walkforward import (
    build_monthly_decision_dates,
    build_walkforward_folds,
    run_walkforward,
)
from taa_project.data_loader import load_prices


def test_build_monthly_decision_dates_uses_actual_last_trading_day() -> None:
    prices = pd.DataFrame(
        {"SPXT": [100.0, 101.0, 102.0, 103.0, float("nan")]},
        index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-28", "2024-02-29", "2024-03-31"]),
    )
    dates = build_monthly_decision_dates(prices, "2024-01-01", "2024-02-29")

    assert dates.tolist() == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]


def test_build_walkforward_folds_creates_contiguous_test_blocks() -> None:
    decision_dates = pd.date_range("2023-01-31", periods=10, freq="M")
    folds = build_walkforward_folds(
        decision_dates=pd.DatetimeIndex(decision_dates),
        train_start=pd.Timestamp("2021-01-01"),
        folds=5,
        embargo_business_days=21,
    )

    assert len(folds) == 5
    assert folds[0].test_start == pd.Timestamp("2023-01-31")
    assert folds[-1].test_end == pd.Timestamp("2023-10-31")
    assert all(spec.test_start <= spec.test_end for spec in folds)


def test_run_walkforward_smoke(tmp_path) -> None:
    prices = load_prices()
    if prices.loc["2003-01-01":"2003-06-30"].empty:
        raise AssertionError("Expected repo price history to cover the 2003 smoke window.")

    artifacts = run_walkforward(
        start="2003-01-01",
        end="2003-06-30",
        folds=2,
        embargo_business_days=21,
        use_timesfm=False,
        output_dir=tmp_path,
    )

    assert not artifacts["folds"].empty
    assert not artifacts["oos_returns"].empty
    assert not artifacts["oos_weights"].empty
    assert not artifacts["oos_regimes"].empty
