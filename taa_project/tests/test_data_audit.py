"""Smoke tests for the Task 1 data-audit logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa_project.data_audit import build_availability_flags, compute_consecutive_log_returns, load_fred_features


def test_consecutive_log_returns_use_adjacent_observed_prices() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"ASSET": [100.0, np.nan, 121.0, 133.1]}, index=dates)

    returns = compute_consecutive_log_returns(prices)

    assert np.isnan(returns.loc[dates[0], "ASSET"])
    assert np.isnan(returns.loc[dates[1], "ASSET"])
    assert np.isclose(returns.loc[dates[2], "ASSET"], np.log(121.0 / 100.0))
    assert np.isclose(returns.loc[dates[3], "ASSET"], np.log(133.1 / 121.0))


def test_availability_flags_match_price_presence() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [1.0, np.nan, 2.0], "B": [np.nan, 3.0, 4.0]}, index=dates)

    flags = build_availability_flags(prices)

    assert flags.loc[dates[0], "A"] == 1
    assert flags.loc[dates[1], "A"] == 0
    assert flags.loc[dates[0], "B"] == 0
    assert flags.loc[dates[2], "B"] == 1


def test_fred_lag_applies_one_business_day_before_alignment(tmp_path) -> None:
    fred = pd.DataFrame(
        {"VIXCLS": [10.0, 20.0, 30.0]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    path = tmp_path / "fred.csv"
    fred.to_csv(path)
    calendar = pd.date_range("2024-01-02", "2024-01-06", freq="D")

    lagged = load_fred_features(path=path, calendar_index=calendar, lag_business_days=1)

    assert np.isnan(lagged.loc["2024-01-02", "VIXCLS"])
    assert lagged.loc["2024-01-03", "VIXCLS"] == 10.0
    assert lagged.loc["2024-01-04", "VIXCLS"] == 20.0
    assert lagged.loc["2024-01-05", "VIXCLS"] == 30.0
    assert lagged.loc["2024-01-06", "VIXCLS"] == 30.0
