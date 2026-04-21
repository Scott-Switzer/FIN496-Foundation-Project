"""Tests for the shared TimesFM parquet cache."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from taa_project.signals.vol_timesfm import (
    DEFAULT_MODEL_VERSION,
    get_or_compute_timesfm_quantiles,
)


class FakeForecaster:
    def __init__(self) -> None:
        self.calls = 0

    def forecast_quantiles(
        self,
        price_history: pd.Series,
        horizon: int = 64,
        context_length: int = 1024,
        min_context: int = 64,
    ) -> dict[str, float]:
        del price_history, horizon, context_length, min_context
        self.calls += 1
        return {
            "q05": -0.20,
            "q10": -0.15,
            "q25": -0.05,
            "q50": 0.02,
            "q75": 0.09,
            "q90": 0.15,
            "q95": 0.20,
        }


def _price_history() -> pd.Series:
    index = pd.date_range("2022-01-01", periods=180, freq="B")
    values = pd.Series(100.0 + np.arange(len(index), dtype=float), index=index, dtype=float)
    return values


def test_cache_miss_then_hit(tmp_path) -> None:
    cache_path = tmp_path / "timesfm_forecasts.parquet"
    history = _price_history()
    decision_date = pd.Timestamp("2022-08-15")
    forecaster = FakeForecaster()

    first = get_or_compute_timesfm_quantiles(
        asset="SPXT",
        decision_date=decision_date,
        price_history=history,
        forecaster=forecaster,
        cache_path=cache_path,
    )
    second = get_or_compute_timesfm_quantiles(
        asset="SPXT",
        decision_date=decision_date,
        price_history=history,
        forecaster=None,
        cache_path=cache_path,
    )

    for key, value in first.items():
        assert second[key] == pytest.approx(value)
    assert forecaster.calls == 1
    assert cache_path.exists()


def test_cache_key_includes_all_params(tmp_path) -> None:
    cache_path = tmp_path / "timesfm_forecasts.parquet"
    history = _price_history()
    decision_date = pd.Timestamp("2022-08-15")
    forecaster = FakeForecaster()

    get_or_compute_timesfm_quantiles(
        asset="SPXT",
        decision_date=decision_date,
        price_history=history,
        forecaster=forecaster,
        horizon=64,
        model_version=DEFAULT_MODEL_VERSION,
        cache_path=cache_path,
    )

    with pytest.raises(RuntimeError, match="Cache miss"):
        get_or_compute_timesfm_quantiles(
            asset="SPXT",
            decision_date=decision_date,
            price_history=history,
            forecaster=None,
            horizon=32,
            model_version=DEFAULT_MODEL_VERSION,
            cache_path=cache_path,
        )


def test_cache_hit_does_not_load_full_parquet(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "timesfm_forecasts.parquet"
    history = _price_history()
    decision_date = pd.Timestamp("2022-08-15")
    forecaster = FakeForecaster()

    get_or_compute_timesfm_quantiles(
        asset="SPXT",
        decision_date=decision_date,
        price_history=history,
        forecaster=forecaster,
        cache_path=cache_path,
    )

    calls: list[object] = []
    original_read_table = pq.read_table

    def wrapped_read_table(*args, **kwargs):
        calls.append(kwargs.get("filters"))
        return original_read_table(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", wrapped_read_table)

    observed = get_or_compute_timesfm_quantiles(
        asset="SPXT",
        decision_date=decision_date,
        price_history=history,
        forecaster=None,
        cache_path=cache_path,
    )

    assert observed["q50"] == pytest.approx(0.02)
    assert calls
    assert calls[-1] == [("cache_key", "=", "SPXT|2022-08-15|1024|64|google/timesfm-2.5-200m-pytorch|timesfm")]
