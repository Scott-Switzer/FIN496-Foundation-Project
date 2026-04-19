"""Smoke tests for the Task 3 benchmark builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa_project.benchmarks import find_first_benchmark_start_date, normalize_target_weights


def test_normalize_target_weights_reindexes_to_full_saa_universe() -> None:
    weights = normalize_target_weights({"SPXT": 0.6, "LBUSTRUU": 0.4})

    assert np.isclose(weights.sum(), 1.0)
    assert weights["SPXT"] == 0.6
    assert weights["LBUSTRUU"] == 0.4
    assert weights.drop(index=["SPXT", "LBUSTRUU"]).eq(0.0).all()


def test_find_first_benchmark_start_date_requires_all_positive_weight_assets() -> None:
    index = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-01-05"])
    prices = pd.DataFrame(
        {
            "SPXT": [100.0, 101.0, 102.0],
            "LBUSTRUU": [np.nan, 200.0, 201.0],
        },
        index=index,
    )
    weights = pd.Series({"SPXT": 0.6, "LBUSTRUU": 0.4})

    start = find_first_benchmark_start_date(prices, weights, "2000-01-01")

    assert start == pd.Timestamp("2000-01-04")
