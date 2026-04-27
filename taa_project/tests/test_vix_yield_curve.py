"""Tests for the VIX + yield-curve macro trip-wire."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa_project.config import ALL_SAA
from taa_project.signals.vix_yield_curve import (
    CURVE_COLUMN,
    VIX_COLUMN,
    VixYieldCurveConfig,
    vix_yield_curve_diagnostics,
    vix_yield_curve_tilt,
)


def _fred_panel(spike_date: pd.Timestamp | None = None) -> pd.DataFrame:
    index = pd.bdate_range("2022-01-03", periods=320)
    phase = np.linspace(0.0, 8.0 * np.pi, len(index))
    vix = 20.0 + 1.5 * np.sin(phase)
    curve = np.full(len(index), 0.75)
    frame = pd.DataFrame({VIX_COLUMN: vix, CURVE_COLUMN: curve}, index=index)
    if spike_date is not None:
        frame.loc[pd.Timestamp(spike_date), VIX_COLUMN] = 45.0
    return frame


def test_vix_yield_curve_tilt_is_indexed_to_all_saa_and_bounded() -> None:
    fred = _fred_panel(spike_date=pd.Timestamp("2023-03-24"))
    as_of = pd.Timestamp("2023-03-24")

    tilt = vix_yield_curve_tilt(fred, as_of)

    assert tilt.index.tolist() == ALL_SAA
    assert tilt.notna().all()
    assert tilt.between(-1.0, 1.0).all()


def test_vix_yield_curve_uses_only_rows_up_to_as_of_date() -> None:
    as_of = pd.Timestamp("2023-02-28")
    fred = _fred_panel()
    mutated_future = fred.copy()
    mutated_future.loc[mutated_future.index > as_of, VIX_COLUMN] = 90.0
    mutated_future.loc[mutated_future.index > as_of, CURVE_COLUMN] = -5.0

    baseline = vix_yield_curve_tilt(fred, as_of)
    after_future_mutation = vix_yield_curve_tilt(mutated_future, as_of)
    base_diag = vix_yield_curve_diagnostics(fred, as_of)
    mutated_diag = vix_yield_curve_diagnostics(mutated_future, as_of)

    pd.testing.assert_series_equal(baseline, after_future_mutation)
    assert base_diag["risk_score"] == mutated_diag["risk_score"]
    assert base_diag["vix_z"] == mutated_diag["vix_z"]
    assert base_diag["curve_level"] == mutated_diag["curve_level"]


def test_high_vix_spike_creates_defensive_asset_tilts() -> None:
    as_of = pd.Timestamp("2023-03-24")
    fred = _fred_panel(spike_date=as_of)

    diagnostics = vix_yield_curve_diagnostics(fred, as_of)
    tilt = vix_yield_curve_tilt(fred, as_of)

    assert diagnostics["risk_score"] < 0.0
    assert tilt["SPXT"] < 0.0
    assert tilt["LBUSTRUU"] > 0.0
    assert tilt["CHF_FRANC"] > 0.0


def test_yield_curve_inversion_only_haircuts_positive_risk_score() -> None:
    as_of = pd.Timestamp("2023-03-24")
    calm_fred = _fred_panel()
    calm_fred.loc[as_of, VIX_COLUMN] = 12.0
    inverted_fred = calm_fred.copy()
    inverted_fred.loc[as_of, CURVE_COLUMN] = -1.50

    config = VixYieldCurveConfig(min_observations=126)
    calm = vix_yield_curve_diagnostics(calm_fred, as_of, config=config)
    inverted = vix_yield_curve_diagnostics(inverted_fred, as_of, config=config)

    assert calm["base_risk_score"] > 0.0
    assert 0.0 < inverted["risk_score"] < calm["risk_score"]
    assert inverted["curve_penalty"] < calm["curve_penalty"]


def test_missing_required_fred_columns_returns_neutral_tilt() -> None:
    fred = pd.DataFrame({"OTHER": [1.0, 2.0]}, index=pd.bdate_range("2024-01-02", periods=2))

    tilt = vix_yield_curve_tilt(fred, pd.Timestamp("2024-01-03"))

    assert tilt.index.tolist() == ALL_SAA
    assert float(tilt.abs().sum()) == 0.0


def test_ten_percent_blend_keeps_signal_scale_stable() -> None:
    fred = _fred_panel(spike_date=pd.Timestamp("2023-03-24"))
    as_of = pd.Timestamp("2023-03-24")
    base_signal = pd.Series(0.02, index=ALL_SAA, dtype=float)
    vix_tilt = vix_yield_curve_tilt(fred, as_of)

    blended = 0.90 * base_signal + 0.10 * vix_tilt

    assert blended.index.tolist() == ALL_SAA
    assert blended.notna().all()
    assert float((blended - base_signal).abs().max()) <= 0.10
    assert float(blended.abs().max()) <= 0.10
