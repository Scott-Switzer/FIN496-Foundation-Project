"""Smoke tests for the Task 4 signal stack."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa_project.signals import SignalBundle
from taa_project.signals.momentum_adm import cross_sectional_rank, period_return
from taa_project.signals.regime_hmm import build_features, classify_states, fit_hmm
from taa_project.signals.trend_faber import trend_score
from taa_project.signals.vol_timesfm import TimesFMForecaster


def test_signal_bundle_holds_expected_fields() -> None:
    series = pd.Series({"SPXT": 0.1})
    bundle = SignalBundle(
        regime_probs=pd.Series({"p_risk_on": 0.7, "p_neutral": 0.2, "p_stress": 0.1}),
        regime_label="risk_on",
        trend=series,
        momo=series,
        timesfm_mu=series,
        timesfm_sigma=series,
        timesfm_dir=series,
    )

    assert bundle.regime_label == "risk_on"
    assert bundle.trend["SPXT"] == 0.1


def test_trend_score_respects_bounds_and_missing_dates() -> None:
    index = pd.date_range("2020-01-01", periods=320, freq="D")
    weekday_mask = index.weekday < 5
    prices = pd.DataFrame({"SPXT": np.nan}, index=index)
    observed_values = np.linspace(100.0, 140.0, int(weekday_mask.sum()))
    prices.loc[weekday_mask, "SPXT"] = observed_values

    score = trend_score(prices)

    assert score.loc[index[~weekday_mask], "SPXT"].isna().all()
    finite = score["SPXT"].dropna()
    assert not finite.empty
    assert finite.between(-1.0, 1.0).all()


def test_period_return_uses_observed_history_not_calendar_rows() -> None:
    index = pd.to_datetime(
        ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06", "2024-01-08", "2024-01-09"]
    )
    prices = pd.DataFrame({"FTSE100": [100.0, 101.0, 102.0, np.nan, 103.0, 104.0]}, index=index)

    returns = period_return(prices, periods=2)

    expected = 103.0 / 101.0 - 1.0
    assert np.isclose(returns.loc["2024-01-08", "FTSE100"], expected)
    assert pd.isna(returns.loc["2024-01-06", "FTSE100"])


def test_cross_sectional_rank_applies_absolute_momentum_guardrail() -> None:
    score = pd.DataFrame(
        {"SPXT": [0.05], "FTSE100": [-0.01], "NIKKEI225": [0.02]},
        index=[pd.Timestamp("2024-01-31")],
    )
    ranked = cross_sectional_rank(score, {"equity": ["SPXT", "FTSE100", "NIKKEI225"]})

    assert ranked.loc[:, "FTSE100"].iloc[0] <= 0.0
    assert ranked.loc[:, "SPXT"].iloc[0] >= ranked.loc[:, "NIKKEI225"].iloc[0]


def test_regime_hmm_fit_and_classify_smoke() -> None:
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2020-01-01", periods=320)
    fred = pd.DataFrame(
        {
            "VIXCLS": 20 + rng.normal(0, 1, len(index)).cumsum() * 0.05,
            "BAMLH0A0HYM2": 4 + rng.normal(0, 1, len(index)).cumsum() * 0.03,
            "T10Y3M": 1 + rng.normal(0, 1, len(index)).cumsum() * 0.02,
            "NFCI": rng.normal(0, 0.2, len(index)),
        },
        index=index,
    )

    features = build_features(fred)
    model = fit_hmm(features, n_states=3, min_observations=252)
    classified = classify_states(model, features)

    assert {"p_risk_on", "p_neutral", "p_stress", "regime"}.issubset(classified.columns)
    np.testing.assert_allclose(
        classified[["p_risk_on", "p_neutral", "p_stress"]].sum(axis=1).to_numpy(),
        np.ones(len(classified)),
        atol=1e-6,
    )


def test_timesfm_helper_transforms_are_finite() -> None:
    forecast = {
        "point": np.array([0.001, -0.0005, 0.0008]),
        "quantiles": np.array(
            [
                [0.001, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
                [0.0, -0.012, -0.006, -0.002, 0.002, 0.006, 0.01, 0.014, 0.018, 0.022],
                [0.0005, -0.011, -0.006, -0.001, 0.003, 0.007, 0.011, 0.015, 0.019, 0.023],
            ]
        ),
    }

    mu = TimesFMForecaster.expected_return(forecast, horizon_days=3)
    sigma = TimesFMForecaster.forecast_vol(forecast)
    direction = TimesFMForecaster.directional_score(forecast)

    assert np.isfinite(mu)
    assert np.isfinite(sigma)
    assert -1.0 <= direction <= 1.0
