"""Smoke tests for the Task 4 signal stack."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa_project.signals import SignalBundle
from taa_project.signals.momentum_adm import cross_sectional_rank, period_return
from taa_project.signals.regime_hmm import FittedRegimeModel, MACRO_FEATURES, build_features, classify_states
from taa_project.signals.trend_faber import trend_score


def test_signal_bundle_holds_expected_fields() -> None:
    series = pd.Series({"SPXT": 0.1})
    bundle = SignalBundle(
        regime_probs=pd.Series({"p_risk_on": 0.7, "p_neutral": 0.2, "p_stress": 0.1}),
        regime_label="risk_on",
        trend=series,
        momo=series,
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
    feature_mean = features.mean()
    feature_std = features.std(ddof=0).replace(0.0, np.nan).fillna(1.0)

    class DummyModel:
        n_components = 3
        means_ = np.array(
            [
                [-0.5, -0.4, 0.3, -0.4],
                [0.0, 0.0, 0.0, 0.0],
                [0.7, 0.6, -0.5, 0.8],
            ]
        )

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            base = np.tile(np.array([[0.2, 0.5, 0.3]]), (len(X), 1))
            base[:, 0] += np.clip(-X[:, 0], 0.0, 0.15)
            base[:, 2] += np.clip(X[:, 0], 0.0, 0.15)
            base[:, 1] = 1.0
            base[:, 1] -= base[:, 0] + base[:, 2]
            return base

    model = FittedRegimeModel(
        model=DummyModel(),
        feature_columns=tuple(MACRO_FEATURES),
        feature_mean=feature_mean,
        feature_std=feature_std,
        state_names={0: "risk_on", 1: "neutral", 2: "stress"},
    )
    classified = classify_states(model, features)

    assert {"p_risk_on", "p_neutral", "p_stress", "regime"}.issubset(classified.columns)
    np.testing.assert_allclose(
        classified[["p_risk_on", "p_neutral", "p_stress"]].sum(axis=1).to_numpy(),
        np.ones(len(classified)),
        atol=1e-6,
    )
