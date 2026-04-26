"""Smoke tests for the Task 6 walk-forward backtester."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taa_project.backtest import walkforward as walkforward_module
from taa_project.backtest.walkforward import (
    build_monthly_decision_dates,
    build_walkforward_folds,
    estimate_taa_covariance,
    run_walkforward,
    simulate_period_returns,
)
from taa_project.config import ALL_SAA
from taa_project.data_loader import load_prices
from taa_project.optimizer.cvxpy_opt import EnsembleConfig, OptimizationResult
from taa_project.signals import SignalBundle


def test_build_monthly_decision_dates_uses_actual_last_trading_day() -> None:
    prices = pd.DataFrame(
        {"SPXT": [100.0, 101.0, 102.0, 103.0, float("nan")]},
        index=pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-28", "2024-02-29", "2024-03-31"]),
    )
    dates = build_monthly_decision_dates(prices, "2024-01-01", "2024-02-29")

    assert dates.tolist() == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]


def test_build_walkforward_folds_creates_contiguous_test_blocks() -> None:
    decision_dates = pd.date_range("2023-01-31", periods=10, freq="ME")
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


def test_run_walkforward_smoke(monkeypatch, tmp_path) -> None:
    prices = load_prices()
    if prices.loc["2003-01-01":"2003-06-30"].empty:
        raise AssertionError("Expected repo price history to cover the 2003 smoke window.")

    def fake_build_signal_bundle_at_date(**kwargs):
        zero = pd.Series(0.0, index=ALL_SAA, dtype=float)
        probs = pd.Series({"risk_on": 0.7, "neutral": 0.2, "stress": 0.1}, dtype=float)
        bundle = SignalBundle(
            regime_probs=probs,
            regime_label="risk_on",
            trend=zero,
            momo=zero,
            timesfm_mu=zero,
            timesfm_sigma=pd.Series(0.2, index=ALL_SAA, dtype=float),
            timesfm_dir=zero,
        )
        return bundle, kwargs.get("hmm_model_cache")

    monkeypatch.setattr(walkforward_module, "build_signal_bundle_at_date", fake_build_signal_bundle_at_date)

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
    assert not artifacts["oos_holdings"].empty
    assert not artifacts["oos_regimes"].empty


def test_run_walkforward_rejects_invalid_vol_budget(tmp_path) -> None:
    with pytest.raises(ValueError, match="vol_budget"):
        run_walkforward(
            start="2003-01-01",
            end="2003-06-30",
            folds=2,
            embargo_business_days=21,
            use_timesfm=False,
            vol_budget=0.16,
            output_dir=tmp_path,
        )


def test_run_walkforward_raises_when_timesfm_requested_but_unavailable(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(walkforward_module, "timesfm_is_available", lambda: False)

    with pytest.raises(RuntimeError, match="TimesFM was requested"):
        run_walkforward(
            start="2003-01-01",
            end="2003-06-30",
            folds=2,
            embargo_business_days=21,
            use_timesfm=True,
            output_dir=tmp_path,
        )


def test_run_walkforward_risk_score_expands_vol_budget(monkeypatch, tmp_path) -> None:
    """Vol budget rises above base when risk_score > 0 (risk-on signal)."""
    captured: list[float] = []

    def fake_build_signal_bundle_at_date(**kwargs):
        zero = pd.Series(0.0, index=ALL_SAA, dtype=float)
        probs = pd.Series({"p_risk_on": 0.8, "p_neutral": 0.1, "p_stress": 0.1}, dtype=float)
        # risk_score = 0.5*1.0 + 0.3*1.0 - 0.2*0.1 = 0.78 (risk-on)
        bundle = SignalBundle(
            regime_probs=probs,
            risk_score=0.78,
            trend=zero,
            momo=zero,
            timesfm_mu=zero,
            timesfm_sigma=pd.Series(0.2, index=ALL_SAA, dtype=float),
            timesfm_dir=zero,
        )
        return bundle, kwargs.get("hmm_model_cache")

    def fake_solve_taa_monthly_result(**kwargs):
        captured.append(float(kwargs["vol_budget"]))
        weights = pd.Series(0.0, index=ALL_SAA, dtype=float)
        weights["SPXT"] = 0.40
        weights["LBUSTRUU"] = 0.60
        return OptimizationResult(
            weights=weights,
            mode="taa_monthly",
            status="optimal",
            used_fallback=False,
            turnover=0.0,
            turnover_cost=0.0,
            ex_ante_vol=float(kwargs["vol_budget"]),
        )

    monkeypatch.setattr(walkforward_module, "build_signal_bundle_at_date", fake_build_signal_bundle_at_date)
    monkeypatch.setattr(walkforward_module, "solve_taa_monthly_result", fake_solve_taa_monthly_result)

    run_walkforward(
        start="2003-01-01",
        end="2003-06-30",
        folds=2,
        embargo_business_days=21,
        use_timesfm=False,
        vol_budget=0.10,
        output_dir=tmp_path,
    )

    assert captured
    # risk_score=0.78 should expand vol budget above the 0.10 base
    assert all(vb > 0.10 for vb in captured), f"Expected vol budgets > 0.10, got {captured}"
    assert all(vb <= 0.15 for vb in captured), f"Vol budget must not exceed VOL_CEILING=0.15, got {captured}"


def test_estimate_taa_covariance_annualizes_historical_variance() -> None:
    index = pd.bdate_range("2024-01-01", periods=80)
    returns = pd.DataFrame(0.0, index=index, columns=ALL_SAA, dtype=float)
    returns["SPXT"] = np.tile([0.03, -0.03], len(index) // 2)
    covariance = estimate_taa_covariance(returns, index[-1], lookback_days=len(index), min_observations=2)

    expected = float(returns["SPXT"].var() * 252.0)
    assert np.isclose(float(covariance.loc["SPXT", "SPXT"]), expected, rtol=1e-6)


def test_simulate_period_returns_executes_compliance_rebalance_after_drift_breach(tmp_path) -> None:
    period_dates = pd.bdate_range("2024-02-01", periods=2)
    returns = pd.DataFrame(0.0, index=period_dates, columns=ALL_SAA, dtype=float)
    returns.loc[period_dates[0], "BITCOIN"] = np.log(2.5)

    starting_weights = pd.Series(0.0, index=ALL_SAA, dtype=float)
    starting_weights["SPXT"] = 0.40
    starting_weights["LBUSTRUU"] = 0.10
    starting_weights["BROAD_TIPS"] = 0.05
    starting_weights["B3REITT"] = 0.10
    starting_weights["XAU"] = 0.15
    starting_weights["SILVER_FUT"] = 0.05
    starting_weights["NIKKEI225"] = 0.05
    starting_weights["BITCOIN"] = 0.05
    starting_weights["CHF_FRANC"] = 0.05

    available = pd.Series(1.0, index=ALL_SAA, dtype=float)
    period_returns, holdings, ending_weights = simulate_period_returns(
        returns=returns,
        period_dates=period_dates,
        starting_weights=starting_weights,
        initial_turnover=0.0,
        initial_turnover_cost=0.0,
        available_assets=available,
        fold_id=1,
        decision_date=pd.Timestamp("2024-01-31"),
        regime_label="risk_on",
        compliance_log_path=tmp_path / "taa_compliance_rebalances.csv",
    )

    assert int(period_returns.loc[period_dates[0], "compliance_rebalance_flag"]) == 1
    assert holdings.loc[period_dates[0], "BITCOIN"] == pytest.approx(0.05)
    assert holdings.loc[period_dates[1], "BITCOIN"] <= 0.10 + 1e-8
    assert ending_weights["BITCOIN"] <= 0.10 + 1e-8
    log = pd.read_csv(tmp_path / "taa_compliance_rebalances.csv")
    assert {"date", "decision_date", "rule", "pre_trade_value", "bound", "turnover"}.issubset(log.columns)
    assert "upper_bound" in set(log["rule"])
