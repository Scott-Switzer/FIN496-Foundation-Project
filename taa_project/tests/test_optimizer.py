"""Smoke tests for the Task 5 optimizer layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa_project.config import ALL_SAA, ALL_TAA, BM2_WEIGHTS, COST_PER_TURNOVER
from taa_project.optimizer.cvxpy_opt import (
    BREACH_LOG_PATH,
    EnsembleConfig,
    cp,
    ensemble_score,
    solve_portfolio,
    solve_saa_annual_result,
    solve_taa_monthly_result,
)


def _identity_covariance(scale: float = 0.04) -> pd.DataFrame:
    return pd.DataFrame(np.eye(len(ALL_SAA)) * scale, index=ALL_SAA, columns=ALL_SAA)


def test_ensemble_score_respects_custom_config() -> None:
    # Pass macro_factor_weight=0.0 so the test is independent of macro data.
    config = EnsembleConfig(
        regime_weight=0.5,
        trend_weight=0.25,
        momo_weight=0.25,
        macro_factor_weight=0.0,
    )
    regime = pd.Series(1.0, index=ALL_SAA)
    trend = pd.Series(0.5, index=ALL_SAA)
    momo = pd.Series(-0.5, index=ALL_SAA)

    # No macro_factor_mu supplied → macro_factor_weight (0.0) folds into regime.
    score = ensemble_score(regime, trend, momo, macro_factor_mu=None, config=config)

    expected = 0.5 * 0.10 + 0.25 * 0.5 * 0.06 + 0.25 * -0.5 * 0.06
    assert np.isclose(score["SPXT"], expected), f"Got {score['SPXT']}, expected {expected}"
    assert score.index.tolist() == ALL_TAA


def test_ensemble_score_redistributes_missing_macro_to_regime() -> None:
    config = EnsembleConfig(
        regime_weight=0.20,
        trend_weight=0.30,
        momo_weight=0.30,
        macro_factor_weight=0.20,
    )
    regime = pd.Series(1.0, index=ALL_SAA)
    trend = pd.Series(1.0, index=ALL_SAA)
    momo = pd.Series(1.0, index=ALL_SAA)

    score = ensemble_score(regime, trend, momo, macro_factor_mu=None, config=config)

    expected = (0.20 + 0.20) * 0.10 + 0.30 * 0.06 + 0.30 * 0.06
    assert np.isclose(score["SPXT"], expected)


def test_saa_annual_result_returns_full_investment_weights() -> None:
    covariance = _identity_covariance()
    prev_weights = pd.Series(BM2_WEIGHTS, dtype=float).reindex(ALL_SAA).fillna(0.0)
    available = pd.Series(1.0, index=ALL_SAA)

    result = solve_saa_annual_result(covariance, prev_weights, available)

    assert np.isclose(result.weights.sum(), 1.0)
    assert bool((result.weights >= -1e-10).all())
    assert result.mode == "saa_annual"
    assert result.status in {"optimal", "fallback"}


def test_dispatch_matches_saa_wrapper() -> None:
    covariance = _identity_covariance()
    prev_weights = pd.Series(BM2_WEIGHTS, dtype=float).reindex(ALL_SAA).fillna(0.0)
    available = pd.Series(1.0, index=ALL_SAA)

    direct = solve_saa_annual_result(covariance, prev_weights, available)
    dispatched = solve_portfolio("saa_annual", covariance, prev_weights, available)

    pd.testing.assert_series_equal(direct.weights, dispatched.weights)


def test_taa_monthly_fallback_logs_when_cvxpy_missing(tmp_path) -> None:
    if cp is not None:
        return

    covariance = _identity_covariance()
    prev_weights = pd.Series(BM2_WEIGHTS, dtype=float).reindex(ALL_SAA).fillna(0.0)
    available = pd.Series(1.0, index=ALL_SAA)
    signal = pd.Series(0.01, index=ALL_SAA)
    breach_log = tmp_path / "breaches.log"

    result = solve_taa_monthly_result(
        signal_score=signal,
        cov_matrix=covariance,
        prev_weights=prev_weights,
        available=available,
        breach_log_path=breach_log,
    )

    assert result.used_fallback
    assert result.status == "missing_dependency"
    assert np.isclose(result.turnover_cost, COST_PER_TURNOVER * result.turnover)
    assert breach_log.exists()
    assert "missing_dependency" in breach_log.read_text(encoding="utf-8")


def test_breach_log_path_default_points_into_outputs() -> None:
    assert BREACH_LOG_PATH.name == "breaches.log"
