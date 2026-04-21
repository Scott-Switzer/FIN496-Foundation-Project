"""Tests for the CVaR-aware TAA optimizer mode."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import psutil
import pytest

from taa_project.config import ALL_SAA, BM2_WEIGHTS, CORE, CORE_FLOOR, NONTRAD, NONTRAD_CAP, SATELLITE, SATELLITE_CAP
from taa_project.optimizer.cvxpy_opt import EnsembleConfig, cp, solve_taa_monthly_result


def _identity_covariance(scale: float = 0.04) -> pd.DataFrame:
    return pd.DataFrame(np.eye(len(ALL_SAA)) * scale, index=ALL_SAA, columns=ALL_SAA)


def _available_subset() -> pd.Series:
    available = pd.Series(0.0, index=ALL_SAA, dtype=float)
    available.loc[["SPXT", "LBUSTRUU", "XAU", "CHF_FRANC"]] = 1.0
    return available


def _previous_weights() -> pd.Series:
    return pd.Series(BM2_WEIGHTS, dtype=float).reindex(ALL_SAA).fillna(0.0)


def _signal_scores() -> pd.Series:
    signal = pd.Series(0.005, index=ALL_SAA, dtype=float)
    signal["SPXT"] = 0.120
    signal["XAU"] = 0.030
    signal["LBUSTRUU"] = 0.012
    signal["CHF_FRANC"] = 0.008
    return signal


def _scenario_matrix(assets: list[str], n_obs: int = 504, crash_every: int = 40) -> np.ndarray:
    rng = np.random.default_rng(42)
    scenarios = pd.DataFrame(0.0, index=pd.RangeIndex(n_obs), columns=assets, dtype=np.float32)
    if "SPXT" in scenarios.columns:
        scenarios["SPXT"] = rng.normal(0.0008, 0.0040, size=n_obs).astype(np.float32)
        tail_idx = np.arange(0, n_obs, crash_every)
        scenarios.loc[tail_idx, "SPXT"] = np.float32(-0.075)
    if "LBUSTRUU" in scenarios.columns:
        scenarios["LBUSTRUU"] = rng.normal(0.00015, 0.0012, size=n_obs).astype(np.float32)
    if "XAU" in scenarios.columns:
        scenarios["XAU"] = rng.normal(0.00035, 0.0020, size=n_obs).astype(np.float32)
    if "CHF_FRANC" in scenarios.columns:
        scenarios["CHF_FRANC"] = rng.normal(0.00010, 0.0010, size=n_obs).astype(np.float32)
    for asset in assets:
        if asset not in {"SPXT", "LBUSTRUU", "XAU", "CHF_FRANC"}:
            scenarios[asset] = rng.normal(0.0002, 0.0025, size=n_obs).astype(np.float32)
    return scenarios.to_numpy(dtype=np.float32, copy=True)


def _portfolio_cvar_from_matrix(scenario_matrix: np.ndarray, weights: pd.Series, alpha: float) -> float:
    losses = -scenario_matrix @ weights.to_numpy(dtype=float)
    var_alpha = float(np.quantile(losses, alpha))
    tail_losses = losses[losses >= var_alpha]
    return float(tail_losses.mean()) if tail_losses.size else var_alpha


@pytest.mark.skipif(cp is None, reason="cvxpy unavailable")
def test_cvar_mode_changes_weights_under_left_tail() -> None:
    covariance = _identity_covariance()
    available = _available_subset()
    previous_weights = _previous_weights()
    signal = _signal_scores()
    active_assets = [asset for asset in ALL_SAA if bool(available.get(asset, 0.0))]
    scenarios = _scenario_matrix(active_assets, crash_every=40)

    vol_result = solve_taa_monthly_result(
        signal_score=signal,
        cov_matrix=covariance,
        prev_weights=previous_weights,
        available=available,
        vol_budget=0.10,
        config=EnsembleConfig(optimizer_mode="vol"),
    )
    cvar_result = solve_taa_monthly_result(
        signal_score=signal,
        cov_matrix=covariance,
        prev_weights=previous_weights,
        available=available,
        vol_budget=0.10,
        config=EnsembleConfig(optimizer_mode="cvar", cvar_alpha=0.95, cvar_budget=0.012),
        scenario_returns=scenarios,
    )

    assert cvar_result.weights["SPXT"] < vol_result.weights["SPXT"]


@pytest.mark.skipif(cp is None, reason="cvxpy unavailable")
def test_cvar_constraint_is_binding() -> None:
    covariance = _identity_covariance()
    available = _available_subset()
    previous_weights = _previous_weights()
    signal = _signal_scores()
    active_assets = [asset for asset in ALL_SAA if bool(available.get(asset, 0.0))]
    scenarios = _scenario_matrix(active_assets, crash_every=35)
    budget = 0.015

    result = solve_taa_monthly_result(
        signal_score=signal,
        cov_matrix=covariance,
        prev_weights=previous_weights,
        available=available,
        vol_budget=0.10,
        config=EnsembleConfig(optimizer_mode="cvar", cvar_alpha=0.95, cvar_budget=budget),
        scenario_returns=scenarios,
    )

    realized_cvar = _portfolio_cvar_from_matrix(scenarios, result.weights.reindex(active_assets).fillna(0.0), alpha=0.95)
    assert realized_cvar <= budget + 1e-3
    assert abs(realized_cvar - budget) <= 2e-3


@pytest.mark.skipif(cp is None, reason="cvxpy unavailable")
def test_cvar_mode_respects_sleeve_constraints() -> None:
    covariance = _identity_covariance()
    available = pd.Series(1.0, index=ALL_SAA, dtype=float)
    previous_weights = _previous_weights()
    signal = _signal_scores()
    scenarios = _scenario_matrix(ALL_SAA, crash_every=40)

    result = solve_taa_monthly_result(
        signal_score=signal,
        cov_matrix=covariance,
        prev_weights=previous_weights,
        available=available,
        vol_budget=0.10,
        config=EnsembleConfig(optimizer_mode="cvar", cvar_alpha=0.95, cvar_budget=0.015),
        scenario_returns=scenarios,
    )

    assert result.weights.loc[CORE].sum() >= CORE_FLOOR - 1e-8
    assert result.weights.loc[SATELLITE].sum() <= SATELLITE_CAP + 1e-8
    assert result.weights.loc[NONTRAD].sum() <= NONTRAD_CAP + 1e-8


@pytest.mark.skipif(cp is None, reason="cvxpy unavailable")
def test_cvar_solver_releases_memory() -> None:
    covariance = _identity_covariance()
    available = _available_subset()
    previous_weights = _previous_weights()
    signal = _signal_scores()
    active_assets = [asset for asset in ALL_SAA if bool(available.get(asset, 0.0))]
    scenarios = _scenario_matrix(active_assets, crash_every=35)
    config = EnsembleConfig(optimizer_mode="cvar", cvar_alpha=0.95, cvar_budget=0.015)

    proc = psutil.Process()
    gc.collect()
    rss_before = proc.memory_info().rss
    for _ in range(20):
        _ = solve_taa_monthly_result(
            signal_score=signal,
            cov_matrix=covariance,
            prev_weights=previous_weights,
            available=available,
            vol_budget=0.10,
            config=config,
            scenario_returns=scenarios,
        )
        gc.collect()
    rss_after = proc.memory_info().rss

    assert rss_after - rss_before < 200 * 1024**2, "memory leak in CVaR solver"
