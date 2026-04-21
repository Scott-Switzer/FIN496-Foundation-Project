"""Tests for nested sleeve risk budgeting."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import psutil
import pytest

from taa_project.config import ALL_SAA, BM2_WEIGHTS, CORE, NONTRAD, SATELLITE
from taa_project.optimizer.cvxpy_opt import EnsembleConfig, cp
from taa_project.optimizer.nested_risk import NestedRiskConfig, solve_nested_taa


def _expected_returns() -> pd.Series:
    return pd.Series(
        {
            "SPXT": 0.14,
            "FTSE100": 0.12,
            "LBUSTRUU": 0.01,
            "BROAD_TIPS": 0.01,
            "B3REITT": 0.20,
            "XAU": 0.05,
            "SILVER_FUT": 0.18,
            "NIKKEI225": 0.17,
            "CSI300_CHINA": 0.25,
            "BITCOIN": 0.50,
            "CHF_FRANC": -0.01,
        },
        dtype=float,
    ).reindex(ALL_SAA)


def _covariance_matrix() -> pd.DataFrame:
    vols = {
        "SPXT": 0.12,
        "FTSE100": 0.13,
        "LBUSTRUU": 0.05,
        "BROAD_TIPS": 0.06,
        "B3REITT": 0.18,
        "XAU": 0.16,
        "SILVER_FUT": 0.20,
        "NIKKEI225": 0.17,
        "CSI300_CHINA": 0.22,
        "BITCOIN": 0.45,
        "CHF_FRANC": 0.08,
    }
    corr = pd.DataFrame(np.eye(len(ALL_SAA)), index=ALL_SAA, columns=ALL_SAA, dtype=float)
    for group, corr_value in ((CORE, 0.25), (SATELLITE, 0.15), (NONTRAD, 0.05)):
        for asset in group:
            for other in group:
                if asset != other:
                    corr.loc[asset, other] = corr_value
    for asset in ALL_SAA:
        for other in ALL_SAA:
            if asset != other and corr.loc[asset, other] == 0.0:
                corr.loc[asset, other] = 0.02

    covariance = pd.DataFrame(index=ALL_SAA, columns=ALL_SAA, dtype=float)
    for asset in ALL_SAA:
        for other in ALL_SAA:
            covariance.loc[asset, other] = corr.loc[asset, other] * vols[asset] * vols[other]
    return covariance


def _asset_log_returns(n_obs: int = 504) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        rng.normal(0.0002, 0.0025, size=(n_obs, len(ALL_SAA))),
        columns=ALL_SAA,
        index=pd.date_range("2022-01-03", periods=n_obs, freq="B"),
        dtype=float,
    )
    frame.loc[frame.index[::40], "BITCOIN"] = -0.06
    return frame


def _previous_weights() -> pd.Series:
    return pd.Series(BM2_WEIGHTS, dtype=float).reindex(ALL_SAA).fillna(0.0)


@pytest.mark.skipif(cp is None, reason="cvxpy unavailable")
def test_nested_blends_at_specified_weights(tmp_path) -> None:
    config = NestedRiskConfig()
    result = solve_nested_taa(
        expected_returns=_expected_returns(),
        cov_matrix=_covariance_matrix(),
        available=pd.Series(1.0, index=ALL_SAA, dtype=float),
        previous_weights=_previous_weights(),
        config=config,
        ensemble_config=EnsembleConfig(use_nested_risk=True, nested_risk_config=config),
        as_of_date=pd.Timestamp("2024-01-31"),
        breach_log_path=tmp_path / "breaches.log",
    )

    assert np.isclose(result.weights.loc[CORE].sum(), 0.55, atol=1e-6)
    assert np.isclose(result.weights.loc[SATELLITE].sum(), 0.35, atol=1e-6)
    assert np.isclose(result.weights.loc[NONTRAD].sum(), 0.10, atol=1e-6)


@pytest.mark.skipif(cp is None, reason="cvxpy unavailable")
def test_nested_falls_back_when_nt_unavailable(tmp_path) -> None:
    config = NestedRiskConfig()
    available = pd.Series(1.0, index=ALL_SAA, dtype=float)
    available.loc[NONTRAD] = 0.0
    result = solve_nested_taa(
        expected_returns=_expected_returns(),
        cov_matrix=_covariance_matrix(),
        available=available,
        previous_weights=_previous_weights(),
        config=config,
        ensemble_config=EnsembleConfig(use_nested_risk=True, nested_risk_config=config),
        as_of_date=pd.Timestamp("2024-01-31"),
        breach_log_path=tmp_path / "breaches.log",
    )

    assert np.isclose(result.weights.loc[NONTRAD].sum(), 0.0, atol=1e-8)
    assert np.isclose(result.weights.loc[CORE].sum(), 0.55 / 0.90, atol=1e-6)
    assert np.isclose(result.weights.loc[SATELLITE].sum(), 0.35 / 0.90, atol=1e-6)
    assert (tmp_path / "nested_renormalizations.csv").exists()


@pytest.mark.skipif(cp is None, reason="cvxpy unavailable")
def test_nested_per_sleeve_vol_is_close_to_target(tmp_path) -> None:
    config = NestedRiskConfig()
    covariance = _covariance_matrix()
    result = solve_nested_taa(
        expected_returns=_expected_returns(),
        cov_matrix=covariance,
        available=pd.Series(1.0, index=ALL_SAA, dtype=float),
        previous_weights=_previous_weights(),
        config=config,
        ensemble_config=EnsembleConfig(use_nested_risk=True, nested_risk_config=config),
        as_of_date=pd.Timestamp("2024-01-31"),
        breach_log_path=tmp_path / "breaches.log",
    )

    for assets, sleeve_weight, target in (
        (CORE, 0.55, config.core_vol_target),
        (SATELLITE, 0.35, config.satellite_vol_target),
        (NONTRAD, 0.10, config.nontraditional_vol_target),
    ):
        normalized = result.weights.loc[assets] / sleeve_weight
        sleeve_cov = covariance.loc[assets, assets].to_numpy(dtype=float)
        sleeve_vol = float(np.sqrt(max(normalized.to_numpy(dtype=float) @ sleeve_cov @ normalized.to_numpy(dtype=float), 0.0)))
        assert abs(sleeve_vol - target) <= 0.01


@pytest.mark.skipif(cp is None, reason="cvxpy unavailable")
def test_nested_solver_releases_memory(tmp_path) -> None:
    config = NestedRiskConfig()
    ensemble_config = EnsembleConfig(use_nested_risk=True, nested_risk_config=config)
    proc = psutil.Process()
    gc.collect()
    rss_before = proc.memory_info().rss
    for _ in range(20):
        _ = solve_nested_taa(
            expected_returns=_expected_returns(),
            cov_matrix=_covariance_matrix(),
            available=pd.Series(1.0, index=ALL_SAA, dtype=float),
            previous_weights=_previous_weights(),
            config=config,
            ensemble_config=ensemble_config,
            as_of_date=pd.Timestamp("2024-01-31"),
            breach_log_path=tmp_path / "breaches.log",
            asset_log_returns=_asset_log_returns(),
        )
        gc.collect()
    rss_after = proc.memory_info().rss

    assert rss_after - rss_before < 200 * 1024**2, "memory leak in nested-risk solver"
