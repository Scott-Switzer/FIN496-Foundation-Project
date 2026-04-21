"""Tests for regime-conditional Black-Litterman stress views."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa_project.config import ALL_SAA, BM2_WEIGHTS, EQUITY_ASSETS
from taa_project.saa.saa_comparison import RF_ANNUAL, bl_with_stress_views


def _covariance_matrix() -> pd.DataFrame:
    vols = np.linspace(0.08, 0.24, len(ALL_SAA))
    covariance = np.diag(vols**2)
    return pd.DataFrame(covariance, index=ALL_SAA, columns=ALL_SAA)


def test_bl_stress_view_shifts_equity_mu_down() -> None:
    covariance = _covariance_matrix()
    policy_weights = pd.Series(BM2_WEIGHTS, dtype=float)

    risk_on = bl_with_stress_views(policy_weights, covariance, regime_label="risk_on", stress_equity_shock_sigmas=1.0)
    stress = bl_with_stress_views(policy_weights, covariance, regime_label="stress", stress_equity_shock_sigmas=1.0)

    for asset in EQUITY_ASSETS:
        assert stress.loc[asset] < risk_on.loc[asset]
    for asset in [asset for asset in ALL_SAA if asset not in EQUITY_ASSETS]:
        assert np.isclose(stress.loc[asset], risk_on.loc[asset])


def test_bl_no_stress_view_in_risk_on_regime() -> None:
    covariance = _covariance_matrix()
    policy_weights = pd.Series(BM2_WEIGHTS, dtype=float).reindex(ALL_SAA).fillna(0.0)
    policy_weights /= float(policy_weights.sum())
    ridge_cov = covariance.to_numpy(dtype=float) + 1e-8 * np.eye(len(ALL_SAA))
    expected_prior = pd.Series(RF_ANNUAL + 2.5 * (ridge_cov @ policy_weights.to_numpy(dtype=float)), index=ALL_SAA)

    observed = bl_with_stress_views(policy_weights, covariance, regime_label="risk_on", stress_equity_shock_sigmas=1.0)

    pd.testing.assert_series_equal(observed, expected_prior)
