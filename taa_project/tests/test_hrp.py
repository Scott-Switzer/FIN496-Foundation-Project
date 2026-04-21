"""Tests for the HRP SAA method."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa_project.config import ALL_SAA
from taa_project.saa.build_saa import build_saa_portfolio
from taa_project.saa import saa_comparison
from taa_project.saa.saa_comparison import _quasi_diagonal, solve_hierarchical_risk_parity


def test_hrp_weights_sum_to_one_and_nonneg() -> None:
    assets = ["SPXT", "LBUSTRUU", "XAU", "B3REITT", "CHF_FRANC"]
    covariance = pd.DataFrame(
        [
            [0.0300, 0.0040, 0.0020, 0.0030, 0.0010],
            [0.0040, 0.0100, 0.0015, 0.0010, 0.0008],
            [0.0020, 0.0015, 0.0250, 0.0050, 0.0012],
            [0.0030, 0.0010, 0.0050, 0.0350, 0.0015],
            [0.0010, 0.0008, 0.0012, 0.0015, 0.0080],
        ],
        index=assets,
        columns=assets,
    )

    weights = solve_hierarchical_risk_parity(assets, covariance)

    assert np.isclose(float(weights.sum()), 1.0)
    assert bool((weights >= -1e-10).all())


def test_hrp_quasi_diagonal_matches_five_asset_tree_example() -> None:
    link = np.array(
        [
            [0, 1, 0.10, 2],
            [2, 3, 0.12, 2],
            [5, 4, 0.20, 3],
            [7, 6, 0.35, 5],
        ],
        dtype=float,
    )

    assert _quasi_diagonal(link) == [0, 1, 4, 2, 3]


def test_hrp_runs_in_saa_pipeline(tmp_path) -> None:
    risk_parity_dir = tmp_path / "risk_parity"
    hrp_dir = tmp_path / "hrp"

    rp_weights, _ = build_saa_portfolio(
        start_date="2000-01-01",
        end_date="2006-12-31",
        output_dir=risk_parity_dir,
        method="risk_parity",
    )
    hrp_weights, _ = build_saa_portfolio(
        start_date="2000-01-01",
        end_date="2006-12-31",
        output_dir=hrp_dir,
        method="hrp",
    )

    assert (hrp_dir / "saa_weights.csv").exists()
    assert (hrp_dir / "saa_returns.csv").exists()
    assert not hrp_weights.loc[:, ALL_SAA].round(10).equals(rp_weights.loc[:, ALL_SAA].round(10))


def test_hrp_fallback_when_fewer_than_4_assets(monkeypatch) -> None:
    assets = ["SPXT", "LBUSTRUU", "XAU"]
    covariance = pd.DataFrame(
        [
            [0.0300, 0.0040, 0.0020],
            [0.0040, 0.0100, 0.0015],
            [0.0020, 0.0015, 0.0250],
        ],
        index=assets,
        columns=assets,
    )

    sentinel = pd.Series(0.0, index=ALL_SAA, dtype=float)
    sentinel["SPXT"] = 0.60
    sentinel["LBUSTRUU"] = 0.20
    sentinel["XAU"] = 0.20

    def fake_inverse_vol(observed_assets, observed_covariance):
        assert observed_assets == assets
        pd.testing.assert_frame_equal(observed_covariance, covariance)
        return sentinel

    monkeypatch.setattr(saa_comparison, "solve_inverse_vol", fake_inverse_vol)

    observed = solve_hierarchical_risk_parity(assets, covariance)

    pd.testing.assert_series_equal(observed, sentinel)
