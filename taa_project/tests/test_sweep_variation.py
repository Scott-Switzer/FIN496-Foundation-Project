"""Regression tests for the canonical configuration sweep outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "taa_project" / "outputs" / "runs"
ASSET_COLUMNS = [
    "SPXT",
    "FTSE100",
    "LBUSTRUU",
    "BROAD_TIPS",
    "B3REITT",
    "XAU",
    "SILVER_FUT",
    "NIKKEI225",
    "CSI300_CHINA",
    "BITCOIN",
    "CHF_FRANC",
]


def _oos_weights(run_id: str) -> pd.DataFrame:
    path = RUNS_ROOT / run_id / "outputs" / "oos_weights.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical run artifact: {path}")
    return pd.read_csv(path).loc[:, ASSET_COLUMNS]


def test_baseline_vs_timesfm_vb07_oos_weights_differ() -> None:
    baseline = _oos_weights("baseline")
    timesfm_vb07 = _oos_weights("timesfm_vb07")

    assert not baseline.round(12).equals(timesfm_vb07.round(12))


def test_regime_vb_vs_regime_dd_oos_weights_differ() -> None:
    regime_vb = _oos_weights("timesfm_regime_vb")
    regime_dd = _oos_weights("timesfm_regime_dd")

    assert not regime_vb.round(12).equals(regime_dd.round(12))


def test_config_comparison_has_multiple_unique_taa_drawdowns() -> None:
    comparison = pd.read_csv(REPO_ROOT / "taa_project" / "outputs" / "config_comparison.csv")
    taa_rows = comparison.loc[~comparison["Run"].isin(["BM1", "BM2"])]

    assert taa_rows["Max DD"].nunique() >= 3
