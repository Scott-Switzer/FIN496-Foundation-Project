"""Shared exports for the Whitmore TAA signal stack."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalBundle:
    """Container for the TAA signal stack at one decision date.

    Inputs:
    - `regime_probs`: posterior regime probabilities from the HMM layer
      (kept for logging and diagnostics; not used as a hard trigger).
    - `risk_score`: continuous risk score in `[-1, +1]` combining HMM stress
      posterior with equity trend and momentum.
    - `regime_label`: interpreted regime label retained for older callers and
      reporting.
    - `trend`: per-asset smooth Faber trend scores.
    - `momo`: per-asset ADM momentum scores.
    - `timesfm_mu`: per-asset TimesFM annualized expected-return forecasts.
    - `timesfm_sigma`: per-asset TimesFM annualized volatility forecasts.
    - `timesfm_dir`: per-asset TimesFM directional scores.

    Outputs:
    - Immutable bundle for downstream optimizer and attribution code.

    Point-in-time safety:
    - Safe when each field is constructed from data observed on or before the
      shared decision date.
    """

    regime_probs: pd.Series
    trend: pd.Series
    momo: pd.Series
    timesfm_mu: pd.Series
    timesfm_sigma: pd.Series
    timesfm_dir: pd.Series
    risk_score: float = 0.0
    regime_label: str = "neutral"


__all__ = ["SignalBundle"]
