"""Shared exports for the Whitmore TAA signal stack."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SignalBundle:
    """Container for the four-layer TAA signal stack at one decision date.

    Inputs:
    - `regime_probs`: posterior regime probabilities from the HMM layer.
    - `regime_label`: interpreted current regime label.
    - `trend`: per-asset smooth Faber trend scores.
    - `momo`: per-asset ADM momentum scores.
    - `timesfm_mu`: per-asset TimesFM annualized expected-return forecasts.
    - `timesfm_sigma`: per-asset TimesFM annualized volatility forecasts.
    - `timesfm_dir`: per-asset TimesFM directional scores.

    Outputs:
    - Immutable bundle for downstream optimizer and attribution code.

    Citation:
    - Whitmore Task 4 signal-stack specification.

    Point-in-time safety:
    - Safe when each field is constructed from data observed on or before the
      shared decision date.
    """

    regime_probs: pd.Series
    regime_label: str
    trend: pd.Series
    momo: pd.Series
    timesfm_mu: pd.Series
    timesfm_sigma: pd.Series
    timesfm_dir: pd.Series


__all__ = ["SignalBundle"]
