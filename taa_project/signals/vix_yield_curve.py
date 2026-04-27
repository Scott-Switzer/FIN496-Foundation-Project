"""Simple VIX + yield-curve macro risk signal.

This module is intentionally separate from the HMM regime layer. It provides a
small, auditable alternative macro signal built only from:

- VIXCLS: equity-market implied volatility / stress.
- T10Y3M: 10-year Treasury yield minus 3-month Treasury yield.

VIX is the timing signal. The yield curve is a sizing penalty only: inversion
can reduce a positive risk score, but it cannot by itself block a calm-VIX
month from being labeled risk-on.

Point-in-time safety:
- Safe when the input ``fred`` panel comes from ``load_fred``. That loader
  applies the one-business-day macro publication lag before this module sees
  the data. Each calculation uses observations dated on or before
  ``as_of_date`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from taa_project.config import ALL_SAA


VIX_COLUMN = "VIXCLS"
CURVE_COLUMN = "T10Y3M"


@dataclass(frozen=True)
class VixYieldCurveConfig:
    """Configuration for the simple VIX + yield-curve signal.

    Inputs:
    - ``zscore_window``: trailing observations used to normalize VIX.
    - ``min_observations``: minimum history required before emitting a signal.
    - ``risk_on_vix_z``: VIX z-score at or below this is risk-on.
    - ``stress_vix_z``: VIX z-score at or above this is stress.
    - ``mild_inversion`` / ``deep_inversion``: curve levels used for positive
      risk-score haircuts.
    - ``mild_curve_penalty`` / ``deep_curve_penalty``: multipliers applied to
      positive risk scores when the curve is inverted.
    - ``vix_z_scale``: VIX z-score move that maps roughly to a meaningful
      stress signal.
    """

    zscore_window: int = 252
    min_observations: int = 126
    risk_on_vix_z: float = -0.50
    stress_vix_z: float = 0.75
    mild_inversion: float = 0.00
    deep_inversion: float = -1.00
    mild_curve_penalty: float = 0.90
    deep_curve_penalty: float = 0.70
    vix_z_scale: float = 1.50


DEFAULT_CONFIG = VixYieldCurveConfig()


_RISK_LOADINGS: dict[str, float] = {
    "SPXT": 0.08,
    "FTSE100": 0.04,
    "NIKKEI225": 0.04,
    "CSI300_CHINA": 0.03,
    "B3REITT": 0.03,
    "BITCOIN": 0.02,
    "SILVER_FUT": 0.01,
    "XAU": -0.01,
    "LBUSTRUU": -0.05,
    "BROAD_TIPS": -0.03,
    "CHF_FRANC": -0.04,
}


def _empty_diagnostics(as_of_date: pd.Timestamp) -> pd.Series:
    return pd.Series(
        {
            "as_of_date": pd.Timestamp(as_of_date),
            "risk_score": 0.0,
            "regime_label": "neutral",
            "vix_level": np.nan,
            "vix_z": np.nan,
            "vix_component": 0.0,
            "curve_level": np.nan,
            "curve_penalty": 1.0,
            "base_risk_score": 0.0,
        }
    )


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    observed = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    mean = observed.rolling(window, min_periods=max(2, window // 2)).mean()
    std = observed.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
    std = std.replace(0.0, np.nan)
    return ((observed - mean) / std).clip(-3.0, 3.0)


def vix_yield_curve_diagnostics(
    fred: pd.DataFrame,
    as_of_date: pd.Timestamp,
    config: VixYieldCurveConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Compute simple macro risk diagnostics at one decision date.

    Inputs:
    - ``fred``: lagged FRED panel containing ``VIXCLS`` and ``T10Y3M``.
    - ``as_of_date``: decision date. Data after this date is ignored.
    - ``config``: signal calibration.

    Outputs:
    - Series with ``risk_score`` in ``[-1, 1]`` plus component diagnostics.
      Positive means risk-on; negative means risk-off.
    """

    as_of = pd.Timestamp(as_of_date)
    if VIX_COLUMN not in fred.columns or CURVE_COLUMN not in fred.columns:
        return _empty_diagnostics(as_of)

    history = fred.loc[:as_of, [VIX_COLUMN, CURVE_COLUMN]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(history) < config.min_observations:
        return _empty_diagnostics(as_of)

    vix = history[VIX_COLUMN]
    curve = history[CURVE_COLUMN]
    vix_z_series = _rolling_zscore(vix, config.zscore_window)
    if vix_z_series.empty:
        return _empty_diagnostics(as_of)

    vix_level = float(vix.iloc[-1])
    curve_level = float(curve.iloc[-1])
    vix_z = float(vix_z_series.iloc[-1])
    if not np.isfinite(vix_z):
        return _empty_diagnostics(as_of)

    vix_component = -float(np.tanh(vix_z / config.vix_z_scale))
    base_risk_score = vix_component
    if curve_level <= config.deep_inversion:
        curve_penalty = config.deep_curve_penalty
    elif curve_level < config.mild_inversion:
        depth = abs(curve_level - config.mild_inversion) / max(
            abs(config.deep_inversion - config.mild_inversion),
            1e-12,
        )
        curve_penalty = config.mild_curve_penalty - depth * (config.mild_curve_penalty - config.deep_curve_penalty)
    else:
        curve_penalty = 1.0
    curve_penalty = float(np.clip(curve_penalty, 0.0, 1.0))

    risk_score = base_risk_score * curve_penalty if base_risk_score > 0.0 else base_risk_score
    risk_score = float(np.clip(risk_score, -1.0, 1.0))

    if vix_z <= config.risk_on_vix_z:
        label = "risk_on"
    elif vix_z >= config.stress_vix_z:
        label = "stress"
    else:
        label = "neutral"

    return pd.Series(
        {
            "as_of_date": as_of,
            "risk_score": risk_score,
            "regime_label": label,
            "vix_level": vix_level,
            "vix_z": vix_z,
            "vix_component": vix_component,
            "curve_level": curve_level,
            "curve_penalty": curve_penalty,
            "base_risk_score": base_risk_score,
        }
    )


def vix_yield_curve_tilt(
    fred: pd.DataFrame,
    as_of_date: pd.Timestamp,
    config: VixYieldCurveConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Convert the simple macro risk score into per-asset expected-return tilts.

    Inputs:
    - ``fred``: lagged FRED panel containing ``VIXCLS`` and ``T10Y3M``.
    - ``as_of_date``: decision date. Data after this date is ignored.
    - ``config``: signal calibration.

    Outputs:
    - Per-asset annualized expected-return proxy indexed to ``ALL_SAA``.
      Positive values favor an asset; negative values discourage it.
    """

    diagnostics = vix_yield_curve_diagnostics(fred=fred, as_of_date=as_of_date, config=config)
    risk_score = float(diagnostics["risk_score"])
    tilt = pd.Series(0.0, index=ALL_SAA, dtype=float)
    for asset, loading in _RISK_LOADINGS.items():
        if asset in tilt.index:
            tilt.loc[asset] = risk_score * loading
    return tilt


def vix_yield_curve_history(
    fred: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    config: VixYieldCurveConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute diagnostics for a sequence of decision dates.

    This helper is for charting and validation. It still calls the one-date
    function for each row, so every row is causal.
    """

    rows = [
        vix_yield_curve_diagnostics(fred=fred, as_of_date=pd.Timestamp(date), config=config)
        for date in pd.DatetimeIndex(decision_dates)
    ]
    if not rows:
        return pd.DataFrame(
            columns=[
                "risk_score",
                "regime_label",
                "vix_level",
                "vix_z",
                "vix_component",
                "curve_level",
                "curve_penalty",
                "base_risk_score",
            ]
        )
    frame = pd.DataFrame(rows)
    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"])
    return frame.set_index("as_of_date").sort_index()
