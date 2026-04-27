"""IPS compliance checks shared by backtests and research sweeps."""

from __future__ import annotations

import math

import pandas as pd

from taa_project.config import (
    ALL_TAA,
    CORE,
    CORE_FLOOR,
    MAX_DD,
    NONTRAD,
    NONTRAD_CAP,
    OPPORTUNISTIC,
    SATELLITE,
    SATELLITE_CAP,
    VOL_CEILING,
)


TRADING_DAYS_PER_YEAR = 252
ROLLING_VOL_WINDOW = 21
IPS_OPPORTUNISTIC_CAP = 0.15


def drawdown_series(returns: pd.Series) -> pd.Series:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return pd.Series(dtype=float)
    wealth = (1.0 + clean).cumprod()
    return wealth / wealth.cummax() - 1.0


def rolling_21d_volatility(returns: pd.Series) -> pd.Series:
    clean = returns.dropna().astype(float)
    return clean.rolling(ROLLING_VOL_WINDOW, min_periods=ROLLING_VOL_WINDOW).std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)


def audit_ips_compliance(
    weights: pd.DataFrame,
    returns: pd.Series,
    *,
    opportunistic_cap: float = IPS_OPPORTUNISTIC_CAP,
) -> pd.DataFrame:
    """Return one row per IPS breach across allocation, rolling vol, and drawdown."""

    rows: list[dict[str, object]] = []
    aligned = weights.reindex(columns=ALL_TAA).fillna(0.0)

    for date, row in aligned.iterrows():
        core_weight = float(row.reindex(CORE).sum())
        satellite_weight = float(row.reindex(SATELLITE).sum())
        nontrad_weight = float(row.reindex(NONTRAD).sum())
        opportunistic_weight = float(row.reindex(OPPORTUNISTIC).sum())

        checks = [
            ("core_floor", core_weight >= CORE_FLOOR - 1e-8, core_weight, CORE_FLOOR),
            ("satellite_cap", satellite_weight <= SATELLITE_CAP + 1e-8, satellite_weight, SATELLITE_CAP),
            ("nontrad_cap", nontrad_weight <= NONTRAD_CAP + 1e-8, nontrad_weight, NONTRAD_CAP),
            ("opportunistic_cap", opportunistic_weight <= opportunistic_cap + 1e-8, opportunistic_weight, opportunistic_cap),
        ]
        for rule, passed, value, bound in checks:
            if not passed:
                rows.append({"date": date, "rule": rule, "value": value, "bound": bound})

    rolling_vol = rolling_21d_volatility(returns)
    for date, value in rolling_vol.loc[rolling_vol > VOL_CEILING + 1e-8].items():
        rows.append({"date": date, "rule": "rolling_21d_vol_ceiling", "value": float(value), "bound": VOL_CEILING})

    drawdown = drawdown_series(returns)
    for date, value in drawdown.loc[drawdown < -MAX_DD - 1e-8].items():
        rows.append({"date": date, "rule": "max_drawdown", "value": float(value), "bound": -MAX_DD})

    return pd.DataFrame(rows, columns=["date", "rule", "value", "bound"])


def compliance_stats(returns: pd.Series) -> dict[str, float]:
    rolling_vol = rolling_21d_volatility(returns)
    drawdown = drawdown_series(returns)
    return {
        "max_rolling_21d_vol": float(rolling_vol.max()) if not rolling_vol.empty else 0.0,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
    }
