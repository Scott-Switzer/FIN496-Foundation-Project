"""IPS compliance checks and logging shared by backtests and research sweeps."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from taa_project.config import (
    ALL_TAA,
    CORE,
    CORE_FLOOR,
    MAX_DD,
    NONTRAD,
    NONTRAD_CAP,
    OPPO_CAP,
    OPPORTUNISTIC,
    SATELLITE,
    SATELLITE_CAP,
    SINGLE_SLEEVE_MAX,
    VOL_CEILING,
)


TRADING_DAYS_PER_YEAR = 252
ROLLING_VOL_WINDOW = 21
IPS_OPPORTUNISTIC_CAP = 0.15


COMPLIANCE_REBALANCE_COLUMNS = [
    "portfolio",
    "date",
    "decision_date",
    "rule",
    "asset",
    "pre_trade_value",
    "bound",
    "turnover",
    "remediation",
]


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


def _append_row(
    rows: list[dict[str, object]],
    *,
    portfolio: str,
    date: pd.Timestamp,
    decision_date: pd.Timestamp | None,
    rule: str,
    asset: str,
    value: float,
    bound: float,
    turnover: float,
    remediation: str,
) -> None:
    rows.append(
        {
            "portfolio": portfolio,
            "date": pd.Timestamp(date).date().isoformat(),
            "decision_date": pd.Timestamp(decision_date).date().isoformat() if decision_date is not None else "",
            "rule": rule,
            "asset": asset,
            "pre_trade_value": float(value),
            "bound": float(bound),
            "turnover": float(turnover),
            "remediation": remediation,
        }
    )


def compliance_breach_rows(
    *,
    portfolio: str,
    date: pd.Timestamp,
    decision_date: pd.Timestamp | None,
    pre_trade_weights: pd.Series,
    post_trade_weights: pd.Series,
    active_assets: list[str],
    band_map: dict[str, tuple[float, float]],
    turnover: float,
    remediation: str,
    tolerance: float = 1e-8,
) -> list[dict[str, object]]:
    """Build structured rows describing why a compliance rebalance occurred."""

    assets = list(band_map.keys())
    active = set(active_assets)
    pre = pre_trade_weights.reindex(assets).fillna(0.0).astype(float)
    rows: list[dict[str, object]] = []

    total_weight = float(pre.sum())
    if abs(total_weight - 1.0) > tolerance:
        _append_row(
            rows,
            portfolio=portfolio,
            date=date,
            decision_date=decision_date,
            rule="sum_to_one",
            asset="TOTAL",
            value=total_weight,
            bound=1.0,
            turnover=turnover,
            remediation=remediation,
        )

    min_weight = float(pre.min()) if not pre.empty else 0.0
    if min_weight < -tolerance:
        asset = str(pre.idxmin())
        _append_row(
            rows,
            portfolio=portfolio,
            date=date,
            decision_date=decision_date,
            rule="no_short",
            asset=asset,
            value=float(pre.loc[asset]),
            bound=0.0,
            turnover=turnover,
            remediation=remediation,
        )

    inactive = [asset for asset in assets if asset not in active]
    for asset in inactive:
        value = float(pre.loc[asset])
        if value > tolerance:
            _append_row(
                rows,
                portfolio=portfolio,
                date=date,
                decision_date=decision_date,
                rule="unavailable_asset",
                asset=asset,
                value=value,
                bound=0.0,
                turnover=turnover,
                remediation=remediation,
            )

    for asset in active_assets:
        if asset not in band_map:
            continue
        lower, upper = band_map[asset]
        value = float(pre.get(asset, 0.0))
        if value < float(lower) - tolerance:
            _append_row(
                rows,
                portfolio=portfolio,
                date=date,
                decision_date=decision_date,
                rule="lower_bound",
                asset=asset,
                value=value,
                bound=float(lower),
                turnover=turnover,
                remediation=remediation,
            )
        if value > float(min(upper, SINGLE_SLEEVE_MAX)) + tolerance:
            _append_row(
                rows,
                portfolio=portfolio,
                date=date,
                decision_date=decision_date,
                rule="upper_bound",
                asset=asset,
                value=value,
                bound=float(min(upper, SINGLE_SLEEVE_MAX)),
                turnover=turnover,
                remediation=remediation,
            )

    aggregate_checks = [
        ("core_floor", "Core", float(pre.reindex(CORE).fillna(0.0).sum()), CORE_FLOOR, "min"),
        ("satellite_cap", "Satellite", float(pre.reindex(SATELLITE).fillna(0.0).sum()), SATELLITE_CAP, "max"),
        ("nontrad_cap", "Non-Traditional", float(pre.reindex(NONTRAD).fillna(0.0).sum()), NONTRAD_CAP, "max"),
        ("opportunistic_cap", "Opportunistic", float(pre.reindex(OPPORTUNISTIC).fillna(0.0).sum()), OPPO_CAP, "max"),
    ]
    for rule, asset, value, bound, direction in aggregate_checks:
        breached = value < bound - tolerance if direction == "min" else value > bound + tolerance
        if breached:
            _append_row(
                rows,
                portfolio=portfolio,
                date=date,
                decision_date=decision_date,
                rule=rule,
                asset=asset,
                value=value,
                bound=bound,
                turnover=turnover,
                remediation=remediation,
            )

    max_weight = float(pre.max()) if not pre.empty else 0.0
    if max_weight > SINGLE_SLEEVE_MAX + tolerance:
        asset = str(pre.idxmax())
        _append_row(
            rows,
            portfolio=portfolio,
            date=date,
            decision_date=decision_date,
            rule="single_sleeve_cap",
            asset=asset,
            value=float(pre.loc[asset]),
            bound=SINGLE_SLEEVE_MAX,
            turnover=turnover,
            remediation=remediation,
        )

    return rows


def append_compliance_rebalance_log(path: Path, rows: list[dict[str, object]]) -> None:
    """Append structured compliance-rebalance rows to a CSV file."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).reindex(columns=COMPLIANCE_REBALANCE_COLUMNS)
    if path.exists() and path.stat().st_size > 0:
        try:
            existing = pd.read_csv(path, on_bad_lines="skip")
            frame = pd.concat([existing, frame], ignore_index=True).reindex(columns=COMPLIANCE_REBALANCE_COLUMNS)
        except Exception:
            pass  # stale / schema-mismatch file — overwrite with fresh data
    frame.to_csv(path, index=False)
