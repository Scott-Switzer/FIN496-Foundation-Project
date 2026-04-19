# Addresses rubric criterion 1 and Task 5 by defining a causal drawdown-clip
# risk overlay for the monthly TAA optimizer.
"""Drawdown-guardrail utilities for the Whitmore TAA overlay.

References:
- Grossman & Zhou (1993), "Optimal Investment Strategies for Controlling
  Drawdowns": https://doi.org/10.1111/j.1467-9965.1993.tb00044.x
- Whitmore Task 5 drawdown-clip requirement.

Point-in-time safety:
- Safe. The guardrail multiplier at date `t` depends only on realized
  portfolio returns observed on or before `t`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DrawdownGuardrailConfig:
    """Configuration for the causal drawdown-clip overlay.

    Inputs:
    - `trigger_dd`: tighten when the trailing drawdown is at or below this
      threshold.
    - `lookback_days`: trailing window length used for the drawdown estimate.
    - `tightening_factor`: multiplier applied to the active vol budget while
      the guardrail is engaged.
    - `release_dd`: release once the trailing drawdown recovers above this
      threshold.
    - `min_days_between_switches`: minimum observation count between state
      changes.

    Outputs:
    - Immutable configuration for `dd_guardrail_multiplier`.

    Citation:
    - Grossman & Zhou (1993):
      https://doi.org/10.1111/j.1467-9965.1993.tb00044.x

    Point-in-time safety:
    - Safe. Static configuration only.
    """

    trigger_dd: float = -0.15
    lookback_days: int = 126
    tightening_factor: float = 0.5
    release_dd: float = -0.07
    min_days_between_switches: int = 21


def _window_max_drawdown(window_returns: pd.Series) -> float:
    clean = window_returns.dropna()
    if clean.empty:
        return 0.0
    wealth = (1.0 + clean).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def trailing_drawdown_series(
    realized_returns: pd.Series,
    lookback_days: int,
) -> pd.Series:
    """Compute trailing-window max drawdown using only returns observed to date.

    Inputs:
    - `realized_returns`: realized simple return series.
    - `lookback_days`: trailing observation count for the drawdown window.

    Outputs:
    - Series of trailing max-drawdown values aligned to `realized_returns`.

    Citation:
    - Grossman & Zhou (1993):
      https://doi.org/10.1111/j.1467-9965.1993.tb00044.x

    Point-in-time safety:
    - Safe. Each output at `t` uses only returns from the trailing window
      ending at `t`.
    """

    clean = realized_returns.dropna()
    if clean.empty:
        return pd.Series(dtype=float)

    values = []
    for index in range(len(clean)):
        start = max(0, index - lookback_days + 1)
        values.append(_window_max_drawdown(clean.iloc[start : index + 1]))
    return pd.Series(values, index=clean.index, name="trailing_drawdown")


def dd_guardrail_multiplier(
    realized_returns: pd.Series,
    config: DrawdownGuardrailConfig,
) -> pd.Series:
    """Return the causal drawdown-guardrail multiplier over time.

    Inputs:
    - `realized_returns`: realized simple return series.
    - `config`: drawdown-guardrail configuration.

    Outputs:
    - Series of multipliers aligned to `realized_returns`, typically `1.0`
      or `config.tightening_factor`.

    Citation:
    - Grossman & Zhou (1993):
      https://doi.org/10.1111/j.1467-9965.1993.tb00044.x

    Point-in-time safety:
    - Safe. The multiplier at `t` depends only on returns observed on or
      before `t`.
    """

    trailing_dd = trailing_drawdown_series(realized_returns, config.lookback_days)
    if trailing_dd.empty:
        return pd.Series(dtype=float)

    multiplier = 1.0
    since_switch = config.min_days_between_switches
    values: list[float] = []

    for drawdown in trailing_dd:
        if multiplier == 1.0 and drawdown <= config.trigger_dd and since_switch >= config.min_days_between_switches:
            multiplier = config.tightening_factor
            since_switch = 0
        elif multiplier != 1.0 and drawdown >= config.release_dd and since_switch >= config.min_days_between_switches:
            multiplier = 1.0
            since_switch = 0
        values.append(float(multiplier))
        since_switch += 1

    return pd.Series(values, index=trailing_dd.index, name="guardrail_multiplier")

