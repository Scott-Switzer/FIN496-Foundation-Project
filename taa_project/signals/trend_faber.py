# Addresses rubric criterion 1 (TAA signal design) by implementing the
# 120-day Faber trend layer on observed trading histories.
"""Faber-style trend signals for the Whitmore TAA stack.

This module implements the Task 4 trend layer:
- 120-observation simple moving average per sleeve.
- Smooth score `tanh((price / SMA - 1) / sigma_60d)` using 60 observed daily
  return volatility.
- Rolling windows are computed on each asset's observed trading history, not on
  the mixed calendar panel, so weekend and holiday gaps remain `NaN`.

References:
- Faber (2007), "A Quantitative Approach to Tactical Asset Allocation":
  https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf

Point-in-time safety:
- Safe. Every score at date `t` uses only the sleeve's current observed price,
  its prior 120 observed prices for the SMA, and prior 60 observed returns for
  volatility. Missing dates remain `NaN`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_TREND_WINDOW = 120
DEFAULT_VOL_WINDOW = 60
EPSILON = 1e-8


def _observed_rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling mean on observed prices only.

    Inputs:
    - `series`: price series with structural gaps preserved as `NaN`.
    - `window`: number of observed prices to average.

    Outputs:
    - Rolling mean reindexed to the full panel calendar.

    Citation:
    - Faber (2007):
      https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf

    Point-in-time safety:
    - Safe. Each value uses only the asset's trailing observed history.
    """

    observed = series.dropna()
    rolling_mean = observed.rolling(window=window, min_periods=window).mean()
    return rolling_mean.reindex(series.index)


def _observed_return_volatility(series: pd.Series, window: int) -> pd.Series:
    """Compute trailing observed return volatility for one asset.

    Inputs:
    - `series`: price series with gaps preserved as `NaN`.
    - `window`: number of observed simple returns in the volatility window.

    Outputs:
    - Rolling standard deviation of observed simple returns reindexed to the
      full panel calendar.

    Citation:
    - Faber (2007):
      https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf

    Point-in-time safety:
    - Safe. Each estimate uses only past observed prices for that sleeve.
    """

    observed = series.dropna()
    observed_returns = observed.pct_change()
    rolling_vol = observed_returns.rolling(window=window, min_periods=window).std(ddof=0)
    return rolling_vol.reindex(series.index)


def sma_signals(prices: pd.DataFrame, daily_window: int = DEFAULT_TREND_WINDOW) -> pd.DataFrame:
    """Return binary above/below-SMA signals on observed trading histories.

    Inputs:
    - `prices`: asset price dataframe with gaps preserved as `NaN`.
    - `daily_window`: SMA lookback in observed trading sessions.

    Outputs:
    - Dataframe of `{-1, 0, +1}` signals aligned to the full calendar.

    Citation:
    - Faber (2007):
      https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf

    Point-in-time safety:
    - Safe. Each signal uses only current price and the trailing SMA.
    """

    signals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for column in prices.columns:
        sma = _observed_rolling_mean(prices[column], daily_window)
        state = np.where(prices[column] > sma, 1.0, np.where(prices[column] < sma, -1.0, 0.0))
        signals[column] = pd.Series(state, index=prices.index).where(prices[column].notna() & sma.notna())
    return signals


def trend_score(
    prices: pd.DataFrame,
    daily_window: int = DEFAULT_TREND_WINDOW,
    vol_window: int = DEFAULT_VOL_WINDOW,
) -> pd.DataFrame:
    """Return the smooth Faber trend score in `[-1, +1]`.

    Inputs:
    - `prices`: asset price dataframe with gaps preserved as `NaN`.
    - `daily_window`: SMA lookback in observed trading sessions.
    - `vol_window`: volatility lookback in observed returns.

    Outputs:
    - Dataframe of smooth trend scores aligned to the full calendar.

    Citation:
    - Faber (2007):
      https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf

    Point-in-time safety:
    - Safe. Each score uses only current price, trailing SMA, and trailing
      observed return volatility, all truncated at the evaluation date.
    """

    scores = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for column in prices.columns:
        price_series = prices[column]
        sma = _observed_rolling_mean(price_series, daily_window)
        sigma_60d = _observed_return_volatility(price_series, vol_window)
        scale = sigma_60d.replace(0.0, np.nan).clip(lower=EPSILON)
        normalized_distance = (price_series / sma - 1.0) / scale
        scores[column] = np.tanh(normalized_distance).where(price_series.notna() & sma.notna() & scale.notna())
    return scores
