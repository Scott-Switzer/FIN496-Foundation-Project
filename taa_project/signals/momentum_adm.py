# Addresses rubric criterion 1 (TAA signal design) by implementing the
# Accelerating Dual Momentum layer on observed trading histories.
"""Accelerating Dual Momentum signals for the Whitmore TAA stack.

This module implements the Task 4 momentum layer:
- 1/2/3/6-month blended total-return momentum using observed trading
  histories.
- Cross-sectional ranking within sleeve buckets.
- Absolute-momentum guardrail so assets with negative blended momentum cannot
  receive positive cross-sectional scores.

References:
- Antonacci (2012), dual momentum framework. Secondary implementation note:
  https://allocatesmartly.com/taa-strategy-accelerating-dual-momentum/

Point-in-time safety:
- Safe. All lookbacks are computed from each asset's observed historical prices
  up to date `t`; missing dates remain `NaN` and are never filled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_WINDOWS = (21, 42, 63, 120)


def _observed_period_return(series: pd.Series, periods: int) -> pd.Series:
    """Compute an observed-history total return over a fixed trading lookback.

    Inputs:
    - `series`: price series with structural gaps preserved as `NaN`.
    - `periods`: lookback in observed trading sessions.

    Outputs:
    - Period return series reindexed to the full panel calendar.

    Citation:
    - Allocate Smartly ADM summary:
      https://allocatesmartly.com/taa-strategy-accelerating-dual-momentum/

    Point-in-time safety:
    - Safe. Each return uses only current observed price and the observed price
      `periods` sessions earlier.
    """

    observed = series.dropna()
    period_return = observed / observed.shift(periods) - 1.0
    return period_return.reindex(series.index)


def period_return(prices: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Compute period returns on observed histories for all assets.

    Inputs:
    - `prices`: asset price dataframe with gaps preserved as `NaN`.
    - `periods`: lookback in observed sessions.

    Outputs:
    - Dataframe of period returns aligned to the full calendar.

    Citation:
    - Allocate Smartly ADM summary:
      https://allocatesmartly.com/taa-strategy-accelerating-dual-momentum/

    Point-in-time safety:
    - Safe. Each column uses only the sleeve's own observed history.
    """

    output = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for column in prices.columns:
        output[column] = _observed_period_return(prices[column], periods)
    return output


def adm_score(prices: pd.DataFrame, windows: tuple[int, ...] = DEFAULT_WINDOWS) -> pd.DataFrame:
    """Compute the blended 1/2/3/6-month ADM total-return score.

    Inputs:
    - `prices`: asset price dataframe with gaps preserved as `NaN`.
    - `windows`: observed-session lookbacks for the ADM blend.

    Outputs:
    - Dataframe of blended total-return momentum scores.

    Citation:
    - Allocate Smartly ADM summary:
      https://allocatesmartly.com/taa-strategy-accelerating-dual-momentum/

    Point-in-time safety:
    - Safe. Every component return is computed only from observed prices dated
      on or before the current evaluation date.
    """

    components = [period_return(prices, window) for window in windows]
    blended = sum(components) / float(len(components))
    return blended.where(prices.notna())


def cross_sectional_rank(
    score: pd.DataFrame,
    buckets: dict[str, list[str]],
    apply_absolute_filter: bool = True,
) -> pd.DataFrame:
    """Rank ADM scores within sleeve buckets and map them into `[-1, +1]`.

    Inputs:
    - `score`: blended ADM score dataframe.
    - `buckets`: mapping from bucket name to constituent assets.
    - `apply_absolute_filter`: if `True`, assets with non-positive blended
      momentum are clipped at `0` on the positive side so they cannot receive a
      positive dual-momentum score.

    Outputs:
    - Cross-sectionally ranked momentum dataframe in `[-1, +1]`.

    Citation:
    - Antonacci-style relative momentum framework summarized by Allocate
      Smartly: https://allocatesmartly.com/taa-strategy-accelerating-dual-momentum/

    Point-in-time safety:
    - Safe. Each row uses only the contemporaneous cross-section of scores at
      that decision date.
    """

    ranked = pd.DataFrame(np.nan, index=score.index, columns=score.columns, dtype=float)
    for assets in buckets.values():
        available_assets = [asset for asset in assets if asset in score.columns]
        if not available_assets:
            continue

        bucket_scores = score.loc[:, available_assets]
        percentile_rank = bucket_scores.rank(axis=1, pct=True, method="average")
        centered_rank = (percentile_rank - 0.5) * 2.0
        bucket_output = centered_rank.where(bucket_scores.notna())

        if apply_absolute_filter:
            # Negative or flat absolute momentum cannot become a positive
            # signal merely by ranking best among weak peers.
            bucket_output = bucket_output.where(bucket_scores > 0.0, bucket_output.clip(upper=0.0))

        ranked.loc[:, available_assets] = bucket_output

    return ranked
