# Addresses tasks.md Steps 2-4 by exposing the Task 1 audited loaders for
# downstream SAA, benchmark, and TAA modules.
"""Shared accessors for the audited Whitmore asset and macro datasets."""

from __future__ import annotations

import pandas as pd

from taa_project.config import FRED_CSV, PRICES_CSV
from taa_project.data_audit import (
    build_availability_flags,
    compute_consecutive_log_returns,
    load_asset_prices,
    load_fred_features,
)


def load_prices(path: str | None = None) -> pd.DataFrame:
    """Load the audited asset price panel.

    Inputs:
    - `path`: optional override for `whitmore_daily.csv`.

    Outputs:
    - Price dataframe containing the asset columns mapped in `data_key.csv`.

    Citation:
    - Internal project files `data/asset_data/whitmore_daily.csv` and
      `data/asset_data/data_key.csv`.

    Point-in-time safety:
    - Safe. This delegates to the Task 1 audited loader and preserves the raw
      missingness pattern from the source file.
    """

    price_path = PRICES_CSV if path is None else path
    prices, _ = load_asset_prices(price_path)
    return prices


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute audited log returns from the asset price panel.

    Inputs:
    - `prices`: asset price dataframe with gaps preserved as `NaN`.

    Outputs:
    - Log-return dataframe indexed to the same calendar as `prices`.

    Citation:
    - `tasks.md`, Step 2 return-calculation rules.

    Point-in-time safety:
    - Safe. Each return at date `t` uses only the current observed price and
      that asset's immediately prior observed price.
    """

    return compute_consecutive_log_returns(prices)


def availability_flag(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the audited availability matrix.

    Inputs:
    - `prices`: asset price dataframe.

    Outputs:
    - Binary dataframe with `1` where a price exists and `0` otherwise.

    Citation:
    - `tasks.md`, Step 4 availability-flag requirement.

    Point-in-time safety:
    - Safe. Flags are derived directly from contemporaneous price presence.
    """

    return build_availability_flags(prices)


def load_fred(path: str | None = None, calendar_index: pd.Index | None = None) -> pd.DataFrame:
    """Load the lagged FRED feature panel for signal construction.

    Inputs:
    - `path`: optional override for `fred_data.csv`.
    - `calendar_index`: optional calendar to align onto after lagging.

    Outputs:
    - FRED feature dataframe shifted by one business day before use.

    Citation:
    - Internal project file `data/consolidated_csvs/fred/master/fred_data.csv`.

    Point-in-time safety:
    - Safe. The loader applies the one-business-day lag mandated in Task 1
    before any forward-fill alignment to a downstream calendar.
    """

    fred_path = FRED_CSV if path is None else path
    return load_fred_features(fred_path, calendar_index=calendar_index)


def build_master(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Bundle the primary audited dataframes for downstream use.

    Inputs:
    - `prices`: asset price dataframe.

    Outputs:
    - Dictionary containing prices, log returns, and availability flags.

    Citation:
    - `tasks.md`, Steps 2-4.

    Point-in-time safety:
    - Safe. This bundles audited causal transformations only.
    """

    rets = log_returns(prices)
    flag = availability_flag(prices)
    return {"prices": prices, "returns": rets, "available": flag}


if __name__ == "__main__":
    prices = load_prices()
    print(f"Prices shape: {prices.shape}")
    print(f"Date range: {prices.index.min()} → {prices.index.max()}")
    print("First-available date per asset:")
    print(prices.apply(lambda series: series.first_valid_index()).sort_values().to_string())

    fred = load_fred()
    print(f"\nLagged FRED shape: {fred.shape}, cols: {list(fred.columns)}")
