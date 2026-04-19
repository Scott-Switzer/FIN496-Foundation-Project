# Addresses IPS §4 benchmark construction and the rubric's benchmark-comparison
# requirement by building the two fixed-weight policy references.
"""Build Whitmore Benchmark 1 and Benchmark 2 series.

The IPS defines both benchmarks as fixed-weight portfolios rebalanced on the
last trading day of each calendar year. This module preserves that definition:

- BM1 is a constant 60/40 SPXT/LBUSTRUU portfolio.
- BM2 is the Diversified Policy Portfolio defined in IPS §4.
- Each benchmark begins on the first date on or after the requested start date
  when every positive-weight constituent has an observed price.
- Turnover costs of 5 bps per unit traded are charged only on scheduled annual
  rebalances, not on initial funding.

References:
- Whitmore IPS `IPS.md` §4 and `Guidelines.md` benchmarks section.
- Internal project file `data/asset_data/whitmore_daily.csv`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from taa_project.config import ALL_SAA, BM1_WEIGHTS, BM2_WEIGHTS, COST_PER_TURNOVER, OUTPUT_DIR
from taa_project.data_loader import load_prices, log_returns


DEFAULT_START = "2000-01-01"
DEFAULT_END = "2025-12-31"
BM1_RETURNS_FILENAME = "bm1_returns.csv"
BM2_RETURNS_FILENAME = "bm2_returns.csv"
BM1_WEIGHTS_FILENAME = "bm1_weights.csv"
BM2_WEIGHTS_FILENAME = "bm2_weights.csv"


def normalize_target_weights(weight_map: Mapping[str, float]) -> pd.Series:
    """Reindex a benchmark weight map onto the full SAA universe.

    Inputs:
    - `weight_map`: benchmark target weights keyed by ticker.

    Outputs:
    - Full-universe target weight series summing to one.

    Citation:
    - Whitmore IPS `IPS.md` §4 benchmark definitions.

    Point-in-time safety:
    - Safe. Benchmark weights are static IPS policy inputs.
    """

    weights = pd.Series(weight_map, dtype=float).reindex(ALL_SAA).fillna(0.0)
    total = float(weights.sum())
    if not np.isclose(total, 1.0):
        raise ValueError(f"Benchmark weights must sum to 1.0, received {total:.12f}.")
    return weights


def positive_weight_assets(target_weights: pd.Series) -> list[str]:
    """List the benchmark sleeves with strictly positive target weights.

    Inputs:
    - `target_weights`: full-universe target weight series.

    Outputs:
    - Ordered list of positive-weight benchmark sleeves.

    Citation:
    - Whitmore IPS `IPS.md` §4 benchmark definitions.

    Point-in-time safety:
    - Safe. This depends only on static benchmark weights.
    """

    return [asset for asset in ALL_SAA if target_weights.get(asset, 0.0) > 0.0]


def find_first_benchmark_start_date(
    prices: pd.DataFrame,
    target_weights: pd.Series,
    start_date: str,
) -> pd.Timestamp:
    """Find the first feasible benchmark inception date.

    Inputs:
    - `prices`: audited price dataframe.
    - `target_weights`: full-universe benchmark target weights.
    - `start_date`: earliest requested start date.

    Outputs:
    - First date on or after `start_date` where every positive-weight sleeve has
      an observed price.

    Citation:
    - Whitmore IPS `IPS.md` §4 fixed-weight benchmark rule.

    Point-in-time safety:
    - Safe. The scan uses only contemporaneous price availability.
    """

    assets = positive_weight_assets(target_weights)
    candidate_index = prices.loc[pd.Timestamp(start_date) :].index
    for date in candidate_index:
        if bool(prices.loc[date, assets].notna().all()):
            return pd.Timestamp(date)
    raise RuntimeError("No feasible benchmark inception date found in the requested history.")


def choose_year_end_rebalance_date(
    prices: pd.DataFrame,
    assets: list[str],
    year: int,
) -> pd.Timestamp | None:
    """Choose the benchmark's last common observed date in a calendar year.

    Inputs:
    - `prices`: audited price dataframe.
    - `assets`: positive-weight benchmark sleeves.
    - `year`: calendar year under inspection.

    Outputs:
    - Last date in `year` where every benchmark constituent has a price, or
      `None` if no such date exists.

    Citation:
    - Whitmore IPS `IPS.md` §4 annual benchmark rebalance rule.

    Point-in-time safety:
    - Safe. The date choice depends only on observed prices within that year.
    """

    year_slice = prices.loc[f"{year}-01-01" : f"{year}-12-31", assets]
    for date in reversed(year_slice.index.tolist()):
        if bool(year_slice.loc[date].notna().all()):
            return pd.Timestamp(date)
    return None


def build_rebalance_schedule(
    prices: pd.DataFrame,
    target_weights: pd.Series,
    start_date: str,
    end_date: str,
) -> list[pd.Timestamp]:
    """Build the benchmark inception date plus annual rebalance dates.

    Inputs:
    - `prices`: audited price dataframe.
    - `target_weights`: full-universe benchmark target weights.
    - `start_date`: earliest requested benchmark start date.
    - `end_date`: last date to include in the benchmark history.

    Outputs:
    - Ordered list of benchmark rebalance dates.

    Citation:
    - Whitmore IPS `IPS.md` §4 annual benchmark rebalance rule.

    Point-in-time safety:
    - Safe. The schedule uses only observed dates on or before each year-end.
    """

    assets = positive_weight_assets(target_weights)
    end_timestamp = pd.Timestamp(end_date)
    bounded_prices = prices.loc[:end_timestamp, assets]
    first_date = find_first_benchmark_start_date(bounded_prices, target_weights, start_date)
    schedule = [first_date]

    for year in range(first_date.year, end_timestamp.year + 1):
        rebalance_date = choose_year_end_rebalance_date(bounded_prices, assets, year)
        if rebalance_date is None or rebalance_date <= first_date:
            continue
        schedule.append(rebalance_date)

    return sorted(dict.fromkeys(schedule))


def simulate_fixed_weight_portfolio(
    returns: pd.DataFrame,
    target_weights: pd.Series,
    schedule: list[pd.Timestamp],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate realized benchmark returns under annual rebalancing.

    Inputs:
    - `returns`: audited log-return panel with gaps preserved as `NaN`.
    - `target_weights`: full-universe benchmark target weights.
    - `schedule`: annual rebalance schedule including inception.
    - `start_date`: benchmark inception date.
    - `end_date`: last date to include.

    Outputs:
    - Tuple `(weights_df, returns_df)` where `weights_df` stores the daily
      benchmark target schedule and `returns_df` stores realized daily returns.

    Citation:
    - Whitmore IPS `IPS.md` §4 benchmark definitions and annual rebalance rule.

    Point-in-time safety:
    - Safe. Realized returns apply holdings forward from each scheduled
      rebalance and use only subsequent observed returns.
    """

    calendar = returns.loc[start_date:end_date].index
    current_holdings = target_weights.copy()
    weight_rows: list[pd.Series] = [target_weights.rename(start_date)]
    return_rows = [
        {
            "Date": start_date,
            "portfolio_return": 0.0,
            "gross_return": 0.0,
            "turnover": 0.0,
            "turnover_cost": 0.0,
            "scheduled_rebalance_flag": 1,
            "rebalance_flag": 1,
        }
    ]

    rebalance_dates = set(schedule)
    for date in calendar[1:]:
        gross_vector = np.exp(returns.loc[date].reindex(ALL_SAA).fillna(0.0))
        gross_return = float((current_holdings * (gross_vector - 1.0)).sum())
        post_move_value = current_holdings * gross_vector
        denominator = float(post_move_value.sum())
        post_move_holdings = post_move_value / denominator if denominator > 0 else current_holdings.copy()

        turnover = 0.0
        turnover_cost = 0.0
        scheduled_rebalance_flag = 0
        rebalance_flag = 0
        if date in rebalance_dates:
            turnover = float((target_weights - post_move_holdings).abs().sum())
            turnover_cost = COST_PER_TURNOVER * turnover
            current_holdings = target_weights.copy()
            scheduled_rebalance_flag = 1
            rebalance_flag = 1
        else:
            current_holdings = post_move_holdings.copy()

        return_rows.append(
            {
                "Date": date,
                "portfolio_return": gross_return - turnover_cost,
                "gross_return": gross_return,
                "turnover": turnover,
                "turnover_cost": turnover_cost,
                "scheduled_rebalance_flag": scheduled_rebalance_flag,
                "rebalance_flag": rebalance_flag,
            }
        )
        weight_rows.append(target_weights.rename(date))

    weights_df = pd.DataFrame(weight_rows)
    returns_df = pd.DataFrame(return_rows).set_index("Date")
    return weights_df, returns_df


def build_benchmark(
    name: str,
    weight_map: Mapping[str, float],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    start_date: str,
    end_date: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one benchmark's weights and return history.

    Inputs:
    - `name`: benchmark label, expected to be `bm1` or `bm2`.
    - `weight_map`: static target weights from the IPS.
    - `prices`: audited price dataframe.
    - `returns`: audited log-return dataframe.
    - `start_date`: earliest requested start date.
    - `end_date`: last date to include.
    - `output_dir`: destination for CSV outputs.

    Outputs:
    - Tuple `(weights_df, returns_df)` also written to disk.

    Citation:
    - Whitmore IPS `IPS.md` §4 benchmark definitions.

    Point-in-time safety:
    - Safe. The builder uses fixed benchmark weights and observed historical
      returns only.
    """

    target_weights = normalize_target_weights(weight_map)
    schedule = build_rebalance_schedule(prices, target_weights, start_date, end_date)
    weights_df, returns_df = simulate_fixed_weight_portfolio(
        returns=returns,
        target_weights=target_weights,
        schedule=schedule,
        start_date=schedule[0],
        end_date=pd.Timestamp(end_date),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_df.to_csv(output_dir / f"{name}_weights.csv")
    returns_df.to_csv(output_dir / f"{name}_returns.csv")
    return weights_df, returns_df


def build_benchmarks(
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Build BM1 and BM2 under the IPS annual rebalance rule.

    Inputs:
    - `start_date`: earliest requested benchmark start date.
    - `end_date`: last date to include.
    - `output_dir`: destination directory for benchmark CSV outputs.

    Outputs:
    - Dictionary with keys `bm1` and `bm2`, each containing
      `(weights_df, returns_df)`.

    Citation:
    - Whitmore IPS `IPS.md` §4 and `Guidelines.md` benchmarks section.

    Point-in-time safety:
    - Safe. This is an orchestration layer over the causal benchmark builders.
    """

    prices = load_prices()
    returns = log_returns(prices)

    return {
        "bm1": build_benchmark("bm1", BM1_WEIGHTS, prices, returns, start_date, end_date, output_dir),
        "bm2": build_benchmark("bm2", BM2_WEIGHTS, prices, returns, start_date, end_date, output_dir),
    }


def main() -> None:
    """CLI entrypoint for building BM1 and BM2.

    Inputs:
    - `--start`: earliest requested benchmark start date.
    - `--end`: last date to include.
    - `--output-dir`: destination for generated CSV files.

    Outputs:
    - Writes benchmark weights and returns CSV files to disk.

    Citation:
    - Whitmore IPS `IPS.md` §4 annual benchmark rule.

    Point-in-time safety:
    - Safe. The CLI only orchestrates the causal benchmark builders.
    """

    parser = argparse.ArgumentParser(description="Build Whitmore benchmark series.")
    parser.add_argument("--start", default=DEFAULT_START, help="Earliest allowed benchmark start date.")
    parser.add_argument("--end", default=DEFAULT_END, help="Last date to include in the benchmark history.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Destination directory for output CSV files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    build_benchmarks(start_date=args.start, end_date=args.end, output_dir=output_dir)
    print(
        "Benchmark outputs written to "
        f"{output_dir / BM1_RETURNS_FILENAME}, {output_dir / BM2_RETURNS_FILENAME}, "
        f"{output_dir / BM1_WEIGHTS_FILENAME}, and {output_dir / BM2_WEIGHTS_FILENAME}"
    )


if __name__ == "__main__":
    main()
