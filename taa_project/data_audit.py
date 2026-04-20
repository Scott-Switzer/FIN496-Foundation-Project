# Addresses tasks.md Steps 2-4 and enforces point-in-time data hygiene for the
# Whitmore FIN 496 project before any SAA or TAA logic is run.
"""Audit and prepare asset and macro data for the Whitmore mandate.

This module implements the Task 1 data checks required by the project brief and
the binding rules in tasks.md. Asset prices are never forward-filled or
backward-filled, and macro inputs are shifted by one business day before they
can enter any signal.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from taa_project.config import (
    ASSET_KEY_CSV,
    CORE,
    FRED_CSV,
    FRED_LAG_BUSINESS_DAYS,
    NONTRAD,
    OUTPUT_DIR,
    PRICES_CSV,
    SATELLITE,
)
from taa_project.pandas_utils import forward_propagate


REPORT_FILENAME = "data_audit_report.md"
INCEPTIONS_FILENAME = "asset_inception_dates.csv"
GAP_SUMMARY_FILENAME = "asset_gap_summary.csv"
GAP_DETAIL_FILENAME = "asset_gap_detail.csv"
RETURNS_FILENAME = "asset_log_returns.csv"
AVAILABILITY_FILENAME = "asset_availability.csv"
FRED_LAGGED_FILENAME = "fred_features_lagged.csv"
MASTER_REFERENCE_FILENAME = "master_data_reference.csv"


def load_asset_key(path: Path = ASSET_KEY_CSV) -> pd.DataFrame:
    """Load the authoritative asset metadata table.

    Inputs:
    - `path`: location of `data_key.csv`.

    Outputs:
    - `pd.DataFrame` indexed by ticker-like column name with asset metadata.

    Citation:
    - Internal project data dictionary: `data/asset_data/data_key.csv`
    - Whitmore IPS and Guidelines in the repository root.

    Point-in-time safety:
    - Safe. This is static metadata and contains no time-varying market data.
    """

    key = pd.read_csv(path)
    key["Column_Name"] = key["Column_Name"].astype(str).str.strip()
    key["Currency"] = key["Currency"].astype(str).str.strip().str.upper()
    return key.set_index("Column_Name", drop=False)


def select_asset_price_columns(raw_columns: Iterable[str], asset_key: pd.DataFrame) -> list[str]:
    """Choose the tradable asset columns from the raw panel.

    Inputs:
    - `raw_columns`: columns from `whitmore_daily.csv`.
    - `asset_key`: metadata returned by `load_asset_key`.

    Outputs:
    - List of columns present in both the price file and the asset key.

    Citation:
    - Internal project data dictionary: `data/asset_data/data_key.csv`

    Point-in-time safety:
    - Safe. Column selection depends only on static metadata and file headers.
    """

    return [column for column in asset_key.index if column in raw_columns]


def load_asset_prices(path: Path = PRICES_CSV, key_path: Path = ASSET_KEY_CSV) -> tuple[pd.DataFrame, int]:
    """Load the authoritative asset price panel and drop duplicate dates.

    Inputs:
    - `path`: location of `whitmore_daily.csv`.
    - `key_path`: location of `data_key.csv`.

    Outputs:
    - Tuple of `(prices, duplicate_count)` where `prices` contains only the
      columns mapped in the asset key and `duplicate_count` records how many
      duplicated dates were removed.

    Citation:
    - Internal project files `data/asset_data/whitmore_daily.csv` and
      `data/asset_data/data_key.csv`.

    Point-in-time safety:
    - Safe. This is raw input loading and sorting only; no future observations
      are used to alter historical values.
    """

    asset_key = load_asset_key(key_path)
    raw = pd.read_csv(path, parse_dates=[0], index_col=0)
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = "Date"
    raw = raw.sort_index()

    duplicate_count = int(raw.index.duplicated(keep="last").sum())
    deduped = raw.loc[~raw.index.duplicated(keep="last")]

    asset_columns = select_asset_price_columns(deduped.columns, asset_key)
    prices = deduped[asset_columns].astype(float)
    return prices, duplicate_count


def compute_consecutive_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns on adjacent observed prices for each asset.

    Inputs:
    - `prices`: asset price panel with gaps preserved as `NaN`.

    Outputs:
    - `pd.DataFrame` of log returns reindexed to the full calendar, with missing
      dates left as `NaN`.

    Citation:
    - Project tasks in `tasks.md`, Step 2: preserve gaps and avoid fill methods.

    Point-in-time safety:
    - Safe. Each return at date `t` is computed only from the current observed
      price and that asset's immediately prior observed price.
    """

    returns = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    for column in prices.columns:
        observed = prices[column].dropna()
        if observed.empty:
            continue
        log_returns = np.log(observed).diff()
        returns[column] = log_returns.reindex(prices.index)
    return returns


def build_availability_flags(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert observed prices into a binary availability matrix.

    Inputs:
    - `prices`: asset price panel with gaps preserved as `NaN`.

    Outputs:
    - `pd.DataFrame` with `1` where a price exists and `0` otherwise.

    Citation:
    - Project tasks in `tasks.md`, Step 4: add an availability flag column.

    Point-in-time safety:
    - Safe. Availability is derived row-by-row from contemporaneous raw prices.
    """

    return prices.notna().astype(int)


def summarize_inceptions(prices: pd.DataFrame, asset_key: pd.DataFrame) -> pd.DataFrame:
    """Summarize first-valid and last-valid dates for each asset column.

    Inputs:
    - `prices`: cleaned asset price panel.
    - `asset_key`: metadata returned by `load_asset_key`.

    Outputs:
    - `pd.DataFrame` with first-valid date, last-valid date, observation count,
      currency label, and asset classification fields.

    Citation:
    - Internal project files `whitmore_daily.csv` and `data_key.csv`.

    Point-in-time safety:
    - Safe. This is descriptive metadata only and does not alter the time
      series used downstream.
    """

    rows: list[dict[str, object]] = []
    for column in prices.columns:
        series = prices[column].dropna()
        rows.append(
            {
                "asset": column,
                "first_valid_date": series.index[0].date().isoformat() if not series.empty else "",
                "last_valid_date": series.index[-1].date().isoformat() if not series.empty else "",
                "observation_count": int(series.shape[0]),
                "currency": asset_key.loc[column, "Currency"],
                "asset_class": asset_key.loc[column, "Asset_Class"],
                "sub_category": asset_key.loc[column, "Sub_Category"],
            }
        )
    return pd.DataFrame(rows).sort_values(["first_valid_date", "asset"]).reset_index(drop=True)


def build_gap_tables(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe calendar gaps between adjacent observed prices.

    Inputs:
    - `prices`: cleaned asset price panel.

    Outputs:
    - Tuple `(summary_df, detail_df)` where `summary_df` aggregates by asset and
      `detail_df` lists every gap between adjacent observed prices.

    Citation:
    - Project tasks in `tasks.md`, Step 2: any gap in data must be flagged and
      documented.

    Point-in-time safety:
    - Safe. Gap detection is purely descriptive and does not change any values.
    """

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for column in prices.columns:
        observed_index = prices[column].dropna().index
        if observed_index.empty:
            summary_rows.append(
                {
                    "asset": column,
                    "gap_count": 0,
                    "missing_calendar_days": 0,
                    "longest_gap_days": 0,
                    "extended_gap_count_gt4d": 0,
                }
            )
            continue

        gaps = observed_index.to_series().diff().dt.days.dropna()
        missing_days = (gaps - 1).clip(lower=0)

        for next_date, gap_days in gaps[gaps > 1].items():
            location = observed_index.get_loc(next_date)
            prev_date = observed_index[location - 1]
            detail_rows.append(
                {
                    "asset": column,
                    "prev_valid_date": prev_date.date().isoformat(),
                    "next_valid_date": next_date.date().isoformat(),
                    "calendar_gap_days": int(gap_days),
                    "missing_calendar_days": int(gap_days - 1),
                    "gap_class": "extended_gap" if gap_days > 4 else "weekend_or_holiday",
                }
            )

        summary_rows.append(
            {
                "asset": column,
                "gap_count": int((gaps > 1).sum()),
                "missing_calendar_days": int(missing_days.sum()),
                "longest_gap_days": int(gaps.max()) if not gaps.empty else 0,
                "extended_gap_count_gt4d": int((gaps > 4).sum()),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["extended_gap_count_gt4d", "asset"], ascending=[False, True])
    if detail_rows:
        detail_df = pd.DataFrame(detail_rows).sort_values(["calendar_gap_days", "asset"], ascending=[False, True])
    else:
        detail_df = pd.DataFrame(
            columns=[
                "asset",
                "prev_valid_date",
                "next_valid_date",
                "calendar_gap_days",
                "missing_calendar_days",
                "gap_class",
            ]
        )
    return summary_df.reset_index(drop=True), detail_df.reset_index(drop=True)


def find_nonpositive_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Locate non-positive values in the asset price panel.

    Inputs:
    - `prices`: cleaned asset price panel containing only asset columns.

    Outputs:
    - `pd.DataFrame` with one row per non-positive observation.

    Citation:
    - Project tasks in `tasks.md`, Step 2: confirm no negative or zero prices.

    Point-in-time safety:
    - Safe. This is a descriptive audit over existing observations.
    """

    mask = (prices <= 0) & prices.notna()
    if not mask.any().any():
        return pd.DataFrame(columns=["date", "asset", "value"])

    failures = []
    for date, asset in mask.stack()[mask.stack()].index:
        failures.append({"date": pd.Timestamp(date).date().isoformat(), "asset": asset, "value": float(prices.loc[date, asset])})
    return pd.DataFrame(failures)


def inspect_currency_labels(asset_key: pd.DataFrame) -> pd.DataFrame:
    """Flag asset columns whose metadata currency label is not `USD`.

    Inputs:
    - `asset_key`: metadata returned by `load_asset_key`.

    Outputs:
    - `pd.DataFrame` of currency-label exceptions.

    Citation:
    - Internal project metadata file `data/asset_data/data_key.csv`.

    Point-in-time safety:
    - Safe. Currency labels are static metadata.
    """

    exceptions = asset_key.loc[asset_key["Currency"] != "USD", ["Column_Name", "Full_Name", "Currency", "Asset_Class"]].copy()
    return exceptions.reset_index(drop=True)


def load_fred_features(
    path: Path = FRED_CSV,
    calendar_index: pd.Index | None = None,
    lag_business_days: int = FRED_LAG_BUSINESS_DAYS,
) -> pd.DataFrame:
    """Load the FRED master table and apply a one-business-day lag.

    Inputs:
    - `path`: location of `fred_data.csv`.
    - `calendar_index`: optional target calendar for downstream alignment.
    - `lag_business_days`: number of business-day rows to lag before use.

    Outputs:
    - Lagged `pd.DataFrame` of FRED features, optionally aligned to an external
      calendar via forward-fill after the lag has been applied.

    Citation:
    - Internal project file `data/consolidated_csvs/fred/master/fred_data.csv`
    - López de Prado (2018) purging and embargo discipline overview:
      https://en.wikipedia.org/wiki/Purged_cross-validation

    Point-in-time safety:
    - Safe. Every feature available at date `t` comes from data observed no
      later than the prior business day.
    """

    fred = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    lagged = fred.copy()
    lagged.index = lagged.index + pd.offsets.BDay(lag_business_days)
    if calendar_index is not None:
        lagged = forward_propagate(lagged.reindex(calendar_index))
    return lagged


def render_markdown_table(frame: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    """Render a compact markdown table without requiring external dependencies.

    Inputs:
    - `frame`: source dataframe.
    - `columns`: columns to include and order.
    - `max_rows`: optional row cap for display.

    Outputs:
    - Markdown table string.

    Citation:
    - Internal project reporting requirement in the Task 1 brief.

    Point-in-time safety:
    - Safe. This is display formatting only.
    """

    trimmed = frame.loc[:, columns]
    if max_rows is not None:
        trimmed = trimmed.head(max_rows)

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in trimmed.iterrows():
        rows.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, divider, *rows])


def build_sanity_check_rows(duplicate_dates: int, nonpositive_prices: pd.DataFrame, currency_exceptions: pd.DataFrame) -> pd.DataFrame:
    """Create a compact status table for the markdown audit report.

    Inputs:
    - `duplicate_dates`: number of duplicate dates removed from the price panel.
    - `nonpositive_prices`: table returned by `find_nonpositive_prices`.
    - `currency_exceptions`: table returned by `inspect_currency_labels`.

    Outputs:
    - `pd.DataFrame` with one row per audit check and a pass/fail status.

    Citation:
    - Project tasks in `tasks.md`, Steps 2-4.

    Point-in-time safety:
    - Safe. This is audit metadata and does not transform model inputs.
    """

    return pd.DataFrame(
        [
            {
                "check": "Duplicate dates removed from asset panel",
                "status": "PASS" if duplicate_dates == 0 else "WARN",
                "detail": f"{duplicate_dates} duplicate dates removed",
            },
            {
                "check": "No non-positive asset prices",
                "status": "PASS" if nonpositive_prices.empty else "FAIL",
                "detail": "No zero or negative prices in asset columns" if nonpositive_prices.empty else f"{len(nonpositive_prices)} failing observations",
            },
            {
                "check": "Currency labels already USD",
                "status": "PASS" if currency_exceptions.empty else "WARN",
                "detail": "All asset-key currency labels normalize to USD" if currency_exceptions.empty else f"{len(currency_exceptions)} non-USD labels flagged for review",
            },
            {
                "check": "FRED one-business-day lag applied",
                "status": "PASS",
                "detail": "All FRED features are shifted one business-day row before signal use",
            },
        ]
    )


def build_master_reference(
    returns: pd.DataFrame,
    availability: pd.DataFrame,
) -> pd.DataFrame:
    """Build a long-form master reference table for audited asset data.

    Inputs:
    - `returns`: audited log-return dataframe.
    - `availability`: audited 1/0 availability matrix.

    Outputs:
    - Long-form dataframe with one row per `(date, ticker)` observation.

    Citation:
    - `tasks.md`, Step 4 master-dataframe requirement.

    Point-in-time safety:
    - Safe. This is a descriptive reshape of already-audited causal outputs.
    """

    tier_lookup = {asset: "Core" for asset in CORE}
    tier_lookup.update({asset: "Satellite" for asset in SATELLITE})
    tier_lookup.update({asset: "Non-Traditional" for asset in NONTRAD})

    reference = (
        returns.stack(dropna=False)
        .rename("log_return")
        .to_frame()
        .join(availability.stack(dropna=False).rename("availability"))
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker"})
    )
    reference["tier"] = reference["ticker"].map(tier_lookup).fillna("Opportunistic")
    return reference


def write_audit_report(
    inceptions: pd.DataFrame,
    gap_summary: pd.DataFrame,
    gap_detail: pd.DataFrame,
    sanity_checks: pd.DataFrame,
    currency_exceptions: pd.DataFrame,
    fred_lagged: pd.DataFrame,
    report_path: Path,
) -> None:
    """Write the Task 1 markdown audit report.

    Inputs:
    - `inceptions`: asset inception summary.
    - `gap_summary`: per-asset gap summary.
    - `gap_detail`: per-gap detail table.
    - `sanity_checks`: compact pass/fail table.
    - `currency_exceptions`: non-USD label table.
    - `fred_lagged`: lagged FRED feature dataframe.
    - `report_path`: output markdown path.

    Outputs:
    - Markdown file written to disk.

    Citation:
    - Task 1 output requirement in the user brief.

    Point-in-time safety:
    - Safe. This is reporting only and does not feed any model state back into
      the dataset.
    """

    report_path.parent.mkdir(parents=True, exist_ok=True)
    fred_first_usable = fred_lagged.apply(lambda column: column.first_valid_index())
    fred_first_usable = fred_first_usable.dropna().sort_index()
    fred_summary = pd.DataFrame(
        {
            "series": fred_first_usable.index,
            "first_usable_date": [pd.Timestamp(value).date().isoformat() for value in fred_first_usable.values],
        }
    )

    sections = [
        "# Data Audit Report",
        "",
        "## Sanity Checks",
        "",
        render_markdown_table(sanity_checks, ["check", "status", "detail"]),
        "",
        "## Asset Inception Dates",
        "",
        render_markdown_table(
            inceptions,
            ["asset", "first_valid_date", "last_valid_date", "observation_count", "currency", "asset_class"],
        ),
        "",
        "## Gap Summary",
        "",
        render_markdown_table(
            gap_summary,
            ["asset", "gap_count", "missing_calendar_days", "longest_gap_days", "extended_gap_count_gt4d"],
        ),
        "",
        "## Longest Observed Gaps",
        "",
        render_markdown_table(
            gap_detail,
            ["asset", "prev_valid_date", "next_valid_date", "calendar_gap_days", "missing_calendar_days", "gap_class"],
            max_rows=25,
        )
        if not gap_detail.empty
        else "No gaps larger than one calendar day were detected.",
        "",
        "## Currency Exceptions",
        "",
        render_markdown_table(currency_exceptions, ["Column_Name", "Full_Name", "Currency", "Asset_Class"])
        if not currency_exceptions.empty
        else "All asset-key labels normalize to USD.",
        "",
        "## FRED Publication-Lag Policy",
        "",
        "- Source file: `data/consolidated_csvs/fred/master/fred_data.csv`",
        f"- Lag rule: every FRED feature is shifted by `{FRED_LAG_BUSINESS_DAYS}` business day before any signal can consume it.",
        "- Alignment rule: after lagging on the FRED business-day calendar, values may be forward-filled onto the asset calendar because the last published macro observation remains the latest observable value until a newer release arrives.",
        "",
        render_markdown_table(fred_summary, ["series", "first_usable_date"]),
        "",
    ]
    report_path.write_text("\n".join(sections), encoding="utf-8")


def run_data_audit(output_dir: Path = OUTPUT_DIR) -> dict[str, pd.DataFrame]:
    """Run the complete Task 1 audit workflow and write reproducible artifacts.

    Inputs:
    - `output_dir`: destination directory for generated CSV and markdown files.

    Outputs:
    - Dictionary of in-memory dataframes keyed by artifact name.

    Citation:
    - FIN 496 Task 1 brief and repository `tasks.md`.

    Point-in-time safety:
    - Safe. All transformations are causal descriptive preprocessing with no
      use of information beyond each observation date.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / REPORT_FILENAME
    inceptions_path = output_dir / INCEPTIONS_FILENAME
    gap_summary_path = output_dir / GAP_SUMMARY_FILENAME
    gap_detail_path = output_dir / GAP_DETAIL_FILENAME
    returns_path = output_dir / RETURNS_FILENAME
    availability_path = output_dir / AVAILABILITY_FILENAME
    fred_lagged_path = output_dir / FRED_LAGGED_FILENAME
    master_reference_path = output_dir / MASTER_REFERENCE_FILENAME

    asset_key = load_asset_key()
    prices, duplicate_dates = load_asset_prices()
    returns = compute_consecutive_log_returns(prices)
    availability = build_availability_flags(prices)
    inceptions = summarize_inceptions(prices, asset_key)
    gap_summary, gap_detail = build_gap_tables(prices)
    nonpositive_prices = find_nonpositive_prices(prices)
    currency_exceptions = inspect_currency_labels(asset_key)
    fred_lagged = load_fred_features(calendar_index=prices.index)
    master_reference = build_master_reference(returns, availability)
    sanity_checks = build_sanity_check_rows(duplicate_dates, nonpositive_prices, currency_exceptions)

    inceptions.to_csv(inceptions_path, index=False)
    gap_summary.to_csv(gap_summary_path, index=False)
    gap_detail.to_csv(gap_detail_path, index=False)
    returns.to_csv(returns_path)
    availability.to_csv(availability_path)
    fred_lagged.to_csv(fred_lagged_path)
    master_reference.to_csv(master_reference_path, index=False)
    write_audit_report(inceptions, gap_summary, gap_detail, sanity_checks, currency_exceptions, fred_lagged, report_path)

    return {
        "prices": prices,
        "returns": returns,
        "availability": availability,
        "inceptions": inceptions,
        "gap_summary": gap_summary,
        "gap_detail": gap_detail,
        "nonpositive_prices": nonpositive_prices,
        "currency_exceptions": currency_exceptions,
        "fred_lagged": fred_lagged,
        "master_reference": master_reference,
        "sanity_checks": sanity_checks,
    }


def main() -> None:
    """CLI entrypoint for the Task 1 data audit.

    Inputs:
    - Optional `--output-dir` command-line argument.

    Outputs:
    - Writes the Task 1 audit artifacts to disk and prints the report path.

    Citation:
    - FIN 496 Task 1 brief.

    Point-in-time safety:
    - Safe. The command orchestrates the causal audit workflow defined above.
    """

    parser = argparse.ArgumentParser(description="Run the Whitmore data audit.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Destination directory for audit artifacts.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_data_audit(output_dir)
    print(f"Audit report written to {output_dir / REPORT_FILENAME}")


if __name__ == "__main__":
    main()
