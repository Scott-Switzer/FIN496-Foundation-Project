#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONSOLIDATED_DIR = DATA_DIR / "consolidated_csvs"
ASSET_DIR = DATA_DIR / "asset_data"
CATALOG_DIR = DATA_DIR / "catalogs"

TARGET_START_DATE = datetime.strptime("2000-01-01", "%Y-%m-%d").date()


CATEGORY_SECTORS = {
    "macro": "Macro",
    "rates": "Rates",
    "credit": "Credit",
    "equity": "Equity",
    "commodity": "Commodity",
    "fx": "FX",
    "liquidity": "Liquidity",
    "recession": "Recession",
}

FRED_TITLE_OVERRIDES = {
    "credit_Chicago_Fed_NFCI": "Chicago Fed National Financial Conditions Index",
    "credit_Chicago_Fed_NFCI_Credit_Subindex": "Chicago Fed NFCI Credit Subindex",
    "macro_Conference_Board_LEI": "Conference Board Leading Economic Index",
    "macro_OECD_CLI_USA": "OECD Composite Leading Indicator - USA",
    "macro_OECD_Consumer_Confidence": "OECD Consumer Confidence - USA",
    "macro_UMich_Consumer_Sentiment": "University of Michigan Consumer Sentiment",
    "macro_Labour_Force_Participation_Rate": "Labor Force Participation Rate",
    "macro_US_GDP_Growth_QoQ_Annualised": "US Real GDP Growth QoQ Annualized",
    "macro_US_Real_GDP_Level": "US Real GDP Level",
    "recession_Hamilton_Recession_Probability": "Hamilton Recession Probability",
    "recession_Smoothed_Recession_Probability": "Smoothed US Recession Probability",
    "liquidity_Fed_Discount_Window_Credit": "Federal Reserve Discount Window Credit",
    "liquidity_Reserve_Balances_at_Fed": "Reserve Balances at the Federal Reserve",
    "fx_JPY_USD_Exchange_Rate": "JPY / USD Exchange Rate",
    "fx_USD_AUD_Exchange_Rate": "USD / AUD Exchange Rate",
    "fx_USD_EUR_Exchange_Rate": "USD / EUR Exchange Rate",
    "fx_USD_Broad_Trade_Weighted_Index": "USD Broad Trade-Weighted Index",
}

TOKEN_OVERRIDES = {
    "bofa": "BofA",
    "cpi": "CPI",
    "core": "Core",
    "eur": "EUR",
    "fed": "Fed",
    "fred": "FRED",
    "fx": "FX",
    "gdp": "GDP",
    "hy": "HY",
    "ice": "ICE",
    "ig": "IG",
    "ism": "ISM",
    "jobless": "Jobless",
    "jp": "JP",
    "jpy": "JPY",
    "lei": "LEI",
    "m2": "M2",
    "nber": "NBER",
    "nfci": "NFCI",
    "oas": "OAS",
    "oecd": "OECD",
    "pce": "PCE",
    "qoq": "QoQ",
    "sofr": "SOFR",
    "sp500": "S&P 500",
    "ted": "TED",
    "tips": "TIPS",
    "usd": "USD",
    "umich": "UMich",
    "us": "US",
    "vix": "VIX",
    "wti": "WTI",
    "y": "Y",
}

POLICY_GROUPS = {
    "SPXT": ("SAA", "Core"),
    "FTSE100": ("SAA", "Core"),
    "LBUSTRUU": ("SAA", "Core"),
    "BROAD_TIPS": ("SAA", "Core"),
    "B3REITT": ("SAA", "Satellite"),
    "XAU": ("SAA", "Satellite"),
    "SILVER_FUT": ("SAA", "Satellite"),
    "NIKKEI225": ("SAA", "Satellite"),
    "CSI300_CHINA": ("SAA", "Satellite"),
    "BITCOIN": ("SAA", "Non-Traditional"),
    "CHF_FRANC": ("SAA", "Non-Traditional"),
    "TA-125_ISRAEL": ("Opportunistic", "Opportunistic"),
    "0_5Y_TIPS": ("Opportunistic", "Opportunistic"),
    "BAIGTRUU_ASIACREDIT": ("Opportunistic", "Opportunistic"),
    "BCEE1T_EUROAREA": ("Opportunistic", "Opportunistic"),
    "I02923JP_JAPAN_BOND": ("Opportunistic", "Opportunistic"),
    "LBEATREU_EUROBONDAGG": ("Opportunistic", "Opportunistic"),
    "COPPERSPOT": ("Opportunistic", "Opportunistic"),
    "NATURALGAS": ("Opportunistic", "Opportunistic"),
    "COFFEE_FUT": ("Opportunistic", "Opportunistic"),
    "COCOAINDEXSPOT": ("Opportunistic", "Opportunistic"),
    "COTTON_FUT": ("Opportunistic", "Opportunistic"),
    "WHEAT_SPOT": ("Opportunistic", "Opportunistic"),
    "SOYBEAN_FUT": ("Opportunistic", "Opportunistic"),
    "ETHEREUM": ("Opportunistic", "Opportunistic"),
    "AUD": ("Opportunistic", "Opportunistic"),
    "CAD": ("Opportunistic", "Opportunistic"),
    "GBP_POUND": ("Opportunistic", "Opportunistic"),
    "EURO": ("Opportunistic", "Opportunistic"),
    "CNY": ("Opportunistic", "Opportunistic"),
    "SHEKEL": ("Opportunistic", "Opportunistic"),
    "USDJPY": ("Opportunistic", "Opportunistic"),
}


def parse_date(value: str):
    value = value.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def date_to_str(value):
    return value.isoformat() if value else "N/A"


def humanize_stem(stem: str) -> str:
    if stem in FRED_TITLE_OVERRIDES:
        return FRED_TITLE_OVERRIDES[stem]

    pieces = stem.split("_")
    if pieces and pieces[0] in CATEGORY_SECTORS:
        pieces = pieces[1:]

    rendered = []
    for piece in pieces:
        token = piece.lower()
        rendered.append(TOKEN_OVERRIDES.get(token, piece.replace("-", " ").title()))
    return " ".join(rendered).replace("  ", " ").strip()


def infer_frequency(sample_dates):
    if len(sample_dates) < 2:
        return "Unknown"
    deltas = []
    for left, right in zip(sample_dates, sample_dates[1:]):
        delta = (right - left).days
        if delta > 0:
            deltas.append(delta)
    if not deltas:
        return "Unknown"
    delta = sorted(deltas)[len(deltas) // 2]
    if delta <= 1:
        return "Daily"
    if delta <= 8:
        return "Weekly"
    if delta <= 35:
        return "Monthly"
    if delta <= 100:
        return "Quarterly"
    return "Irregular"


def clean_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def format_markdown_table(rows, columns):
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(clean_table_cell(str(row.get(column, ""))) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def write_csv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_date_span(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        sample_dates = []
        first = None
        last = None
        for row in reader:
            if not row:
                continue
            date_value = parse_date(row[0])
            if not date_value:
                continue
            if any(cell.strip() for cell in row[1:]):
                if first is None:
                    first = date_value
                last = date_value
                if len(sample_dates) < 64:
                    sample_dates.append(date_value)
        return first, last, infer_frequency(sample_dates), header


def measurement_from_text(text: str, sector: str) -> str:
    lowered = text.lower()
    if lowered.endswith("_vol") or "volume" in lowered:
        return "Trading volume"
    if sector == "FX" or "exchange rate" in lowered:
        return "FX spot rate"
    if "logret" in lowered or "log return" in lowered:
        return "Daily log return"
    if "spread" in lowered or "oas" in lowered:
        return "Spread / percentage points"
    if "yield curve" in lowered or "yieldcurve" in lowered:
        return "Spread / percentage points"
    if "yield" in lowered or "rate" in lowered or "unemployment" in lowered or "participation" in lowered:
        return "Percent / rate"
    if "probability" in lowered:
        return "Probability"
    if "indicator" in lowered:
        return "Binary indicator (0/1)"
    if "price" in lowered or "spot" in lowered:
        return "Price level"
    if "return index" in lowered or "total return index" in lowered:
        return "Index level"
    if sector in {"FX", "Commodity", "Equity", "Crypto", "Real Estate", "Fixed Income"}:
        return "Price / index level"
    if "index" in lowered or "sentiment" in lowered or "confidence" in lowered or "conditions" in lowered:
        return "Index / level"
    return "Level"


def describe_fred_file(path: Path):
    stem = path.stem
    if stem == "_pull_log":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            starts = []
            ends = []
            for row in reader:
                if row["status"] == "SUCCESS":
                    start = parse_date(row["start_date"])
                    end = parse_date(row["end_date"])
                    if start:
                        starts.append(start)
                    if end:
                        ends.append(end)
        return {
            "dataset_name": "FRED Pull Log",
            "description": "Audit log of successful and failed FRED extractions.",
            "start_date": date_to_str(min(starts) if starts else None),
            "end_date": date_to_str(max(ends) if ends else None),
            "frequency": "Per pull",
            "measurement": "Metadata log",
            "sector": "Metadata",
            "contents_summary": "fred_code, series_name, start_date, end_date, observations, status",
        }

    first, last, frequency, header = read_date_span(path)
    sector = CATEGORY_SECTORS.get(stem.split("_", 1)[0], "Unknown")
    dataset_name = humanize_stem(stem)
    measurement = measurement_from_text(stem, sector)
    return {
        "dataset_name": dataset_name,
        "description": f"Single-series FRED extract for {dataset_name}.",
        "start_date": date_to_str(first),
        "end_date": date_to_str(last),
        "frequency": frequency,
        "measurement": measurement,
        "sector": sector,
        "contents_summary": header[1] if len(header) > 1 else dataset_name,
    }


def describe_bloomberg_file(path: Path):
    first, last, frequency, header = read_date_span(path)
    filename = path.name
    if filename == "_data_manifest_bloomberg.csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            starts = []
            ends = []
            series = []
            for row in reader:
                start = parse_date(row["data_start"])
                end = parse_date(row["data_end"])
                if start:
                    starts.append(start)
                if end:
                    ends.append(end)
                series.append(row["series_name"])
        return {
            "dataset_name": "Bloomberg Data Manifest",
            "description": "Series-level metadata manifest for the Bloomberg input universe.",
            "start_date": date_to_str(min(starts) if starts else None),
            "end_date": date_to_str(max(ends) if ends else None),
            "frequency": "Metadata",
            "measurement": "Metadata table",
            "sector": "Metadata",
            "contents_summary": f"{len(series)} Bloomberg series",
        }
    if filename == "_data_quality_bloomberg.csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            starts = [parse_date(row["data_start"]) for row in reader if row["data_start"]]
        return {
            "dataset_name": "Bloomberg Data Quality Report",
            "description": "Coverage and quality diagnostics for the Bloomberg input universe.",
            "start_date": date_to_str(min(starts) if starts else None),
            "end_date": "N/A",
            "frequency": "Metadata",
            "measurement": "Metadata table",
            "sector": "Metadata",
            "contents_summary": "series, n_obs, pct_valid, data_start, issues, status",
        }

    label = filename.replace(".csv", "")
    if "logret" in label or "log_return" in label:
        description = "Daily Bloomberg-derived log return panel."
        measurement = "Daily log return"
    else:
        description = "Daily Bloomberg-derived price and level panel."
        measurement = "Price / index level"
    return {
        "dataset_name": label,
        "description": description,
        "start_date": date_to_str(first),
        "end_date": date_to_str(last),
        "frequency": frequency,
        "measurement": measurement,
        "sector": "Multi-Asset",
        "contents_summary": f"{max(len(header) - 1, 0)} series",
    }


def describe_asset_files():
    key_path = ASSET_DIR / "data_key.csv"
    panel_path = ASSET_DIR / "whitmore_daily.csv"

    key_rows = []
    with key_path.open(newline="", encoding="utf-8") as handle:
        key_rows = list(csv.DictReader(handle))

    panel_first, panel_last, panel_frequency, panel_header = read_date_span(panel_path)

    file_rows = [
        {
            "source_group": "Asset Data",
            "relative_path": "data/asset_data/data_key.csv",
            "dataset_name": "Asset Data Key",
            "description": "Lookup table for the Whitmore daily asset panel columns.",
            "start_date": "N/A",
            "end_date": "N/A",
            "frequency": "Metadata",
            "measurement": "Metadata table",
            "sector": "Metadata",
            "contents_summary": f"{len(key_rows)} mapped asset columns",
        },
        {
            "source_group": "Asset Data",
            "relative_path": "data/asset_data/whitmore_daily.csv",
            "dataset_name": "Whitmore Daily Asset Panel",
            "description": "Daily multi-asset price panel used for SAA and opportunistic asset collection.",
            "start_date": date_to_str(panel_first),
            "end_date": date_to_str(panel_last),
            "frequency": panel_frequency,
            "measurement": "Price / index level, plus *_VOL trading volume columns",
            "sector": "Multi-Asset",
            "contents_summary": f"{max(len(panel_header) - 1, 0)} columns",
        },
    ]

    return file_rows


def build_file_inventory():
    inventory_path = CONSOLIDATED_DIR / "csv_inventory.csv"
    rows = []
    with inventory_path.open(newline="", encoding="utf-8") as handle:
        inventory = csv.DictReader(handle)
        for item in inventory:
            if item["included"] != "yes":
                continue
            relative_part = item["consolidated_path"].replace("Data/consolidated_csvs/", "", 1)
            repo_relative = Path("data/consolidated_csvs") / relative_part
            absolute_path = ROOT / repo_relative
            source_group = "FRED" if "fred/" in relative_part else "Bloomberg"
            if "fred/raw/" in relative_part or item["filename"] == "_pull_log.csv":
                metadata = describe_fred_file(absolute_path)
            elif "fred/master/" in relative_part:
                first, last, frequency, header = read_date_span(absolute_path)
                metadata = {
                    "dataset_name": "FRED Master Table",
                    "description": "Canonical combined FRED input table for the pipeline.",
                    "start_date": date_to_str(first),
                    "end_date": date_to_str(last),
                    "frequency": frequency,
                    "measurement": "Mixed levels, rates, spreads, and indicators",
                    "sector": "Multi-Sector",
                    "contents_summary": f"{max(len(header) - 1, 0)} FRED factors",
                }
            else:
                metadata = describe_bloomberg_file(absolute_path)

            rows.append(
                {
                    "source_group": source_group,
                    "relative_path": repo_relative.as_posix(),
                    **metadata,
                }
            )

    rows.extend(describe_asset_files())
    rows.sort(key=lambda row: (row["source_group"], row["relative_path"]))
    return rows


def load_asset_key():
    key_path = ASSET_DIR / "data_key.csv"
    with key_path.open(newline="", encoding="utf-8") as handle:
        return {row["Column_Name"]: row for row in csv.DictReader(handle)}


def scan_panel_columns(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        columns = header[1:]
        first_dates = {column: None for column in columns}
        last_dates = {column: None for column in columns}
        sample_dates = []

        for row in reader:
            if not row:
                continue
            date_value = parse_date(row[0])
            if not date_value:
                continue
            if len(sample_dates) < 64:
                sample_dates.append(date_value)
            for index, column in enumerate(columns, start=1):
                value = row[index].strip() if index < len(row) else ""
                if value:
                    if first_dates[column] is None:
                        first_dates[column] = date_value
                    last_dates[column] = date_value

    return columns, first_dates, last_dates, infer_frequency(sample_dates)


def build_asset_series_inventory():
    key_lookup = load_asset_key()
    panel_path = ASSET_DIR / "whitmore_daily.csv"
    columns, first_dates, last_dates, frequency = scan_panel_columns(panel_path)

    rows = []
    for column in columns:
        if column in key_lookup:
            key_row = key_lookup[column]
            description = key_row["Full_Name"]
            sector = key_row["Asset_Class"]
            currency = key_row["Currency"]
            measurement = measurement_from_text(description, sector)
            notes = key_row["Sub_Category"]
        elif column.endswith("_VOL"):
            base = column[: -len("_VOL")]
            base_row = key_lookup.get(base, {})
            description = f"{base_row.get('Full_Name', base)} trading volume"
            sector = f"{base_row.get('Asset_Class', 'Unknown')} volume"
            currency = base_row.get("Currency", "N/A")
            measurement = "Trading volume"
            notes = "Volume companion column in whitmore_daily.csv"
        else:
            description = column
            sector = "Unknown"
            currency = "Unknown"
            measurement = "Level"
            notes = ""

        rows.append(
            {
                "column_name": column,
                "description": description,
                "start_date": date_to_str(first_dates[column]),
                "end_date": date_to_str(last_dates[column]),
                "frequency": frequency,
                "measurement": measurement,
                "sector": sector,
                "currency": currency,
                "notes": notes,
            }
        )

    rows.sort(key=lambda row: row["column_name"])
    return rows


def find_portfolio_start(asset_rows):
    eligibility = {}
    for row in asset_rows:
        if row["column_name"] not in POLICY_GROUPS:
            continue
        start = parse_date(row["start_date"]) if row["start_date"] != "N/A" else None
        end = parse_date(row["end_date"]) if row["end_date"] != "N/A" else None
        eligibility[row["column_name"]] = (start, end)

    panel_path = ASSET_DIR / "whitmore_daily.csv"
    with panel_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if not row:
                continue
            date_value = parse_date(row[0])
            if not date_value or date_value < TARGET_START_DATE:
                continue

            class_counts = defaultdict(int)
            for asset, (start, end) in eligibility.items():
                if start and start <= date_value and (end is None or date_value <= end):
                    _, classification = POLICY_GROUPS[asset]
                    class_counts[classification] += 1

            core_capacity = 45 * class_counts["Core"]
            satellite_capacity = min(45 * class_counts["Satellite"], 45)
            non_traditional_capacity = min(45 * class_counts["Non-Traditional"], 20)

            if core_capacity >= 40 and (core_capacity + satellite_capacity + non_traditional_capacity) >= 100:
                return date_value
    return None


def build_policy_asset_availability(asset_rows):
    portfolio_start = find_portfolio_start(asset_rows)
    portfolio_start_text = date_to_str(portfolio_start)

    rows = []
    for row in asset_rows:
        column_name = row["column_name"]
        if column_name not in POLICY_GROUPS:
            continue
        group, classification = POLICY_GROUPS[column_name]
        start = parse_date(row["start_date"]) if row["start_date"] != "N/A" else None
        at_start = "yes" if portfolio_start and start and start <= portfolio_start else "no"
        rows.append(
            {
                "asset": column_name,
                "description": row["description"],
                "group": group,
                "classification": classification,
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "measurement": row["measurement"],
                "sector": row["sector"],
                "available_at_portfolio_start": at_start,
                "portfolio_start_date": portfolio_start_text,
            }
        )

    risk_free_path = CONSOLIDATED_DIR / "fred" / "raw" / "rates_US_3M_Treasury_Yield.csv"
    first, last, frequency, _ = read_date_span(risk_free_path)
    rows.append(
        {
            "asset": "US_3M_TREASURY_YIELD",
            "description": "US 3-Month Treasury Yield (risk-free rate)",
            "group": "Risk-Free",
            "classification": "Risk-Free",
            "start_date": date_to_str(first),
            "end_date": date_to_str(last),
            "measurement": "Percent / rate",
            "sector": "Rates",
            "available_at_portfolio_start": "yes" if first and portfolio_start and first <= portfolio_start else "no",
            "portfolio_start_date": portfolio_start_text,
        }
    )

    rows.sort(key=lambda row: (row["group"], row["asset"]))
    return rows, portfolio_start_text


def write_markdown_outputs(file_rows, asset_rows, policy_rows, portfolio_start_text):
    file_table_columns = [
        "source_group",
        "dataset_name",
        "start_date",
        "end_date",
        "frequency",
        "measurement",
        "sector",
        "relative_path",
    ]
    asset_table_columns = [
        "column_name",
        "description",
        "start_date",
        "end_date",
        "frequency",
        "measurement",
        "sector",
        "currency",
    ]
    policy_table_columns = [
        "asset",
        "group",
        "classification",
        "start_date",
        "end_date",
        "measurement",
        "sector",
        "available_at_portfolio_start",
    ]

    summary = [
        "# Data Catalog",
        "",
        f"- Consolidated CSV files cataloged: {sum(1 for row in file_rows if row['source_group'] != 'Asset Data')}",
        f"- Asset data CSV files cataloged: {sum(1 for row in file_rows if row['source_group'] == 'Asset Data')}",
        f"- Asset panel columns cataloged: {len(asset_rows)}",
        f"- Portfolio start date satisfying hard availability constraints: {portfolio_start_text}",
        "",
        "## File Inventory",
        "",
        format_markdown_table(file_rows, file_table_columns),
        "",
        "## Asset Panel Contents",
        "",
        format_markdown_table(asset_rows, asset_table_columns),
        "",
        "## Policy Asset Availability",
        "",
        format_markdown_table(policy_rows, policy_table_columns),
        "",
    ]

    (CATALOG_DIR / "data_catalog.md").write_text("\n".join(summary), encoding="utf-8")


def main():
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    file_rows = build_file_inventory()
    asset_rows = build_asset_series_inventory()
    policy_rows, portfolio_start_text = build_policy_asset_availability(asset_rows)

    write_csv(
        CATALOG_DIR / "file_inventory.csv",
        file_rows,
        [
            "source_group",
            "dataset_name",
            "description",
            "start_date",
            "end_date",
            "frequency",
            "measurement",
            "sector",
            "contents_summary",
            "relative_path",
        ],
    )
    write_csv(
        CATALOG_DIR / "asset_series_inventory.csv",
        asset_rows,
        [
            "column_name",
            "description",
            "start_date",
            "end_date",
            "frequency",
            "measurement",
            "sector",
            "currency",
            "notes",
        ],
    )
    write_csv(
        CATALOG_DIR / "policy_asset_availability.csv",
        policy_rows,
        [
            "asset",
            "description",
            "group",
            "classification",
            "start_date",
            "end_date",
            "measurement",
            "sector",
            "available_at_portfolio_start",
            "portfolio_start_date",
        ],
    )
    write_markdown_outputs(file_rows, asset_rows, policy_rows, portfolio_start_text)


if __name__ == "__main__":
    main()
