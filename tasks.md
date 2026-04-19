## Step 1: Data Collection
- Data references:
      - `data/catalogs/data_catalog.md`
      - `data/catalogs/file_inventory.csv`
      - `data/catalogs/policy_asset_availability.csv`
- [x] Pull price data for all SAA assets:
      SPXT, FTSE100, LBUSTRUU, BROAD_TIPS,
      B3REITT, XAU, SILVER_FUT, NIKKEI225,
      CSI300_CHINA, BITCOIN, CHF_FRANC
- [x] Pull price data for all Opportunistic assets (Appendix A)
- [x] Pull 3-month US Treasury yield (risk-free rate)
- [x] Target start date reviewed from **early 2000s**; first feasible trading date is **2000-01-03**
- [x] Log the earliest available date for every asset

## Step 2: Data Cleaning
- [x] Convert all prices to **USD terms**
- [x] Handle missing values:
      - **Do not forward-fill any values**
      - **Do not backward-fill any values**
      - Missing days (e.g. holidays) → treat as **no trading day**
        and exclude from return calculations entirely
      - Any gap in data must be flagged and documented
      - If a gap is structural (asset suspended, market closed)
        → treat as no return for that period, not zero return
- [x] Remove duplicate dates
- [x] Confirm no negative or zero prices
- [x] Calculate **daily returns** only between consecutive
      **available** observations — never across gaps
- [x] All cleaning decisions must be made using only
      **information available at that point in time**
      — no future data may influence any cleaning step

## Step 3: Determine Start Date & Asset Eligibility
- [x] Do **not** require all assets to exist simultaneously
- [x] Starting from early 2000s, identify which assets are available
- [x] At the target start date check if IPS hard constraints
      can be satisfied with available assets only:
      - Core ≥ 40% (SPXT and/or LBUSTRUU must be available)
      - Satellite ≤ 45%
      - Non-Traditional ≤ 20%
      - Single sleeve ≤ 45%
      - Portfolio sums to 100%
      - No short positions
- [x] If constraints can be met → use that date as **portfolio start date**
- [x] If constraints cannot be met → step forward day by day
      until they can be met
- [x] Assets not yet available at start date are **excluded** until
      their data begins — do not backfill or substitute
- [x] When a new asset becomes available mid-history:
      - Introduce it at the **next scheduled rebalance date**
      - Confirm constraints still satisfied after introduction
      - Document the date it was added and weight assigned
- [x] Log final start date and which assets were included
      at inception vs added later
- [x] Confirm BITCOIN and CHF_FRANC timing relative to
      other assets — confirm their introduction dates and
      note impact on Non-Traditional cap (20%, Resolution 2026-02)

### Step 3 Notes
- Final portfolio start date from the availability scan: **2000-01-03**
- SAA assets available at inception: `SPXT`, `FTSE100`, `LBUSTRUU`, `BROAD_TIPS`, `NIKKEI225`, `XAU`, `SILVER_FUT`, `CHF_FRANC`
- SAA assets added later by data availability: `CSI300_CHINA` on **2002-01-04**, `B3REITT` on **2003-03-31**, `BITCOIN` on **2010-07-19**
- `CHF_FRANC` was already available at inception on **2000-01-03**; only `BITCOIN` is a later non-traditional entrant

## Step 4: Combine into Master DataFrame
- [x] Merge all assets into a single **returns DataFrame**
- [x] Columns = asset tickers, rows = dates
- [x] Assets not yet available show **NaN** — these must remain
      as NaN and must **never be filled by any method**
- [x] Add a **classification column** mapping each ticker to:
      Core / Satellite / Non-Traditional / Opportunistic
- [x] Add an **availability flag column** per asset per date
      (1 = available, 0 = not yet available or data gap)
- [x] At every date, portfolio weights must sum to 100%
      using **only assets flagged as available on that date**
- [x] Any signal or metric calculations must use only
      data available **up to and including** that date —
      no future observations may enter any calculation
- [x] Save master DataFrame as reference

### Step 2–4 Notes
- `taa_project/outputs/data_audit_report.md` documents the duplicate-date check, non-positive-price check, gap tables, and the one-business-day FRED lag.
- `taa_project/outputs/asset_log_returns.csv` is the audited wide returns panel; `taa_project/outputs/asset_availability.csv` is the aligned availability matrix.
- `taa_project/outputs/master_data_reference.csv` is the long-form reference export with `date`, `ticker`, `log_return`, `availability`, and Whitmore tier classification.
- The authoritative `whitmore_daily.csv` file is already USD-denominated for the pipeline inputs; four non-USD metadata labels in `data_key.csv` remain flagged for review in the audit report rather than being silently transformed.
