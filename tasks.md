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
- [ ] Convert all prices to **USD terms**
- [ ] Handle missing values:
      - **Do not forward-fill any values**
      - **Do not backward-fill any values**
      - Missing days (e.g. holidays) → treat as **no trading day**
        and exclude from return calculations entirely
      - Any gap in data must be flagged and documented
      - If a gap is structural (asset suspended, market closed)
        → treat as no return for that period, not zero return
- [ ] Remove duplicate dates
- [ ] Confirm no negative or zero prices
- [ ] Calculate **daily returns** only between consecutive
      **available** observations — never across gaps
- [ ] All cleaning decisions must be made using only
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
- [ ] Assets not yet available at start date are **excluded** until
      their data begins — do not backfill or substitute
- [ ] When a new asset becomes available mid-history:
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
- [ ] Merge all assets into a single **returns DataFrame**
- [ ] Columns = asset tickers, rows = dates
- [ ] Assets not yet available show **NaN** — these must remain
      as NaN and must **never be filled by any method**
- [ ] Add a **classification column** mapping each ticker to:
      Core / Satellite / Non-Traditional / Opportunistic
- [ ] Add an **availability flag column** per asset per date
      (1 = available, 0 = not yet available or data gap)
- [ ] At every date, portfolio weights must sum to 100%
      using **only assets flagged as available on that date**
- [ ] Any signal or metric calculations must use only
      data available **up to and including** that date —
      no future observations may enter any calculation
- [ ] Save master DataFrame as reference
