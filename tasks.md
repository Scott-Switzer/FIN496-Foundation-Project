## Step 1: Data Collection
- [ ] Pull price data for all SAA assets:
      SPXT, FTSE100, LBUSTRUU, BROAD_TIPS,
      B3REITT, XAU, SILVER_FUT, NIKKEI225,
      CSI300_CHINA, BITCOIN, CHF_FRANC
- [ ] Pull price data for all Opportunistic assets (Appendix A)
- [ ] Pull 3-month US Treasury yield (risk-free rate)
- [ ] Target start date: **early 2000s** (aim for 2000-01-01)
- [ ] Log the earliest available date for every asset

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
- [ ] Do **not** require all assets to exist simultaneously
- [ ] Starting from early 2000s, identify which assets are available
- [ ] At the target start date check if IPS hard constraints
      can be satisfied with available assets only:
      - Core ≥ 40% (SPXT and/or LBUSTRUU must be available)
      - Satellite ≤ 45%
      - Non-Traditional ≤ 20%
      - Single sleeve ≤ 45%
      - Portfolio sums to 100%
      - No short positions
- [ ] If constraints can be met → use that date as **portfolio start date**
- [ ] If constraints cannot be met → step forward day by day
      until they can be met
- [ ] Assets not yet available at start date are **excluded** until
      their data begins — do not backfill or substitute
- [ ] When a new asset becomes available mid-history:
      - Introduce it at the **next scheduled rebalance date**
      - Confirm constraints still satisfied after introduction
      - Document the date it was added and weight assigned
- [ ] Log final start date and which assets were included
      at inception vs added later
- [ ] BITCOIN and CHF_FRANC expected to enter later than
      other assets — confirm their introduction dates and
      note impact on Non-Traditional cap (20%, Resolution 2026-02)

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