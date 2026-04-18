# Data Layout

- `consolidated_csvs/`: local CSV bundle copied into the repo.
- `asset_data/`: shared Whitmore asset panel pulled from Google Drive.
- `catalogs/file_inventory.csv`: file-level inventory for every CSV now tracked in the repo.
- `catalogs/asset_series_inventory.csv`: column-level inventory for `asset_data/whitmore_daily.csv`.
- `catalogs/policy_asset_availability.csv`: earliest available dates for SAA, opportunistic, and risk-free series.
- `catalogs/data_catalog.md`: markdown version of the same inventory tables.

Regenerate the catalog tables with:

```bash
python3 scripts/build_data_catalog.py
```
