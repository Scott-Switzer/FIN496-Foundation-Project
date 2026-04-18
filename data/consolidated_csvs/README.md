# Consolidated CSVs

- Included CSVs: 79
- Excluded CSVs: 112
- FRED raw pulls copied: 72
- FRED master tables copied: 1
- Bloomberg/master tables copied: 6
- Approx copied size: 10.27 MB

## Folder layout

- `fred/raw/`: original CSV pulls from `Data/fred/`
- `fred/master/`: canonical combined FRED input table used by the pipeline
- `bloomberg/master/`: Bloomberg-derived master inputs and manifests
- `csv_inventory.csv`: every original CSV in `Data/`, marked included or excluded with a reason

## Explicit exclusions

- Zion CSVs
- Combined tables with Zion columns
- Generated signal, chart, portfolio, Monte Carlo, result, and report CSVs
- Derived feature and label tables
