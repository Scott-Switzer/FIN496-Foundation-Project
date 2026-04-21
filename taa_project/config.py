"""
Whitmore IPS config — single source of truth.
All constraints flow from Guidelines.md / IPS.md §6-7.
"""
from pathlib import Path
from typing import Dict


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = PACKAGE_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
NOTEBOOK_DIR = PACKAGE_ROOT / "notebooks"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"
MPLCONFIG_DIR = OUTPUT_DIR / ".mplconfig"
TIMESFM_CACHE_PATH = CACHE_DIR / "timesfm_forecasts.parquet"
MEMORY_BREACH_LOG = OUTPUT_DIR / "memory_breaches.log"
TRIAL_LEDGER_CSV = REPO_ROOT / "TRIAL_LEDGER.csv"
SUBMISSION_ZIP = REPO_ROOT / "whitmore_taa_submission.zip"
DEFAULT_RANDOM_SEED = 42
MAX_PROCESS_RSS_GB = 4.0

# --------------------------------------------------------------------------
# Asset universe (SAA only — Opportunistic handled separately)
# --------------------------------------------------------------------------
CORE        = ["SPXT", "FTSE100", "LBUSTRUU", "BROAD_TIPS"]
SATELLITE   = ["B3REITT", "XAU", "SILVER_FUT", "NIKKEI225", "CSI300_CHINA"]
NONTRAD     = ["BITCOIN", "CHF_FRANC"]
EQUITY_ASSETS = ["SPXT", "FTSE100", "NIKKEI225", "CSI300_CHINA"]
ALL_SAA     = CORE + SATELLITE + NONTRAD

SAA_BANDS: Dict[str, tuple[float, float]] = {
    "SPXT": (0.30, 0.45),
    "FTSE100": (0.00, 0.10),
    "LBUSTRUU": (0.05, 0.15),
    "BROAD_TIPS": (0.00, 0.10),
    "B3REITT": (0.05, 0.20),
    "XAU": (0.10, 0.25),
    "NIKKEI225": (0.00, 0.15),
    "BITCOIN": (0.00, 0.05),
    "SILVER_FUT": (0.00, 0.15),
    "CSI300_CHINA": (0.00, 0.15),
    "CHF_FRANC": (0.00, 0.10),
}

SAA_TARGETS: Dict[str, float] = {
    "SPXT": 0.40,
    "FTSE100": 0.00,
    "LBUSTRUU": 0.10,
    "BROAD_TIPS": 0.05,
    "B3REITT": 0.10,
    "XAU": 0.15,
    "NIKKEI225": 0.05,
    "BITCOIN": 0.00,
    "SILVER_FUT": 0.05,
    "CSI300_CHINA": 0.05,
    "CHF_FRANC": 0.05,
}

# Per-sleeve TAA bands (Guidelines §TAA)
TAA_BANDS: Dict[str, tuple] = {
    "SPXT":          (0.20, 0.45),
    "FTSE100":       (0.00, 0.15),
    "LBUSTRUU":      (0.00, 0.35),
    "BROAD_TIPS":    (0.00, 0.25),
    "B3REITT":       (0.00, 0.25),
    "XAU":           (0.00, 0.30),
    "NIKKEI225":     (0.00, 0.20),
    "BITCOIN":       (0.00, 0.10),
    "SILVER_FUT":    (0.00, 0.20),
    "CSI300_CHINA":  (0.00, 0.20),
    "CHF_FRANC":     (0.00, 0.15),
}

# Benchmark 2 target weights (IPS §4)
BM1_WEIGHTS: Dict[str, float] = {
    "SPXT": 0.60,
    "LBUSTRUU": 0.40,
}
assert abs(sum(BM1_WEIGHTS.values()) - 1.0) < 1e-9, "BM1 must sum to 1"

BM2_WEIGHTS: Dict[str, float] = {
    "SPXT":         0.40,
    "NIKKEI225":    0.05,
    "CSI300_CHINA": 0.05,
    "LBUSTRUU":     0.10,
    "BROAD_TIPS":   0.05,
    "B3REITT":      0.10,
    "XAU":          0.15,
    "SILVER_FUT":   0.05,
    "CHF_FRANC":    0.05,
    # FTSE100, BITCOIN start at 0% in BM2
}
assert abs(sum(BM2_WEIGHTS.values()) - 1.0) < 1e-9, "BM2 must sum to 1"

# Aggregate caps (IPS §7, Amendment 2026-02)
CORE_FLOOR       = 0.40
SATELLITE_CAP    = 0.45
NONTRAD_CAP      = 0.20      # amended 2026-02
OPPO_CAP         = 0.15
OPPO_PER_ASSET   = 0.05
SINGLE_SLEEVE_MAX = 0.45

# Risk
VOL_CEILING      = 0.15
MAX_DD           = 0.25
TARGET_VOL       = 0.10      # internal target, < ceiling

# Costs
ROUNDTRIP_COST_BPS = 5
COST_PER_TURNOVER  = ROUNDTRIP_COST_BPS / 1e4  # 0.0005 per unit turnover (L1)

# Rebalance cadence
TAA_FREQ = "ME"    # month-end; TAA may fire intra-month on regime change
SAA_FREQ = "YE"    # year-end

# Paths
PRICES_CSV = DATA_DIR / "asset_data" / "whitmore_daily.csv"
ASSET_KEY_CSV = DATA_DIR / "asset_data" / "data_key.csv"
FRED_CSV = DATA_DIR / "consolidated_csvs" / "fred" / "master" / "fred_data.csv"
FRED_LAG_BUSINESS_DAYS = 1
