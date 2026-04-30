# Whitmore Capital Partners — FIN 496 Reproduction Guide

> **Purpose**: Definitive step-by-step instructions to reproduce every artifact, metric, chart, and report submitted for the Chapman University FIN 496 Foundation Project. If you follow these instructions exactly, you will obtain the same numerical results we reported.

---

## Quick Start (TL;DR)

```bash
# 1. Verify you are in the repo root
cd /path/to/FIN496-Foundation-Project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (final submission configuration)
python -m taa_project.main --start 2003-01-01 --end 2026-04-15 --folds 5 --no-timesfm

# 4. Check the metrics
python -c "import pandas as pd; df=pd.read_csv('taa_project/outputs/portfolio_metrics.csv'); print(df[df['portfolio']=='SAA+TAA'][['annualized_return','annualized_volatility','max_drawdown','sharpe_rf_2pct']])"
```

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | **3.11** (enforced by `.python-version`; 3.10 or 3.12 may work but are not tested) |
| **OS** | macOS or Linux. Windows is untested. |
| **RAM** | **8 GB minimum**. The pipeline alone uses ~2-3 GB. The canonical 13-run sweep peaks at ~4 GB per subprocess. |
| **Disk** | ~2 GB free (mostly for generated figures, reports, and per-run outputs). |
| **Time** | **~25-35 minutes** for a single pipeline run. **~3-4 hours** for the full canonical sweep. |
| **Internet** | Required only for `pip install`. All data is local CSV; no live FRED/API calls during execution. |

### macOS-Specific Prerequisites
If you are on Apple Silicon (M1/M2/M3) and `pip install` fails while compiling wheels, ensure you have the Xcode command-line tools:
```bash
xcode-select --install
```

---

## Step 1: Verify Required Data Files

Before running anything, confirm these three files exist. The pipeline will refuse to start if any are missing.

```bash
ls data/asset_data/whitmore_daily.csv
ls data/asset_data/data_key.csv
ls data/consolidated_csvs/fred/master/fred_data.csv
```

| File | Purpose |
|------|---------|
| `data/asset_data/whitmore_daily.csv` | Daily price panel for all SAA, TAA, and opportunistic assets |
| `data/asset_data/data_key.csv` | Asset metadata (names, currencies, tiers) |
| `data/consolidated_csvs/fred/master/fred_data.csv` | Macro features: VIX, HY OAS, Yield Curve, NFCI, TIPS real yield |

**Do not modify these files.** The pipeline expects their exact contents.

---

## Step 2: Install Python Dependencies

We strongly recommend a virtual environment.

```bash
# Create and activate a virtual environment (optional but recommended)
python3.11 -m venv venv
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

**Expected behavior**: `cvxpy`, `clarabel`, `numpy`, `scipy`, `pandas`, `hmmlearn`, `matplotlib`, and `reportlab` will install. `timesfm` and `torch` are listed for optional experiments but are **not used** in the final submission.

**If pip hangs on `timesfm` or `torch`**: these are only needed for the optional TimesFM sweep configurations. The final submission (`--no-timesfm`) does not import them.

---

## Step 3: Run the Final Submission Pipeline

This is the **exact command** that produced our reported results. Every flag is explicit so there is no ambiguity.

```bash
python -m taa_project.main \
  --start 2003-01-01 \
  --end 2026-04-15 \
  --folds 5 \
  --vol-budget 0.12 \
  --saa-method min_variance \
  --optimizer-mode vol \
  --no-timesfm \
  --no-dd-guardrail \
  --no-daily-risk-governor \
  --output-dir taa_project/outputs \
  --figure-dir taa_project/outputs/figures \
  --report-dir taa_project/outputs/reports \
  --notebook-dir taa_project/outputs/notebooks
```

**What you should see in the terminal**:
```
[YYYY-MM-DD HH:MM:SS] Task 1: data audit
[YYYY-MM-DD HH:MM:SS] Task 2: SAA portfolio
[YYYY-MM-DD HH:MM:SS] Task 3: benchmarks
[YYYY-MM-DD HH:MM:SS] Task 6: walk-forward backtest
[YYYY-MM-DD HH:MM:SS] Task 7: attribution
[YYYY-MM-DD HH:MM:SS] Task 8: metrics, figures, IPS audit, trial ledger
[YYYY-MM-DD HH:MM:SS] Task 9: diagnostics notebook
[YYYY-MM-DD HH:MM:SS] Task 10: report PDF
[YYYY-MM-DD HH:MM:SS] Task 11: presentation deck PPTX
[YYYY-MM-DD HH:MM:SS] Pipeline complete
```

**Note**: The pipeline seeds NumPy's RNG with a fixed seed (`DEFAULT_RANDOM_SEED = 42`) for deterministic HMM initialization. Results should be bitwise-identical across runs on the same machine.

---

## Step 4: Validate the Outputs

After the pipeline finishes, verify the following files and values.

### 4.1 Portfolio Metrics (The Most Important Check)

Run:
```bash
python -c "
import pandas as pd
df = pd.read_csv('taa_project/outputs/portfolio_metrics.csv')
print(df[df['portfolio'] == 'SAA+TAA'][[
    'annualized_return',
    'annualized_volatility',
    'max_drawdown',
    'sharpe_rf_2pct',
    'sortino_rf_2pct',
    'calmar'
]].to_string(index=False))
"
```

**You should see** (values within ±0.001 are acceptable due to floating-point differences across platforms):

| Metric | Value | IPS Requirement |
|--------|-------|-----------------|
| `annualized_return` | **~0.0839** (8.39%) | >= 8.0% target |
| `annualized_volatility` | **~0.0725** (7.25%) | <= 15.0% ceiling |
| `max_drawdown` | **~-0.2192** (-21.92%) | >= -25.0% tolerance |
| `sharpe_rf_2pct` | **~0.881** | — |
| `sortino_rf_2pct` | **~1.245** | — |
| `calmar` | **~0.383** | — |

### 4.2 IPS Compliance Audit

Run:
```bash
python -c "
import pandas as pd
df = pd.read_csv('taa_project/outputs/ips_compliance.csv')
hard = df[~df['rule'].isin(['rolling_vol_21d','rolling_vol_63d','rolling_vol_252d','max_drawdown','saa_taa_upper','saa_taa_lower'])]
print(f'Hard violations: {len(hard)}')
if len(hard) > 0:
    print(hard.head())
"
```

**Expected**: `Hard violations: 0`

(Realized vol and drawdown breaches are logged as "soft" violations because they are market-driven and not optimizer-controllable. A small number of soft rows is expected during crises like 2008 and 2022.)

### 4.3 Figure Checklist (18 PNGs)

Run:
```bash
ls taa_project/outputs/figures/fig*.png | wc -l
```

**Expected**: `18` (or more, depending on optional runs).

Key figures to inspect:
- `fig01_cumgrowth.png` — cumulative wealth of BM1, BM2, SAA, and SAA+TAA
- `fig02_drawdown.png` — underwater chart showing max drawdown ~-22%
- `fig03_rolling_vol.png` — rolling 252-day volatility staying well below 15%
- `fig07_attribution_bar.png` — TAA attribution bars
- `fig14_risk_return_scatter.png` — SAA+TAA in the upper-left vs benchmarks

### 4.4 Report Artifacts

```bash
ls taa_project/outputs/reports/*.pdf
ls taa_project/outputs/reports/*.pptx
```

You should find:
- A PDF report (`report_*.pdf` or similar, generated by `build_report.py`)
- A PowerPoint deck (`deck_*.pptx` or similar, generated by `build_pptx.py`)

### 4.5 Diagnostics Notebook

```bash
ls taa_project/outputs/notebooks/diagnostics.ipynb
```

Open this in Jupyter to inspect intermediate dataframes, signal histories, and fold boundaries interactively.

### 4.6 Trial Ledger

```bash
ls TRIAL_LEDGER.csv
head -5 TRIAL_LEDGER.csv
```

This CSV appends one row per pipeline run. It is used for the Deflated Sharpe Ratio calculation.

---

## Step 5: Run the Test Suite (Recommended)

Before trusting the outputs, confirm the test suite passes. This validates data integrity, signal logic, optimizer constraints, and walk-forward mechanics.

```bash
make test
```

Or equivalently:
```bash
python -m pytest -q
```

**Expected**: All 20+ tests pass. A few deprecation warnings from `hmmlearn` or `sklearn` are harmless.

---

## Step 6: Run the Canonical Sweep (Optional, ~3-4 Hours)

The canonical sweep exercises 13 configurations to demonstrate robustness. Each runs in an isolated subprocess with a 4 GB memory limit.

```bash
python -m taa_project.scripts.run_sweep
```

**Outputs**:
- `taa_project/outputs/runs/<run_id>/outputs/` — per-run CSVs
- `taa_project/outputs/runs/<run_id>/figures/` — per-run figures
- `taa_project/outputs/sweep_results.json` — summary JSON with exit codes and runtimes

**Expected behavior**: The `baseline` run (no TimesFM, 10% vol budget) and the other 12 variants will complete. TimesFM-enabled runs will fall back to non-TimesFM behavior because TimesFM is not wired into the main pipeline, but they will still produce valid outputs for comparison.

---

## Step 7: Build the Submission Bundle (Optional)

```bash
make zip
```

This:
1. Runs the full pipeline (if not already run)
2. Creates `whitmore_taa_submission.zip` in the repo root containing source code, data, and documentation
3. Excludes generated outputs and `__pycache__`

To verify the zip is self-contained:
```bash
make verify-zip
```

This extracts the zip to `/tmp/whitmore_taa_verify` and reruns the pipeline.

---

## Output Directory Map

After a successful run, your directory tree will look like this:

```
FIN496-Foundation-Project/
├── TRIAL_LEDGER.csv                          # Audit trail of all pipeline runs
├── whitmore_taa_submission.zip               # Generated by `make zip` (optional)
│
└── taa_project/outputs/
    ├── portfolio_metrics.csv                 # KEY FILE: BM1, BM2, SAA, SAA+TAA metrics
    ├── ips_compliance.csv                    # Daily IPS audit log
    ├── oos_returns.csv                       # Daily portfolio returns
    ├── oos_weights.csv                       # Monthly rebalance targets
    ├── oos_holdings.csv                      # Daily drifted holdings
    ├── oos_regimes.csv                       # HMM regime labels & probabilities
    ├── walkforward_folds.csv                 # 5-fold OOS metadata
    ├── saa_weights.csv                       # Annual SAA targets
    ├── saa_returns.csv                       # Daily SAA portfolio returns
    ├── bm1_returns.csv                       # Daily BM1 returns
    ├── bm2_returns.csv                       # Daily BM2 returns
    ├── attribution_*.csv                     # SAA vs BM2, TAA vs SAA attribution
    ├── breaches.log                          # Optimizer fallback incidents
    │
    ├── figures/
    │   ├── fig01_cumgrowth.png
    │   ├── fig02_drawdown.png
    │   ├── fig03_rolling_vol.png
    │   ├── fig04_taa_weights_stacked.png
    │   ├── fig05_regime_shading.png
    │   ├── fig06_oos_folds.png
    │   ├── fig07_attribution_bar.png
    │   ├── fig08_per_fold_oos.png
    │   ├── fig09_signal_history.png
    │   ├── fig10_contribution.png
    │   ├── fig11_rolling_alpha.png
    │   ├── fig12_regime_forward_returns.png
    │   ├── fig13_annual_returns.png
    │   ├── fig14_risk_return_scatter.png
    │   ├── fig15_monthly_heatmap.png
    │   ├── fig16_annual_costs.png
    │   ├── fig17_correlation_heatmap.png
    │   └── fig18_cumulative_alpha.png
    │
    ├── reports/
    │   ├── report.pdf                          # Generated PDF report
    │   └── deck.pptx                           # Generated PowerPoint deck
    │
    └── notebooks/
        └── diagnostics.ipynb                   # Interactive diagnostics notebook
```

---

## Command-Line Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--start` | `2003-01-01` | First out-of-sample date for walk-forward |
| `--end` | `2026-04-15` | Last date for generated outputs |
| `--folds` | `5` | Number of contiguous OOS folds |
| `--vol-budget` | `0.12` | Internal ex-ante vol target (12% for final submission) |
| `--saa-method` | `min_variance` | SAA construction: `min_variance`, `risk_parity`, `hrp` |
| `--optimizer-mode` | `vol` | `vol` = mean-variance; `cvar` = Rockafellar-Uryasev CVaR |
| `--timesfm` / `--no-timesfm` | `False` | **Final submission uses `--no-timesfm`** |
| `--nested-risk` | `False` | Enable per-sleeve nested risk budgeting |
| `--bl-stress-views` | `False` | Enable Black-Litterman stress views |
| `--dd-guardrail` | `False` | Enable drawdown guardrail overlay |
| `--daily-risk-governor` | `False` | Enable daily realized-risk defensive governor |

---

## Troubleshooting

### "OpenMP" Warning on macOS
You may see:
```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```
This is **harmless**. It occurs because scikit-learn loads OpenMP independently from NumPy's BLAS. We suppress it via environment variables (`KMP_DUPLICATE_LIB_OK=TRUE`). It does not affect results.

### `pip install` fails on `clarabel` or `cvxpy`
These packages compile C extensions. Ensure you have:
- macOS: `xcode-select --install`
- Linux: `sudo apt-get install python3-dev build-essential`

### Out of Memory During Sweep
The sweep limits each subprocess to 4 GB RSS. If your machine has < 8 GB total, the orchestrator itself may be killed by the OS. In that case, run the pipeline once (Step 3) instead of the full sweep.

### Slow Execution
The HMM refits on an expanding window at the start of each fold. This is the slowest step. Expected times per step on a modern laptop:
- Task 1 (Data audit): < 10s
- Task 2 (SAA): ~30s
- Task 3 (Benchmarks): ~20s
- Task 6 (Walk-forward): **~15-20 minutes**
- Task 7-8 (Attribution & Reporting): ~2-3 minutes
- Task 9-11 (Notebook, PDF, PPTX): ~1 minute

### Results Differ Slightly from Reported Values
If your metrics differ by more than ±0.001 (e.g., Sharpe 0.87 vs 0.88), check:
1. Are you using Python 3.11? Different NumPy/SciPy versions can produce slightly different HMM state orderings.
2. Did you modify any data files?
3. Did you use a different `--vol-budget`? The default is `0.12`.

---

## Questions?

If any step fails, include the following in your report:
1. The exact command you ran
2. The last 50 lines of terminal output
3. The output of `python --version` and `pip freeze | grep -E "numpy|scipy|pandas|hmmlearn|cvxpy"`

---

*This reproduction guide was generated to accompany the Whitmore FIN 496 Foundation Project submission.*
