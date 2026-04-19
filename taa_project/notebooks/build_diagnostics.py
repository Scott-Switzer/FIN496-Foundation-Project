# Addresses rubric criteria 1-3 and Task 9 by generating the diagnostics
# notebook that documents the data, methods, signals, folds, and trial ledger.
"""Build the Whitmore diagnostics notebook.

References:
- Whitmore Task 9 notebook requirement.
- nbformat documentation: https://nbformat.readthedocs.io/

Point-in-time safety:
- Safe. The notebook reads already-generated pipeline outputs and lightweight
  diagnostics derived from those outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

from taa_project.config import NOTEBOOK_DIR, OUTPUT_DIR, REPO_ROOT


NOTEBOOK_FILENAME = "diagnostics.ipynb"


def build_diagnostics_notebook(
    output_dir: Path = OUTPUT_DIR,
    notebook_dir: Path = NOTEBOOK_DIR,
) -> Path:
    """Generate the Task 9 diagnostics notebook.

    Inputs:
    - `output_dir`: directory containing the generated CSV and figure outputs.
    - `notebook_dir`: destination directory for the `.ipynb` file.

    Outputs:
    - Path to the created notebook.

    Citation:
    - Whitmore Task 9 notebook requirement.

    Point-in-time safety:
    - Safe. The notebook consumes already-generated outputs only.
    """

    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebook_dir / NOTEBOOK_FILENAME

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(
            "# Whitmore Diagnostics Notebook\n\n"
            "This notebook is intentionally lightweight: it loads the pipeline outputs "
            "from `taa_project/outputs/` rather than rerunning the expensive backtests."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "\n"
            f"REPO_ROOT = Path({str(REPO_ROOT)!r})\n"
            f"OUTPUT_DIR = Path({str(output_dir)!r})\n"
            "plt.style.use('default')\n"
            "pd.set_option('display.max_columns', 50)\n"
            "pd.set_option('display.width', 160)\n"
        ),
        nbf.v4.new_markdown_cell("## Data Profile"),
        nbf.v4.new_code_cell(
            "inceptions = pd.read_csv(OUTPUT_DIR / 'asset_inception_dates.csv')\n"
            "gap_summary = pd.read_csv(OUTPUT_DIR / 'asset_gap_summary.csv')\n"
            "gap_detail = pd.read_csv(OUTPUT_DIR / 'asset_gap_detail.csv')\n"
            "display(inceptions)\n"
            "display(gap_summary.head(15))\n"
        ),
        nbf.v4.new_code_cell(
            "if not gap_detail.empty:\n"
            "    plt.figure(figsize=(8, 4))\n"
            "    gap_detail['missing_calendar_days'].hist(bins=30)\n"
            "    plt.title('Gap Histogram')\n"
            "    plt.xlabel('Missing Calendar Days')\n"
            "    plt.ylabel('Count')\n"
            "    plt.show()\n"
            "else:\n"
            "    print('No gap detail rows to plot.')\n"
        ),
        nbf.v4.new_markdown_cell("## SAA Method Comparison"),
        nbf.v4.new_code_cell(
            "saa_methods = pd.read_csv(OUTPUT_DIR / 'saa_method_comparison.csv')\n"
            "display(saa_methods)\n"
            "plt.figure(figsize=(8, 4))\n"
            "plt.bar(saa_methods['method'], saa_methods['sharpe'])\n"
            "plt.title('SAA Method Sharpe Comparison')\n"
            "plt.xticks(rotation=20)\n"
            "plt.show()\n"
        ),
        nbf.v4.new_markdown_cell("## Signal Diagnostics"),
        nbf.v4.new_code_cell(
            "oos_regimes = pd.read_csv(OUTPUT_DIR / 'oos_regimes.csv', parse_dates=['date'])\n"
            "display(oos_regimes['regime'].value_counts(dropna=False).rename_axis('regime').reset_index(name='count'))\n"
        ),
        nbf.v4.new_code_cell(
            "from taa_project.data_loader import load_prices\n"
            "from taa_project.signals.trend_faber import trend_score\n"
            "from taa_project.signals.momentum_adm import adm_score, cross_sectional_rank\n"
            "from taa_project.backtest.walkforward import SLEEVE_BUCKETS\n"
            "prices = load_prices()\n"
            "trend = trend_score(prices)\n"
            "future_21d = np.log(prices).diff(21).shift(-21)\n"
            "trend_hits = ((trend * future_21d).reindex(columns=trend.columns) > 0).mean().sort_values(ascending=False)\n"
            "display(trend_hits.rename('trend_hit_rate').to_frame())\n"
            "adm = cross_sectional_rank(adm_score(prices), SLEEVE_BUCKETS)\n"
            "aligned_future = future_21d.reindex(adm.index)\n"
            "ics = []\n"
            "for dt in adm.index:\n"
            "    x = adm.loc[dt]\n"
            "    y = aligned_future.loc[dt]\n"
            "    valid = x.notna() & y.notna()\n"
            "    if valid.sum() >= 3:\n"
            "        ics.append(x[valid].corr(y[valid], method='spearman'))\n"
            "print('ADM mean IC:', float(pd.Series(ics).mean()) if ics else 'n/a')\n"
        ),
        nbf.v4.new_code_cell(
            "dsr_summary = pd.read_csv(OUTPUT_DIR / 'dsr_summary.csv')\n"
            "if int(dsr_summary.loc[0, 'timesfm_enabled']) == 0:\n"
            "    print('TimesFM diagnostics skipped: baseline run used --no-timesfm.')\n"
            "else:\n"
            "    print('TimesFM was enabled; inspect saved forecast cache if present.')\n"
        ),
        nbf.v4.new_markdown_cell("## Walk-Forward Validation"),
        nbf.v4.new_code_cell(
            "folds = pd.read_csv(OUTPUT_DIR / 'walkforward_folds.csv', parse_dates=['train_start','train_end','embargo_start','embargo_end','test_start','test_end'])\n"
            "per_fold = pd.read_csv(OUTPUT_DIR / 'per_fold_metrics.csv')\n"
            "display(folds)\n"
            "display(per_fold)\n"
        ),
        nbf.v4.new_markdown_cell("## Attribution Decomposition"),
        nbf.v4.new_code_cell(
            "attr_saa = pd.read_csv(OUTPUT_DIR / 'attribution_saa_vs_bm2.csv')\n"
            "attr_taa = pd.read_csv(OUTPUT_DIR / 'attribution_taa_vs_saa.csv')\n"
            "attr_signal = pd.read_csv(OUTPUT_DIR / 'attribution_per_signal.csv')\n"
            "display(attr_saa.head(20))\n"
            "display(attr_taa.head(20))\n"
            "display(attr_signal)\n"
            "plot_df = attr_signal[attr_signal['layer'] != 'baseline']\n"
            "plt.figure(figsize=(8, 4))\n"
            "plt.bar(plot_df['layer'], plot_df['marginal_oos_sharpe'])\n"
            "plt.title('Per-Signal Marginal OOS Sharpe')\n"
            "plt.show()\n"
        ),
        nbf.v4.new_markdown_cell("## Turnover and Cost Profile"),
        nbf.v4.new_code_cell(
            "metrics = pd.read_csv(OUTPUT_DIR / 'portfolio_metrics.csv')\n"
            "display(metrics[['portfolio','turnover_pa','cost_drag_pa','hit_rate']])\n"
        ),
        nbf.v4.new_markdown_cell("## Trial Ledger"),
        nbf.v4.new_code_cell(
            f"trial_ledger = pd.read_csv(Path({str(REPO_ROOT / 'TRIAL_LEDGER.csv')!r}))\n"
            "display(trial_ledger)\n"
            "display(pd.read_csv(OUTPUT_DIR / 'dsr_summary.csv'))\n"
        ),
    ]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "version": "3.11"}
    notebook_path.write_text(nbf.writes(nb), encoding="utf-8")
    return notebook_path


def main() -> None:
    """CLI entrypoint for generating the diagnostics notebook.

    Inputs:
    - `--output-dir`: directory containing generated CSV artifacts.
    - `--notebook-dir`: destination directory for the `.ipynb` file.

    Outputs:
    - Writes `diagnostics.ipynb`.

    Citation:
    - Whitmore Task 9 notebook requirement.

    Point-in-time safety:
    - Ex-post reporting only.
    """

    parser = argparse.ArgumentParser(description="Build the Whitmore diagnostics notebook.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory containing generated CSV artifacts.")
    parser.add_argument("--notebook-dir", default=str(NOTEBOOK_DIR), help="Destination notebook directory.")
    args = parser.parse_args()

    notebook_path = build_diagnostics_notebook(output_dir=Path(args.output_dir), notebook_dir=Path(args.notebook_dir))
    print(f"Diagnostics notebook written to {notebook_path}")


if __name__ == "__main__":
    main()
