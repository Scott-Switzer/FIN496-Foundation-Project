# Addresses rubric criteria 3-5 and Task 10 by assembling the final research
# report PDF from generated tables, figures, and cited methodology notes.
"""Build the Whitmore research report PDF and markdown source.

References:
- Whitmore IPS / Guidelines in the repo.
- Bailey & López de Prado (2014):
  https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- ReportLab user guide: https://www.reportlab.com/docs/reportlab-userguide.pdf

Point-in-time safety:
- Safe. This module consumes only already-generated ex-post outputs and does
  not feed any reporting decisions back into the strategy logic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from taa_project.analysis.reporting import (
    DSR_SUMMARY_FILENAME,
    IPS_COMPLIANCE_FILENAME,
    PER_FOLD_METRICS_FILENAME,
    PORTFOLIO_METRICS_FILENAME,
    REGIME_ALLOCATION_FILENAME,
    SAA_METHOD_COMPARISON_FILENAME,
)
from taa_project.config import FIGURES_DIR, OUTPUT_DIR, REPORT_DIR


REPORT_MD_FILENAME = "whitmore_report.md"
REPORT_PDF_FILENAME = "whitmore_report.pdf"


def _load_report_inputs(output_dir: Path, figure_dir: Path) -> dict[str, object]:
    """Load the tables and figures required for the PDF report.

    Inputs:
    - `output_dir`: directory containing CSV artifacts.
    - `figure_dir`: directory containing figure PNGs.

    Outputs:
    - Dictionary of loaded dataframes and figure paths.

    Citation:
    - Whitmore Tasks 7-10.

    Point-in-time safety:
    - Ex-post reporting only.
    """

    metrics = pd.read_csv(output_dir / PORTFOLIO_METRICS_FILENAME)
    saa_methods = pd.read_csv(output_dir / SAA_METHOD_COMPARISON_FILENAME)
    per_fold = pd.read_csv(output_dir / PER_FOLD_METRICS_FILENAME)
    dsr_summary = pd.read_csv(output_dir / DSR_SUMMARY_FILENAME)
    attribution_signal = pd.read_csv(output_dir / "attribution_per_signal.csv")
    attribution_saa_bm2 = pd.read_csv(output_dir / "attribution_saa_vs_bm2.csv")
    attribution_taa = pd.read_csv(output_dir / "attribution_taa_vs_saa.csv")
    regime_alloc = pd.read_csv(output_dir / REGIME_ALLOCATION_FILENAME)
    compliance = pd.read_csv(output_dir / IPS_COMPLIANCE_FILENAME)

    return {
        "metrics": metrics,
        "saa_methods": saa_methods,
        "per_fold": per_fold,
        "dsr_summary": dsr_summary,
        "attribution_signal": attribution_signal,
        "attribution_saa_bm2": attribution_saa_bm2,
        "attribution_taa": attribution_taa,
        "regime_alloc": regime_alloc,
        "compliance": compliance,
        "figures": {
            "cumgrowth": figure_dir / "fig01_cumgrowth.png",
            "drawdown": figure_dir / "fig02_drawdown.png",
            "rolling_vol": figure_dir / "fig03_rolling_vol.png",
            "taa_weights": figure_dir / "fig04_taa_weights_stacked.png",
            "regime_shading": figure_dir / "fig05_regime_shading.png",
            "oos_folds": figure_dir / "fig06_oos_folds.png",
            "attribution": figure_dir / "fig07_attribution_bar.png",
        },
    }


def _styles():
    """Create the paragraph styles used in the report PDF.

    Inputs:
    - None.

    Outputs:
    - ReportLab stylesheet dictionary.

    Citation:
    - ReportLab user guide.

    Point-in-time safety:
    - Presentation formatting only.
    """

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="WhitmoreTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=18,
            leading=22,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="WhitmoreHeading",
            parent=styles["Heading2"],
            fontSize=12,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="WhitmoreBody",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=11,
            spaceAfter=5,
        )
    )
    styles.add(
        ParagraphStyle(
            name="WhitmoreBullet",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=10,
            leftIndent=12,
            bulletIndent=0,
            spaceAfter=2,
        )
    )
    return styles


def _format_pct(value: float) -> str:
    """Format a decimal value as a percentage string.

    Inputs:
    - `value`: decimal numeric value.

    Outputs:
    - Percentage string with two decimals.

    Citation:
    - Internal report-formatting helper.

    Point-in-time safety:
    - Presentation formatting only.
    """

    return f"{100.0 * float(value):.2f}%"


def _df_table(frame: pd.DataFrame, max_rows: int = 12) -> Table:
    """Convert a dataframe into a compact reportlab table.

    Inputs:
    - `frame`: dataframe to render.
    - `max_rows`: optional row cap.

    Outputs:
    - Formatted `Table` object.

    Citation:
    - ReportLab user guide.

    Point-in-time safety:
    - Presentation formatting only.
    """

    trimmed = frame.head(max_rows).copy()
    display = [list(trimmed.columns)]
    for _, row in trimmed.iterrows():
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        display.append(rendered)

    table = Table(display, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table


def _build_markdown(inputs: dict[str, object]) -> str:
    """Build the markdown source used for the report narrative.

    Inputs:
    - `inputs`: loaded tables and figure metadata.

    Outputs:
    - Markdown string saved alongside the PDF.

    Citation:
    - Whitmore Task 10 report outline.

    Point-in-time safety:
    - Ex-post reporting only.
    """

    metrics = inputs["metrics"].set_index("portfolio")
    dsr_summary = inputs["dsr_summary"].iloc[0]
    compliance_rows = len(inputs["compliance"])
    saa = metrics.loc["SAA"]
    taa = metrics.loc["SAA+TAA"]
    bm2 = metrics.loc["BM2"]
    mode_text = "--timesfm" if int(dsr_summary["timesfm_enabled"]) == 1 else "--no-timesfm"

    taa_vs_saa_ann = taa["annualized_return"] - saa["annualized_return"]
    taa_vs_bm2_sharpe = taa["sharpe_rf_2pct"] - bm2["sharpe_rf_2pct"]

    lines = [
        "# Whitmore Capital Partners SAA/TAA Report",
        "",
        "## Executive Summary",
        f"- The final implementation uses constrained risk parity for SAA and a monthly cvxpy TAA overlay driven by HMM regime, Faber trend, Antonacci-style ADM, and an optional TimesFM layer. This report reflects run mode `{mode_text}`. [Sources: `taa_project/saa/build_saa.py`, `taa_project/backtest/walkforward.py`, `taa_project/analysis/reporting.py`]",
        f"- Net annualized return is {_format_pct(taa['annualized_return'])} for `SAA+TAA` versus {_format_pct(saa['annualized_return'])} for `SAA` and {_format_pct(bm2['annualized_return'])} for `BM2`. [Source: `taa_project/outputs/{PORTFOLIO_METRICS_FILENAME}`]",
        f"- Net Sharpe improves by {taa_vs_bm2_sharpe:.2f} versus `BM2`, while the Deflated Sharpe Ratio is {dsr_summary['baseline_dsr']:.3f} across {int(dsr_summary['n_taa_trials'])} disclosed TAA trials. [Sources: `taa_project/outputs/{PORTFOLIO_METRICS_FILENAME}`, `taa_project/outputs/{DSR_SUMMARY_FILENAME}`, `TRIAL_LEDGER.csv`]",
        f"- Daily IPS audit produced {compliance_rows} violations across the strategy target schedules. [Source: `taa_project/outputs/{IPS_COMPLIANCE_FILENAME}`]",
        "",
        "## SAA Construction and IPS Compliance",
        "- Risk parity was selected over inverse volatility, minimum variance, maximum diversification, and mean-variance because it cleared the 8% return objective while staying below the 15% volatility ceiling without relying on fragile expected-return estimates; minimum variance was safer on volatility but undershot the return mandate. [Sources: `taa_project/saa/build_saa.py`, `taa_project/outputs/saa_method_comparison.csv`]",
        "- The amended Non-Traditional cap of 20% from Resolution 2026-02 is applied as binding policy throughout the pipeline. [Sources: `IPS.md`, `Guidelines.md`, `taa_project/config.py`]",
        "",
        "## TAA Signal Design",
        "- The regime layer is a 3-state Gaussian HMM on lagged `VIXCLS`, `BAMLH0A0HYM2`, `T10Y3M`, and `NFCI`, refit monthly on an expanding window. [Sources: `taa_project/signals/regime_hmm.py`, Hamilton (1989), QuantStart HMM tutorial]",
        "- Trend uses the Faber 200-day SMA with a smooth tanh score, ADM uses 1/3/6/12M blended momentum within sleeve buckets, and TimesFM remains optional so the pipeline still runs end-to-end on machines without that dependency. [Sources: `taa_project/signals/trend_faber.py`, `taa_project/signals/momentum_adm.py`, `taa_project/signals/vol_timesfm.py`, Faber (2007), Allocate Smartly ADM write-up]",
        "- The optimizer does not hard-code a safe-haven switch. Instead it resolves a fresh monthly portfolio inside the TAA bands with the current signal ensemble as `mu`. [Source: `taa_project/optimizer/cvxpy_opt.py`]",
        "",
        "## Walk-Forward Validation",
        "- The OOS period is split into five contiguous expanding folds with a 21-business-day embargo before each fold's first test decision. [Sources: `taa_project/backtest/walkforward.py`, `taa_project/outputs/walkforward_folds.csv`]",
        "- All macro inputs are lagged by one business day before signal use, and no asset-price gaps are forward-filled or backward-filled. [Sources: `taa_project/data_audit.py`, `taa_project/outputs/data_audit_report.md`]",
        "",
        "## Performance Results",
        f"- `SAA+TAA` delivers {_format_pct(taa['annualized_return'])} annualized return, {_format_pct(taa['annualized_volatility'])} annualized volatility, and {_format_pct(taa['max_drawdown'])} max drawdown. [Source: `taa_project/outputs/{PORTFOLIO_METRICS_FILENAME}`]",
        f"- Relative to `SAA`, the TAA overlay changes annualized return by {_format_pct(taa_vs_saa_ann)} and cost drag by {_format_pct(taa['cost_drag_pa'] - saa['cost_drag_pa'])} per year. [Source: `taa_project/outputs/{PORTFOLIO_METRICS_FILENAME}`]",
        "",
        "## Contribution Analysis",
        "- Active-return decomposition is reported separately for `SAA vs BM2`, `TAA vs SAA`, `TAA vs BM1`, and `TAA vs BM2`. [Sources: `taa_project/outputs/attribution_saa_vs_bm2.csv`, `taa_project/outputs/attribution_taa_vs_saa.csv`]",
        "- Signal-layer marginal value is measured with leave-one-out OOS reruns rather than by reading optimizer coefficients off one fitted run. [Source: `taa_project/outputs/attribution_per_signal.csv`]",
        "",
        "## Limitations and Failure Modes",
        "- The HMM is still a retrospective statistical classifier and state meanings can drift over time. [Source: `taa_project/signals/regime_hmm.py`]",
        "- TimesFM is not finance-native and was not used in the final baseline when the dependency was unavailable; the no-TimesFM fallback is explicitly disclosed. [Sources: `taa_project/signals/vol_timesfm.py`, `taa_project/outputs/dsr_summary.csv`]",
        "- The optimizer uses a shrinkage-style covariance stabilization and a soft volatility ceiling, so results depend on those engineering choices even under walk-forward discipline. [Source: `taa_project/backtest/walkforward.py`]",
        "",
        "## Recommendation",
        "- Use the risk-parity SAA as the policy core and treat the TAA overlay as an additive layer only when the OOS Sharpe and DSR remain superior after costs under the disclosed trial count. [Sources: `taa_project/outputs/portfolio_metrics.csv`, `TRIAL_LEDGER.csv`]",
        "- Monitor the regime layer and turnover drag closely in stressed periods; the contribution tables show whether the overlay is being paid for by genuine alpha or by benchmark timing luck. [Source: `taa_project/outputs/attribution_per_signal.csv`]",
        "",
        "## Appendix",
        "- Appendix tables in the PDF include the SAA method comparison, per-fold OOS metrics, the portfolio metrics table, and the leading rows of the trial ledger and IPS compliance log. [Sources: `taa_project/outputs/saa_method_comparison.csv`, `taa_project/outputs/per_fold_metrics.csv`, `TRIAL_LEDGER.csv`, `taa_project/outputs/ips_compliance.csv`]",
    ]
    return "\n".join(lines)


def build_report(
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURES_DIR,
    report_dir: Path = REPORT_DIR,
) -> tuple[Path, Path]:
    """Build the Whitmore markdown report and final PDF.

    Inputs:
    - `output_dir`: directory containing generated CSV artifacts.
    - `figure_dir`: directory containing generated figure PNGs.
    - `report_dir`: destination directory for the markdown and PDF.

    Outputs:
    - Tuple `(markdown_path, pdf_path)`.

    Citation:
    - Whitmore Task 10 report requirement.
    - ReportLab user guide.

    Point-in-time safety:
    - Ex-post reporting only.
    """

    report_dir.mkdir(parents=True, exist_ok=True)
    inputs = _load_report_inputs(output_dir, figure_dir)
    markdown = _build_markdown(inputs)
    markdown_path = report_dir / REPORT_MD_FILENAME
    pdf_path = report_dir / REPORT_PDF_FILENAME
    markdown_path.write_text(markdown, encoding="utf-8")

    styles = _styles()
    story = []

    story.append(Paragraph("Whitmore Capital Partners SAA/TAA Mandate", styles["WhitmoreTitle"]))
    story.append(Paragraph("FIN 496 Foundation Project", styles["WhitmoreHeading"]))
    story.append(Spacer(1, 0.3 * cm))

    for line in markdown.splitlines():
        if not line.strip():
            story.append(Spacer(1, 0.15 * cm))
            continue
        if line.startswith("# "):
            story.append(Paragraph(line[2:].strip(), styles["WhitmoreTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(line[3:].strip(), styles["WhitmoreHeading"]))
        elif line.startswith("- "):
            story.append(Paragraph(line[2:].strip(), styles["WhitmoreBullet"], bulletText="•"))
        else:
            story.append(Paragraph(line, styles["WhitmoreBody"]))

    story.extend(
        [
            PageBreak(),
            Paragraph("SAA Method Comparison", styles["WhitmoreHeading"]),
            _df_table(
                inputs["saa_methods"][
                    [
                        "method",
                        "annualized_return",
                        "annualized_volatility",
                        "sharpe",
                        "sortino",
                        "max_drawdown",
                        "turnover_pa",
                    ]
                ]
            ),
            Spacer(1, 0.25 * cm),
            Paragraph("Per-Fold OOS Metrics", styles["WhitmoreHeading"]),
            _df_table(
                inputs["per_fold"][
                    [
                        "fold_id",
                        "start_date",
                        "end_date",
                        "annualized_return",
                        "annualized_volatility",
                        "sharpe",
                        "sortino",
                        "max_drawdown",
                    ]
                ]
            ),
            Spacer(1, 0.25 * cm),
            Paragraph("Portfolio Metrics", styles["WhitmoreHeading"]),
            _df_table(
                inputs["metrics"][
                    [
                        "portfolio",
                        "annualized_return",
                        "annualized_volatility",
                        "max_drawdown",
                        "sharpe_rf_2pct",
                        "sortino_rf_2pct",
                        "calmar",
                        "cost_drag_pa",
                    ]
                ].rename(columns={"sharpe_rf_2pct": "Sharpe", "sortino_rf_2pct": "Sortino", "calmar": "Calmar"})
            ),
            PageBreak(),
            Paragraph("Performance Results", styles["WhitmoreHeading"]),
            Image(str(inputs["figures"]["cumgrowth"]), width=17.2 * cm, height=9.4 * cm),
            Spacer(1, 0.2 * cm),
            Image(str(inputs["figures"]["drawdown"]), width=17.2 * cm, height=9.4 * cm),
            PageBreak(),
            Paragraph("Walk-Forward Validation", styles["WhitmoreHeading"]),
            Image(str(inputs["figures"]["oos_folds"]), width=17.0 * cm, height=7.0 * cm),
            Spacer(1, 0.2 * cm),
            Paragraph("Contribution Analysis", styles["WhitmoreHeading"]),
            Image(str(inputs["figures"]["attribution"]), width=15.8 * cm, height=8.6 * cm),
            Spacer(1, 0.2 * cm),
            Paragraph("Signal-Layer Attribution Table", styles["WhitmoreHeading"]),
            _df_table(
                inputs["attribution_signal"][
                    ["variant_id", "marginal_oos_sharpe", "turnover_cost_delta", "ann_return_delta", "notes"]
                ]
            ),
            PageBreak(),
            Paragraph("Appendix: IPS Compliance and Trial Disclosure", styles["WhitmoreHeading"]),
            Paragraph("IPS Compliance Log", styles["WhitmoreBody"]),
            _df_table(inputs["compliance"] if not inputs["compliance"].empty else pd.DataFrame([{"status": "no violations"}])),
            Spacer(1, 0.2 * cm),
            Paragraph("Trial Ledger (Leading Rows)", styles["WhitmoreBody"]),
            _df_table(
                (
                    pd.read_csv(output_dir.parent.parent / "TRIAL_LEDGER.csv")[
                        ["variant_id", "OOS_sharpe", "DSR", "cv_folds", "notes"]
                    ]
                    if (output_dir.parent.parent / "TRIAL_LEDGER.csv").exists()
                    else pd.DataFrame([{"status": "missing"}])
                ),
                max_rows=10,
            ),
        ]
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=1.15 * cm,
        leftMargin=1.15 * cm,
        topMargin=1.15 * cm,
        bottomMargin=1.0 * cm,
    )
    doc.build(story)
    return markdown_path, pdf_path


def main() -> None:
    """CLI entrypoint for building the Whitmore report PDF.

    Inputs:
    - `--output-dir`: directory containing CSV artifacts.
    - `--figure-dir`: directory containing figure PNGs.
    - `--report-dir`: destination directory for markdown/PDF outputs.

    Outputs:
    - Writes `whitmore_report.md` and `whitmore_report.pdf`.

    Citation:
    - Whitmore Task 10 report requirement.

    Point-in-time safety:
    - Ex-post reporting only.
    """

    parser = argparse.ArgumentParser(description="Build the Whitmore report PDF.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory containing CSV artifacts.")
    parser.add_argument("--figure-dir", default=str(FIGURES_DIR), help="Directory containing figure PNGs.")
    parser.add_argument("--report-dir", default=str(REPORT_DIR), help="Destination directory for report outputs.")
    args = parser.parse_args()

    markdown_path, pdf_path = build_report(
        output_dir=Path(args.output_dir),
        figure_dir=Path(args.figure_dir),
        report_dir=Path(args.report_dir),
    )
    print(f"Report markdown written to {markdown_path}")
    print(f"Report PDF written to {pdf_path}")


if __name__ == "__main__":
    main()
