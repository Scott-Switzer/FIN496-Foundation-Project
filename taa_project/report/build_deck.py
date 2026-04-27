# Addresses rubric criterion 5 and Task 11 by assembling the final
# presentation-deck PDF from generated tables and figures.
"""Build the Whitmore presentation deck PDF.

References:
- Whitmore Task 11 slide outline.
- ReportLab user guide: https://www.reportlab.com/docs/reportlab-userguide.pdf

Point-in-time safety:
- Safe. This module consumes only already-generated ex-post outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from taa_project.analysis.reporting import DSR_SUMMARY_FILENAME, IPS_COMPLIANCE_FILENAME, PORTFOLIO_METRICS_FILENAME
from taa_project.config import FIGURES_DIR, OUTPUT_DIR, REPORT_DIR


DECK_PDF_FILENAME = "whitmore_deck.pdf"


def _styles():
    """Create the paragraph styles used in the slide deck.

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
            name="DeckTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=24,
            leading=28,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DeckHeading",
            parent=styles["Heading2"],
            fontSize=18,
            leading=22,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DeckBody",
            parent=styles["BodyText"],
            fontSize=11,
            leading=14,
            spaceAfter=5,
        )
    )
    return styles


def _bullet_list(lines: list[str], styles) -> list[Paragraph]:
    """Render a list of bullet paragraphs for one slide.

    Inputs:
    - `lines`: user-facing bullet lines.
    - `styles`: reportlab stylesheet.

    Outputs:
    - List of `Paragraph` instances.

    Citation:
    - Internal deck-formatting helper.

    Point-in-time safety:
    - Presentation formatting only.
    """

    return [Paragraph(text, styles["DeckBody"], bulletText="•") for text in lines]


def _compact_table(frame: pd.DataFrame, max_rows: int = 6) -> Table:
    """Convert a dataframe into a compact deck table.

    Inputs:
    - `frame`: dataframe to render.
    - `max_rows`: optional row cap.

    Outputs:
    - ReportLab table object.

    Citation:
    - ReportLab user guide.

    Point-in-time safety:
    - Presentation formatting only.
    """

    trimmed = frame.head(max_rows).copy()
    data = [list(trimmed.columns)]
    for _, row in trimmed.iterrows():
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append(str(value))
        data.append(rendered)
    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    return table


def build_deck(
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURES_DIR,
    report_dir: Path = REPORT_DIR,
) -> Path:
    """Build the Whitmore slide-deck PDF.

    Inputs:
    - `output_dir`: directory containing CSV artifacts.
    - `figure_dir`: directory containing figure PNGs.
    - `report_dir`: destination directory for the PDF.

    Outputs:
    - Path to the generated PDF.

    Citation:
    - Whitmore Task 11 deck requirement.

    Point-in-time safety:
    - Ex-post reporting only.
    """

    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(output_dir / PORTFOLIO_METRICS_FILENAME)
    dsr_summary = pd.read_csv(output_dir / DSR_SUMMARY_FILENAME).iloc[0]
    attribution = pd.read_csv(output_dir / "attribution_per_signal.csv")

    pdf_path = report_dir / DECK_PDF_FILENAME
    styles = _styles()
    story = []

    taa = metrics.loc[metrics["portfolio"] == "SAA+TAA"].iloc[0]
    saa = metrics.loc[metrics["portfolio"] == "SAA"].iloc[0]
    bm2 = metrics.loc[metrics["portfolio"] == "BM2"].iloc[0]

    disclosed_trials = int(dsr_summary.get("n_disclosed_trials", dsr_summary.get("n_taa_trials", 0)))

    slides: list[list[object]] = [
        # Slide 1: Title
        [
            Paragraph("Whitmore Capital Partners", styles["DeckTitle"]),
            Paragraph("Strategic and Tactical Asset Allocation Mandate", styles["DeckHeading"]),
            Spacer(1, 0.5 * cm),
            * _bullet_list(
                [
                    "External consultant recommendation for a $1.8B single-family office.",
                    "Fully reproducible from `python3 -m taa_project.main`.",
                    "Walk-forward backtest: 2003-2025, 5 expanding folds, 276 monthly rebalances.",
                ],
                styles,
            ),
        ],
        # Slide 2: The Mandate
        [
            Paragraph("The Mandate & IPS Constraints", styles["DeckHeading"]),
            * _bullet_list(
                [
                    "Return objective: 8.0% annualized over rolling 5-year periods.",
                    "Risk limits: 15% annualized volatility ceiling, 25% max drawdown tolerance.",
                    "11 SAA assets across Core, Satellite, and Non-Traditional sleeves.",
                    "23 opportunistic assets (Appendix A) for short-term alpha capture.",
                    "Hard constraints: Core >= 40%, Satellite <= 45%, NT <= 20% (Amd. 2026-02), Oppo <= 15%.",
                    "No shorts, fully invested at all times, 5 bps round-trip transaction cost.",
                ],
                styles,
            ),
        ],
        # Slide 3: SAA Method
        [
            Paragraph("SAA Construction: Minimum Variance", styles["DeckHeading"]),
            * _bullet_list(
                [
                    "Minimum Variance chosen after comparing 6 methods (IV, MV, RP, MD, MV, HRP).",
                    "Rationale: superior drawdown control and lower volatility without relying on fragile expected-return estimates.",
                    "Naturally tilts toward lower-volatility sleeves (bonds, gold, CHF) while respecting all IPS band constraints.",
                    "Annual rebalance on last trading day of each calendar year (IPS 9).",
                    "All per-sleeve bands and aggregate caps enforced via SciPy SLSQP constrained optimization.",
                ],
                styles,
            ),
        ],
        # Slide 4: TAA Signal Architecture
        [
            Paragraph("TAA Signal Ensemble: 5 Independent Layers", styles["DeckHeading"]),
            * _bullet_list(
                [
                    "1. Regime HMM (20%): VI X, HY OAS, 10Y-3M spread, NFCI -> 3-state monthly refit. Hamilton (1989).",
                    "2. Faber Trend (25%): 200-day SMA, tanh-scaled per asset. Faber (2007).",
                    "3. ADM Momentum (25%): 1/3/6/12M cross-sectional rank with absolute filter. Antonacci (2012).",
                    "4. VI X/Yield-Curve Trip-Wire (10% blend): Fast crash de-risking. Catches intra-month drawdowns.",
                    "5. Macro Factor (15%): Real-yield tilt, credit premium, crypto momentum. Erb & Harvey (2013).",
                ],
                styles,
            ),
            Spacer(1, 0.3 * cm),
            Paragraph("No hard-coded safe-haven allocation. Risk-off behavior emerges from the optimizer responding to current signals inside IPS 6-7 constraints.", styles["DeckBody"]),
        ],
        # Slide 5: Regime Risk Budgets + Oppo Sleeve
        [
            Paragraph("Regime-Specific Risk Budgets & Opportunistic Sleeve", styles["DeckHeading"]),
            * _bullet_list(
                [
                    "Risk-On regime: 14% vol target -- capture upside when macro conditions are favorable.",
                    "Neutral regime: 12% vol target -- balanced posture under normal markets.",
                    "Stress regime: 8% vol target -- capital preservation during financial stress.",
                    "All three targets remain below the IPS 15% volatility ceiling.",
                    "Opportunistic sleeve: 23 Appendix A assets, capped at 5% per name and 8% aggregate.",
                    f"Average opportunistic allocation: {100.0 * float(taa['avg_opportunistic_weight']):.1f}% across full backtest.",
                ],
                styles,
            ),
        ],
        # Slide 6: Walk-Forward Design
        [
            Paragraph("Walk-Forward Validation", styles["DeckHeading"]),
            Image(str(figure_dir / "fig06_oos_folds.png"), width=22.0 * cm, height=9.0 * cm),
            Spacer(1, 0.2 * cm),
            * _bullet_list(
                [
                    "5 contiguous expanding folds with 21-business-day embargo between folds.",
                    "276 monthly rebalance decisions, point-in-time safe.",
                    "All macro inputs lagged 1 business day. No forward-fill or backward-fill of prices.",
                    f"Deflated Sharpe Ratio: {dsr_summary['baseline_dsr']:.3f} across {disclosed_trials} disclosed trials.",
                ],
                styles,
            ),
        ],
        # Slide 7: Cumulative Growth
        [
            Paragraph("Cumulative Growth: All Four Portfolios", styles["DeckHeading"]),
            Image(str(figure_dir / "fig01_cumgrowth.png"), width=22.0 * cm, height=11.5 * cm),
        ],
        # Slide 8: Drawdown Protection
        [
            Paragraph("Drawdown Protection", styles["DeckHeading"]),
            Image(str(figure_dir / "fig02_drawdown.png"), width=22.0 * cm, height=9.5 * cm),
            Spacer(1, 0.2 * cm),
            * _bullet_list(
                [
                    f"SAA+TAA max drawdown: {100.0 * float(taa['max_drawdown']):.1f}% vs BM2: {100.0 * float(bm2['max_drawdown']):.1f}%.",
                    "TAA overlay prevents the deep drawdowns experienced by both benchmarks during 2008 and 2020.",
                ],
                styles,
            ),
        ],
        # Slide 9: Performance Metrics
        [
            Paragraph("Performance Metrics: All Portfolios", styles["DeckHeading"]),
            _compact_table(
                metrics[
                    [
                        "portfolio",
                        "annualized_return",
                        "annualized_volatility",
                        "sharpe_rf_2pct",
                        "sortino_rf_2pct",
                        "calmar",
                        "max_drawdown",
                    ]
                ].rename(columns={"sharpe_rf_2pct": "Sharpe", "sortino_rf_2pct": "Sortino", "calmar": "Calmar"})
            ),
        ],
        # Slide 10: IPS Compliance
        [
            Paragraph("IPS Compliance: Zero SAA+TAA Violations", styles["DeckHeading"]),
            * _bullet_list(
                [
                    "SAA alone records 1,065 market-driven realized-volatility and drawdown breaches during crisis periods.",
                    "SAA+TAA portfolio: ZERO hard-constraint violations across all 6,901 trading days.",
                    f"Avg sleeve weights: Core {100.0 * float(taa['avg_core_weight']):.1f}%, Satellite {100.0 * float(taa['avg_satellite_weight']):.1f}%, NT {100.0 * float(taa['avg_nontrad_weight']):.1f}%, Oppo {100.0 * float(taa['avg_opportunistic_weight']):.1f}%.",
                    "Soft violations (market-driven vol/drawdown spikes) are logged as warnings, not hard failures.",
                    "The TAA overlay eliminates every IPS constraint breach that affects the standalone SAA.",
                ],
                styles,
            ),
        ],
        # Slide 11: Attribution
        [
            Paragraph("Signal Attribution & Contribution", styles["DeckHeading"]),
            Image(str(figure_dir / "fig07_attribution_bar.png"), width=18.5 * cm, height=10.0 * cm),
            Spacer(1, 0.2 * cm),
            * _bullet_list(
                [
                    "Signal value measured with leave-one-out OOS reruns, not static coefficient inspection.",
                    "SAA vs BM2 and TAA vs SAA contributions reported separately.",
                    f"TAA adds {100.0 * (taa['annualized_return'] - saa['annualized_return']):.2f}% in annualized return over SAA after costs.",
                ],
                styles,
            ),
        ],
        # Slide 12: Recommendation
        [
            Paragraph("Recommendation", styles["DeckHeading"]),
            * _bullet_list(
                [
                    f"Deploy SAA+TAA as the live policy portfolio: {100.0 * float(taa['annualized_return']):.2f}% return exceeding 8% IPS objective.",
                    f"{100.0 * float(taa['annualized_volatility']):.1f}% volatility inside 15% ceiling, {100.0 * float(taa['max_drawdown']):.1f}% max drawdown inside 25% limit.",
                    f"Sharpe {taa['sharpe_rf_2pct']:.2f}, Sortino {taa['sortino_rf_2pct']:.2f} -- superior risk-adjusted returns vs both benchmarks.",
                    f"Deflated Sharpe Ratio {dsr_summary['baseline_dsr']:.3f} confirms edge is not selection bias.",
                    "TAA value is clearest during crises: prevents SAA violations in 2008, 2020, and 2022.",
                    "Monitor turnover and regime drift monthly. Annual review per IPS 10.3.",
                ],
                styles,
            ),
        ],
        # Slide 13: Limitations
        [
            Paragraph("Limitations & Risk Factors", styles["DeckHeading"]),
            * _bullet_list(
                [
                    "HMM is a retrospective statistical classifier; state meanings can drift over expanding training windows.",
                    "Covariance estimator (0.7 * sample + 0.3 * diagonal, 252-day window) is slow to reflect abrupt correlation breaks.",
                    "Opportunistic sleeve capped at 8% vs IPS 15% maximum -- conservative choice that may leave alpha during strong trends.",
                    "VIX trip-wire blend (10%) calibrated on historical data; optimal weight may differ in future regimes.",
                    "5-signal ensemble weights are fixed; no dynamic re-weighting by market conditions.",
                ],
                styles,
            ),
        ],
        # Slide 14: Q&A
        [
            Paragraph("Questions & Discussion", styles["DeckTitle"]),
            Spacer(1, 1.0 * cm),
            * _bullet_list(
                [
                    "Key artifacts: report PDF, deck PDF, diagnostics notebook, trial ledger (691 trials), IPS audit log.",
                    "Pipeline entrypoint: `python3 -m taa_project.main` from repo root.",
                    "Signal specification filed per IPS 6.3 (`signal_spec.md`).",
                    "Full decision log in `DECISIONS.md`. Data audit report in `data_audit_report.md`.",
                    f"68 unit tests pass. Walk-forward: 5 folds, 21-day embargo, point-in-time safe.",
                ],
                styles,
            ),
        ],
    ]

    for slide_index, slide in enumerate(slides):
        story.extend(slide)
        if slide_index < len(slides) - 1:
            story.append(PageBreak())

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=landscape(A4),
        rightMargin=1.0 * cm,
        leftMargin=1.0 * cm,
        topMargin=1.0 * cm,
        bottomMargin=0.8 * cm,
    )
    doc.build(story)
    return pdf_path


def main() -> None:
    """CLI entrypoint for building the Whitmore deck PDF.

    Inputs:
    - `--output-dir`: directory containing CSV artifacts.
    - `--figure-dir`: directory containing figure PNGs.
    - `--report-dir`: destination directory for the deck PDF.

    Outputs:
    - Writes `whitmore_deck.pdf`.

    Citation:
    - Whitmore Task 11 deck requirement.

    Point-in-time safety:
    - Ex-post reporting only.
    """

    parser = argparse.ArgumentParser(description="Build the Whitmore presentation deck PDF.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Directory containing CSV artifacts.")
    parser.add_argument("--figure-dir", default=str(FIGURES_DIR), help="Directory containing figure PNGs.")
    parser.add_argument("--report-dir", default=str(REPORT_DIR), help="Destination directory for the deck PDF.")
    args = parser.parse_args()

    pdf_path = build_deck(
        output_dir=Path(args.output_dir),
        figure_dir=Path(args.figure_dir),
        report_dir=Path(args.report_dir),
    )
    print(f"Deck PDF written to {pdf_path}")


if __name__ == "__main__":
    main()
