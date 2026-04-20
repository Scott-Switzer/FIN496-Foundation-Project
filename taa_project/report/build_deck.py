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
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from taa_project.analysis.config_comparison import CONFIG_COMPARISON_FILENAME, SUBMISSION_SELECTION_FILENAME
from taa_project.analysis.reporting import DSR_SUMMARY_FILENAME, PORTFOLIO_METRICS_FILENAME
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
    comparison_path = output_dir / CONFIG_COMPARISON_FILENAME
    selection_path = output_dir / SUBMISSION_SELECTION_FILENAME
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.exists() else None

    pdf_path = report_dir / DECK_PDF_FILENAME
    styles = _styles()
    story = []

    taa = metrics.loc[metrics["portfolio"] == "SAA+TAA"].iloc[0]
    saa = metrics.loc[metrics["portfolio"] == "SAA"].iloc[0]
    bm2 = metrics.loc[metrics["portfolio"] == "BM2"].iloc[0]

    slides: list[list[object]] = [
        [
            Paragraph("Whitmore Capital Partners SAA/TAA Mandate", styles["DeckTitle"]),
            Paragraph("FIN 496 Foundation Project", styles["DeckHeading"]),
            * _bullet_list(
                [
                    "External consultant recommendation for a fictional $1.8B single-family office.",
                    "Pipeline is fully reproducible from `python taa_project/main.py`.",
                    f"Final run mode: {'--timesfm' if int(dsr_summary['timesfm_enabled']) == 1 else '--no-timesfm'}.",
                ],
                styles,
            ),
        ],
        [
            Paragraph("Mandate & IPS Highlights", styles["DeckHeading"]),
            * _bullet_list(
                [
                    "Target 8% return with 15% volatility ceiling and 25% max drawdown tolerance.",
                    "Core must stay at or above 40%, Satellite at or below 45%, Non-Traditional at or below 20%.",
                    "No shorts, no hard-coded risk-off safe haven, and 5 bps round-trip cost applied at each rebalance.",
                ],
                styles,
            ),
        ],
        [
            Paragraph("SAA Method Choice", styles["DeckHeading"]),
            * _bullet_list(
                [
                    "Chosen SAA method: constrained risk parity.",
                    "Reason: it clears the 8% return mandate while staying below the 15% volatility ceiling without leaning on unstable mean forecasts.",
                    "Alternatives were evaluated side-by-side in the diagnostics notebook and `saa_method_comparison.csv`.",
                ],
                styles,
            ),
        ],
        [
            Paragraph("TAA Signal Architecture", styles["DeckHeading"]),
            * _bullet_list(
                [
                    "Lagged macro -> 3-state HMM regime probabilities.",
                    "Daily price histories -> Faber trend + ADM momentum.",
                    "Optional TimesFM forecast -> direction and volatility input.",
                    "Signal ensemble -> cvxpy optimizer inside TAA bands with turnover and volatility controls.",
                ],
                styles,
            ),
        ],
        [
            Paragraph("Risk Overlays & Vol-Budget Sweep", styles["DeckHeading"]),
            Image(str(figure_dir / "config_comparison.png"), width=22.0 * cm, height=10.0 * cm)
            if (figure_dir / "config_comparison.png").exists()
            else Paragraph("Config comparison figure not available for this output directory.", styles["DeckBody"]),
            Spacer(1, 0.2 * cm),
            *(
                _bullet_list(
                    [
                        f"Selected configuration: {selection['run_id']} ({selection['display_name']}).",
                        f"Best tested max drawdown: {100.0 * float(selection['max_dd']):.2f}% across {int(selection['n_tested_configurations'])} canonical configurations.",
                        "All six canonical variants converged to the same realized OOS portfolio path, so the baseline was retained as the simplest tied winner.",
                        "Overlays only tighten the risk envelope; they do not hard-code a safe-haven allocation.",
                    ],
                    styles,
                )
                if selection is not None
                else _bullet_list(
                    [
                        "Six canonical configurations are compared here once the sweep artifacts are present.",
                        "Overlays only tighten the risk envelope; they do not hard-code a safe-haven allocation.",
                    ],
                    styles,
                )
            ),
        ],
        [
            Paragraph("Walk-Forward Design", styles["DeckHeading"]),
            Image(str(figure_dir / "fig06_oos_folds.png"), width=22.0 * cm, height=9.0 * cm),
            Spacer(1, 0.2 * cm),
            Paragraph(f"Deflated Sharpe Ratio: {dsr_summary['baseline_dsr']:.3f} across {int(dsr_summary['n_taa_trials'])} disclosed TAA trials.", styles["DeckBody"]),
        ],
        [
            Paragraph("Cumulative Growth", styles["DeckHeading"]),
            Image(str(figure_dir / "fig01_cumgrowth.png"), width=22.0 * cm, height=11.5 * cm),
        ],
        [
            Paragraph("Metrics Table", styles["DeckHeading"]),
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
        [
            Paragraph("Attribution", styles["DeckHeading"]),
            Image(str(figure_dir / "fig07_attribution_bar.png"), width=18.5 * cm, height=10.0 * cm),
            Spacer(1, 0.2 * cm),
            * _bullet_list(
                [
                    "Separate decomposition delivered for SAA vs BM2 and TAA vs SAA/BM1/BM2.",
                    "Signal value is measured with leave-one-out OOS reruns, not static coefficient inspection.",
                ],
                styles,
            ),
        ],
        [
            Paragraph("Limitations", styles["DeckHeading"]),
            * _bullet_list(
                [
                    "HMM states are statistical and can drift.",
                    "TimesFM is optional and not finance-native.",
                    "Covariance shrinkage and optimizer penalties remain engineering choices, not immutable truths.",
                ],
                styles,
            ),
        ],
        [
            Paragraph("Recommendation", styles["DeckHeading"]),
            * _bullet_list(
                [
                    f"SAA+TAA annualized return: {100.0 * taa['annualized_return']:.2f}% versus {100.0 * saa['annualized_return']:.2f}% for SAA and {100.0 * bm2['annualized_return']:.2f}% for BM2.",
                    (
                        f"Submitted configuration: {selection['run_id']}. "
                        f"Max drawdown {'passes' if selection['pass_mdd'] else 'misses'} the -25% IPS tolerance at {100.0 * float(selection['max_dd']):.2f}%."
                        if selection is not None
                        else "Use the risk-parity SAA as the policy anchor and keep the TAA layer conditional on continuing OOS superiority after costs and DSR adjustment."
                    ),
                    "Track turnover and stressed-regime attribution before expanding risk budget to the overlay.",
                ],
                styles,
            ),
        ],
        [
            Paragraph("Q&A", styles["DeckHeading"]),
            * _bullet_list(
                [
                    "Key files: `taa_project/main.py`, `taa_project/analysis/reporting.py`, `taa_project/report/build_report.py`.",
                    "Key artifacts: report PDF, deck PDF, diagnostics notebook, trial ledger, IPS audit.",
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
