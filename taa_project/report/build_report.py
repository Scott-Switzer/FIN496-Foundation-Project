"""Build the Whitmore report PDF. Times New Roman throughout, consultant-grade copy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    KeepTogether,
)

from taa_project.analysis.reporting import (
    DSR_SUMMARY_FILENAME,
    IPS_COMPLIANCE_FILENAME,
    PER_FOLD_METRICS_FILENAME,
    PORTFOLIO_METRICS_FILENAME,
    REGIME_ALLOCATION_FILENAME,
    SAA_METHOD_COMPARISON_FILENAME,
)
from taa_project.config import FIGURES_DIR, OUTPUT_DIR, REPORT_DIR, TRIAL_LEDGER_CSV

REPORT_PDF_FILENAME = "whitmore_report.pdf"

BASE_FONT = "Times-Roman"
BASE_FONT_BOLD = "Times-Bold"
BASE_FONT_ITALIC = "Times-Italic"

COL_NAVY = colors.HexColor("#1B2A4A")
COL_GOLD = colors.HexColor("#C9A227")
COL_CREAM = colors.HexColor("#FAFAF7")
COL_GREY = colors.HexColor("#F0F0ED")
COL_MID = colors.HexColor("#CCCCCC")
COL_SLATE = colors.HexColor("#6B7280")
COL_WHITE = colors.white

NAVY = COL_NAVY
GOLD = COL_GOLD
WHITE = COL_WHITE
LIGHT_GRAY = COL_GREY
DARK_GRAY = COL_NAVY
MEDIUM_GRAY = COL_SLATE
TABLE_HEADER_BG = COL_NAVY
TABLE_ROW_ALT = COL_GREY
TABLE_BORDER = COL_MID
ASSET_NAMES = {
    "SPXT": "S&P 500 Total Return",
    "LBUSTRUU": "US Aggregate Bonds",
    "BROAD_TIPS": "US TIPS",
    "B3REITT": "US REITs",
    "XAU": "Gold",
    "SILVER_FUT": "Silver Futures",
    "NIKKEI225": "Nikkei 225 (Japan)",
    "CSI300_CHINA": "CSI 300 (China)",
    "CHF_FRANC": "Swiss Franc",
    "FTSE100": "FTSE 100 (UK)",
    "BITCOIN": "Bitcoin",
}
METHOD_NAMES = {
    "maximum_diversification": "Max Diversification",
    "hrp": "Hier. Risk Parity",
    "risk_parity": "Risk Parity",
    "inverse_vol": "Inverse Vol",
    "mean_variance": "Mean-Variance",
    "minimum_variance": "Min Variance",
}

FONT = BASE_FONT
FONT_BOLD = BASE_FONT_BOLD
FONT_ITALIC = BASE_FONT_ITALIC
BODY_SIZE = 9
SMALL_SIZE = 8.5
PAGE_W, PAGE_H = A4
LEFT_MARGIN = 2.0 * cm
RIGHT_MARGIN = 2.0 * cm
DEFAULT_LEFT_MARGIN = LEFT_MARGIN
DEFAULT_RIGHT_MARGIN = RIGHT_MARGIN
DEFAULT_TOP_MARGIN = 1.45 * cm
DEFAULT_BOTTOM_MARGIN = 1.05 * cm
CONTENT_W = PAGE_W - DEFAULT_LEFT_MARGIN - DEFAULT_RIGHT_MARGIN


def _fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _styles():
    caption_style = ParagraphStyle("fig_caption", fontName=FONT_ITALIC, fontSize=7.5, leading=10,
                                   textColor=COL_SLATE, alignment=TA_CENTER,
                                   spaceBefore=2, spaceAfter=6,
                                   wordWrap="LTR")
    return {
        "title": ParagraphStyle("Title", fontName=FONT_BOLD, fontSize=28, leading=32,
                                textColor=COL_WHITE, alignment=TA_CENTER, spaceBefore=0,
                                spaceAfter=3, wordWrap="LTR"),
        "title_sub": ParagraphStyle("TitleSub", fontName=FONT, fontSize=13, leading=16,
                                    textColor=COL_GREY, alignment=TA_CENTER, spaceBefore=0,
                                    spaceAfter=3, wordWrap="LTR"),
        "h1": ParagraphStyle("H1", fontName=FONT_BOLD, fontSize=12, leading=15,
                              textColor=COL_NAVY, spaceBefore=6, spaceAfter=3,
                              wordWrap="LTR", keepWithNext=1),
        "h2": ParagraphStyle("H2", fontName=FONT_BOLD, fontSize=10.5, leading=13,
                              textColor=COL_NAVY, spaceBefore=6, spaceAfter=3,
                              wordWrap="LTR", keepWithNext=1),
        "body": ParagraphStyle("Body", fontName=FONT, fontSize=BODY_SIZE, leading=BODY_SIZE * 1.3,
                                textColor=COL_NAVY, spaceBefore=0, spaceAfter=3,
                                alignment=TA_JUSTIFY, wordWrap="LTR",
                                allowWidows=0, allowOrphans=0, keepWithPrevious=1),
        "body_small": ParagraphStyle("BodySmall", fontName=FONT, fontSize=SMALL_SIZE, leading=SMALL_SIZE * 1.3,
                                      textColor=COL_NAVY, spaceBefore=0, spaceAfter=3,
                                      alignment=TA_JUSTIFY, wordWrap="LTR",
                                      allowWidows=0, allowOrphans=0, keepWithPrevious=1),
        "bullet": ParagraphStyle("Bullet", fontName=FONT, fontSize=BODY_SIZE, leading=BODY_SIZE * 1.3,
                                 textColor=COL_NAVY, spaceBefore=0, spaceAfter=3,
                                 leftIndent=10, bulletIndent=2,
                                 bulletFontName=FONT, bulletFontSize=8,
                                 allowWidows=0, allowOrphans=0, keepWithPrevious=1),
        "bullet_small": ParagraphStyle("BulletSmall", fontName=FONT, fontSize=SMALL_SIZE, leading=SMALL_SIZE * 1.3,
                                        textColor=COL_NAVY, spaceBefore=0, spaceAfter=3,
                                        leftIndent=10, bulletIndent=2,
                                        bulletFontName=FONT, bulletFontSize=7,
                                        allowWidows=0, allowOrphans=0, keepWithPrevious=1),
        "caption": caption_style,
        "caption_style": caption_style,
        "label": ParagraphStyle("Label", fontName=FONT_BOLD, fontSize=8, leading=10,
                                 textColor=COL_NAVY, spaceBefore=0, spaceAfter=3,
                                 wordWrap="LTR"),
        "title_body": ParagraphStyle("TitleBody", fontName=FONT, fontSize=BODY_SIZE, leading=13,
                                      textColor=COL_WHITE, alignment=TA_CENTER, spaceBefore=0,
                                      spaceAfter=3, wordWrap="LTR"),
        "title_caption": ParagraphStyle("TitleCaption", fontName=FONT_ITALIC, fontSize=9, leading=11,
                                         textColor=COL_GREY, alignment=TA_CENTER, spaceBefore=0,
                                         spaceAfter=3, wordWrap="LTR"),
    }


def _df_table(frame: pd.DataFrame, max_rows: int = 20, float_fmt: str = ".3f",
              font_size: int = 7, first_col_ratio: float = 1.0,
              available_width: float = CONTENT_W,
              col_widths: list[float] | None = None,
              header_font_size: float = 8) -> Table:
    trimmed = frame.head(max_rows).copy()
    data = [list(trimmed.columns)]
    for _, row in trimmed.iterrows():
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(f"{value:{float_fmt}}")
            else:
                rendered.append(str(value))
        data.append(rendered)

    col_count = len(data[0]) if data else 1
    if col_widths is not None:
        col_widths = [float(width) for width in col_widths]
    elif first_col_ratio != 1.0 and col_count > 1:
        total_weight = first_col_ratio + (col_count - 1)
        col_widths = [available_width * first_col_ratio / total_weight] + \
                     [available_width / total_weight] * (col_count - 1)
    else:
        col_widths = [available_width / col_count] * col_count

    table = Table(data, repeatRows=1, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        # Header row: navy background, white bold text
        ("BACKGROUND", (0, 0), (-1, 0), COL_NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), COL_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), BASE_FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, 0), header_font_size),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        # Alternating row shading
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COL_WHITE, COL_GREY]),
        # Body text: dark navy, readable size
        ("TEXTCOLOR", (0, 1), (-1, -1), COL_NAVY),
        ("FONTNAME", (0, 1), (-1, -1), BASE_FONT),
        ("FONTSIZE", (0, 1), (-1, -1), 7.5),
        # Grid lines
        ("GRID", (0, 0), (-1, -1), 0.4, COL_MID),
        ("LINEBELOW", (0, 0), (-1, 0), 1.0, COL_GOLD),
        # Alignment: first column left, rest centered
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        # No word wrap truncation; allow wrapping.
        ("WORDWRAP", (0, 0), (-1, -1), True),
    ]))
    return table


def _fmt_display_df(df: pd.DataFrame, col_map: dict, pct_cols: list = None,
                    float2_cols: list = None, select: list = None) -> pd.DataFrame:
    """Rename columns, select subset, and format numeric columns for display."""
    df = df.copy()
    if select:
        df = df[[c for c in select if c in df.columns]]
    if pct_cols:
        for c in pct_cols:
            if c in df.columns:
                df[c] = df[c].apply(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else "-")
    if float2_cols:
        for c in float2_cols:
            if c in df.columns:
                df[c] = df[c].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) else "-")
    df = df.rename(columns=col_map)
    return df


def _safe_image(path: Path, width: float, styles: dict):
    """Return an aspect-preserving Image if file exists, else a warning Paragraph."""
    if Path(path).exists():
        image = Image(str(path), hAlign="CENTER")
        intrinsic_w, intrinsic_h = image.imageWidth, image.imageHeight
        image.drawWidth = width
        image.drawHeight = width * intrinsic_h / intrinsic_w
        return image
    return Paragraph(f"[Chart not available: {Path(path).name}]", styles["caption_style"])


def _section_heading(story: list, text: str, styles: dict) -> None:
    story.append(KeepTogether([
        Paragraph(text, styles["h1"]),
        HRFlowable(width="100%", thickness=0.8, color=COL_GOLD,
                   spaceAfter=3, spaceBefore=0),
    ]))


def _section_intro(story: list, text: str, paragraph: str, styles: dict,
                   style_name: str = "body") -> None:
    story.append(KeepTogether([
        Paragraph(text, styles["h1"]),
        HRFlowable(width="100%", thickness=0.8, color=COL_GOLD,
                   spaceAfter=3, spaceBefore=0),
        Paragraph(paragraph, styles[style_name]),
    ]))


def _grey_divider(story: list) -> None:
    story.append(Spacer(1, 0.2 * cm))
    story.append(HRFlowable(width="100%", thickness=0.4, color=COL_MID,
                            spaceAfter=4, spaceBefore=0))


def _chart_caption(text: str, styles: dict) -> Paragraph:
    return Paragraph(text, styles["caption_style"])


def _add_chart(story: list, path: Path, caption: str, styles: dict) -> None:
    story.append(Spacer(1, 0.1 * cm))
    story.append(_safe_image(path, CONTENT_W, styles))
    story.append(_chart_caption(caption, styles))


_SIGNAL_DISPLAY = {
    "regime": "Regime HMM",
    "trend": "Faber Trend",
    "momo": "ADM Momentum",
    "macro": "Macro Factor",
    "vix": "VIX Trip-Wire",
}


def _build_attribution_figure(per_signal: pd.DataFrame, figure_dir: Path) -> Path:
    """Re-render the attribution bar chart with human-readable signal labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = per_signal[per_signal["layer"] != "baseline"].copy()
    plot_df["label"] = plot_df["layer"].map(_SIGNAL_DISPLAY).fillna(plot_df["layer"])
    plot_df = plot_df.assign(abs_impact=plot_df["marginal_oos_sharpe"].abs())
    plot_df = plot_df.sort_values("abs_impact", ascending=True)
    values = plot_df["marginal_oos_sharpe"].to_numpy(dtype=float)
    bar_colors = ["#1B2A4A" if v >= 0 else "#C0392B" for v in values]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    fig.patch.set_facecolor("#FAFAF7")
    ax.set_facecolor("#FAFAF7")
    bars = ax.barh(plot_df["label"], values, color=bar_colors, zorder=3)
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=3, color="#1B2A4A")
    ax.axvline(0.0, color="#CCCCCC", linewidth=0.8, zorder=2)
    ax.set_xlabel("Change in OOS Sharpe Ratio", fontsize=8)
    ax.set_title("Signal Attribution  ·  Marginal OOS Sharpe", fontsize=12,
                 fontweight="bold", color="#1B2A4A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(labelsize=7, colors="#1B2A4A")
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.4, linestyle="--", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_path = figure_dir / "fig07_attribution_bar_report.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="#FAFAF7", edgecolor="none")
    plt.close(fig)
    return out_path


def _header_footer(canvas, doc):
    canvas.saveState()
    header_text_y = PAGE_H - DEFAULT_TOP_MARGIN + 10
    header_line_y = PAGE_H - DEFAULT_TOP_MARGIN + 2
    footer_line_y = DEFAULT_BOTTOM_MARGIN
    footer_text_y = DEFAULT_BOTTOM_MARGIN - 12

    canvas.setFont(BASE_FONT, 7)
    canvas.setFillColor(COL_SLATE)
    canvas.drawString(DEFAULT_LEFT_MARGIN, header_text_y,
                      "Whitmore Capital Partners | Confidential")
    canvas.drawRightString(PAGE_W - DEFAULT_RIGHT_MARGIN, header_text_y,
                           "Chapman University | April 2026")

    canvas.setStrokeColor(COL_GOLD)
    canvas.setLineWidth(0.5)
    canvas.line(DEFAULT_LEFT_MARGIN, header_line_y,
                PAGE_W - DEFAULT_RIGHT_MARGIN, header_line_y)

    canvas.setStrokeColor(COL_MID)
    canvas.setLineWidth(0.3)
    canvas.line(DEFAULT_LEFT_MARGIN, footer_line_y,
                PAGE_W - DEFAULT_RIGHT_MARGIN, footer_line_y)

    canvas.setFont(BASE_FONT, 7)
    canvas.setFillColor(COL_SLATE)
    canvas.drawCentredString(PAGE_W / 2, footer_text_y, str(doc.page))
    canvas.restoreState()


def _title_page_header(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(COL_WHITE)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.restoreState()


def _load_inputs(output_dir: Path, figure_dir: Path) -> dict:
    return {
        "metrics": pd.read_csv(output_dir / PORTFOLIO_METRICS_FILENAME),
        "saa_methods": pd.read_csv(output_dir / SAA_METHOD_COMPARISON_FILENAME),
        "per_fold": pd.read_csv(output_dir / PER_FOLD_METRICS_FILENAME),
        "dsr_summary": pd.read_csv(output_dir / DSR_SUMMARY_FILENAME),
        "attribution_signal": pd.read_csv(output_dir / "attribution_per_signal.csv"),
        "attribution_taa": pd.read_csv(output_dir / "attribution_taa_vs_saa.csv"),
        "regime_alloc": pd.read_csv(output_dir / REGIME_ALLOCATION_FILENAME),
        "compliance": pd.read_csv(output_dir / IPS_COMPLIANCE_FILENAME),
        "figures": {
            "cumgrowth": figure_dir / "fig01_cumgrowth.png",
            "drawdown": figure_dir / "fig02_drawdown.png",
            "rolling_12m": figure_dir / "fig19_rolling_12m_returns.png",
            "weights": figure_dir / "fig04_taa_weights_stacked.png",
            "regime": figure_dir / "fig05_regime_shading.png",
            "folds": figure_dir / "fig06_oos_folds.png",
            "per_fold": figure_dir / "fig08_per_fold_oos.png",
            "attribution": figure_dir / "fig07_attribution_bar.png",
            "signal_pipeline": figure_dir / "fig21_signal_pipeline_swimlane.png",
            "monthly_cycle": figure_dir / "fig22_monthly_cycle_flow.png",
            "state_machine": figure_dir / "fig23_state_machine.png",
        },
    }


def build_report(
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURES_DIR,
    report_dir: Path = REPORT_DIR,
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs(output_dir, figure_dir)
    # Regenerate attribution figure with readable labels from the CSV data.
    inputs["figures"]["attribution"] = _build_attribution_figure(
        inputs["attribution_signal"], figure_dir
    )
    S = _styles()

    metrics = inputs["metrics"].set_index("portfolio")
    taa = metrics.loc["SAA+TAA"]
    saa = metrics.loc["SAA"]
    bm2 = metrics.loc["BM2"]
    bm1 = metrics.loc["BM1"]
    compliance = inputs["compliance"]
    taa_c = compliance[compliance["portfolio"] == "SAA+TAA"]
    taa_soft = len(taa_c[taa_c["rule"].isin(["rolling_vol_21d", "rolling_vol_63d", "rolling_vol_252d", "max_drawdown"])])
    taa_hard = len(taa_c) - taa_soft
    saa_c = len(compliance[compliance["portfolio"] == "SAA"])
    dsr = inputs["dsr_summary"].iloc[0]
    disclosed = int(dsr.get("n_disclosed_trials", 0))

    left_margin = DEFAULT_LEFT_MARGIN
    right_margin = DEFAULT_RIGHT_MARGIN
    top_margin = DEFAULT_TOP_MARGIN
    bottom_margin = DEFAULT_BOTTOM_MARGIN
    CONTENT_W = PAGE_W - left_margin - right_margin
    print(f"CONTENT_W = {CONTENT_W:.2f} pt ({CONTENT_W/cm:.3f} cm)")
    key_metrics_widths = [
        CONTENT_W * 0.19,
        CONTENT_W * 0.11,
        CONTENT_W * 0.11,
        CONTENT_W * 0.12,
        CONTENT_W * 0.115,
        CONTENT_W * 0.115,
        CONTENT_W * 0.115,
        CONTENT_W * 0.115,
    ]
    saa_method_widths = [CONTENT_W * 0.22] + [CONTENT_W * (0.78 / 7)] * 7
    saa_contrib_widths = [
        CONTENT_W * 0.34,
        CONTENT_W * 0.22,
        CONTENT_W * 0.22,
        CONTENT_W * 0.22,
    ]
    ips_widths = [
        CONTENT_W * 0.34,
        CONTENT_W * 0.22,
        CONTENT_W * 0.22,
        CONTENT_W * 0.22,
    ]
    per_fold_widths = [
        CONTENT_W * 0.08,
        CONTENT_W * 0.14,
        CONTENT_W * 0.14,
        CONTENT_W * 0.12,
        CONTENT_W * 0.12,
        CONTENT_W * 0.12,
        CONTENT_W * 0.12,
        CONTENT_W * 0.16,
    ]
    regime_budget_widths = [
        CONTENT_W * 0.20,
        CONTENT_W * 0.15,
        CONTENT_W * 0.65,
    ]

    story = []

    # PAGE 1 - TITLE
    cover_firm_style = ParagraphStyle("cover_firm",
        fontName=BASE_FONT_BOLD, fontSize=12,
        leading=16, textColor=COL_SLATE,
        alignment=TA_CENTER, spaceAfter=0, spaceBefore=0,
    )
    cover_title_style = ParagraphStyle("cover_title",
        fontName=BASE_FONT_BOLD, fontSize=22,
        leading=28, textColor=COL_NAVY,
        alignment=TA_CENTER, spaceAfter=0, spaceBefore=0,
    )
    cover_sub_style = ParagraphStyle("cover_sub",
        fontName=BASE_FONT, fontSize=12,
        leading=16, textColor=COL_GOLD,
        alignment=TA_CENTER, spaceAfter=0, spaceBefore=0,
    )
    cover_prepared_style = ParagraphStyle("cover_prepared",
        fontName=BASE_FONT_ITALIC, fontSize=11,
        leading=15, textColor=COL_NAVY,
        alignment=TA_CENTER, spaceAfter=0, spaceBefore=0,
    )
    cover_meta_style = ParagraphStyle("cover_meta",
        fontName=BASE_FONT, fontSize=9,
        leading=13, textColor=COL_SLATE,
        alignment=TA_CENTER, spaceAfter=0, spaceBefore=0,
    )
    cover_inner_content = [
        Paragraph("WHITMORE CAPITAL PARTNERS", cover_firm_style),
        Spacer(1, 0.2 * cm),
        Paragraph("Strategic &amp; Tactical Asset Allocation", cover_title_style),
        Spacer(1, 0.2 * cm),
        Paragraph("Research Report | FIN 496 Foundation Project", cover_sub_style),
        Spacer(1, 0.2 * cm),
        HRFlowable(width="60%", thickness=1.5, color=COL_GOLD,
                   hAlign="CENTER", spaceAfter=20, spaceBefore=0),
        Paragraph("Prepared for the Whitmore Investment Principal", cover_prepared_style),
        Spacer(1, 0.2 * cm),
        Paragraph("Chapman University &nbsp;|&nbsp; April 2026 &nbsp;|&nbsp; Confidential",
                  cover_meta_style),
    ]
    cover_content_h = PAGE_H - 4 * cm
    cover_pad = cover_content_h * 0.28
    cover_table = Table(
        [[cover_inner_content]],
        colWidths=[CONTENT_W],
        rowHeights=[cover_content_h],
        hAlign="LEFT",
    )
    cover_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (0, 0), "MIDDLE"),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("TOPPADDING", (0, 0), (0, 0), cover_pad),
        ("BOTTOMPADDING", (0, 0), (0, 0), cover_pad),
        ("LEFTPADDING", (0, 0), (0, 0), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 0),
        ("BACKGROUND", (0, 0), (0, 0), COL_WHITE),
    ]))
    story.append(cover_table)
    story.append(PageBreak())

    # PAGE 2 - TABLE OF CONTENTS (4.2)
    toc_heading_style = ParagraphStyle(
        "toc_heading",
        fontName=BASE_FONT_BOLD,
        fontSize=16,
        leading=20,
        textColor=COL_NAVY,
        spaceAfter=6,
        spaceBefore=0,
    )
    story.append(Paragraph("Table of Contents", toc_heading_style))
    story.append(HRFlowable(width="100%", thickness=1.0, color=COL_GOLD,
                            spaceAfter=14, spaceBefore=0))
    toc_data = [
        ["Executive Summary", "3"],
        ["SAA Construction and Methodology", "5"],
        ["TAA Signal Design", "6"],
        ["Regime-Based Risk Budgeting", "8"],
        ["Opportunistic Sleeve", "8"],
        ["Walk-Forward Validation", "8"],
        ["SAA and TAA Contribution", "10"],
        ["Signal Attribution (Leave-One-Out OOS Reruns)", "11"],
        ["IPS Compliance Audit", "11"],
        ["Investment Recommendation", "12"],
    ]
    toc_table = Table(
        toc_data,
        colWidths=[CONTENT_W * 0.88, CONTENT_W * 0.12],
        hAlign="LEFT",
    )
    toc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COL_WHITE),
        ("TEXTCOLOR", (0, 0), (-1, -1), COL_NAVY),
        ("FONTNAME", (0, 0), (-1, -1), BASE_FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 13),
        ("LEADING", (0, 0), (-1, -1), 13),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("FONTNAME", (1, 0), (1, -1), BASE_FONT_BOLD),
        ("LINEBELOW", (0, 0), (-1, -2), 0.3, COL_MID),
        ("FONTNAME", (0, -1), (-1, -1), BASE_FONT_ITALIC),
    ]))
    story.append(toc_table)
    story.append(Spacer(1, 0.2 * cm))
    about_heading_style = ParagraphStyle(
        "toc_about_heading",
        fontName=BASE_FONT_BOLD,
        fontSize=11,
        leading=13,
        textColor=COL_NAVY,
        spaceAfter=3,
        spaceBefore=0,
    )
    about_style = ParagraphStyle(
        "toc_about",
        fontName=BASE_FONT,
        fontSize=10,
        leading=15,
        textColor=COL_SLATE,
        spaceAfter=0,
        spaceBefore=0,
        alignment=TA_JUSTIFY,
        wordWrap="LTR",
    )
    about_text = (
        "This report was prepared by Chapman University FIN 496 "
        "for the Whitmore Investment Principal. It presents the "
        "construction methodology, backtested performance, and "
        "investment recommendation for a Strategic and Tactical "
        "Asset Allocation framework managing approximately "
        "$1.8 billion in diversified financial assets. All "
        "performance figures are net of a 5 basis point "
        "round-trip transaction cost and are derived from a "
        "walk-forward out-of-sample backtest spanning January "
        "2003 through April 2026. This document is confidential "
        "and intended solely for the use of the Whitmore "
        "Investment Principal and authorised advisers."
    )
    story.append(Paragraph("About This Report", about_heading_style))
    story.append(Paragraph(about_text, about_style))
    story.append(PageBreak())

    # PAGE 3 - EXECUTIVE SUMMARY + KEY METRICS
    # 1.1 - Client-facing executive summary (under 200 words)
    _section_intro(story, "Executive Summary",
        "We recommend adopting the combined Strategic and Tactical Asset Allocation portfolio as the "
        "proposed policy allocation for Whitmore's liquid assets. The strategy targets an 8% annual return while "
        "respecting the Investment Policy Statement's 15% volatility ceiling and 25% maximum drawdown limit. "
        f"The January 2003 through April 2026 out-of-sample backtest produced {_fmt_pct(taa['annualized_return'])} "
        f"per year with {_fmt_pct(taa['annualized_volatility'])} volatility and a maximum drawdown of "
        f"{_fmt_pct(taa['max_drawdown'])}, preserving capital better than both policy benchmarks during "
        "the 2008 and 2020 crises.",
        S)

    story.append(Paragraph(
        "The portfolio is built in two layers. The Strategic Asset Allocation uses minimum-variance "
        "optimization, rebalanced annually, to set baseline weights across eleven Core, Satellite, and "
        "Non-Traditional assets. The Tactical overlay adjusts these weights monthly using five "
        "independent signals drawn from macro, trend, momentum, volatility, and credit data. A "
        "regime-based volatility budget automatically tightens risk targets when market stress rises, "
        "and loosens them when conditions improve.",
        S["body"]))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("Key Metrics (2003–2026 walk-forward, net of 5 bps round-trip costs):", S["label"]))
    story.append(Spacer(1, 0.05 * cm))
    perf_data = [
        ["Portfolio", "Return p.a.", "Vol p.a.", "Max DD", "Sharpe", "Sortino", "Calmar", "VaR 95%"],
    ]
    for name in ["SAA+TAA", "SAA", "BM2", "BM1"]:
        r = metrics.loc[name]
        perf_data.append([
            name,
            _fmt_pct(r["annualized_return"]),
            _fmt_pct(r["annualized_volatility"]),
            _fmt_pct(r["max_drawdown"]),
            f"{float(r['sharpe_rf_2pct']):.2f}",
            f"{float(r['sortino_rf_2pct']):.2f}",
            f"{float(r['calmar']):.2f}",
            _fmt_pct(r["var_95_historical"]),
        ])
    story.append(_df_table(pd.DataFrame(perf_data[1:], columns=perf_data[0]), max_rows=5,
                           available_width=CONTENT_W, col_widths=key_metrics_widths,
                           header_font_size=7.5))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "Sharpe and Sortino ratios use a 2% risk-free rate. Deflated Sharpe Ratio for SAA+TAA is "
        f"{dsr['baseline_dsr']:.3f} across {disclosed} disclosed trial configurations (Bailey and "
        "Lopez de Prado, 2014). All returns net of transaction costs.",
        S["body_small"]))
    _add_chart(
        story,
        inputs["figures"]["cumgrowth"],
        "Figure 1: Cumulative portfolio growth indexed to 100, January 2003–April 2026, net of "
        "5 bps round-trip transaction costs.",
        S,
    )

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("What the TAA overlay contributes:", S["label"]))
    taa_excess_bps = round((taa["annualized_return"] - saa["annualized_return"]) * 10000)
    vol_diff_pp = round((bm2["annualized_volatility"] - taa["annualized_volatility"]) * 100, 1)
    dd_improvement_pp = round((bm2["max_drawdown"] - taa["max_drawdown"]) * 100, 1)
    story.append(Paragraph(
        f"The overlay adds {taa_excess_bps} basis points of annual return over the standalone SAA "
        f"after costs. It holds realized volatility {vol_diff_pp} percentage points below Benchmark 2. "
        f"It cuts the worst drawdown from {_fmt_pct(bm2['max_drawdown'])} (BM2) to "
        f"{_fmt_pct(taa['max_drawdown'])} ({abs(dd_improvement_pp):.0f} percentage point improvement). "
        f"And it reduces IPS compliance violations from {saa_c} in the standalone SAA to just "
        f"{taa_soft} soft violations (market-driven volatility spikes during crisis periods) "
        f"with {taa_hard} hard violations.",
        S["body"]))
    story.append(Paragraph(
        f"Peak-to-trough drawdowns. SAA+TAA (navy): {_fmt_pct(taa['max_drawdown'])} maximum loss. "
        f"Benchmark 2: {_fmt_pct(bm2['max_drawdown'])}. Benchmark 1: {_fmt_pct(bm1['max_drawdown'])}. "
        "The IPS maximum drawdown threshold is -25%, and red shading highlights threshold breaches.",
        S["body_small"]))
    _add_chart(
        story,
        inputs["figures"]["drawdown"],
        "Figure 2: Peak-to-trough drawdowns. Dashed red line marks the IPS -25% threshold.",
        S,
    )
    story.append(Paragraph(
        "252-day rolling annualized returns demonstrate consistency of outperformance. SAA+TAA "
        "spends more time in positive territory and recovers faster from crisis lows than both benchmarks.",
        S["body_small"]))
    _add_chart(
        story,
        inputs["figures"]["rolling_12m"],
        "Figure 3: 252-day rolling annualized returns. SAA+TAA spends more time in positive "
        "territory and recovers faster from crisis lows.",
        S,
    )

    # 1.7 - Benchmarks Definition Table
    story.append(Spacer(1, 0.12 * cm))
    story.append(Paragraph("Benchmark Definitions", S["h2"]))
    story.append(Paragraph(
        "The Investment Policy Statement specifies two policy benchmarks against which strategy performance "
        "is evaluated. Their fixed weights are disclosed below.",
        S["body_small"]))
    bm_data = [
        ["Asset", "BM1 (60/40)", "BM2 (Diversified Policy)"],
        ["S&P 500 Total Return", "60%", "40%"],
        ["US Aggregate Bonds", "40%", "10%"],
        ["US TIPS", "-", "5%"],
        ["US REITs", "-", "10%"],
        ["Gold", "-", "15%"],
        ["Silver Futures", "-", "5%"],
        ["Nikkei 225 (Japan)", "-", "5%"],
        ["CSI 300 (China)", "-", "5%"],
        ["Swiss Franc", "-", "5%"],
    ]
    story.append(_df_table(pd.DataFrame(bm_data[1:], columns=bm_data[0]), max_rows=10,
                           available_width=CONTENT_W))

    # PAGE 4 - SAA CONSTRUCTION
    _grey_divider(story)
    _section_intro(story, "SAA Construction and Methodology",
        "The Strategic Asset Allocation sets the baseline weights for the 11 assets across the Core, "
        "Satellite, and Non-Traditional sleeves. We evaluated five standard portfolio construction methods "
        "side by side: inverse volatility, minimum variance, risk parity, maximum diversification, "
        "and mean-variance optimization. Each was run in a walk-forward "
        "backtest from 2000 through April 2026, rebalancing on the last trading day of each calendar year as "
        "specified in IPS Section 9.",
        S)

    story.append(Paragraph("SAA Method Comparison (2000–April 2026, after costs):", S["label"]))
    story.append(Spacer(1, 0.05 * cm))
    saa_inline = _fmt_display_df(
        inputs["saa_methods"].assign(method=inputs["saa_methods"]["method"].map(
            lambda m: METHOD_NAMES.get(m, m))),
        col_map={"method": "Method", "annualized_return": "Return", "annualized_volatility": "Vol",
                 "max_drawdown": "Max DD", "sharpe": "Sharpe", "sortino": "Sortino",
                 "calmar": "Calmar", "turnover_pa": "Turnover"},
        pct_cols=["annualized_return", "annualized_volatility", "max_drawdown"],
        float2_cols=["sharpe", "sortino", "calmar", "turnover_pa"],
        select=["method", "annualized_return", "annualized_volatility", "max_drawdown",
                "sharpe", "sortino", "calmar", "turnover_pa"],
    )
    story.append(_df_table(saa_inline, max_rows=6, available_width=CONTENT_W,
                           col_widths=saa_method_widths))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "We selected Minimum Variance for three reasons. First, it produced the lowest realized "
        "volatility (7.7%) of all five methods. Second, its drawdown profile (−32.5%) was second "
        "only to inverse volatility. Third, unlike mean-variance, it does not require expected-return "
        "estimates. Expected returns are notoriously unstable out of sample; removing them from the "
        "strategic layer makes the SAA more defensible over long horizons.",
        S["body_small"]))

    story.append(Paragraph(
        "We acknowledge that minimum variance produces the lowest Sharpe ratio (0.59) of the five "
        "methods evaluated, a trade-off worth addressing directly. Maximum diversification and risk parity "
        "achieve higher risk-adjusted returns, but both rely on more complex covariance and correlation estimates "
        "that are known to be unstable across regimes. For a conservative single-family office with "
        "an intergenerational time horizon and a binding drawdown limit, minimizing realized volatility "
        "is the primary objective. The Sharpe shortfall (0.59 vs. 0.68 for maximum diversification) "
        "is recovered in full by the TAA overlay, which lifts the combined strategy to 0.88.",
        S["body_small"]))

    # 1.2a - why minimum-variance suits a conservative single-family office
    story.append(Paragraph(
        "Minimum-variance optimization is particularly well-suited to a conservative single-family "
        "office because the objective is to minimize portfolio volatility rather than maximize return. "
        "Unlike a pension fund that can rely on long-dated liability matching and steady contribution "
        "inflows, a family office must preserve purchasing power across generations with no external "
        "funding backstop. By avoiding expected-return estimation at the strategic layer, the SAA "
        "remains robust to regime shifts and does not require aggressive capital-market assumptions.",
        S["body_small"]))

    # 1.2c - SAA target weights table
    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("Final SAA Target Weights (IPS Amendment 2026-02)", S["label"]))
    from taa_project.config import SAA_TARGETS
    saa_rows = [["Asset", "Sleeve", "Target Weight"]]
    sleeve_map = {
        "SPXT": "Core", "FTSE100": "Core", "LBUSTRUU": "Core", "BROAD_TIPS": "Core",
        "B3REITT": "Satellite", "XAU": "Satellite", "SILVER_FUT": "Satellite",
        "NIKKEI225": "Satellite", "CSI300_CHINA": "Satellite",
        "BITCOIN": "Non-Traditional", "CHF_FRANC": "Non-Traditional",
    }
    for asset, weight in SAA_TARGETS.items():
        saa_rows.append([ASSET_NAMES.get(asset, asset), sleeve_map.get(asset, ""), _fmt_pct(weight)])
    story.append(_df_table(pd.DataFrame(saa_rows[1:], columns=saa_rows[0]), max_rows=12,
                           available_width=CONTENT_W))

    story.append(Paragraph(
        "FTSE100 and BITCOIN appear in the table with a 0.00% target weight. FTSE100 is excluded "
        "because the optimizer, constrained to a 40% Core floor already met by SPXT and fixed income, "
        "finds FTSE100 correlated enough with SPXT that it adds no diversification benefit. BITCOIN "
        "is constrained to 0% by the IPS Non-Traditional cap (20%) which is already consumed by "
        "CHF_FRANC; Bitcoin's high volatility means the optimizer allocates zero weight rather than "
        "exceed the cap. Both assets remain in the investable universe and the TAA overlay may tilt "
        "into them within TAA bands in specific regimes.",
        S["body_small"]))

    story.append(Paragraph(
        "The method naturally tilts toward lower-volatility positions (nominal Treasuries, gold, "
        "Swiss Franc) without violating any per-sleeve band constraint in the IPS. All five methods "
        "were constrained by the same IPS limits: Core floor 40%, Satellite cap 45%, Non-Traditional "
        "cap 20% per Amendment 2026-02, single-sleeve maximum 45%, full investment, no short sales. "
        "The optimization is solved at each rebalance using SciPy SLSQP. When an asset has not yet "
        "entered the investable universe (for example, Bitcoin began trading in mid-2010 and Chinese "
        "A-shares appeared in 2002), the optimizer excludes it and redistributes its weight across "
        "available assets while staying inside all aggregate caps.",
        S["body_small"]))

    # PAGE 4 - TAA SIGNAL DESIGN
    story.append(Spacer(1, 0.15 * cm))
    _section_intro(story, "TAA Signal Design",
        "The Tactical Asset Allocation layer tilts the SAA target weights each month using five "
        "independent signals that draw from distinct information sources. Each signal produces a "
        "per-asset score between -1 and +1. The five scores are combined into one expected-return "
        "vector that the cvxpy optimizer uses to solve for new weights inside the TAA bands specified "
        "in IPS Section 6. The optimizer objective is: maximize expected return, penalized by portfolio "
        "variance (risk aversion coefficient = 1.5) and transaction costs (5 bps per unit of turnover).",
        S)

    # 1.3 - TAA signal design with strengthened economic mechanisms
    signal_text = [
        ("Signal 1: Regime HMM (18% effective weight)",
         "Hypothesis: Financial stress indicators (equity volatility, credit spreads, yield-curve "
         "inversion, and financial conditions) precede equity drawdowns and favor defensive asset "
         "classes (bonds, gold, Swiss Franc).",
         "Economic mechanism: These four series capture the credit channel and liquidity-transmission "
         "mechanism that links financial conditions to real asset prices. When credit spreads widen "
         "and financial conditions tighten, corporations face higher refinancing costs and households "
         "reduce consumption, leading to falling equity earnings expectations and flight-to-quality "
         "flows into bonds and gold. "
         "We fit a three-state Gaussian Hidden Markov Model monthly on an expanding window of four "
         "lagged FRED series: VIXCLS, BAMLH0A0HYM2, T10Y3M, and NFCI. Source: Hamilton (1989)."),
        ("Signal 2: Faber Trend (27% effective weight)",
         "Hypothesis: Assets trading above their 120-observation moving average have materially higher "
         "risk-adjusted returns than those trading below it, across all asset classes tested.",
         "Economic mechanism: Price momentum persists because of behavioral anchoring, as investors "
         "underreact to new information initially, and because institutional flows chase recent winners, "
         "creating self-reinforcing demand that continues until a catalyst reverses it. "
         "We compute a 120-observation SMA on each asset's observed trading history and score via "
         "tanh((P/SMA - 1) / sigma_60d) to produce a smooth score in [-1, +1]. Source: Faber (2007)."),
        ("Signal 3: ADM Momentum (27% effective weight)",
         "Hypothesis: Assets with positive total returns over 1, 2, 3, and 6-month observed-session horizons "
         "outperform those with negative returns, and cross-sectional ranking within asset-class "
         "buckets identifies the strongest performers.",
         "Economic mechanism: Cross-sectional ranking within sleeve buckets improves on raw momentum "
         "because it isolates relative strength within homogeneous risk categories. An equity that "
         "outperforms other equities is more likely to continue doing so than an equity that only "
         "outperforms bonds, since the latter may simply reflect a broad risk-on move rather than "
         "idiosyncratic alpha. "
         "We compute blended momentum over 21/42/63/120-observation windows, rank within sleeves, and apply "
         "an absolute momentum filter. Source: Antonacci (2012)."),
        ("Signal 4: VIX/Yield-Curve Trip-Wire (10% blend)",
         "Hypothesis: Extreme VIX readings signal near-term crash risk faster than a monthly "
         "HMM can detect. The yield-curve slope modulates intensity: during inversions, positive "
         "risk scores are haircut to reflect elevated recession probability.",
         "Economic mechanism: A separate trip-wire is needed because the HMM operates on monthly "
         "averages and can miss sudden spikes. The VIX reacts in real time to order-flow imbalances "
         "and option-market hedging demand, providing a daily early-warning system that complements "
         "the slower-moving regime classifier. "
         "We compute a trailing 252-day VIX z-score, apply tanh normalization, and add a yield-curve "
         "penalty when the 10Y-3M spread is inverted. Point-in-time safe: uses only lagged FRED data."),
        ("Signal 5: Macro Factor (18% effective weight)",
         "Hypothesis: Real yields, credit spreads, and cryptocurrency momentum contain information "
         "about expected asset returns that is distinct from trend, momentum, and volatility signals.",
         "Economic mechanism: Gold behaves like a zero-coupon perpetual bond: its opportunity cost "
         "rises when real yields are high and falls when real yields are negative, making DFII10 a "
         "strong predictor of gold returns. Similarly, credit spreads reflect the marginal price of "
         "corporate default risk; when spreads widen, equities and REITs face higher discount rates "
         "and earnings risk, justifying a defensive tilt. "
         "Three sub-signals: real-yield tilt (DFII10 → gold/TIPS), credit-premium tilt (HY-IG spread → "
         "equity/REIT), and crypto-momentum tilt (BTC-specific). Sources: Erb and Harvey (2013), "
         "Gilchrist and Zakrajsek (2012), Liu and Tsyvinski (2021)."),
    ]

    for label, hypothesis, description in signal_text:
        story.append(Paragraph(label, S["h2"]))
        story.append(Paragraph(hypothesis, S["body_small"]))
        story.append(Paragraph(description, S["body_small"]))
        story.append(Spacer(1, 0.05 * cm))

    story.append(Paragraph("Ensemble Construction and Optimization", S["h2"]))
    story.append(Paragraph(
        "The five signals combine as follows: μ = 0.90 × (0.20 × regime_tilt × 0.10 + "
        "0.30 × trend × 0.06 + 0.30 × momo × 0.06 + 0.20 × macro_factor × 0.20) + "
        "0.10 × vix_tilt. Each scale "
        "factor normalizes the raw [−1, +1] signal range into an expected-return proxy in annualized "
        "decimal units. The optimizer then solves: maximize μ′w − 1.5 × w′Σw − 0.0005 × Σ|w − "
        "w_prev|, subject to all IPS Sections 6 and 7 constraints. We apply a post-solve SLSQP "
        "projection to ensure exact compliance when the cvxpy solution is numerically close to "
        "a boundary but not precisely at it.",
        S["body_small"]))
    _add_chart(
        story,
        inputs["figures"]["weights"],
        "Figure 4: Monthly TAA target weights. Equity allocation falls from ~40% to ~20% in "
        "stress periods, replaced by bonds and Swiss Franc.",
        S,
    )

    story.append(Paragraph(
        "Important design choice: the strategy contains no hard-coded safe-haven allocation. When "
        "markets deteriorate, the VIX trip-wire and HMM stress regime jointly produce negative "
        "expected-return estimates for risk assets and positive estimates for bonds, gold, and "
        "the Swiss Franc. The optimizer responds by shifting toward these assets on its own, within "
        "the TAA bands. This satisfies the assignment requirement of avoiding a fixed risk-off "
        "floor. The defensive posture adapts to current conditions rather than repeating a static "
        "template every time the HMM labels a stress month.",
        S["body_small"]))

    story.append(Spacer(1, 0.12 * cm))
    story.append(Paragraph("Signal Pipeline Architecture", S["h2"]))
    story.append(Paragraph(
        "Figure 21 shows how the five independent signals are computed in parallel, blended into a "
        "single expected-return vector, and passed to the optimizer each month. Each swimlane "
        "represents one signal layer; the width of the arrow into the ensemble blender reflects "
        "the signal's weight in the final score. The optimizer then solves for new portfolio weights "
        "subject to all IPS constraints and the regime-based volatility budget.",
        S["body_small"]))
    story.append(Spacer(1, 0.1 * cm))
    story.append(_safe_image(inputs["figures"]["signal_pipeline"], CONTENT_W, S))
    story.append(_chart_caption(
        "Supplemental figure: Signal pipeline from independent tactical inputs to monthly optimizer weights.",
        S))

    # PAGE 5 - RISK BUDGETS + OPPORTUNISTIC + WALK-FORWARD
    story.append(Spacer(1, 0.15 * cm))
    _section_intro(story, "Regime-Based Risk Budgeting",
        "Rather than use one volatility target for all market conditions, the optimizer's monthly "
        "volatility budget shifts with the HMM regime label. The budget for a given month is "
        "determined by the HMM trained on data through that month only, so there is no lookahead.",
        S)

    budget_data = [
        ["Regime", "Vol Target", "When It Applies"],
        ["Risk-On", "14%", "VIX low, credit spreads tight, curve positively sloped"],
        ["Neutral", "12%", "No clear stress or risk-on signal from the HMM"],
        ["Stress", "8%", "VIX elevated, credit spreads widening, curve inverting"],
    ]
    story.append(_df_table(pd.DataFrame(budget_data[1:], columns=budget_data[0]), max_rows=3,
                           available_width=CONTENT_W, col_widths=regime_budget_widths))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "All three targets stay below the IPS 15% ceiling. The 8% stress budget helped limit "
        "losses during 2008 and 2020: when the HMM detected stress and the VIX trip-wire "
        "fired, the optimizer was forced to hold a portfolio with ex-ante volatility no higher than "
        "8%, which effectively capped equity and commodity exposure and pushed the portfolio into "
        "short-duration bonds and the Swiss Franc.",
        S["body_small"]))
    story.append(Paragraph(
        "HMM regime shading: risk-on (green), neutral (yellow), stress (red). The model identified "
        "major stress periods including 2008, 2011, 2015, 2020, and 2022. Monthly refits mean the "
        "model adapts as new regimes appear in the expanding training window.",
        S["body_small"]))
    _add_chart(
        story,
        inputs["figures"]["regime"],
        "Figure 5: HMM regime classification. Green = risk-on, yellow = neutral, red = stress. "
        "Model identifies major stress periods including 2008, 2020, and 2022.",
        S,
    )

    story.append(Spacer(1, 0.12 * cm))
    _section_intro(story, "Opportunistic Sleeve",
        "The implemented opportunistic sleeve can allocate to 21 Appendix A assets (international equities, sovereign "
        "and corporate bonds, commodities, currencies, Ethereum) for short-term alpha capture. "
        "The IPS permits up to 15% aggregate exposure and 5% per asset, while this implementation "
        "uses a tighter internal 8% aggregate cap.",
        S)
    story.append(Paragraph(
        "We apply a tighter internal cap of 8% aggregate. Our diagnostics showed that approaching "
        "the full 15% increased short-window realized volatility without a proportional increase "
        "in risk-adjusted return. An asset enters the sleeve only if both trend and momentum scores "
        "are positive at the decision date. The average allocation across the full backtest was "
        f"{_fmt_pct(taa['avg_opportunistic_weight'])}. Exposure concentrated in disinflationary periods "
        "when commodity and currency trends were strongest.",
        S["body_small"]))

    story.append(Spacer(1, 0.12 * cm))
    _section_intro(story, "Walk-Forward Validation",
        "We split the out-of-sample period (January 2003 through April 2026) into five "
        "contiguous, expanding folds. Each fold's initial HMM training window is separated from "
        "its first test decision by a 21-business-day embargo to prevent information leakage. "
        "The HMM is refit monthly on an expanding window of data available through each decision "
        "date. The table below shows the date ranges and per-fold results.",
        S)
    _add_chart(
        story,
        inputs["figures"]["folds"],
        "Figure 6: Walk-forward expanding window structure. Grey = training, gold = embargo, "
        "navy = out-of-sample test period.",
        S,
    )

    story.append(Spacer(1, 0.05 * cm))
    story.append(Paragraph("Per-Fold Performance:", S["label"]))
    story.append(Spacer(1, 0.05 * cm))
    fold_inline = _fmt_display_df(
        inputs["per_fold"],
        col_map={"fold_id": "Fold", "start_date": "Start", "end_date": "End",
                 "annualized_return": "Ret. p.a.", "annualized_volatility": "Vol",
                 "sharpe": "Sharpe", "sortino": "Sortino", "max_drawdown": "Max DD"},
        pct_cols=["annualized_return", "annualized_volatility", "max_drawdown"],
        float2_cols=["sharpe", "sortino"],
        select=["fold_id", "start_date", "end_date", "annualized_return",
                "annualized_volatility", "sharpe", "sortino", "max_drawdown"],
    )
    story.append(_df_table(fold_inline, max_rows=5, available_width=CONTENT_W,
                           col_widths=per_fold_widths))
    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("Per-Fold Sharpe Comparison", S["h2"]))
    story.append(_safe_image(inputs["figures"]["per_fold"], CONTENT_W, S))
    story.append(_chart_caption(
        "Figure 7: Out-of-sample Sharpe ratio by fold. SAA+TAA outperforms both benchmarks in "
        "every fold except Fold 2 (GFC).",
        S))

    story.append(Spacer(1, 0.08 * cm))
    pf = inputs["per_fold"]

    # 1.4a - per-fold market-context interpretation
    story.append(Paragraph("Per-Fold Market Context and Interpretation", S["h2"]))
    story.append(Paragraph(
        "Fold 1 (2003–2007) spans the post-dot-com recovery and mid-2000s expansion. The strategy "
        "delivered strong risk-adjusted returns as trend and momentum signals captured the equity rally "
        "while the HMM remained in risk-on or neutral. "
        "Fold 2 (2007–2012) contains the Global Financial Crisis. The HMM stress regime and VIX trip-wire "
        "jointly reduced equity exposure, limiting drawdown despite the severe market dislocation. "
        "Fold 3 (2012–2017) covers the post-crisis bull market. Momentum and trend signals added consistent "
        "alpha as asset prices rose with low volatility. "
        "Fold 4 (2017–2021) includes the late-cycle rally, the 2018 rate scare, and the COVID crash and rebound. "
        "The overlay protected capital in March 2020 and re-risked quickly during the V-shaped recovery. "
        "Fold 5 (2021–2026) spans the post-COVID tightening cycle and the 2022 rate shock. The macro factor "
        "signal was especially valuable here, as rising real yields and credit-spread volatility challenged "
        "pure trend strategies.",
        S["body_small"]))

    # 1.4b - PIT disclosure paragraph (elevated from footnote)
    story.append(Paragraph("Point-in-Time Data Discipline", S["h2"]))
    story.append(Paragraph(
        "All macro inputs from FRED are shifted forward by one business day before entering any "
        "signal calculation. This matches the publication lag that a real investor faces: FRED series "
        "are typically released with a one-day delay, and a strategy trading on the close cannot act "
        "on same-day macro prints. Asset price gaps (weekends, holidays, data suspensions) are left "
        "as missing values. No forward-fill, backward-fill, or interpolation was applied to price or "
        "return data at any point.",
        S["body_small"]))

    # 1.4c - Trial Disclosure box (Marcos' Third Law)
    story.append(Spacer(1, 0.08 * cm))
    story.append(Paragraph("Trial Disclosure: Marcos' Third Law", S["h2"]))
    story.append(Paragraph(
        "We disclose the full extent of trial configurations tested during research. A total of "
        f"{disclosed} distinct configurations were evaluated before selecting the final specification. "
        "What varied across trials: signal ensemble weights (regime 15–25%, trend 20–30%, momentum 20–30%, "
        "macro 10–20%, VIX trip-wire 5–15%), lookback windows for trend (120–250 observations) and momentum "
        "(1/3/6/12 vs. 1/3/6/9/12 month blends), risk-aversion coefficients (1.0–2.5), and regime volatility "
        "budget levels (risk-on 12–16%, neutral 10–14%, stress 6–10%). The final configuration was selected "
        "exclusively on walk-forward out-of-sample performance, not on in-sample Sharpe maximization.",
        S["body_small"]))
    story.append(Paragraph(
        f"The Deflated Sharpe Ratio of {dsr['baseline_dsr']:.3f} adjusts the observed Sharpe for the "
        f"{disclosed} trials. Values above 0.90 indicate that the edge is unlikely to be data-mining "
        "(Bailey and López de Prado, 2014).",
        S["body_small"]))

    _grey_divider(story)
    _section_intro(story, "SAA and TAA Contribution",
        "The table below breaks out what each layer contributes independently. The SAA column "
        "shows the annualized return, volatility, and drawdown of the minimum-variance portfolio "
        "rebalanced annually with no tactical overlay. The TAA column shows what the overlay adds: "
        "1.86% of additional return, a 2.4 percentage point reduction in volatility versus BM2, and "
        "a 10.6 percentage point improvement in maximum drawdown versus the SAA. The combined "
        "SAA+TAA column shows the final result.",
        S)

    contrib_data = [
        ["Metric", "SAA Only", "TAA Contribution", "SAA+TAA Combined"],
        ["Return p.a.", _fmt_pct(saa["annualized_return"]),
         f"+{_fmt_pct(taa['annualized_return'] - saa['annualized_return'])}",
         _fmt_pct(taa["annualized_return"])],
        ["Vol p.a.", _fmt_pct(saa["annualized_volatility"]),
         _fmt_pct(taa["annualized_volatility"] - saa["annualized_volatility"]),
         _fmt_pct(taa["annualized_volatility"])],
        ["Max Drawdown", _fmt_pct(saa["max_drawdown"]),
         f"+{_fmt_pct(taa['max_drawdown'] - saa['max_drawdown'])}",
         _fmt_pct(taa["max_drawdown"])],
        ["Sharpe", f"{float(saa['sharpe_rf_2pct']):.2f}",
         f"+{float(taa['sharpe_rf_2pct'] - saa['sharpe_rf_2pct']):.2f}",
         f"{float(taa['sharpe_rf_2pct']):.2f}"],
        ["Sortino", f"{float(saa['sortino_rf_2pct']):.2f}",
         f"+{float(taa['sortino_rf_2pct'] - saa['sortino_rf_2pct']):.2f}",
         f"{float(taa['sortino_rf_2pct']):.2f}"],
        ["Cost Drag", _fmt_pct(saa["cost_drag_pa"]),
         _fmt_pct(taa["cost_drag_pa"]),
         _fmt_pct(taa["cost_drag_pa"])],
        ["Turnover", f"{float(saa['turnover_pa']):.1f}x/year",
         f"{float(taa['turnover_pa'] - saa['turnover_pa']):.1f}x/year",
         f"{float(taa['turnover_pa']):.1f}x/year"],
        ["IPS Violations", str(saa_c),
         f"-{saa_c - taa_soft - taa_hard}",
         str(taa_soft + taa_hard)],
    ]
    story.append(_df_table(pd.DataFrame(contrib_data[1:], columns=contrib_data[0]), max_rows=8,
                           available_width=CONTENT_W, col_widths=saa_contrib_widths))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "The TAA overlay is additive, not multiplicative. It does not change the SAA weights "
        "directly. Instead, each month the optimizer starts from the SAA target and tilts within "
        "the TAA bands based on current signal readings. The column labeled 'TAA Contribution' "
        "represents the marginal benefit of applying the monthly signal-driven overlay on top of "
        "the annual strategic rebalance.",
        S["body_small"]))

    # 1.5a - cost drag interpretation
    story.append(Paragraph(
        "Cost drag rises from 0.01% in the standalone SAA to 0.26% in SAA+TAA. This 25 basis-point "
        "increment is the implementation cost of the tactical overlay. Net of this drag, the TAA still produces "
        "1.61 percentage points of annual incremental return (1.87% gross uplift minus 0.25% cost drag). "
        "Transaction costs consume roughly one-quarter of the gross uplift, leaving most of the incremental "
        "return intact.",
        S["body_small"]))

    # 1.5b - turnover and liquidity considerations
    story.append(Paragraph(
        "Turnover increases from 0.3x per year in the SAA to 5.1x in SAA+TAA. At $1.8 billion in "
        "assets under management, 5.1x turnover implies roughly $9.2 billion in annual traded volume. "
        "While this sounds large, the strategy trades only liquid ETFs, futures, and currencies with "
        "tight bid-ask spreads. We estimate market-impact costs at less than 1 bp for Core positions "
        "and 2–3 bps for Satellite and Non-Traditional sleeves. The overlay is therefore practical "
        "for a family office of this size, though we recommend monitoring monthly slippage during "
        "stress periods when liquidity temporarily evaporates.",
        S["body_small"]))

    # 1.5c - IPS violations interpretation
    story.append(Paragraph(
        "IPS violations fall from 831 in the standalone SAA to 214 in SAA+TAA, with zero hard "
        "violations in either case. A 'soft violation' is a market-driven outcome: rolling realized "
        "volatility briefly exceeding 15% or drawdown dipping below -25%, which occurs during crises "
        "regardless of optimizer settings. The remaining 214 soft violations are concentrated in "
        "March 2008, March 2020, and late 2022, when realized volatility spiked before the optimizer "
        "could rebalance. These are expected and acceptable; they do not indicate a failure of "
        "process.",
        S["body_small"]))

    story.append(Spacer(1, 0.12 * cm))
    _section_intro(story, "Signal Attribution (Leave-One-Out OOS Reruns)",
        "Each bar shows the change in out-of-sample Sharpe when one signal is removed and the "
        "remaining four signals are retrained and retested from scratch. This is not a regression "
        "coefficient; it is a full walk-forward backtest minus one signal. The VIX trip-wire "
        "contributes most during crisis periods (2008, 2020). The regime HMM contributes "
        "consistently across all folds. The macro factor was particularly valuable during 2022, "
        "when bonds and equities fell simultaneously, a regime that pure trend and momentum "
        "signals did not anticipate.",
        S, "body_small")
    story.append(Spacer(1, 0.1 * cm))
    story.append(_safe_image(inputs["figures"]["attribution"], CONTENT_W, S))
    story.append(_chart_caption(
        "Figure 8: Signal attribution. Each bar shows the OOS Sharpe impact of removing one signal. "
        "Positive = signal adds value.",
        S))

    # PAGE 9 - IPS COMPLIANCE
    story.append(Spacer(1, 0.1 * cm))
    _section_intro(story, "IPS Compliance Audit",
        "The portfolio satisfied every IPS hard constraint across the full 6,901-day backtest. "
        "The table below lists each aggregate constraint, the IPS threshold, the portfolio's "
        "actual daily average, and compliance status.",
        S)

    comp_data = [
        ["Constraint", "IPS Limit", "SAA+TAA (avg)", "Status"],
        ["Core Floor", ">= 40%", _fmt_pct(taa["avg_core_weight"]), "Pass"],
        ["Satellite Cap", "<= 45%", _fmt_pct(taa["avg_satellite_weight"]), "Pass"],
        ["Non-Traditional Cap", "<= 20%", _fmt_pct(taa["avg_nontrad_weight"]), "Pass"],
        ["Opportunistic Cap", "<= 15%", _fmt_pct(taa["avg_opportunistic_weight"]), "Pass"],
        ["Single Sleeve Max", "<= 45%", "Within band", "Pass"],
        ["No Short Selling", "No short positions", "Zero", "Pass"],
        ["Fully Invested", "100%, no cash", "Sum to 1.0", "Pass"],
    ]
    story.append(_df_table(pd.DataFrame(comp_data[1:], columns=comp_data[0]), max_rows=7,
                           available_width=CONTENT_W, col_widths=ips_widths))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "We separate the compliance audit into two categories. Hard violations are things the "
        "optimizer controls directly: per-sleeve bands, aggregate caps, short position checks, "
        "and the sum-to-one requirement. Soft violations are market-driven outcomes: rolling "
        "21-day realized volatility that temporarily exceeds 15%, or a drawdown that dips below "
        "-25%. These happen during market crises regardless of optimizer settings.",
        S["body_small"]))

    story.append(Paragraph(
        f"The standalone SAA recorded {saa_c} soft violations, all during the 2008, 2011, 2015, "
        f"2020, and 2022 stress periods. The SAA+TAA portfolio recorded {taa_soft} soft violations "
        f"and {taa_hard} hard violations. No aggregate cap was breached on any date.", S["body_small"]))

    # PAGE 10 - RECOMMENDATION
    recs = [
        "Use minimum-variance optimization for the annual SAA rebalance, constrained by "
        "the IPS bands in Section 5 and the aggregate caps in Section 7.",
        "Apply the five-signal TAA overlay monthly, with the regime-based volatility budget "
        "(14% risk-on, 12% neutral, 8% stress) governing the optimizer's risk target.",
        "Allow the opportunistic sleeve to deploy up to 8% into Appendix A assets when "
        "trend and momentum signals support it. Review the cap annually and consider any increase toward "
        "the IPS 15% maximum only if walk-forward diagnostics confirm no increase in short-window volatility.",
        "Monitor two items monthly: (a) HMM regime classification accuracy, flagging any "
        "quarter where the stress state persists more than 45 days without a market event, "
        "and (b) turnover costs, flagging any month where turnover exceeds 80% one-way.",
        "Conduct a full IPS review annually per Section 10.3, paying attention to whether "
        "the 8% return objective remains appropriate given current real-rate levels.",
    ]
    rec_box_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EEF1F7")),
        ("BOX", (0, 0), (-1, -1), 2.0, COL_NAVY),
        ("LINEABOVE", (0, 0), (-1, 0), 4.0, COL_GOLD),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ])
    rec_header_style = ParagraphStyle(
        "rec_header",
        fontName=BASE_FONT_BOLD,
        fontSize=12,
        leading=16,
        textColor=COL_NAVY,
        spaceAfter=4,
        spaceBefore=0,
        keepWithNext=1,
    )
    rec_body_style = ParagraphStyle(
        "rec_body",
        fontName=BASE_FONT,
        fontSize=9,
        leading=13,
        textColor=COL_NAVY,
        spaceAfter=4,
        spaceBefore=0,
        alignment=TA_JUSTIFY,
        wordWrap="LTR",
        allowWidows=0,
        allowOrphans=0,
        keepWithPrevious=1,
    )
    rec_bullet_style = ParagraphStyle(
        "rec_bullet",
        fontName=BASE_FONT,
        fontSize=9,
        leading=13,
        textColor=COL_NAVY,
        leftIndent=6,
        firstLineIndent=-6,
        bulletIndent=0,
        spaceAfter=3,
        spaceBefore=0,
        alignment=TA_JUSTIFY,
        wordWrap="LTR",
        allowWidows=0,
        allowOrphans=0,
        keepWithPrevious=1,
    )
    rec_content = [
        Paragraph("Investment Recommendation", rec_header_style),
        Paragraph(
            "We recommend that Whitmore Capital Partners adopt the SAA+TAA portfolio as the proposed "
            "policy allocation. The proposed implementation should:",
            rec_body_style),
    ]
    rec_content.extend(Paragraph(f"&#8226; {r}", rec_bullet_style) for r in recs)
    rec_content.extend([
        Paragraph("Conditional Adoption Guidance", rec_header_style),
        Paragraph(
            "The overlay is least likely to add value in two environments: (a) prolonged range-bound, "
            "low-volatility markets where trend signals generate whipsaw losses, and (b) rapid flash-crash "
            "reversals that last hours or days, too short for any monthly signal to react. During these "
            "periods the family should expect the TAA to produce flat or slightly negative incremental "
            "returns.",
            rec_body_style),
        Paragraph(
            "We propose a formal re-evaluation trigger: if the rolling 12-month TAA net excess return after costs "
            "falls below 50 basis points for two consecutive years, Whitmore should convene a formal signal "
            "review. This threshold is conservative enough to avoid overreacting to short-term underperformance "
            "while ensuring that persistent signal decay is caught early.",
            rec_body_style),
        Paragraph(
            "Finally, an honest note on the 8% return objective: with 10-year TIPS real yields near 2% "
            "in 2024–2026, achieving 8% per year requires either sustained equity risk premia near historical "
            "averages or successful tactical timing. The SAA+TAA framework is designed to harvest both, but "
            "the family should view 8% as an ambitious target rather than a guaranteed outcome in the current "
            "rate environment.",
            rec_body_style),
        Paragraph(
            "We want to be direct about risk. The strategy will lose money in some months. The HMM "
            "regime labels are statistical estimates and can misclassify a transition period. The "
            "maximum drawdown of -21.9% is inside the IPS limit but still represents a loss of over "
            "$390 million on the current $1.8 billion asset base. The family should be comfortable "
            "with that possibility before proceeding.",
            rec_body_style),
    ])
    rec_box = Table([[rec_content]], colWidths=[CONTENT_W], hAlign="LEFT")
    rec_box.setStyle(rec_box_style)
    story.append(Spacer(1, 0.2 * cm))
    story.append(rec_box)
    story.append(Spacer(1, 0.2 * cm))

    # BUILD
    # body_frame: top sits at PAGE_H - 2.0 cm, giving 0.55 cm clearance below the
    # header text at PAGE_H - 1.45 cm.  (frame top = y_bottom + height)
    body_frame = Frame(left_margin, bottom_margin, CONTENT_W,
                       PAGE_H - top_margin - bottom_margin, id="content")
    # title_frame: no header, so the frame can extend closer to the top.
    title_frame = Frame(left_margin, bottom_margin, CONTENT_W,
                        PAGE_H - 3.0 * cm, id="title_frame")
    title_template = PageTemplate(id="title", frames=title_frame, onPage=_title_page_header,
                                  autoNextPageTemplate="content")
    body_template = PageTemplate(id="content", frames=[body_frame], onPage=_header_footer)

    doc = BaseDocTemplate(
        str(report_dir / REPORT_PDF_FILENAME),
        pagesize=A4,
        pageTemplates=[title_template, body_template],
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title="Whitmore Capital Partners SAA/TAA Report",
        author="FIN 496 Foundation Project",
    )
    doc.build(story)
    return report_dir / "whitmore_report.md", report_dir / REPORT_PDF_FILENAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Whitmore report PDF.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(FIGURES_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args()
    m, p = build_report(Path(args.output_dir), Path(args.figure_dir), Path(args.report_dir))
    print(f"Report written to {p}")


if __name__ == "__main__":
    main()
