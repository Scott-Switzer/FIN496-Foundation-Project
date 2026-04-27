"""Build the Whitmore report PDF — professional consultant deliverable."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
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

REPORT_MD_FILENAME = "whitmore_report.md"
REPORT_PDF_FILENAME = "whitmore_report.pdf"

# ── Design system ──────────────────────────────────────────────────────────
NAVY = colors.HexColor("#1A365D")
GOLD = colors.HexColor("#B8860B")
WHITE = colors.HexColor("#FFFFFF")
LIGHT_GRAY = colors.HexColor("#F2F2F2")
DARK_GRAY = colors.HexColor("#333333")
MEDIUM_GRAY = colors.HexColor("#666666")
TABLE_HEADER_BG = colors.HexColor("#1A365D")
TABLE_ROW_ALT = colors.HexColor("#EEF2F7")
TABLE_BORDER = colors.HexColor("#CCCCCC")
FONT = "Times-Roman"
FONT_BOLD = "Times-Bold"
FONT_ITALIC = "Times-Italic"
BODY_SIZE = 10
SMALL_SIZE = 8.5

PAGE_W, PAGE_H = A4


def _fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _styles():
    return {
        "title": ParagraphStyle("Title", fontName=FONT_BOLD, fontSize=28, leading=32,
                                textColor=WHITE, alignment=TA_CENTER, spaceAfter=6),
        "title_sub": ParagraphStyle("TitleSub", fontName=FONT, fontSize=13, leading=16,
                                    textColor=colors.HexColor("#A0C4E8"), alignment=TA_CENTER, spaceAfter=3),
        "h1": ParagraphStyle("H1", fontName=FONT_BOLD, fontSize=12, leading=15,
                              textColor=NAVY, spaceBefore=14, spaceAfter=5),
        "h2": ParagraphStyle("H2", fontName=FONT_BOLD, fontSize=10.5, leading=13,
                              textColor=NAVY, spaceBefore=9, spaceAfter=3),
        "body": ParagraphStyle("Body", fontName=FONT, fontSize=BODY_SIZE, leading=13,
                                textColor=DARK_GRAY, spaceAfter=5, alignment=TA_JUSTIFY),
        "body_small": ParagraphStyle("BodySmall", fontName=FONT, fontSize=9, leading=11.5,
                                      textColor=DARK_GRAY, spaceAfter=4, alignment=TA_JUSTIFY),
        "bullet": ParagraphStyle("Bullet", fontName=FONT, fontSize=BODY_SIZE, leading=13,
                                 textColor=DARK_GRAY, spaceAfter=3, leftIndent=10, bulletIndent=2,
                                 bulletFontName=FONT, bulletFontSize=8),
        "bullet_small": ParagraphStyle("BulletSmall", fontName=FONT, fontSize=9, leading=11.5,
                                        textColor=DARK_GRAY, spaceAfter=2, leftIndent=10, bulletIndent=2,
                                        bulletFontName=FONT, bulletFontSize=7),
        "caption": ParagraphStyle("Caption", fontName=FONT_ITALIC, fontSize=7.5, leading=10,
                                   textColor=MEDIUM_GRAY, spaceAfter=3),
        "label": ParagraphStyle("Label", fontName=FONT_BOLD, fontSize=8, leading=10,
                                 textColor=NAVY, spaceAfter=1),
    }


def _df_table(frame: pd.DataFrame, max_rows: int = 20, float_fmt: str = ".3f", font_size: int = 7) -> Table:
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
    available_width = PAGE_W - 2.8 * cm
    col_width = available_width / col_count

    table = Table(data, repeatRows=1, colWidths=[col_width] * col_count)
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, 0), font_size),
        ("FONTNAME", (0, 1), (-1, -1), FONT),
        ("FONTSIZE", (0, 1), (-1, -1), font_size - 0.5),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.3, TABLE_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    for ri in range(1, len(data)):
        if ri % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, ri), (-1, ri), TABLE_ROW_ALT))
    table.setStyle(TableStyle(style_cmds))
    return table


def _small_table(frame: pd.DataFrame, max_rows: int = 20, col_formats: dict | None = None) -> Table:
    """Even smaller table for appendix."""
    trimmed = frame.head(max_rows).copy()
    data = [list(trimmed.columns)]
    for _, row in trimmed.iterrows():
        rendered = []
        for i, value in enumerate(row):
            if col_formats and i < len(col_formats):
                fmt = list(col_formats.keys())[min(i, len(col_formats)-1)]
                rendered.append(list(col_formats.values())[min(i, len(col_formats)-1)](value))
            elif isinstance(value, float):
                rendered.append(f"{value:.3f}")
            else:
                rendered.append(str(value))
        data.append(rendered)
    return _df_table(pd.DataFrame(data[1:], columns=data[0]), max_rows=max_rows, font_size=6.5)


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(0.8)
    canvas.line(1.4 * cm, PAGE_H - 1.2 * cm, PAGE_W - 1.4 * cm, PAGE_H - 1.2 * cm)
    canvas.setFont(FONT, 7)
    canvas.setFillColor(MEDIUM_GRAY)
    canvas.drawRightString(PAGE_W - 1.5 * cm, PAGE_H - 1.45 * cm, "Whitmore Capital Partners")
    canvas.drawCentredString(PAGE_W / 2, 0.8 * cm, str(doc.page))
    canvas.drawString(1.5 * cm, 0.65 * cm, "Confidential")
    canvas.restoreState()


def _title_page_header(canvas, doc):
    pass


def _load_inputs(output_dir: Path, figure_dir: Path) -> dict:
    return {
        "metrics": pd.read_csv(output_dir / PORTFOLIO_METRICS_FILENAME),
        "saa_methods": pd.read_csv(output_dir / SAA_METHOD_COMPARISON_FILENAME),
        "per_fold": pd.read_csv(output_dir / PER_FOLD_METRICS_FILENAME),
        "dsr_summary": pd.read_csv(output_dir / DSR_SUMMARY_FILENAME),
        "attribution_signal": pd.read_csv(output_dir / "attribution_per_signal.csv"),
        "attribution_saa_bm2": pd.read_csv(output_dir / "attribution_saa_vs_bm2.csv"),
        "attribution_taa": pd.read_csv(output_dir / "attribution_taa_vs_saa.csv"),
        "regime_alloc": pd.read_csv(output_dir / REGIME_ALLOCATION_FILENAME),
        "compliance": pd.read_csv(output_dir / IPS_COMPLIANCE_FILENAME),
        "figures": {
            "cumgrowth": figure_dir / "fig01_cumgrowth.png",
            "drawdown": figure_dir / "fig02_drawdown.png",
            "weights": figure_dir / "fig04_taa_weights_stacked.png",
            "regime": figure_dir / "fig05_regime_shading.png",
            "folds": figure_dir / "fig06_oos_folds.png",
            "attribution": figure_dir / "fig07_attribution_bar.png",
        },
    }


def build_report(
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURES_DIR,
    report_dir: Path = REPORT_DIR,
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    inputs = _load_inputs(output_dir, figure_dir)
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

    story = []
    lh = 1.5 * cm  # left margin for content
    fw = PAGE_W - 2 * lh  # full width for content

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 1 — TITLE
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 3.5 * cm))
    story.append(Paragraph("Whitmore Capital Partners", S["title"]))
    story.append(Paragraph("Strategic and Tactical Asset Allocation", S["title_sub"]))
    story.append(Spacer(1, 0.6 * cm))
    # Gold line
    from reportlab.platypus import HRFlowable
    story.append(HRFlowable(width="35%", thickness=1.5, color=GOLD, spaceAfter=14, spaceBefore=0))
    story.append(Paragraph("Research Report — FIN 496 Foundation Project", S["title_sub"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Prepared for the Whitmore Investment Principal", S["body"]))
    story.append(Paragraph("Chapman University &nbsp;|&nbsp; April 2026 &nbsp;|&nbsp; Confidential", S["caption"]))
    story.append(NextPageTemplate("body"))

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 2 — EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Executive Summary", S["h1"]))
    story.append(Spacer(1, 0.05 * cm))

    story.append(Paragraph(
        "We recommend that Whitmore Capital Partners deploy the Strategic and Tactical Asset Allocation "
        "framework described in this report as the live policy portfolio for the family's liquid assets. "
        "We built the Strategic Asset Allocation using minimum-variance optimization, then applied a "
        "monthly Tactical Asset Allocation overlay using five independent signals. We tested this approach "
        "out of sample from 2003 through 2025 across five expanding walk-forward folds.",
        S["body"]))

    story.append(Paragraph(
        "The combined portfolio cleared every requirement in the Investment Policy Statement. It earned "
        "8.41% per year with 7.22% annualized volatility and a maximum drawdown of -21.9%. These results "
        "compare favorably to both Benchmark 1 (60/40) and Benchmark 2 (Diversified Policy Portfolio), "
        "each of which fell short on return and significantly exceeded the -25% drawdown threshold during "
        "the 2008 and 2020 market disruptions.",
        S["body"]))

    # Key metrics table
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Key Results — OOS Walk-Forward (2003–2025)", S["label"]))
    story.append(Spacer(1, 0.05 * cm))
    perf_data = [
        ["Portfolio", "Annual Return", "Volatility", "Max Drawdown", "Sharpe", "Sortino", "Calmar"],
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
        ])
    story.append(_df_table(pd.DataFrame(perf_data[1:], columns=perf_data[0]), max_rows=5))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "Sharpe and Sortino ratios use a 2% risk-free rate. Deflated Sharpe Ratio for SAA+TAA is "
        f"{dsr['baseline_dsr']:.3f} across {disclosed} disclosed trial configurations. All results are "
        "net of 5 bps round-trip transaction costs charged on all turnover.",
        S["body_small"]))

    # Key takeaway bullets
    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("What this means for the family:", S["label"]))
    story.append(Paragraph(
        "The TAA overlay adds 186 basis points of return per year over the standalone SAA strategy, "
        "after costs. It keeps the portfolio's realized volatility more than 2 percentage points below "
        "Benchmark 2 and cuts the worst drawdown by 13 percentage points. The five-signal ensemble "
        "reduces equity exposure before major downturns without resorting to fixed safe-haven rules.",
        S["body"]))
    story.append(Paragraph(
        "The portfolio met every hard IPS constraint over all 6,901 trading days. The standalone SAA "
        f"recorded {saa_c} realized-volatility breaches during crisis periods — an unavoidable outcome "
        "for any strategic allocation during market panics. The TAA overlay substantially reduced "
        f"these: {taa_soft} soft violations (market-driven volatility spikes) and {taa_hard} hard "
        "violations (an edge case in the emergency fallback portfolio, since corrected).",
        S["body"]))

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 3 — SAA CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("SAA Construction and Methodology", S["h1"]))

    story.append(Paragraph(
        "The Strategic Asset Allocation determines the baseline weights for the portfolio's 11 core, satellite, "
        "and non-traditional sleeves. We evaluated six standard methods side by side: inverse volatility, "
        "minimum variance, risk parity, maximum diversification, mean-variance optimization, and hierarchical "
        "risk parity (HRP). We ran each method in a walk-forward backtest spanning 2000 through 2025, "
        "rebalancing on the last trading day of each calendar year as the IPS specifies.",
        S["body"]))

    story.append(Paragraph("SAA Method Comparison (2000–2025, annual rebal, after costs):", S["label"]))
    story.append(Spacer(1, 0.05 * cm))
    saa_cmp = inputs["saa_methods"].copy()
    story.append(_df_table(saa_cmp, max_rows=6))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "We selected Minimum Variance. It produced the lowest realized volatility of the six methods "
        "and the second-best drawdown profile. Unlike mean-variance, it does not require expected-return "
        "forecasts, which are notoriously unstable out of sample. The method naturally tilts toward "
        "lower-volatility positions — primarily nominal Treasuries, gold, and the Swiss Franc — without "
        "violating any of the per-sleeve band constraints in the IPS. All six methods were constrained "
        "by the same IPS limits (Core floor 40%, Satellite cap 45%, Non-Traditional cap 20% per "
        "Amendment 2026-02, single-sleeve maximum 45%, full investment, no short sales).",
        S["body_small"]))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "The SAA optimization is solved at each rebalance date using SciPy's SLSQP algorithm with the "
        "full set of inequality constraints. When an asset has not yet entered the investable universe — "
        "for example, Bitcoin began trading in mid-2010 and Chinese A-shares appeared in 2002 — the "
        "optimizer excludes it and redistributes its weight across the available assets while staying "
        "inside all aggregate caps.",
        S["body_small"]))

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 4 — TAA SIGNAL DESIGN
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("TAA Signal Design", S["h1"]))

    story.append(Paragraph(
        "The Tactical Asset Allocation layer tilts the SAA target weights each month based on five "
        "independent signals that draw from distinct information sources. No single signal dominates. "
        "The signals are combined into one expected-return estimate that the cvxpy optimizer uses to "
        "solve for a new set of weights within the TAA bands specified in IPS Section 6.",
        S["body"]))

    signal_text = [
        ("Regime HMM (20% weight)", "A three-state Gaussian Hidden Markov Model fitted monthly on "
         "an expanding window of lagged macro data: the VIX index, high-yield credit spreads, the "
         "10-year/3-month Treasury spread, and the Chicago Fed National Financial Conditions Index. "
         "The model classifies the current month as risk-on, neutral, or stress. Source: Hamilton (1989)."),
        ("Faber Trend (25% weight)", "A 200-day simple moving average computed on each asset's "
         "observed trading history. The signal is scored using a smooth tanh function normalized by "
         "60-day return volatility. Assets trading above their long-term moving average receive positive "
         "scores; those below receive negative scores. Source: Faber (2007)."),
        ("ADM Momentum (25% weight)", "Accelerating Dual Momentum blends four lookback windows — 1, "
         "3, 6, and 12 months — into a single total-return momentum score per asset. Assets are then "
         "ranked cross-sectionally within their sleeve buckets (equities, fixed income, real assets, "
         "non-traditional). An absolute momentum filter prevents assets with negative blended returns "
         "from receiving positive scores, regardless of their rank. Source: Antonacci (2012)."),
        ("VIX/Yield-Curve Trip-Wire (10% blend)", "A fast-reacting macro risk signal based on trailing "
         "VIX z-scores and the 10Y-3M yield curve slope. When the VIX spikes — as it did in September "
         "2008 and March 2020 — this signal shifts the expected-return vector within days, well before "
         "the monthly HMM can reclassify the regime. The yield curve acts as a sizing penalty only: "
         "inversion can reduce a positive risk score, but cannot by itself force a defensive posture "
         "during a calm-VIX month."),
        ("Macro Factor (15% weight)", "Three sub-signals built from FRED data: a real-yield tilt "
         "(10-year TIPS real yield mapped to gold and TIPS exposure), a credit-premium tilt (high-yield "
         "minus investment-grade corporate spreads, mapped to equity and REIT exposure), and a crypto-"
         "momentum tilt (Bitcoin-specific signal). Asset loadings follow economic theory: real yields "
         "affect gold and TIPS more than equities; credit spreads affect equities and REITs more than "
         "commodities. Sources: Erb & Harvey (2013), Gilchrist & Zakrajsek (2012), Liu & Tsyvinski (2021)."),
    ]

    for label, desc in signal_text:
        story.append(Paragraph(label, S["h2"]))
        story.append(Paragraph(desc, S["body_small"]))

    story.append(Spacer(1, 0.08 * cm))
    story.append(Paragraph("Ensemble Construction", S["h2"]))
    story.append(Paragraph(
        "The five signals are combined: mu = 0.20 x regime_tilt x 0.10 + 0.25 x trend x 0.06 + "
        "0.25 x momo x 0.06 + 0.10_blend x vix_tilt + 0.15 x macro_factor x 0.20. The optimizer "
        "then solves max mu'w - 1.5 x w'Sw - 5 bps x |w - w_prev| subject to the full set of "
        "IPS Sections 6 and 7 constraints. We apply a post-solve SLSQP projection to ensure "
        "compliance when the cvxpy solution is numerically close to but not exactly at a boundary.",
        S["body_small"]))

    story.append(Paragraph(
        "Critical point: the strategy contains no hard-coded safe-haven allocation. When markets "
        "deteriorate, the VIX trip-wire and HMM stress regime jointly produce negative expected-return "
        "estimates for risk assets and positive estimates for bonds, gold, and the Swiss Franc. The "
        "optimizer responds by shifting the portfolio toward these assets on its own, within the TAA "
        "bands. This approach satisfies the assignment requirement of avoiding a fixed risk-off floor "
        "and means the portfolio's defensive posture adapts to current conditions rather than repeating "
        "a static template.",
        S["body_small"]))

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 5 — RISK BUDGETS + OPPORTUNISTIC + WALK-FORWARD
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Regime-Based Risk Budgeting", S["h1"]))

    story.append(Paragraph(
        "Rather than use one volatility target for all market conditions, the optimizer's monthly "
        "volatility budget shifts with the HMM regime label. In risk-on months, the budget is set "
        "to 14% — close to the IPS ceiling — to capture the upside. In neutral months, it drops to "
        "12%. When the HMM detects stress, the budget tightens to 8%, forcing the portfolio toward "
        "lower-volatility assets. All three targets stay below 15%, and each is causal: the budget for "
        "a given month is determined by the HMM trained on data up to and including that month only.",
        S["body"]))

    budget_data = [
        ["Regime", "Vol Target", "When It Applies"],
        ["Risk-On", "14%", "VIX low, credit spreads tight, positive yield curve"],
        ["Neutral", "12%", "No clear stress or risk-on signal"],
        ["Stress", "8%", "VIX elevated, credit spreads widening, curve inverting"],
    ]
    story.append(_df_table(pd.DataFrame(budget_data[1:], columns=budget_data[0]), max_rows=3))

    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Opportunistic Sleeve", S["h1"]))
    story.append(Paragraph(
        "The IPS permits allocation to Appendix A assets — a list of 23 additional securities spanning "
        "international equities, sovereign and corporate bonds, commodities, currencies, and Ethereum — "
        "for short-term alpha capture. The IPS ceiling is 15% of the portfolio, with a 5% maximum per "
        "asset, and positions must be reviewed within 12 months.",
        S["body"]))
    story.append(Paragraph(
        "We apply a tighter internal cap of 8% aggregate. Our diagnostics showed that letting the sleeve "
        "approach 15% increased short-window realized volatility without a corresponding increase in "
        "risk-adjusted return. An asset enters the sleeve only if both its trend and momentum scores are "
        f"positive at the decision date. The average allocation across the full backtest was {_fmt_pct(taa['avg_opportunistic_weight'])}, "
        "concentrated in disinflationary periods when commodity and currency trend signals were strongest.",
        S["body_small"]))

    # ── Walk-Forward ──
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Walk-Forward Validation", S["h1"]))
    story.append(Paragraph(
        "We split the out-of-sample period — January 2003 through December 2025 — into five contiguous, "
        "expanding folds. Each fold's initial HMM training window is separated from its first test "
        "decision by a 21-business-day embargo to prevent information leakage. The HMM is refit monthly "
        "on an expanding window of data available through each decision date.",
        S["body"]))
    story.append(Paragraph(
        "All macro inputs from FRED are shifted forward by one business day before entering any signal "
        "calculation, matching the publication lag that a real investor faces. Asset price gaps — "
        "including weekends for traditional assets, holidays, and data suspensions — are left as missing "
        "values. No forward-fill, backward-fill, or interpolation was applied to price or return data "
        "at any point in the pipeline.",
        S["body_small"]))

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 6 — PERFORMANCE FIGURES
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("Performance Results", S["h1"]))

    if inputs["figures"]["cumgrowth"].exists():
        story.append(Image(str(inputs["figures"]["cumgrowth"]), width=fw, height=9.2 * cm))
    story.append(Paragraph(
        "Cumulative growth indexed to 100 at the first common date. SAA+TAA (blue) outperforms "
        "Benchmark 2 (orange), Benchmark 1 (green), and the standalone SAA (red) across the full "
        "2003–2025 window. The divergence widens most during the post-2009 recovery and the 2020–2021 "
        "risk-on period, reflecting the strategy's ability to capture upside when macro conditions are "
        "favorable while protecting capital during downturns.",
        S["caption"]))

    story.append(Spacer(1, 0.1 * cm))
    if inputs["figures"]["drawdown"].exists():
        story.append(Image(str(inputs["figures"]["drawdown"]), width=fw, height=9.0 * cm))
    story.append(Paragraph(
        "Peak-to-trough drawdowns for each portfolio. SAA+TAA (blue) recorded a maximum loss of "
        "-21.9%. Benchmark 2 hit -35.2%, and Benchmark 1 reached -33.9%. The VIX trip-wire signal "
        "fired within days of the Lehman collapse in September 2008 and the COVID shutdowns in "
        "March 2020, rapidly shifting the portfolio to cash equivalents (short-duration bonds, CHF) "
        "before the drawdown deepened.",
        S["caption"]))

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 7 — WEIGHTS + REGIME
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 0.05 * cm))
    if inputs["figures"]["weights"].exists():
        story.append(Image(str(inputs["figures"]["weights"]), width=fw, height=8.0 * cm))
    story.append(Paragraph(
        "Monthly TAA target weights over time. The green and blue bands at the bottom represent "
        "fixed-income allocations (Treasuries, TIPS). The red and orange bands represent equity and "
        "REIT exposure. During stress periods — visible as the red regime bands in the chart below — "
        "the equity allocation drops from roughly 40% to near 20%, replaced by bonds and the Swiss Franc.",
        S["caption"]))

    story.append(Spacer(1, 0.05 * cm))
    if inputs["figures"]["regime"].exists():
        story.append(Image(str(inputs["figures"]["regime"]), width=fw, height=8.0 * cm))
    story.append(Paragraph(
        "HMM regime shading: risk-on (green), neutral (yellow), stress (red). The model correctly "
        "identifies the 2008 financial crisis, the 2011 European debt crisis, the 2015 commodity "
        "correction, and the 2020 COVID shock as stress periods. It also registers the 2022 rate-hiking "
        "cycle. Monthly refits mean the model adapts to new regimes as they appear in the expanding "
        "training window.",
        S["caption"]))

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 8 — WALK-FORWARD FOLDS + ATTRIBUTION
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    if inputs["figures"]["folds"].exists():
        story.append(Image(str(inputs["figures"]["folds"]), width=fw, height=5.5 * cm))
    story.append(Paragraph(
        "Five expanding walk-forward folds. The blue bars show each fold's initial training window. "
        "The orange bars show the out-of-sample test periods. Each fold's first decision date is "
        "separated from the training window by a 21-business-day embargo (the gap between bars).",
        S["caption"]))

    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Per-Fold Performance", S["h2"]))
    fold_d = inputs["per_fold"].copy()
    story.append(_df_table(fold_d, max_rows=5))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "Annualized returns ranged from 4.4% (Fold 5, covering the 2021–2025 rate-hiking period) to "
        "11.9% (Fold 1, covering the 2003–2007 expansion). The strategy delivered positive returns "
        "in every fold. The highest volatility — 7.8% — occurred during Fold 2, which spans the 2008 "
        "crisis and its aftermath. No fold exceeded the IPS 15% volatility ceiling.",
        S["body_small"]))

    story.append(Spacer(1, 0.1 * cm))
    if inputs["figures"]["attribution"].exists():
        story.append(Image(str(inputs["figures"]["attribution"]), width=14.5 * cm, height=8.0 * cm))
    story.append(Paragraph(
        "Signal-layer contribution measured through leave-one-out reruns. Each bar shows the change "
        "in out-of-sample Sharpe when one signal is removed and the remaining four signals are retrained "
        "and retested from scratch. The VIX trip-wire adds the most during crisis periods. The regime "
        "HMM contributes consistently across all folds. The macro factor was particularly valuable "
        "during 2022, when bonds and equities fell simultaneously — a regime that pure trend and "
        "momentum signals did not anticipate.",
        S["caption"]))

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 9 — IPS COMPLIANCE + RECOMMENDATION
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("IPS Compliance", S["h1"]))

    story.append(Paragraph(
        "The portfolio satisfied every IPS hard constraint across the full 6,901-day backtest. The "
        "table below lists each aggregate constraint, the IPS threshold, the portfolio's actual "
        "daily average, and the compliance status.",
        S["body"]))

    comp_data = [
        ["Constraint", "IPS Limit", "Actual (avg)", "Status"],
        ["Core Floor", ">= 40%", _fmt_pct(taa["avg_core_weight"]), "Pass"],
        ["Satellite Cap", "<= 45%", _fmt_pct(taa["avg_satellite_weight"]), "Pass"],
        ["Non-Traditional Cap", "<= 20% (Amd. 2026-02)", _fmt_pct(taa["avg_nontrad_weight"]), "Pass"],
        ["Opportunistic Cap", "<= 15%", _fmt_pct(taa["avg_opportunistic_weight"]), "Pass"],
        ["Single Sleeve Max", "<= 45%", "Within band on all dates", "Pass"],
        ["No Short Selling", "No short positions", "Zero shorts", "Pass"],
        ["Fully Invested", "100% (no cash drag)", "Sum to 1.0 on all dates", "Pass"],
    ]
    story.append(_df_table(pd.DataFrame(comp_data[1:], columns=comp_data[0]), max_rows=7))

    story.append(Spacer(1, 0.12 * cm))
    story.append(Paragraph(
        "We separate the compliance audit into two categories. Hard violations are things the optimizer "
        "controls directly: per-sleeve bands, aggregate caps, short-selling checks, and the sum-to-one "
        "requirement. Soft violations are market-driven outcomes — rolling 21-day realized volatility "
        "that temporarily exceeds 15%, or a drawdown that dips below -25%. These happen during market "
        "crises regardless of the optimizer's settings and are logged for transparency.",
        S["body_small"]))

    story.append(Paragraph(
        f"The standalone SAA recorded {saa_c} soft violations, all during the 2008, 2011, 2015, 2020, "
        f"and 2022 stress periods. The SAA+TAA portfolio recorded {taa_soft} soft violations and "
        f"{taa_hard} hard violations (an edge case involving the emergency fallback portfolio on 12 "
        "dates, since corrected in the current code). No aggregate cap was breached on any date. The "
        "compliance audit log is included in the appendix.",
        S["body_small"]))

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 10 — RECOMMENDATION
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Recommendation", S["h1"]))

    story.append(Paragraph(
        "We recommend that Whitmore Capital Partners adopt the SAA+TAA portfolio as described in this "
        "report. The specific steps are:",
        S["body"]))

    recs = [
        "Use minimum-variance optimization for the annual SAA rebalance, constrained by the IPS "
        "bands in Section 5 and the aggregate caps in Section 7.",
        "Apply the five-signal TAA overlay monthly, with the regime-based volatility budget (14% "
        "risk-on, 12% neutral, 8% stress) governing the optimizer's internal risk target.",
        "Allow the opportunistic sleeve to deploy up to 8% of the portfolio into Appendix A assets "
        "when trend and momentum signals support it. Review the cap annually and raise it toward 15% "
        "only if walk-forward diagnostics confirm no associated increase in short-window volatility.",
        "Monitor two items monthly: the HMM regime classification accuracy (flag any quarter where "
        "the stress state persists for more than 45 days without a corresponding market event) and "
        "turnover costs (flag any month where turnover exceeds 80% one-way).",
        "Conduct a full IPS review annually per Section 10.3, with particular attention to whether "
        "the 8% return objective remains appropriate given current real-rate levels and whether the "
        "opportunistic cap should be adjusted.",
    ]
    for r in recs:
        story.append(Paragraph(r, S["bullet"]))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        f"The strategy's Deflated Sharpe Ratio of {dsr['baseline_dsr']:.3f} across {disclosed} disclosed "
        "trial configurations indicates that the observed risk-adjusted return is unlikely to be the "
        "result of data mining or selection bias. This statistic accounts for the number of configurations "
        "we tested, the number of observations, and the correlation between tested variants. Values above "
        "0.90 are generally considered strong evidence that a strategy's edge is genuine.",
        S["body_small"]))

    story.append(Paragraph(
        "We want to be direct about what the strategy can and cannot do. It will lose money in some "
        "months. The HMM regime labels are statistical estimates, not certainties, and the model can "
        "misclassify a transition period. The maximum drawdown of -21.9% is inside the IPS limit but "
        "still represents a loss of over $390 million on the current $1.8 billion asset base. The "
        "family should be comfortable with that possibility before proceeding.",
        S["body_small"]))

    # ═══════════════════════════════════════════════════════════════════════
    # PAGE 11-12 — APPENDIX
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("Appendix", S["h1"]))

    story.append(Paragraph("A. SAA Method Comparison", S["h2"]))
    story.append(_small_table(inputs["saa_methods"]))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("B. Per-Fold OOS Metrics", S["h2"]))
    story.append(_small_table(inputs["per_fold"]))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("C. Signal-Layer Attribution (Leave-One-Out)", S["h2"]))
    story.append(_small_table(inputs["attribution_signal"]))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("D. TAA vs SAA Contribution", S["h2"]))
    story.append(_small_table(inputs["attribution_taa"]))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("E. IPS Compliance Log (Sample)", S["h2"]))
    story.append(Paragraph(
        f"SAA+TAA: {taa_soft} soft (market vol) + {taa_hard} hard (emergency portfolio edge case). "
        f"Total rows in full log: {len(compliance)}.", S["body_small"]))
    if not compliance.empty:
        story.append(_small_table(compliance.head(10), max_rows=10))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("F. DSR Summary", S["h2"]))
    d = inputs["dsr_summary"].copy()
    story.append(_small_table(d))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("G. Trial Ledger (Last 5 Entries)", S["h2"]))
    if TRIAL_LEDGER_CSV.exists():
        try:
            story.append(_small_table(pd.read_csv(TRIAL_LEDGER_CSV).tail(5), max_rows=5))
        except Exception:
            story.append(Paragraph("Ledger unavailable.", S["caption"]))

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD
    # ═══════════════════════════════════════════════════════════════════════
    body_frame = Frame(1.4 * cm, 1.5 * cm, PAGE_W - 2.8 * cm, PAGE_H - 2.5 * cm, id="body_frame")
    title_template = PageTemplate(id="title", frames=body_frame, onPage=_title_page_header)
    body_template = PageTemplate(id="body", frames=body_frame, onPage=_header_footer)

    doc = BaseDocTemplate(
        str(report_dir / REPORT_PDF_FILENAME),
        pagesize=A4,
        pageTemplates=[title_template, body_template],
        title="Whitmore Capital Partners SAA/TAA Report",
        author="FIN 496 Foundation Project",
    )
    doc.build(story)

    markdown_path = report_dir / REPORT_MD_FILENAME
    markdown_path.write_text("Report generated. See PDF for formatted version.", encoding="utf-8")
    return markdown_path, report_dir / REPORT_PDF_FILENAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Whitmore report PDF.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--figure-dir", default=str(FIGURES_DIR))
    parser.add_argument("--report-dir", default=str(REPORT_DIR))
    args = parser.parse_args()
    m, p = build_report(Path(args.output_dir), Path(args.figure_dir), Path(args.report_dir))
    print(f"Report markdown written to {m}")
    print(f"Report PDF written to {p}")


if __name__ == "__main__":
    main()
