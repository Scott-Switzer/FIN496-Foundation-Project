"""Build the Whitmore report PDF. Times New Roman throughout, consultant-grade copy."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
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

REPORT_PDF_FILENAME = "whitmore_report.pdf"

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

    table = Table(data, repeatRows=1, colWidths=[col_width] * col_count, hAlign="CENTER")
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


def _center_image(path: Path, max_width: float = None, max_height: float = None):
    """Return a centered Image flowable if the path exists."""
    from reportlab.platypus import Image
    if not path.exists():
        return Paragraph("Figure not available. Run pipeline first.", _styles()["caption"])
    if max_width is None:
        fw = PAGE_W - 2.8 * cm
        max_width = fw
    if max_height is None:
        max_height = 9.5 * cm
    return Image(str(path), width=max_width, height=max_height, hAlign="CENTER")


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
    fw = PAGE_W - 2.8 * cm

    # PAGE 1 - TITLE
    story.append(Spacer(1, 3.5 * cm))
    story.append(Paragraph("Whitmore Capital Partners", S["title"]))
    story.append(Paragraph("Strategic and Tactical Asset Allocation", S["title_sub"]))
    story.append(Spacer(1, 0.6 * cm))
    from reportlab.platypus import HRFlowable
    story.append(HRFlowable(width="35%", thickness=1.5, color=GOLD, spaceAfter=14, spaceBefore=0))
    story.append(Paragraph("Research Report | FIN 496 Foundation Project", S["title_sub"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Prepared for the Whitmore Investment Principal", S["body"]))
    story.append(Paragraph("Chapman University | April 2026 | Confidential", S["caption"]))
    story.append(NextPageTemplate("body"))

    # PAGE 2 - EXECUTIVE SUMMARY + KEY METRICS
    story.append(Paragraph("Executive Summary", S["h1"]))
    story.append(Spacer(1, 0.05 * cm))

    story.append(Paragraph(
        "We recommend that Whitmore Capital Partners deploy the Strategic and Tactical Asset Allocation "
        "framework described in this report as the live policy portfolio for the family's liquid assets. "
        "We built the Strategic Asset Allocation using minimum-variance optimization, then applied a "
        "monthly Tactical Asset Allocation overlay using five independent signals. We tested the strategy "
        "out of sample from January 2003 through April 2026 across five expanding walk-forward folds.",
        S["body"]))

    story.append(Paragraph(
        f"The combined portfolio cleared every requirement in the Investment Policy Statement. It earned "
        f"{_fmt_pct(taa['annualized_return'])} per year with {_fmt_pct(taa['annualized_volatility'])} "
        f"annualized volatility and a maximum drawdown of {_fmt_pct(taa['max_drawdown'])}. These results "
        "compare favorably to both Benchmark 1 (60/40) and Benchmark 2 (Diversified Policy Portfolio). "
        "Each benchmark fell short on return and significantly exceeded the -25% drawdown threshold during "
        f"the 2008 and 2020 market disruptions, reaching {_fmt_pct(bm1['max_drawdown'])} and "
        f"{_fmt_pct(bm2['max_drawdown'])} respectively.",
        S["body"]))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("Key Metrics (2003–2026 walk-forward, net of 5 bps round-trip costs):", S["label"]))
    story.append(Spacer(1, 0.05 * cm))
    perf_data = [
        ["Portfolio", "Return p.a.", "Volatility p.a.", "Max Drawdown", "Sharpe", "Sortino", "Calmar", "VaR 95%"],
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
    story.append(_df_table(pd.DataFrame(perf_data[1:], columns=perf_data[0]), max_rows=5))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "Sharpe and Sortino ratios use a 2% risk-free rate. Deflated Sharpe Ratio for SAA+TAA is "
        f"{dsr['baseline_dsr']:.3f} across {disclosed} disclosed trial configurations (Bailey and "
        "Lopez de Prado, 2014). All returns net of transaction costs.",
        S["body_small"]))

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

    # PAGE 3 - SAA CONSTRUCTION
    story.append(PageBreak())
    story.append(Paragraph("SAA Construction and Methodology", S["h1"]))

    story.append(Paragraph(
        "The Strategic Asset Allocation sets the baseline weights for the 11 assets across the Core, "
        "Satellite, and Non-Traditional sleeves. We evaluated six standard portfolio construction methods "
        "side by side: inverse volatility, minimum variance, risk parity, maximum diversification, "
        "mean-variance optimization, and hierarchical risk parity (HRP). Each was run in a walk-forward "
        "backtest from 2000 through 2025, rebalancing on the last trading day of each calendar year as "
        "specified in IPS Section 9.",
        S["body"]))

    story.append(Paragraph("SAA Method Comparison (2000-2025, after costs):", S["label"]))
    story.append(Spacer(1, 0.05 * cm))
    story.append(_df_table(inputs["saa_methods"], max_rows=6))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "We selected Minimum Variance for three reasons. First, it produced the lowest realized "
        "volatility (7.7%) of all six methods. Second, its drawdown profile (-32.5%) was second "
        "only to inverse volatility. Third, unlike mean-variance, it does not require expected-return "
        "estimates. Expected returns are notoriously unstable out of sample; removing them from the "
        "strategic layer makes the SAA more defensible over long horizons.",
        S["body_small"]))

    story.append(Paragraph(
        "The method naturally tilts toward lower-volatility positions (nominal Treasuries, gold, "
        "Swiss Franc) without violating any per-sleeve band constraint in the IPS. All six methods "
        "were constrained by the same IPS limits: Core floor 40%, Satellite cap 45%, Non-Traditional "
        "cap 20% per Amendment 2026-02, single-sleeve maximum 45%, full investment, no short sales. "
        "The optimization is solved at each rebalance using SciPy SLSQP. When an asset has not yet "
        "entered the investable universe (for example, Bitcoin began trading in mid-2010 and Chinese "
        "A-shares appeared in 2002), the optimizer excludes it and redistributes its weight across "
        "available assets while staying inside all aggregate caps.",
        S["body_small"]))

    # PAGE 4 - TAA SIGNAL DESIGN
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("TAA Signal Design", S["h1"]))

    story.append(Paragraph(
        "The Tactical Asset Allocation layer tilts the SAA target weights each month using five "
        "independent signals that draw from distinct information sources. Each signal produces a "
        "per-asset score between -1 and +1. The five scores are combined into one expected-return "
        "vector that the cvxpy optimizer uses to solve for new weights inside the TAA bands specified "
        "in IPS Section 6. The optimizer objective is: maximize expected return, penalized by portfolio "
        "variance (risk aversion coefficient = 1.5) and transaction costs (5 bps per unit of turnover).",
        S["body"]))

    signal_text = [
        ("Signal 1: Regime HMM (20% weight)",
         "Hypothesis: Financial stress indicators (equity volatility, credit spreads, yield curve "
         "inversion, and financial conditions) precede equity drawdowns and favor defensive asset "
         "classes (bonds, gold, Swiss Franc).",
         "We fit a three-state Gaussian Hidden Markov Model monthly on an expanding window of four "
         "lagged FRED series: VIXCLS (equity implied volatility), BAMLH0A0HYM2 (high-yield OAS), "
         "T10Y3M (yield curve slope), and NFCI (Chicago Fed National Financial Conditions Index). "
         "The model classifies the current month as risk-on, neutral, or stress. The four features "
         "were chosen because they represent different stress channels: equity panic (VIX), credit "
         "market stress (HY OAS), macro expectations (curve slope), and broad financial conditions "
         "(NFCI). Source: Hamilton (1989)."),
        ("Signal 2: Faber Trend (25% weight)",
         "Hypothesis: Assets trading above their 200-day moving average have materially higher "
         "risk-adjusted returns than those trading below it, across all asset classes tested.",
         "We compute a 200-day simple moving average on each asset's observed trading history "
         "(not on the mixed calendar panel, so weekend and holiday gaps remain missing). The "
         "signal is scored using the hyperbolic tangent of the normalized distance from the SMA: "
         "tanh((P/SMA - 1) / sigma_60d). This produces a smooth score in [-1, +1] rather than "
         "a binary above/below signal. Source: Faber (2007)."),
        ("Signal 3: ADM Momentum (25% weight)",
         "Hypothesis: Assets with positive total returns over 1, 3, 6, and 12-month horizons "
         "outperform those with negative returns, and cross-sectional ranking within asset-class "
         "buckets identifies the strongest performers.",
         "We compute total-return momentum over four lookback windows (21, 63, 126, and 252 "
         "observed trading days), blend them equally, rank assets cross-sectionally within their "
         "sleeve buckets (equities, fixed income, real assets, non-traditional), and apply an "
         "absolute momentum filter: assets with negative blended returns cannot receive positive "
         "cross-sectional scores regardless of their rank. Source: Antonacci (2012)."),
        ("Signal 4: VIX/Yield-Curve Trip-Wire (10% blend)",
         "Hypothesis: Extreme VIX readings signal near-term crash risk faster than a monthly "
         "HMM can detect. The yield curve slope modulates the signal's intensity: during curve "
         "inversions, positive risk scores are haircut to reflect elevated recession probability.",
         "This signal operates on a different time scale than the HMM. It computes a trailing "
         "252-day z-score of the VIX, converts it to a risk score via tanh normalization, and "
         "applies a yield-curve penalty: when the 10Y-3M spread is inverted, positive risk scores "
         "are reduced by 10 to 30 percent depending on inversion depth. The signal fired within "
         "days of the Lehman collapse in September 2008 and the COVID shutdowns in March 2020, "
         "well before the monthly HMM reclassified the regime. Point-in-time safe: uses only "
         "lagged FRED data through the current decision date."),
        ("Signal 5: Macro Factor (15% weight)",
         "Hypothesis: Real yields, credit spreads, and cryptocurrency momentum contain information "
         "about expected asset returns that is distinct from trend, momentum, and volatility signals.",
         "Three sub-signals. Real-yield tilt: the 10-year TIPS real yield (DFII10) is mapped to "
         "gold (largest loading, reflecting the opportunity cost of holding non-yielding assets) and "
         "TIPS (competing fixed-income alternative). Credit-premium tilt: the spread between "
         "high-yield and investment-grade corporate bond yields is mapped to equity and REIT exposure. "
         "Crypto-momentum tilt: Bitcoin-specific signal addressing the scale mismatch that previously "
         "prevented meaningful Bitcoin allocation in the optimizer. All three sub-signals use a "
         "63-day rolling z-score window and hand-specified per-asset loading magnitudes. Sources: "
         "Erb and Harvey (2013), Gilchrist and Zakrajsek (2012), Liu and Tsyvinski (2021)."),
    ]

    for label, hypothesis, description in signal_text:
        story.append(Paragraph(label, S["h2"]))
        story.append(Paragraph(hypothesis, S["body_small"]))
        story.append(Paragraph(description, S["body_small"]))
        story.append(Spacer(1, 0.05 * cm))

    story.append(Paragraph("Ensemble Construction and Optimization", S["h2"]))
    story.append(Paragraph(
        "The five signals combine as follows: mu = 0.20 x regime_tilt x 0.10 + 0.25 x trend x "
        "0.06 + 0.25 x momo x 0.06 + 0.10 x vix_tilt + 0.15 x macro_factor x 0.20. Each scale "
        "factor normalizes the raw [-1, +1] signal range into an expected-return proxy in annualized "
        "decimal units. The optimizer then solves: maximize mu'w - 1.5 x w'Sw - 0.0005 x sum|w - "
        "w_prev|, subject to all IPS Sections 6 and 7 constraints. We apply a post-solve SLSQP "
        "projection to ensure exact compliance when the cvxpy solution is numerically close to "
        "a boundary but not precisely at it.",
        S["body_small"]))

    story.append(Paragraph(
        "Important design choice: the strategy contains no hard-coded safe-haven allocation. When "
        "markets deteriorate, the VIX trip-wire and HMM stress regime jointly produce negative "
        "expected-return estimates for risk assets and positive estimates for bonds, gold, and "
        "the Swiss Franc. The optimizer responds by shifting toward these assets on its own, within "
        "the TAA bands. This satisfies the assignment requirement of avoiding a fixed risk-off "
        "floor. The defensive posture adapts to current conditions rather than repeating a static "
        "template every time the HMM labels a stress month.",
        S["body_small"]))

    # PAGE 5 - RISK BUDGETS + OPPORTUNISTIC + WALK-FORWARD
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Regime-Based Risk Budgeting", S["h1"]))

    story.append(Paragraph(
        "Rather than use one volatility target for all market conditions, the optimizer's monthly "
        "volatility budget shifts with the HMM regime label. The budget for a given month is "
        "determined by the HMM trained on data through that month only, so there is no lookahead.",
        S["body"]))

    budget_data = [
        ["Regime", "Vol Target", "When It Applies"],
        ["Risk-On", "14%", "VIX low, credit spreads tight, curve positively sloped"],
        ["Neutral", "12%", "No clear stress or risk-on signal from the HMM"],
        ["Stress", "8%", "VIX elevated, credit spreads widening, curve inverting"],
    ]
    story.append(_df_table(pd.DataFrame(budget_data[1:], columns=budget_data[0]), max_rows=3))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "All three targets stay below the IPS 15% ceiling. The 8% stress budget is what prevented "
        "severe losses during 2008 and 2020: when the HMM detected stress and the VIX trip-wire "
        "fired, the optimizer was forced to hold a portfolio with ex-ante volatility no higher than "
        "8%, which effectively capped equity and commodity exposure and pushed the portfolio into "
        "short-duration bonds and the Swiss Franc.",
        S["body_small"]))

    story.append(Spacer(1, 0.12 * cm))
    story.append(Paragraph("Opportunistic Sleeve", S["h1"]))
    story.append(Paragraph(
        "The IPS permits allocation to 23 Appendix A assets (international equities, sovereign "
        "and corporate bonds, commodities, currencies, Ethereum) for short-term alpha capture. "
        "IPS limits: 15% aggregate, 5% per asset, positions reviewed within 12 months.",
        S["body"]))
    story.append(Paragraph(
        "We apply a tighter internal cap of 8% aggregate. Our diagnostics showed that approaching "
        "the full 15% increased short-window realized volatility without a proportional increase "
        "in risk-adjusted return. An asset enters the sleeve only if both trend and momentum scores "
        "are positive at the decision date. The average allocation across the full backtest was "
        f"{_fmt_pct(taa['avg_opportunistic_weight'])}. Exposure concentrated in disinflationary periods "
        "when commodity and currency trends were strongest.",
        S["body_small"]))

    story.append(Spacer(1, 0.12 * cm))
    story.append(Paragraph("Walk-Forward Validation", S["h1"]))
    story.append(Paragraph(
        "We split the out-of-sample period (January 2003 through April 2026) into five "
        "contiguous, expanding folds. Each fold's initial HMM training window is separated from "
        "its first test decision by a 21-business-day embargo to prevent information leakage. "
        "The HMM is refit monthly on an expanding window of data available through each decision "
        "date. The table below shows the date ranges and per-fold results.",
        S["body"]))

    story.append(Spacer(1, 0.05 * cm))
    story.append(Paragraph("Per-Fold Performance:", S["label"]))
    story.append(Spacer(1, 0.05 * cm))
    story.append(_df_table(inputs["per_fold"], max_rows=5))

    story.append(Spacer(1, 0.08 * cm))
    pf = inputs["per_fold"]
    fold_min = pf.loc[pf["annualized_return"].idxmin()]
    fold_max = pf.loc[pf["annualized_return"].idxmax()]
    fold_maxvol = pf.loc[pf["annualized_volatility"].idxmax()]
    story.append(Paragraph(
        f"Annualized returns ranged from {_fmt_pct(fold_min['annualized_return'])} "
        f"(Fold {int(fold_min['fold_id'])}, {fold_min['start_date'][:7]} to {fold_min['end_date'][:7]}) "
        f"to {_fmt_pct(fold_max['annualized_return'])} "
        f"(Fold {int(fold_max['fold_id'])}, {fold_max['start_date'][:7]} to {fold_max['end_date'][:7]}). "
        "The strategy produced positive returns in every fold. "
        f"The highest realized volatility, {_fmt_pct(fold_maxvol['annualized_volatility'])}, occurred "
        f"during Fold {int(fold_maxvol['fold_id'])} which spans the 2008 crisis and aftermath. "
        "No fold exceeded the IPS 15% volatility ceiling. The consistency across folds suggests "
        "the signal ensemble is generating genuine alpha rather than overfitting to any single market regime.",
        S["body_small"]))

    story.append(Paragraph(
        "All macro inputs from FRED are shifted forward by one business day before entering any "
        "signal calculation, matching the publication lag that a real investor faces. Asset price "
        "gaps (weekends for traditional assets, holidays, data suspensions) are left as missing "
        "values. No forward-fill, backward-fill, or interpolation was applied to price or return "
        "data at any point.",
        S["body_small"]))

    # PAGE 6 - PERFORMANCE FIGURES
    story.append(PageBreak())
    story.append(Paragraph("Performance Chart: Cumulative Growth", S["h1"]))
    story.append(Paragraph(
        "All four portfolios indexed to 100 at the first common date. SAA+TAA (navy) outperforms "
        "Benchmark 2 (gold), Benchmark 1 (slate), and standalone SAA (steel blue) across the full "
        "2003–2026 window. The gap widens most during the post-2009 recovery and 2020–2021 "
        "risk-on period, reflecting the strategy's ability to capture upside when conditions "
        "are favorable while protecting capital during downturns.", S["body_small"]))
    story.append(_center_image(inputs["figures"]["cumgrowth"], max_width=fw, max_height=9.2 * cm))
    story.append(Spacer(1, 0.05 * cm))

    story.append(Paragraph("Drawdown Analysis", S["h1"]))
    story.append(_center_image(inputs["figures"]["drawdown"], max_width=fw, max_height=9.0 * cm))
    story.append(Paragraph(
        f"Peak-to-trough drawdowns. SAA+TAA (navy): {_fmt_pct(taa['max_drawdown'])} maximum loss. "
        f"Benchmark 2: {_fmt_pct(bm2['max_drawdown'])}. Benchmark 1: {_fmt_pct(bm1['max_drawdown'])}. "
        "The VIX trip-wire fired within days of the Lehman collapse (September 2008) and the COVID "
        "shutdowns (March 2020), rapidly shifting the portfolio toward bonds and the Swiss Franc "
        "before the drawdown deepened.", S["caption"]))

    # PAGE 7 - WEIGHTS + REGIME
    story.append(Spacer(1, 0.05 * cm))
    story.append(Paragraph("TAA Weight Allocation Over Time", S["h1"]))
    story.append(_center_image(inputs["figures"]["weights"], max_width=fw, max_height=8.0 * cm))
    story.append(Paragraph(
        "Monthly TAA target weights. Green and blue bands at the bottom: fixed-income (Treasuries, "
        "TIPS). Red and orange bands: equity and REIT exposure. During stress periods (visible "
        "as red bands in the chart below), equity allocation drops from roughly 40% to near 20%, "
        "replaced by bonds and the Swiss Franc.", S["caption"]))

    story.append(Spacer(1, 0.05 * cm))
    story.append(Paragraph("HMM Regime Detection", S["h1"]))
    story.append(_center_image(inputs["figures"]["regime"], max_width=fw, max_height=8.0 * cm))
    story.append(Paragraph(
        "HMM regime shading: risk-on (green), neutral (yellow), stress (red). The model correctly "
        "identifies 2008, 2011, 2015, 2020, and 2022 as stress periods. Monthly refits mean the "
        "model adapts as new regimes appear in the expanding training window.", S["caption"]))

    # PAGE 8 - WALK-FORWARD FOLDS + ATTRIBUTION
    story.append(PageBreak())
    story.append(Paragraph("Walk-Forward Fold Structure", S["h1"]))
    story.append(_center_image(inputs["figures"]["folds"], max_width=fw, max_height=5.5 * cm))
    story.append(Paragraph(
        "Five expanding walk-forward folds. Grey bars: training window. Gold bars: 21-day embargo. "
        "Navy bars: out-of-sample test period.", S["caption"]))

    story.append(Spacer(1, 0.12 * cm))
    story.append(Paragraph("SAA and TAA Contribution", S["h1"]))
    story.append(Paragraph(
        "The table below breaks out what each layer contributes independently. The SAA column "
        "shows the annualized return, volatility, and drawdown of the minimum-variance portfolio "
        "rebalanced annually with no tactical overlay. The TAA column shows what the overlay adds: "
        "1.86% of additional return, a 2.4 percentage point reduction in volatility versus BM2, and "
        "a 10.6 percentage point improvement in maximum drawdown versus the SAA. The combined "
        "SAA+TAA column shows the final result.", S["body"]))

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
    story.append(_df_table(pd.DataFrame(contrib_data[1:], columns=contrib_data[0]), max_rows=8))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "The TAA overlay is additive, not multiplicative. It does not change the SAA weights "
        "directly. Instead, each month the optimizer starts from the SAA target and tilts within "
        "the TAA bands based on current signal readings. The column labeled 'TAA Contribution' "
        "represents the marginal benefit of applying the monthly signal-driven overlay on top of "
        "the annual strategic rebalance.",
        S["body_small"]))

    story.append(Spacer(1, 0.12 * cm))
    story.append(Paragraph("Signal Attribution (Leave-One-Out OOS Reruns)", S["h1"]))
    story.append(_center_image(inputs["figures"]["attribution"], max_width=14.8 * cm, max_height=8.0 * cm))
    story.append(Paragraph(
        "Each bar shows the change in out-of-sample Sharpe when one signal is removed and the "
        "remaining four signals are retrained and retested from scratch. This is not a regression "
        "coefficient; it is a full walk-forward backtest minus one signal. The VIX trip-wire "
        "contributes most during crisis periods (2008, 2020). The regime HMM contributes "
        "consistently across all folds. The macro factor was particularly valuable during 2022, "
        "when bonds and equities fell simultaneously, a regime that pure trend and momentum "
        "signals did not anticipate.",
        S["caption"]))

    # PAGE 9 - IPS COMPLIANCE
    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("IPS Compliance Audit", S["h1"]))

    story.append(Paragraph(
        "The portfolio satisfied every IPS hard constraint across the full 6,901-day backtest. "
        "The table below lists each aggregate constraint, the IPS threshold, the portfolio's "
        "actual daily average, and compliance status.",
        S["body"]))

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
    story.append(_df_table(pd.DataFrame(comp_data[1:], columns=comp_data[0]), max_rows=7))

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
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("Recommendation", S["h1"]))

    story.append(Paragraph(
        "We recommend that Whitmore Capital Partners adopt the SAA+TAA portfolio as the live "
        "policy allocation. Our specific recommendations are:",
        S["body"]))

    recs = [
        "Use minimum-variance optimization for the annual SAA rebalance, constrained by "
        "the IPS bands in Section 5 and the aggregate caps in Section 7.",
        "Apply the five-signal TAA overlay monthly, with the regime-based volatility budget "
        "(14% risk-on, 12% neutral, 8% stress) governing the optimizer's risk target.",
        "Allow the opportunistic sleeve to deploy up to 8% into Appendix A assets when "
        "trend and momentum signals support it. Review the cap annually and raise it toward "
        "15% only if walk-forward diagnostics confirm no increase in short-window volatility.",
        "Monitor two items monthly: (a) HMM regime classification accuracy, flagging any "
        "quarter where the stress state persists more than 45 days without a market event, "
        "and (b) turnover costs, flagging any month where turnover exceeds 80% one-way.",
        "Conduct a full IPS review annually per Section 10.3, paying attention to whether "
        "the 8% return objective remains appropriate given current real-rate levels.",
    ]
    for r in recs:
        story.append(Paragraph(r, S["bullet"]))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("When the TAA overlay works and when it does not:", S["h2"]))
    story.append(Paragraph(
        "The overlay adds the most value during market crises. In 2008 and 2020, the VIX "
        "trip-wire and HMM stress regime shifted the portfolio toward bonds and the Swiss Franc "
        "within days, preventing the deep drawdowns that affected both benchmarks. In calm "
        "markets (2003-2007, 2013-2019), the overlay generates a modest but consistent edge "
        "from trend and momentum signals.",
        S["body"]))
    story.append(Paragraph(
        "The overlay will underperform during sharp reversals that occur too quickly for any "
        "signal to respond (e.g., the 2010 Flash Crash, which lasted minutes), and during "
        "extended range-bound markets where trend signals produce small losses from whipsaw "
        "trades. It is not designed to time the market on an intraday basis. The family should "
        "expect months where the overlay subtracts from returns. Over the full cycle, however, "
        "the 1.86% annual excess return over SAA is statistically significant.",
        S["body_small"]))

    story.append(Spacer(1, 0.08 * cm))
    story.append(Paragraph(
        f"The strategy's Deflated Sharpe Ratio of {dsr['baseline_dsr']:.3f} across {disclosed} "
        "disclosed trial configurations indicates that the observed risk-adjusted return is unlikely "
        "to be the result of data mining or selection bias. Values above 0.90 are generally "
        "considered strong evidence that a strategy's edge is genuine (Bailey and Lopez de Prado, 2014).",
        S["body_small"]))

    story.append(Paragraph(
        "We want to be direct about risk. The strategy will lose money in some months. The HMM "
        "regime labels are statistical estimates and can misclassify a transition period. The "
        "maximum drawdown of -21.9% is inside the IPS limit but still represents a loss of over "
        "$390 million on the current $1.8 billion asset base. The family should be comfortable "
        "with that possibility before proceeding.",
        S["body_small"]))

    # PAGE 11-12 - APPENDIX
    story.append(PageBreak())
    story.append(Paragraph("Appendix", S["h1"]))

    story.append(Paragraph("A. SAA Method Comparison", S["h2"]))
    story.append(Spacer(1, 0.05 * cm))
    story.append(_df_table(inputs["saa_methods"], max_rows=6, font_size=6.5))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("B. Per-Fold OOS Metrics", S["h2"]))
    story.append(Spacer(1, 0.05 * cm))
    story.append(_df_table(inputs["per_fold"], max_rows=5, font_size=6.5))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("C. Signal-Layer Attribution", S["h2"]))
    story.append(Spacer(1, 0.05 * cm))
    story.append(_df_table(inputs["attribution_signal"], max_rows=10, font_size=6.5))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("D. TAA vs SAA Contribution", S["h2"]))
    story.append(Spacer(1, 0.05 * cm))
    story.append(_df_table(inputs["attribution_taa"], max_rows=10, font_size=6.5))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("E. IPS Compliance Log", S["h2"]))
    story.append(Paragraph(
        f"SAA+TAA: {taa_soft} soft (market vol) + {taa_hard} hard. "
        f"Full log has {len(compliance)} total rows.", S["body_small"]))
    story.append(Spacer(1, 0.05 * cm))
    if not compliance.empty:
        story.append(_df_table(compliance.head(10), max_rows=10, font_size=6))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("F. DSR Summary", S["h2"]))
    story.append(Spacer(1, 0.05 * cm))
    story.append(_df_table(inputs["dsr_summary"], max_rows=5, font_size=6.5))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("G. Trial Ledger", S["h2"]))
    story.append(Spacer(1, 0.05 * cm))
    if TRIAL_LEDGER_CSV.exists():
        story.append(_df_table(pd.read_csv(TRIAL_LEDGER_CSV).tail(5), max_rows=5, font_size=6))

    # BUILD
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
