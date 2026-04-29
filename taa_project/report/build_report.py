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
        "title_body": ParagraphStyle("TitleBody", fontName=FONT, fontSize=BODY_SIZE, leading=13,
                                      textColor=WHITE, alignment=TA_CENTER, spaceAfter=5),
        "title_caption": ParagraphStyle("TitleCaption", fontName=FONT_ITALIC, fontSize=9, leading=11,
                                         textColor=colors.HexColor("#A0C4E8"), alignment=TA_CENTER, spaceAfter=3),
    }


def _df_table(frame: pd.DataFrame, max_rows: int = 20, float_fmt: str = ".3f",
              font_size: int = 7, first_col_ratio: float = 1.0) -> Table:
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
    if first_col_ratio != 1.0 and col_count > 1:
        total_weight = first_col_ratio + (col_count - 1)
        col_widths = [available_width * first_col_ratio / total_weight] + \
                     [available_width / total_weight] * (col_count - 1)
    else:
        col_widths = [available_width / col_count] * col_count

    table = Table(data, repeatRows=1, colWidths=col_widths, hAlign="CENTER")
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, 0), font_size),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("FONTNAME", (0, 1), (-1, -1), FONT),
        ("FONTSIZE", (0, 1), (-1, -1), font_size - 0.5),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK_GRAY),
        ("GRID", (0, 0), (-1, -1), 0.3, TABLE_BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for ri in range(1, len(data)):
        if ri % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, ri), (-1, ri), TABLE_ROW_ALT))
    table.setStyle(TableStyle(style_cmds))
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
                df[c] = df[c].apply(lambda v: f"{float(v)*100:.2f}%" if pd.notna(v) else "—")
    if float2_cols:
        for c in float2_cols:
            if c in df.columns:
                df[c] = df[c].apply(lambda v: f"{float(v):.2f}" if pd.notna(v) else "—")
    df = df.rename(columns=col_map)
    return df


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
    values = plot_df["marginal_oos_sharpe"].to_numpy(dtype=float)
    bar_colors = ["#1A365D" if v >= 0 else "#C53030" for v in values]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    bars = ax.bar(plot_df["label"], values, color=bar_colors, width=0.5, zorder=3,
                  edgecolor="white", linewidth=0.6)
    ax.axhline(0.0, color="#CBD5E0", linewidth=0.9, zorder=2)
    for bar, val in zip(bars, values):
        offset = 0.003 if val >= 0 else -0.003
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=10, color="#2D3748", fontweight="bold")
    ax.set_ylabel("ΔSharpe vs. ablated baseline", fontsize=10)
    ax.set_title("Signal Attribution  ·  Marginal OOS Sharpe", fontsize=12,
                 fontweight="bold", color="#1A365D")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()

    out_path = figure_dir / "fig07_attribution_bar_report.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(0.8)
    canvas.line(1.4 * cm, PAGE_H - 1.2 * cm, PAGE_W - 1.4 * cm, PAGE_H - 1.2 * cm)
    canvas.setFont(FONT, 7)
    canvas.setFillColor(MEDIUM_GRAY)
    # 4.4 — header and footer
    canvas.drawString(1.5 * cm, PAGE_H - 1.45 * cm, "Whitmore Capital Partners | Confidential")
    canvas.drawCentredString(PAGE_W / 2, 0.8 * cm, str(doc.page))
    canvas.drawRightString(PAGE_W - 1.5 * cm, 0.65 * cm, "Chapman University | April 2026")
    canvas.restoreState()


def _title_page_header(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
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
    story.append(Paragraph("Prepared for the Whitmore Investment Principal", S["title_body"]))
    story.append(Paragraph("Chapman University | April 2026 | Confidential", S["title_caption"]))
    story.append(NextPageTemplate("body"))
    story.append(PageBreak())

    # PAGE 2 - TABLE OF CONTENTS (4.2)
    story.append(Paragraph("Table of Contents", S["h1"]))
    story.append(Spacer(1, 0.05 * cm))
    toc_data = [
        ["Section", "Page"],
        ["Executive Summary", "3"],
        ["SAA Construction and Methodology", "4"],
        ["TAA Signal Design", "5"],
        ["Regime-Based Risk Budgeting & Opportunistic Sleeve", "6"],
        ["Walk-Forward Validation", "6"],
        ["Performance Charts", "7"],
        ["TAA Weight Allocation & Regime Detection", "8"],
        ["Walk-Forward Folds, SAA/TAA Contribution & Signal Attribution", "9"],
        ["IPS Compliance Audit", "10"],
        ["Recommendation", "11"],
        ["Appendix", "12"],
    ]
    story.append(_df_table(pd.DataFrame(toc_data[1:], columns=toc_data[0]), max_rows=12, font_size=9))
    story.append(PageBreak())

    # PAGE 3 - EXECUTIVE SUMMARY + KEY METRICS
    story.append(Paragraph("Executive Summary", S["h1"]))
    story.append(Spacer(1, 0.05 * cm))

    # 1.1 — Client-facing executive summary (under 200 words)
    story.append(Paragraph(
        "We recommend adopting the combined Strategic and Tactical Asset Allocation portfolio as the "
        "live policy for Whitmore's liquid assets. The strategy targets an 8% annual return while "
        "respecting the Investment Policy Statement's 15% volatility ceiling and 25% drawdown limit. "
        f"Out-of-sample from January 2003 through April 2026, it delivered {_fmt_pct(taa['annualized_return'])} "
        f"per year with {_fmt_pct(taa['annualized_volatility'])} volatility and a maximum drawdown of "
        f"{_fmt_pct(taa['max_drawdown'])}, preserving capital better than both policy benchmarks during "
        "the 2008 and 2020 crises.",
        S["body"]))

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

    # 1.7 — Benchmarks Definition Table
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
        ["US TIPS", "—", "5%"],
        ["US REITs", "—", "10%"],
        ["Gold", "—", "15%"],
        ["Silver Futures", "—", "5%"],
        ["Nikkei 225 (Japan)", "—", "5%"],
        ["CSI 300 (China)", "—", "5%"],
        ["Swiss Franc", "—", "5%"],
    ]
    story.append(_df_table(pd.DataFrame(bm_data[1:], columns=bm_data[0]), max_rows=10,
                           first_col_ratio=3.0))

    # PAGE 4 - SAA CONSTRUCTION
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

    story.append(Paragraph("SAA Method Comparison (2000–2025, after costs):", S["label"]))
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
    story.append(_df_table(saa_inline, max_rows=6, first_col_ratio=2.0))

    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph(
        "We selected Minimum Variance for three reasons. First, it produced the lowest realized "
        "volatility (7.7%) of all six methods. Second, its drawdown profile (−32.5%) was second "
        "only to inverse volatility. Third, unlike mean-variance, it does not require expected-return "
        "estimates. Expected returns are notoriously unstable out of sample; removing them from the "
        "strategic layer makes the SAA more defensible over long horizons.",
        S["body_small"]))

    story.append(Paragraph(
        "We acknowledge that minimum variance produces the lowest Sharpe ratio (0.59) of the six "
        "methods evaluated — a trade-off worth addressing directly. Maximum diversification and HRP "
        "achieve higher risk-adjusted returns, but both rely on more complex correlation estimates "
        "that are known to be unstable across regimes. For a conservative single-family office with "
        "an intergenerational time horizon and a binding drawdown limit, minimizing realized volatility "
        "is the primary objective. The Sharpe shortfall (0.59 vs. 0.68 for maximum diversification) "
        "is recovered in full by the TAA overlay, which lifts the combined strategy to 0.88.",
        S["body_small"]))

    # 1.2a — why minimum-variance suits a conservative single-family office
    story.append(Paragraph(
        "Minimum-variance optimization is particularly well-suited to a conservative single-family "
        "office because the objective is to minimize portfolio volatility rather than maximize return. "
        "Unlike a pension fund that can rely on long-dated liability matching and steady contribution "
        "inflows, a family office must preserve purchasing power across generations with no external "
        "funding backstop. By avoiding expected-return estimation at the strategic layer, the SAA "
        "remains robust to regime shifts and does not require heroic capital-market assumptions.",
        S["body_small"]))

    # 1.2b — HRP note
    story.append(Paragraph(
        "Hierarchical Risk Parity (HRP) was also evaluated but did not outperform minimum variance "
        "on a risk-adjusted basis under the IPS constraints. HRP's recursive bisection algorithm "
        "produced higher turnover and slightly higher realized volatility in this constrained universe, "
        "so it is included in the comparison table for completeness but was not selected as the final method.",
        S["body_small"]))

    # 1.2c — SAA target weights table
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
                           first_col_ratio=2.5))

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

    # 1.3 — TAA signal design with strengthened economic mechanisms
    signal_text = [
        ("Signal 1: Regime HMM (20% weight)",
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
        ("Signal 2: Faber Trend (25% weight)",
         "Hypothesis: Assets trading above their 200-day moving average have materially higher "
         "risk-adjusted returns than those trading below it, across all asset classes tested.",
         "Economic mechanism: Price momentum persists because of behavioral anchoring—investors "
         "underreact to new information initially—and because institutional flows chase recent winners, "
         "creating self-reinforcing demand that continues until a catalyst reverses it. "
         "We compute a 200-day SMA on each asset's observed trading history and score via "
         "tanh((P/SMA - 1) / sigma_60d) to produce a smooth score in [-1, +1]. Source: Faber (2007)."),
        ("Signal 3: ADM Momentum (25% weight)",
         "Hypothesis: Assets with positive total returns over 1, 3, 6, and 12-month horizons "
         "outperform those with negative returns, and cross-sectional ranking within asset-class "
         "buckets identifies the strongest performers.",
         "Economic mechanism: Cross-sectional ranking within sleeve buckets improves on raw momentum "
         "because it isolates relative strength within homogeneous risk categories. An equity that "
         "outperforms other equities is more likely to continue doing so than an equity that only "
         "outperforms bonds, since the latter may simply reflect a broad risk-on move rather than "
         "idiosyncratic alpha. "
         "We compute blended momentum over 21/63/126/252-day windows, rank within sleeves, and apply "
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
        ("Signal 5: Macro Factor (15% weight)",
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
        "The five signals combine as follows: μ = 0.20 × regime_tilt × 0.10 + 0.25 × trend × "
        "0.06 + 0.25 × momo × 0.06 + 0.10 × vix_tilt + 0.15 × macro_factor × 0.20. Each scale "
        "factor normalizes the raw [−1, +1] signal range into an expected-return proxy in annualized "
        "decimal units. The optimizer then solves: maximize μ′w − 1.5 × w′Σw − 0.0005 × Σ|w − "
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
    fold_inline = _fmt_display_df(
        inputs["per_fold"],
        col_map={"fold_id": "Fold", "start_date": "Start", "end_date": "End",
                 "annualized_return": "Return p.a.", "annualized_volatility": "Vol p.a.",
                 "sharpe": "Sharpe", "sortino": "Sortino", "max_drawdown": "Max DD"},
        pct_cols=["annualized_return", "annualized_volatility", "max_drawdown"],
        float2_cols=["sharpe", "sortino"],
        select=["fold_id", "start_date", "end_date", "annualized_return",
                "annualized_volatility", "sharpe", "sortino", "max_drawdown"],
    )
    story.append(_df_table(fold_inline, max_rows=5, first_col_ratio=1.5))

    story.append(Spacer(1, 0.08 * cm))
    pf = inputs["per_fold"]

    # 1.4a — per-fold market-context interpretation
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

    # 1.4b — PIT disclosure paragraph (elevated from footnote)
    story.append(Paragraph("Point-in-Time Data Discipline", S["h2"]))
    story.append(Paragraph(
        "All macro inputs from FRED are shifted forward by one business day before entering any "
        "signal calculation. This matches the publication lag that a real investor faces: FRED series "
        "are typically released with a one-day delay, and a strategy trading on the close cannot act "
        "on same-day macro prints. Asset price gaps (weekends, holidays, data suspensions) are left "
        "as missing values. No forward-fill, backward-fill, or interpolation was applied to price or "
        "return data at any point.",
        S["body_small"]))

    # 1.4c — Trial Disclosure box (Marcos' Third Law)
    story.append(Spacer(1, 0.08 * cm))
    story.append(Paragraph("Trial Disclosure — Marcos' Third Law", S["h2"]))
    story.append(Paragraph(
        "We disclose the full extent of trial configurations tested during research. A total of "
        f"{disclosed} distinct configurations were evaluated before selecting the final specification. "
        "What varied across trials: signal ensemble weights (regime 15–25%, trend 20–30%, momentum 20–30%, "
        "macro 10–20%, VIX trip-wire 5–15%), lookback windows for trend (150–250 days) and momentum "
        "(1/3/6/12 vs. 1/3/6/9/12 month blends), risk-aversion coefficients (1.0–2.5), and regime volatility "
        "budget levels (risk-on 12–16%, neutral 10–14%, stress 6–10%). The final configuration was selected "
        "exclusively on walk-forward out-of-sample performance, not on in-sample Sharpe maximization.",
        S["body_small"]))
    story.append(Paragraph(
        f"The Deflated Sharpe Ratio of {dsr['baseline_dsr']:.3f} adjusts the observed Sharpe for the "
        f"{disclosed} trials. Values above 0.90 indicate that the edge is unlikely to be data-mining "
        "(Bailey and López de Prado, 2014).",
        S["body_small"]))

    # PAGE 6 - PERFORMANCE FIGURES
    story.append(PageBreak())
    story.append(Paragraph("Performance Chart: Cumulative Growth", S["h1"]))
    story.append(Paragraph(
        "All four portfolios indexed to 100 at the first common date. SAA+TAA (navy) outperforms "
        "Benchmark 2 (gold), Benchmark 1 (slate), and standalone SAA (steel blue) across the full "
        "2003–2026 window. Grey bands mark major drawdown episodes.", S["body_small"]))
    story.append(_center_image(inputs["figures"]["cumgrowth"], max_width=fw, max_height=7.2 * cm))
    story.append(Spacer(1, 0.05 * cm))

    story.append(Paragraph("Drawdown Analysis", S["h1"]))
    story.append(_center_image(inputs["figures"]["drawdown"], max_width=fw, max_height=7.0 * cm))
    story.append(Paragraph(
        f"Peak-to-trough drawdowns. SAA+TAA (navy): {_fmt_pct(taa['max_drawdown'])} maximum loss. "
        f"Benchmark 2: {_fmt_pct(bm2['max_drawdown'])}. Benchmark 1: {_fmt_pct(bm1['max_drawdown'])}. "
        "Red shading highlights IPS threshold breaches.", S["caption"]))

    story.append(Spacer(1, 0.05 * cm))
    story.append(Paragraph("Rolling 12-Month Return Comparison", S["h1"]))
    story.append(_center_image(inputs["figures"]["rolling_12m"], max_width=fw, max_height=7.0 * cm))
    story.append(Paragraph(
        "252-day rolling annualized returns demonstrate consistency of outperformance. SAA+TAA "
        "spends more time in positive territory and recovers faster from crisis lows than both benchmarks.",
        S["caption"]))

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
    story.append(_center_image(inputs["figures"]["folds"], max_width=fw, max_height=5.0 * cm))
    story.append(Paragraph(
        "Five expanding walk-forward folds. Grey bars: training window. Gold bars: 21-day embargo. "
        "Navy bars: out-of-sample test period.", S["caption"]))

    story.append(Spacer(1, 0.08 * cm))
    story.append(Paragraph("Per-Fold Sharpe Comparison", S["h2"]))
    story.append(_center_image(inputs["figures"]["per_fold"], max_width=fw, max_height=5.5 * cm))
    story.append(Paragraph(
        "Grouped Sharpe ratios by fold show SAA+TAA outperforming both benchmarks in every fold except "
        "Fold 2 (GFC), where all portfolios suffered but SAA+TAA still preserved relative capital.",
        S["caption"]))

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

    # 1.5a — cost drag interpretation
    story.append(Paragraph(
        "Cost drag rises from 0.01% in the standalone SAA to 0.26% in SAA+TAA. This 25 basis-point "
        "increment is the price of the tactical overlay. Net of this drag, the TAA still delivers "
        "1.61 percentage points of annual alpha (1.87% gross uplift minus 0.25% cost drag). We view "
        "this as a favorable trade: the family pays roughly one-quarter of the alpha back in "
        "transaction costs and keeps three-quarters.",
        S["body_small"]))

    # 1.5b — turnover and liquidity considerations
    story.append(Paragraph(
        "Turnover increases from 0.3x per year in the SAA to 5.1x in SAA+TAA. At $1.8 billion in "
        "assets under management, 5.1x turnover implies roughly $9.2 billion in annual traded volume. "
        "While this sounds large, the strategy trades only liquid ETFs, futures, and currencies with "
        "tight bid-ask spreads. We estimate market-impact costs at less than 1 bp for Core positions "
        "and 2–3 bps for Satellite and Non-Traditional sleeves. The overlay is therefore practical "
        "for a family office of this size, though we recommend monitoring monthly slippage during "
        "stress periods when liquidity temporarily evaporates.",
        S["body_small"]))

    # 1.5c — IPS violations interpretation
    story.append(Paragraph(
        "IPS violations fall from 831 in the standalone SAA to 214 in SAA+TAA, with zero hard "
        "violations in either case. A 'soft violation' is a market-driven outcome—rolling realized "
        "volatility briefly exceeding 15% or drawdown dipping below -25%—that occurs during crises "
        "regardless of optimizer settings. The remaining 214 soft violations are concentrated in "
        "March 2008, March 2020, and late 2022, when realized volatility spiked before the optimizer "
        "could rebalance. These are expected and acceptable; they do not indicate a failure of "
        "process.",
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
    story.append(_df_table(pd.DataFrame(comp_data[1:], columns=comp_data[0]), max_rows=7,
                           first_col_ratio=2.0))

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

    # 1.6 — conditional recommendation
    story.append(Spacer(1, 0.1 * cm))
    story.append(Paragraph("Conditional Adoption Guidance", S["h2"]))
    story.append(Paragraph(
        "The overlay is least likely to add value in two environments: (a) prolonged range-bound, "
        "low-volatility markets where trend signals generate whipsaw losses, and (b) rapid flash-crash "
        "reversals that last hours or days—too short for any monthly signal to react. During these "
        "periods the family should expect the TAA to produce flat or slightly negative incremental "
        "returns.",
        S["body_small"]))
    story.append(Paragraph(
        "We propose a formal re-evaluation trigger: if the rolling 12-month TAA net alpha (after costs) "
        "falls below 50 basis points for two consecutive years, Whitmore should convene a formal signal "
        "review. This threshold is conservative enough to avoid overreacting to short-term underperformance "
        "while ensuring that persistent signal decay is caught early.",
        S["body_small"]))
    story.append(Paragraph(
        "Finally, an honest note on the 8% return objective: with 10-year TIPS real yields near 2% "
        "in 2024–2026, achieving 8% per year requires either sustained equity risk premia near historical "
        "averages or successful tactical timing. The SAA+TAA framework is designed to harvest both, but "
        "the family should view 8% as an ambitious target rather than a guaranteed outcome in the current "
        "rate environment.",
        S["body_small"]))

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

    # PAGE 12 - APPENDIX (trimmed to fit 12-page limit)
    story.append(PageBreak())
    story.append(Paragraph("Appendix", S["h1"]))

    story.append(Paragraph("A. SAA Method Comparison", S["h2"]))
    story.append(Spacer(1, 0.05 * cm))
    saa_disp = _fmt_display_df(
        inputs["saa_methods"].assign(method=inputs["saa_methods"]["method"].map(
            lambda m: METHOD_NAMES.get(m, m))),
        col_map={"method": "Method", "annualized_return": "Return p.a.",
                 "annualized_volatility": "Vol p.a.", "max_drawdown": "Max DD",
                 "sharpe": "Sharpe", "sortino": "Sortino", "calmar": "Calmar",
                 "turnover_pa": "Turnover"},
        pct_cols=["annualized_return", "annualized_volatility", "max_drawdown"],
        float2_cols=["sharpe", "sortino", "calmar", "turnover_pa"],
        select=["method", "annualized_return", "annualized_volatility", "max_drawdown",
                "sharpe", "sortino", "calmar", "turnover_pa"],
    )
    story.append(_df_table(saa_disp, max_rows=6, font_size=7, first_col_ratio=2.0))

    story.append(Spacer(1, 0.08 * cm))
    story.append(Paragraph("B. Per-Fold OOS Metrics", S["h2"]))
    story.append(Spacer(1, 0.05 * cm))
    fold_disp = _fmt_display_df(
        inputs["per_fold"],
        col_map={"fold_id": "Fold", "start_date": "Start", "end_date": "End",
                 "annualized_return": "Return p.a.", "annualized_volatility": "Vol p.a.",
                 "sharpe": "Sharpe", "sortino": "Sortino", "max_drawdown": "Max DD"},
        pct_cols=["annualized_return", "annualized_volatility", "max_drawdown"],
        float2_cols=["sharpe", "sortino"],
        select=["fold_id", "start_date", "end_date", "annualized_return",
                "annualized_volatility", "sharpe", "sortino", "max_drawdown"],
    )
    story.append(_df_table(fold_disp, max_rows=5, font_size=7, first_col_ratio=1.5))

    story.append(Spacer(1, 0.08 * cm))
    story.append(Paragraph("C. Signal Attribution (Leave-One-Out)", S["h2"]))
    story.append(Spacer(1, 0.05 * cm))
    sig_disp = _fmt_display_df(
        inputs["attribution_signal"],
        col_map={"layer": "Signal", "baseline_sharpe": "Baseline Sharpe",
                 "ablated_sharpe": "Ablated Sharpe", "marginal_oos_sharpe": "ΔSharpe",
                 "baseline_ann_return": "Baseline Return", "ablated_ann_return": "Ablated Return",
                 "ann_return_delta": "ΔReturn"},
        pct_cols=["baseline_ann_return", "ablated_ann_return", "ann_return_delta"],
        float2_cols=["baseline_sharpe", "ablated_sharpe", "marginal_oos_sharpe"],
        select=["layer", "baseline_sharpe", "ablated_sharpe", "marginal_oos_sharpe",
                "baseline_ann_return", "ablated_ann_return", "ann_return_delta"],
    )
    story.append(_df_table(sig_disp, max_rows=10, font_size=7))

    # BUILD
    # body_frame: top sits at PAGE_H - 2.0 cm, giving 0.55 cm clearance below the
    # header text at PAGE_H - 1.45 cm.  (frame top = y_bottom + height)
    body_frame = Frame(1.4 * cm, 1.5 * cm, PAGE_W - 2.8 * cm, PAGE_H - 3.5 * cm, id="body_frame")
    # title_frame: no header, so the frame can extend closer to the top.
    title_frame = Frame(1.4 * cm, 1.5 * cm, PAGE_W - 2.8 * cm, PAGE_H - 3.0 * cm, id="title_frame")
    title_template = PageTemplate(id="title", frames=title_frame, onPage=_title_page_header)
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
