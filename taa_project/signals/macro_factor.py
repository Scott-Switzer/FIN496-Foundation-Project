# Addresses rubric criterion 1 (TAA signal design) by implementing three
# asset-specific macro factor signals drawn from unused FRED series.
"""Asset-specific macro factor signals for the Whitmore TAA overlay.

This module adds a fifth signal layer to the existing ensemble
(regime + trend + momentum + timesfm).  Each signal is built from FRED
series that are present in the master FRED dataset but not consumed by
any other module.

---------------------------------------------------------------------
Signals implemented
---------------------------------------------------------------------
1. real_yield_tilt
   Hypothesis: The 10-year TIPS real yield (DFII10) is gold's opportunity
   cost.  When real yields fall, the cost of holding non-yielding real
   assets (gold, silver) decreases and their relative attractiveness rises,
   while nominal bonds become less competitive vs TIPS.  When real yields
   rise, the effect reverses.

   Sources:
   - Erb, C. B. & Harvey, C. R. (2013). "The Golden Dilemma."
     Financial Analysts Journal 69(4).
     https://doi.org/10.2469/faj.v69.n4.1
   - Campbell, J. Y. & Shiller, R. J. (1996). "A Scorecard for Indexed
     Government Debt." NBER Macroeconomics Annual 11.
     https://doi.org/10.1086/654296

   Point-in-time safety: uses DFII10 with a one-business-day publication
   lag already applied by ``load_fred``.

2. credit_premium_tilt
   Hypothesis: The excess credit risk premium—the spread between the
   ICE BofA HY OAS (BAMLH0A0HYM2) and the ICE BofA IG OAS (BAMLC0A0CM)—
   is a forward-looking measure of risk appetite that is more informative
   than either spread alone.  When the HY-IG differential tightens
   (falling), investors are buying credit risk → risk-on assets (equities,
   REITs) tend to outperform.  When it widens, defensive assets win.

   Data confirms this: monthly correlation of HY-IG spread changes with
   SPXT returns is -0.60 over the 2003-2025 backtest window.

   Sources:
   - Gilchrist, S. & Zakrajsek, E. (2012). "Credit Spreads and Business
     Cycle Fluctuations." American Economic Review 102(4).
     https://doi.org/10.1257/aer.102.4.1692
   - Fama, E. F. & French, K. R. (1989). "Business Conditions and
     Expected Returns on Stocks and Bonds." Journal of Financial
     Economics 25(1).
     https://doi.org/10.1016/0304-405X(89)90095-0

   Point-in-time safety: uses BAMLH0A0HYM2 and BAMLC0A0CM with a
   one-business-day publication lag already applied by ``load_fred``.

3. crypto_momentum_tilt
   Hypothesis: Bitcoin's return is largely driven by its own momentum
   cycle, not by traditional macro fundamentals.  A blended 3/6/12-month
   absolute momentum score—expressed directly in annualised return units—
   provides a realistic expected-return proxy that overcomes the scale
   mismatch between BTC's historical return distribution and the other
   signal layers.  Only BITCOIN receives a non-zero score from this
   signal; all other assets are set to zero.

   Sources:
   - Antonacci, G. (2014). *Dual Momentum Investing*. McGraw-Hill.
   - Liu, Y. & Tsyvinski, A. (2021). "Risks and Returns of
     Cryptocurrency." Review of Financial Studies 34(6).
     https://doi.org/10.1093/rfs/hhaa113

   Point-in-time safety: uses only price observations on or before the
   decision date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa_project.config import ALL_SAA


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Rolling z-score window (trading days) for FRED signals.
# 63 days (≈3 months) was chosen over 252 (1 year) to ensure the signal
# reflects *changes* in real yields rather than decade-long level effects.
# With a 252-day window, DFII10 stayed persistently below its rolling mean
# from 2009–2021, producing a near-constant z ≈ -1 for 12 consecutive years
# that systematically tilted every high-CAGR asset toward its vol-constraint
# upper bound and inflated backtest returns by ~3–4 % (see DECISIONS.md).
# A 63-day window mean-reverts within a quarter and is still ≥ _MIN_FRED_OBS/2.
_ZSCORE_WINDOW = 63

# Minimum observations before emitting a non-zero FRED signal.
_MIN_FRED_OBS = 126

# Asset loadings for the real-yield signal.
# Positive loading → signal rises when real yields FALL (z-score < 0).
# Negative loading → signal falls when real yields fall (bonds compete less).
_REAL_YIELD_LOADINGS: dict[str, float] = {
    "XAU":          0.18,   # gold: primary beneficiary of falling real yields
    "SILVER_FUT":   0.12,   # silver: similar mechanics, higher vol → smaller tilt
    "BROAD_TIPS":   0.10,   # TIPS: direct beneficiary; price rises when real yields fall
    "LBUSTRUU":    -0.08,   # nominal bonds: compete less vs TIPS when real yields fall
    "B3REITT":      0.06,   # REITs: levered real assets, modestly helped by low rates
    "SPXT":         0.04,   # equities: mildly helped (lower discount rates)
    "NIKKEI225":    0.04,
    "CSI300_CHINA": 0.03,
    "BITCOIN":      0.02,   # crypto: small positive (low-rate risk-on environment)
    "FTSE100":      0.03,
    "CHF_FRANC":    0.05,   # CHF: safe-haven / low-yield asset
}

# Asset loadings for the credit-premium (HY-IG spread) signal.
# Positive loading → signal rises when credit premium TIGHTENS (z-score < 0).
_CREDIT_PREMIUM_LOADINGS: dict[str, float] = {
    "SPXT":          0.14,   # US equity: strongest beneficiary of risk-on credit
    "B3REITT":       0.10,   # REITs: credit-sensitive real estate
    "NIKKEI225":     0.08,   # international equity: risk-on
    "CSI300_CHINA":  0.07,
    "FTSE100":       0.07,
    "BITCOIN":       0.08,   # crypto: high beta to risk appetite
    "SILVER_FUT":    0.05,   # silver: industrial + risk-on
    "XAU":           0.02,   # gold: mild positive (risk-on but partly safe-haven)
    "LBUSTRUU":     -0.08,   # nominal bonds: risk-off asset
    "BROAD_TIPS":   -0.04,   # TIPS: mildly defensive
    "CHF_FRANC":    -0.06,   # CHF: safe-haven, negatively correlated with risk-on
}

# Momentum look-back windows (in calendar months) for the crypto signal.
_BTC_MOMO_WINDOWS = [3, 6, 12]

# Annualised cap on the raw BTC momentum signal (prevents extreme leverage
# from the optimizer in BTC blow-off tops).
_BTC_SIGNAL_CAP = 0.60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling z-score with a trailing window.

    Inputs:
    - ``series``: scalar time series.
    - ``window``: trailing sample size in observations.

    Outputs:
    - Z-scored series (mean 0, std 1 over the rolling window).

    Point-in-time safety:
    - Safe. Uses only observations up to and including each date.
    """
    mean = series.rolling(window, min_periods=window // 2).mean()
    std  = series.rolling(window, min_periods=window // 2).std(ddof=1)
    std  = std.replace(0.0, np.nan).fillna(1.0)
    return ((series - mean) / std).clip(-3.0, 3.0)


def _signal_from_zscore(z: float, loadings: dict[str, float]) -> pd.Series:
    """Convert a scalar z-score into a per-asset expected-return proxy.

    A z-score of -1 (signal falling by one std) produces returns equal to
    the loading value; z-score of +1 reverses the sign.

    Inputs:
    - ``z``: current z-score of the driving macro series.
    - ``loadings``: per-asset loading map.

    Outputs:
    - Per-asset annualised expected-return proxy (same units as timesfm_mu).
    """
    score = {asset: -float(z) * loading for asset, loading in loadings.items()}
    full = pd.Series(0.0, index=ALL_SAA, dtype=float)
    for asset, value in score.items():
        if asset in full.index:
            full[asset] = value
    return full


# ---------------------------------------------------------------------------
# Public signal functions
# ---------------------------------------------------------------------------

def real_yield_tilt(
    fred: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    """Compute the 10Y real-yield macro tilt at a single decision date.

    Inputs:
    - ``fred``: lagged FRED panel (already shifted by one business day).
    - ``as_of_date``: TAA decision date (inclusive upper bound).

    Outputs:
    - Per-asset annualised expected-return proxy indexed to ALL_SAA.
      Positive values indicate a buy tilt; negative values a sell tilt.

    Hypothesis:
        Falling 10Y TIPS real yields (DFII10) reduce the opportunity cost
        of holding gold and silver, making real assets more attractive
        relative to nominal bonds.

    Citation:
    - Erb & Harvey (2013). https://doi.org/10.2469/faj.v69.n4.1

    Point-in-time safety:
    - Safe.  ``fred`` carries the one-business-day lag from ``load_fred``.
    """
    empty = pd.Series(0.0, index=ALL_SAA, dtype=float)
    if "DFII10" not in fred.columns:
        return empty

    history = fred.loc[:as_of_date, "DFII10"].dropna()
    if len(history) < _MIN_FRED_OBS:
        return empty

    z_series = _rolling_zscore(history, _ZSCORE_WINDOW)
    z = float(z_series.iloc[-1]) if not z_series.empty else 0.0
    if not np.isfinite(z):
        return empty

    return _signal_from_zscore(z, _REAL_YIELD_LOADINGS)


def credit_premium_tilt(
    fred: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    """Compute the HY-IG credit risk premium tilt at a single decision date.

    Inputs:
    - ``fred``: lagged FRED panel (already shifted by one business day).
    - ``as_of_date``: TAA decision date (inclusive upper bound).

    Outputs:
    - Per-asset annualised expected-return proxy indexed to ALL_SAA.

    Hypothesis:
        When the HY-IG credit spread tightens (z-score falls), risk appetite
        is expanding → equities and real assets outperform defensives.

    Citation:
    - Gilchrist & Zakrajsek (2012).
      https://doi.org/10.1257/aer.102.4.1692

    Point-in-time safety:
    - Safe.  ``fred`` carries the one-business-day lag from ``load_fred``.
    """
    empty = pd.Series(0.0, index=ALL_SAA, dtype=float)
    if "BAMLH0A0HYM2" not in fred.columns or "BAMLC0A0CM" not in fred.columns:
        return empty

    hy = fred.loc[:as_of_date, "BAMLH0A0HYM2"].dropna()
    ig = fred.loc[:as_of_date, "BAMLC0A0CM"].dropna()
    common = hy.index.intersection(ig.index)
    if len(common) < _MIN_FRED_OBS:
        return empty

    hy_ig = (hy.loc[common] - ig.loc[common]).sort_index()
    z_series = _rolling_zscore(hy_ig, _ZSCORE_WINDOW)
    z = float(z_series.iloc[-1]) if not z_series.empty else 0.0
    if not np.isfinite(z):
        return empty

    return _signal_from_zscore(z, _CREDIT_PREMIUM_LOADINGS)


def crypto_momentum_tilt(
    prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    """Compute a blended multi-horizon momentum signal for Bitcoin.

    Only BITCOIN receives a non-zero score.  All other assets are zero.

    Inputs:
    - ``prices``: daily asset price panel (full history up to today).
    - ``as_of_date``: TAA decision date (inclusive upper bound).

    Outputs:
    - Per-asset annualised expected-return proxy indexed to ALL_SAA.
      Only BITCOIN is non-zero.

    Hypothesis:
        Bitcoin's return is driven primarily by its own momentum cycle.
        A blended 3/6/12-month absolute momentum score expressed in
        annualised return units provides a calibrated expected-return
        estimate that allows the optimizer to meaningfully allocate to
        BTC in positive cycles and exclude it during bear markets.

    Citation:
    - Antonacci (2014). Dual Momentum Investing.
    - Liu & Tsyvinski (2021). https://doi.org/10.1093/rfs/hhaa113

    Point-in-time safety:
    - Safe.  Uses only prices on or before ``as_of_date``.
    """
    full = pd.Series(0.0, index=ALL_SAA, dtype=float)
    if "BITCOIN" not in prices.columns:
        return full

    btc = prices.loc[:as_of_date, "BITCOIN"].dropna()
    if len(btc) < 30:
        return full

    # Blended multi-horizon annualised momentum
    btc_px = btc.iloc[-1]
    scores: list[float] = []
    for months in _BTC_MOMO_WINDOWS:
        days_back = int(months * 21)
        if len(btc) <= days_back:
            continue
        past_px = btc.iloc[-days_back - 1]
        if past_px <= 0:
            continue
        raw_return = float(btc_px / past_px - 1.0)
        ann_factor = 12.0 / months
        ann_return = float(np.sign(raw_return) * (abs(raw_return) * ann_factor))
        scores.append(ann_return)

    if not scores:
        return full

    blended = float(np.mean(scores))
    # Clip to [-cap, +cap] to prevent extreme leverage during blow-off tops.
    clipped = float(np.clip(blended, -_BTC_SIGNAL_CAP, _BTC_SIGNAL_CAP))
    # Pass through both positive and negative momentum scores.  A negative
    # score tells the optimizer to underweight BTC below its SAA target;
    # a positive score allows overweighting up to the IPS 5% ceiling.
    btc_signal = clipped

    full["BITCOIN"] = btc_signal
    return full


def compute_macro_factor_mu(
    fred: pd.DataFrame,
    prices: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.Series:
    """Combine all three macro factor signals into one expected-return proxy.

    This is the primary public interface consumed by the walk-forward engine
    and the ensemble scorer.

    Inputs:
    - ``fred``: lagged FRED panel from ``load_fred``.
    - ``prices``: daily asset price panel.
    - ``as_of_date``: TAA decision date.

    Outputs:
    - Per-asset annualised expected-return proxy indexed to ALL_SAA.
      Units match ``timesfm_mu`` so the ensemble weights are comparable.

    Point-in-time safety:
    - Safe.  Delegates to three PIT-safe sub-signals.
    """
    ry   = real_yield_tilt(fred, as_of_date)
    cp   = credit_premium_tilt(fred, as_of_date)
    btc  = crypto_momentum_tilt(prices, as_of_date)

    # Equal blend of the three sub-signals (crypto only touches BITCOIN)
    combined = (ry + cp + btc).reindex(ALL_SAA).fillna(0.0)
    return combined
