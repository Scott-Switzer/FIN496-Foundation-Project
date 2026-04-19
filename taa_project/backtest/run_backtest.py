"""
Walk-forward TAA backtester.

At each month-end t:
  1. Layers 1–4 build signals using data ≤ t
  2. Optimizer produces TAA weights w_t (respecting IPS constraints)
  3. Apply to next-day returns, deduct 5bps × turnover cost
  4. Log: weights, realized return, regime label, attribution vs BM2
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from taa_project.config import (
    ALL_SAA, BM2_WEIGHTS, TAA_BANDS, COST_PER_TURNOVER, VOL_CEILING, MAX_DD, TARGET_VOL,
    OUTPUT_DIR,
)
from taa_project.data_loader import load_prices, load_fred, log_returns, availability_flag
from taa_project.signals.regime_hmm import (
    build_features, fit_hmm, classify_states, REGIME_TILT,
)
from taa_project.signals.trend_faber import trend_score
from taa_project.signals.momentum_adm import adm_score, cross_sectional_rank
from taa_project.optimizer.cvxpy_opt import solve_taa, ensemble_score


SLEEVE_BUCKETS = {
    "equity":     ["SPXT", "FTSE100", "NIKKEI225", "CSI300_CHINA"],
    "fixed_inc":  ["LBUSTRUU", "BROAD_TIPS"],
    "real":       ["XAU", "SILVER_FUT", "B3REITT"],
    "nontrad":    ["BITCOIN", "CHF_FRANC"],
}


def run(
    start: str = "2005-01-01",
    end: str | None = None,
    use_timesfm: bool = False,       # flip on if you've pip-installed timesfm
    refit_hmm_freq: str = "ME",
    rebalance_freq: str = "ME",
    hmm_states: int = 3,
    vol_budget: float = TARGET_VOL,
):
    if vol_budget > VOL_CEILING:
        raise ValueError(
            f"vol_budget={vol_budget:.4f} exceeds VOL_CEILING={VOL_CEILING:.4f}. "
            "Use an internal target at or below the IPS volatility ceiling."
        )
    if vol_budget < 0.02:
        raise ValueError(
            f"vol_budget={vol_budget:.4f} is below 0.0200. "
            "This is likely a typo; the standalone backtest refuses to run with unrealistically tight budgets."
        )
    # ----- data -----
    prices = load_prices()
    fred = load_fred()
    rets = log_returns(prices)
    avail = availability_flag(prices)

    # ----- pre-compute slow signals that are vectorized & causal -----
    trend = trend_score(prices)
    momo  = cross_sectional_rank(adm_score(prices), SLEEVE_BUCKETS)

    # ----- rebalance schedule -----
    rebalance_dates = (
        prices.loc[start:end]
        .resample(rebalance_freq).last()
        .dropna(how="all").index
    )

    # ----- optional TimesFM -----
    forecaster = None
    if use_timesfm:
        from taa_project.signals.vol_timesfm import (
            TimesFMForecaster, compute_vol_and_direction_signals,
        )
        forecaster = TimesFMForecaster(max_context=1024, max_horizon=64)

    # ----- main loop -----
    weight_log, regime_log, return_log = [], [], []
    prev_w = pd.Series(BM2_WEIGHTS).reindex(ALL_SAA).fillna(0.0)
    last_hmm_refit = None
    hmm_model = None

    for t in tqdm(rebalance_dates):
        # 1) regime — refit HMM at most once per month on expanding window
        if last_hmm_refit is None or (t - last_hmm_refit).days > 25:
            feats = build_features(fred).loc[:t]
            if len(feats) > 252:
                try:
                    hmm_model = fit_hmm(feats, n_states=hmm_states)
                    last_hmm_refit = t
                except Exception as e:
                    print(f"[hmm] {t}: {e}")
        regime_label = "neutral"
        if hmm_model is not None:
            feats_now = build_features(fred).loc[:t]
            states = classify_states(hmm_model, feats_now)
            regime_label = states["regime"].iloc[-1]
        regime_tilt = pd.Series(REGIME_TILT.get(regime_label, REGIME_TILT["neutral"]))
        regime_tilt = regime_tilt.reindex(ALL_SAA).fillna(0.0)

        # 2) trend & momo signals at t
        trend_t = trend.loc[:t].iloc[-1].reindex(ALL_SAA).fillna(0.0)
        momo_t  = momo.loc[:t].iloc[-1].reindex(ALL_SAA).fillna(0.0)

        # 3) TimesFM signals (optional)
        if forecaster is not None:
            from taa_project.signals.vol_timesfm import compute_vol_and_direction_signals
            tfm = compute_vol_and_direction_signals(rets, t, forecaster, horizon=21)
            timesfm_mu = tfm["mu_ann"].reindex(ALL_SAA).fillna(0.0)
            fcst_sigma = tfm["sigma_ann_fcst"].reindex(ALL_SAA)
        else:
            timesfm_mu = pd.Series(0.0, index=ALL_SAA)
            fcst_sigma = None

        # 4) ensemble → mu proxy
        mu = ensemble_score(regime_tilt, trend_t, momo_t, timesfm_mu)

        # 5) covariance from historical returns — Ledoit-Wolf style shrinkage
        recent = rets.loc[:t].iloc[-252:].dropna(axis=1, how="all")
        cov = recent.cov() * 252
        # shrink toward diag to stabilize
        diag = np.diag(np.diag(cov))
        cov = 0.7 * cov + 0.3 * diag
        # If TimesFM gave us vol, override the diagonal
        if fcst_sigma is not None:
            for a in cov.columns:
                if pd.notna(fcst_sigma.get(a)) and fcst_sigma[a] > 0:
                    cov.loc[a, a] = fcst_sigma[a] ** 2

        cov = cov.reindex(index=ALL_SAA, columns=ALL_SAA).fillna(0)
        np.fill_diagonal(cov.values, np.where(np.diag(cov) > 0, np.diag(cov), 0.04))

        # 6) solve
        w = solve_taa(
            signal_score=mu,
            cov_matrix=cov,
            prev_weights=prev_w,
            available=avail.loc[:t].iloc[-1].reindex(ALL_SAA).fillna(0),
            as_of_date=t,
            vol_budget=vol_budget,
        )

        # 7) realize returns to next rebalance
        next_idx = rebalance_dates.get_loc(t)
        if next_idx + 1 < len(rebalance_dates):
            t_next = rebalance_dates[next_idx + 1]
            period_rets = rets.loc[t:t_next].iloc[1:]  # skip decision day itself
            port_rets = period_rets.reindex(columns=ALL_SAA).fillna(0).dot(w)
            turnover = (w - prev_w).abs().sum()
            # charge 5bps once at rebalance on notional traded
            port_rets.iloc[0] -= COST_PER_TURNOVER * turnover
            return_log.append(port_rets)

        weight_log.append(pd.Series(w, name=t))
        regime_log.append((t, regime_label))
        prev_w = w

    weights_df = pd.DataFrame(weight_log)
    regime_df = pd.DataFrame(regime_log, columns=["date", "regime"]).set_index("date")
    returns_s = pd.concat(return_log) if return_log else pd.Series(dtype=float)
    return weights_df, regime_df, returns_s


def tearsheet(rets: pd.Series, label: str = "TAA"):
    cum = (1 + rets).cumprod()
    ann_ret = (1 + rets.mean()) ** 252 - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe  = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else np.nan
    dd = cum / cum.cummax() - 1
    calmar  = ann_ret / abs(dd.min()) if dd.min() < 0 else np.nan
    print(f"""
{label} tearsheet
  Ann return : {ann_ret:6.2%}
  Ann vol    : {ann_vol:6.2%}   (IPS ceiling 15%)
  Sharpe     : {sharpe:6.2f}
  Max DD     : {dd.min():6.2%}  (IPS limit -25%)
  Calmar     : {calmar:6.2f}
  Vol breach?: {'YES' if ann_vol > VOL_CEILING else 'no'}
  DD  breach?: {'YES' if dd.min() < -MAX_DD else 'no'}
""")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end",   default=None)
    ap.add_argument("--timesfm", action="store_true", help="use TimesFM vol/dir signals")
    ap.add_argument("--vol-budget", type=float, default=TARGET_VOL, help="Internal ex-ante vol target used by the TAA optimizer.")
    args = ap.parse_args()

    w, r, rets = run(start=args.start, end=args.end, use_timesfm=args.timesfm, vol_budget=args.vol_budget)
    tearsheet(rets, "TAA overlay")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = OUTPUT_DIR / "taa_weights.csv"
    regimes_path = OUTPUT_DIR / "taa_regimes.csv"
    returns_path = OUTPUT_DIR / "taa_returns.csv"
    w.to_csv(weights_path)
    r.to_csv(regimes_path)
    rets.to_csv(returns_path)
    print(f"Saved: {weights_path}, {regimes_path}, {returns_path}")
