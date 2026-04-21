"""Compare all six SAA construction methods for the Whitmore portfolio.

Methods tested:
1. Equal Weight (1/N)
2. Inverse Volatility
3. Minimum Variance
4. Risk Parity (IPS-target budgets)
5. Maximum Diversification
6. Mean-Variance (Markowitz, trailing-mean mu)

All methods share:
- The same annual rebalance schedule (last trading day of each year)
- All IPS hard constraints (per-sleeve bands, aggregate caps)
- The same expanding universe (only assets available at each rebalance date)
- 5 bps round-trip turnover cost

Outputs (taa_project/outputs/):
- saa_<method>_returns.csv  and  saa_<method>_weights.csv  per method
- figures/saa_cumulative_wealth.png
- figures/saa_drawdown.png
- figures/saa_risk_return.png
- figures/saa_summary_table.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize
from sklearn.covariance import LedoitWolf

from taa_project.config import (
    ALL_SAA,
    EQUITY_ASSETS,
    COST_PER_TURNOVER,
    OUTPUT_DIR,
    PRICES_CSV,
)
from taa_project.data_audit import load_asset_prices
from taa_project.data_loader import log_returns
from taa_project.config import BM2_WEIGHTS
from taa_project.saa.build_saa import (
    SAAOptimizationInputs,
    available_assets_on,
    bounds_for_assets,
    build_linear_constraints,
    build_rebalance_schedule,
    estimate_covariance,
    first_valid_dates,
    project_weights_to_feasible_set,
    simulate_portfolio,
    solve_target_risk_parity,
    target_risk_budgets,
)

DEFAULT_START = "2000-01-01"
DEFAULT_END = "2025-12-31"
LOOKBACK_DAYS = 756          # ~3 years for covariance estimation
MV_RISK_AVERSION = 3.0       # lambda in BL equilibrium implied returns
DIAGONAL_FLOOR = 1e-6
RF_ANNUAL = 0.02             # risk-free rate proxy for BL equilibrium
IPS_TARGET_VOL = 0.08        # 8% ex-ante target accounts for estimation bias; realized lands ~9-10%
MOMENTUM_WINDOW = 252        # 12-month lookback for momentum signal
MOMENTUM_SKIP = 21           # skip last month (reversal avoidance)
MOMENTUM_BLEND = 0.15        # weight on momentum vs BL equilibrium — keep close to IPS equilibrium

METHODS = [
    "equal_weight",
    "inverse_vol",
    "min_variance",
    "risk_parity",
    "max_diversification",
    "hrp",
    "mean_variance",
]

METHOD_LABELS = {
    "equal_weight":       "Equal Weight (1/N)",
    "inverse_vol":        "Inverse Volatility",
    "min_variance":       "Minimum Variance",
    "risk_parity":        "Risk Parity",
    "max_diversification":"Max Diversification",
    "hrp":                "Hierarchical Risk Parity",
    "mean_variance":      "Mean-Variance",
    "bm1":                "BM1 (60/40)",
    "bm2":                "BM2 (Policy)",
}

METHOD_COLORS = {
    "equal_weight":        "#E63946",
    "inverse_vol":         "#F4A261",
    "min_variance":        "#2A9D8F",
    "risk_parity":         "#457B9D",
    "max_diversification": "#6A4C93",
    "hrp":                 "#264653",
    "mean_variance":       "#F72585",
    "bm1":                 "#888888",
    "bm2":                 "#333333",
}

# Approximate NBER US recession bands visible in the sample
RECESSIONS = [
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
}

LOGGER = logging.getLogger(__name__)


# LedoitWolf imported but used only for MV — other methods use the project's
# validated estimate_covariance() which blends sample + diagonal for stability.


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _feasible_projection(raw_weights: np.ndarray, assets: list[str]) -> pd.Series:
    lower, upper = bounds_for_assets(assets)
    projected = project_weights_to_feasible_set(raw_weights, lower, upper, assets)
    full = pd.Series(0.0, index=ALL_SAA)
    full.loc[assets] = projected
    return full


def _slsqp(objective, x0: np.ndarray, assets: list[str]) -> np.ndarray:
    lower, upper = bounds_for_assets(assets)
    sum_to_one, inequalities = build_linear_constraints(assets)
    result = minimize(
        fun=objective,
        x0=x0,
        method="SLSQP",
        bounds=Bounds(lower.values, upper.values),
        constraints=[sum_to_one, *inequalities],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    solution = result.x if result.success else x0
    return project_weights_to_feasible_set(solution, lower, upper, assets)


def solve_equal_weight(assets: list[str]) -> pd.Series:
    raw = np.full(len(assets), 1.0 / len(assets))
    return _feasible_projection(raw, assets)


def solve_inverse_vol(assets: list[str], covariance: pd.DataFrame) -> pd.Series:
    vols = np.sqrt(np.diag(covariance.loc[assets, assets].values))
    vols = np.maximum(vols, DIAGONAL_FLOOR)
    raw = (1.0 / vols) / (1.0 / vols).sum()
    return _feasible_projection(raw, assets)


def solve_min_variance(assets: list[str], covariance: pd.DataFrame) -> pd.Series:
    cov = covariance.loc[assets, assets].to_numpy(dtype=float)
    lower, upper = bounds_for_assets(assets)
    x0 = project_weights_to_feasible_set(np.full(len(assets), 1.0 / len(assets)), lower, upper, assets)
    solution = _slsqp(lambda w: float(w @ cov @ w), x0, assets)
    full = pd.Series(0.0, index=ALL_SAA)
    full.loc[assets] = solution
    return full


def solve_risk_parity(assets: list[str], covariance: pd.DataFrame) -> pd.Series:
    """IPS-target risk parity: each asset contributes risk proportional to its IPS target weight.

    Using IPS targets as risk budgets (rather than 1/N) is the correct choice here
    because: (a) the IPS targets encode the CIO's policy views on desired risk exposure,
    and (b) true 1/N ERC cannot be achieved under the binding IPS minimum constraints
    on high-volatility assets (SPXT ≥ 30%, XAU ≥ 10%) — the optimizer would violate
    the spirit of the mandate trying to equalize contributions across an infeasible set.
    """
    lower, upper = bounds_for_assets(assets)
    budgets = target_risk_budgets(assets)
    w = solve_target_risk_parity(
        SAAOptimizationInputs(
            covariance=covariance,
            lower_bounds=lower,
            upper_bounds=upper,
            risk_budgets=budgets,
            assets=assets,
        )
    )
    full = pd.Series(0.0, index=ALL_SAA)
    full.loc[assets] = w
    return full


def solve_max_diversification(assets: list[str], covariance: pd.DataFrame) -> pd.Series:
    cov = covariance.loc[assets, assets].to_numpy(dtype=float)
    vols = np.sqrt(np.maximum(np.diag(cov), DIAGONAL_FLOOR))
    lower, upper = bounds_for_assets(assets)
    x0 = project_weights_to_feasible_set(np.full(len(assets), 1.0 / len(assets)), lower, upper, assets)

    def neg_dr(w: np.ndarray) -> float:
        port_vol = float(np.sqrt(max(w @ cov @ w, DIAGONAL_FLOOR)))
        return -float(w @ vols) / port_vol

    solution = _slsqp(neg_dr, x0, assets)
    full = pd.Series(0.0, index=ALL_SAA)
    full.loc[assets] = solution
    return full


def solve_hierarchical_risk_parity(
    assets: list[str],
    covariance: pd.DataFrame,
) -> pd.Series:
    """López de Prado (2016) Hierarchical Risk Parity.

    Algorithm:
    1. Compute the correlation distance matrix `d_ij = sqrt(0.5 * (1 - rho_ij))`.
    2. Run single-linkage hierarchical clustering.
    3. Quasi-diagonalize the covariance matrix by the cluster leaf order.
    4. Apply recursive bisection using inverse-variance cluster allocations.

    Citation:
    - López de Prado, M. (2016), "Building Diversified Portfolios that
      Outperform Out-of-Sample":
      https://jpm.pm-research.com/content/42/4/59
    - Quantopian public HRP reference implementation:
      https://github.com/quantopian/research_public/blob/master/research/hierarchical_risk_parity.py

    Point-in-time safety:
    - Safe. The function uses only the supplied covariance matrix, which is
      built upstream from data observed on or before the rebalance date.
    """

    if len(assets) < 4:
        LOGGER.warning("HRP fallback to inverse volatility because only %d assets are available.", len(assets))
        return solve_inverse_vol(assets, covariance)

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    cov = covariance.loc[assets, assets].to_numpy(dtype=float)
    corr = _cov_to_corr(cov)
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    sort_idx = _quasi_diagonal(link)
    hrp_weights = _recursive_bisection(cov, sort_idx)
    out = pd.Series(0.0, index=assets, dtype=float)
    for position, cov_index in enumerate(sort_idx):
        out.iloc[cov_index] = hrp_weights[position]
    return _feasible_projection(out.to_numpy(dtype=float), assets)


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.maximum(np.diag(cov), DIAGONAL_FLOOR))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _quasi_diagonal(link: np.ndarray) -> list[int]:
    sort_idx = pd.Series([int(link[-1, 0]), int(link[-1, 1])])
    num_items = int(link[-1, 3])
    while int(sort_idx.max()) >= num_items:
        sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
        cluster_items = sort_idx[sort_idx >= num_items]
        indices = cluster_items.index.to_list()
        values = (cluster_items.to_numpy(dtype=int) - num_items).tolist()
        sort_idx.loc[indices] = [int(link[value, 0]) for value in values]
        right_items = pd.Series([int(link[value, 1]) for value in values], index=[index + 1 for index in indices])
        sort_idx = pd.concat([sort_idx, right_items]).sort_index()
        sort_idx.index = range(sort_idx.shape[0])
    return sort_idx.astype(int).tolist()


def _cluster_variance(cov: np.ndarray, cluster_items: list[int]) -> float:
    cluster_cov = cov[np.ix_(cluster_items, cluster_items)]
    inverse_variance = 1.0 / np.maximum(np.diag(cluster_cov), DIAGONAL_FLOOR)
    inverse_variance /= inverse_variance.sum()
    return float(inverse_variance @ cluster_cov @ inverse_variance)


def _recursive_bisection(cov: np.ndarray, sort_idx: list[int]) -> np.ndarray:
    weights = pd.Series(1.0, index=sort_idx, dtype=float)
    clusters = [sort_idx]
    while clusters:
        clusters = [
            cluster[start:stop]
            for cluster in clusters
            for start, stop in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
            if len(cluster) > 1
        ]
        for index in range(0, len(clusters), 2):
            left_cluster = clusters[index]
            right_cluster = clusters[index + 1]
            left_var = _cluster_variance(cov, left_cluster)
            right_var = _cluster_variance(cov, right_cluster)
            alpha = 1.0 - left_var / (left_var + right_var)
            weights[left_cluster] *= alpha
            weights[right_cluster] *= 1.0 - alpha
    return weights.loc[sort_idx].to_numpy(dtype=float)


def momentum_returns(
    returns: pd.DataFrame,
    as_of_date: pd.Timestamp,
    assets: list[str],
    window: int = MOMENTUM_WINDOW,
    skip: int = MOMENTUM_SKIP,
) -> pd.Series:
    """12-1 month cross-sectional momentum signal, annualized.

    Uses the standard Jegadeesh-Titman (1993) specification: cumulative return
    over the past 12 months excluding the most recent month (to avoid short-term
    reversal contaminating the signal).  Winsorized at the 10/90th percentile to
    prevent any single asset's extreme history from dominating the estimate.
    """
    history = returns.loc[:as_of_date, assets]
    end_idx = len(history) - skip
    start_idx = max(0, end_idx - window)
    if end_idx <= start_idx:
        return pd.Series(0.0, index=assets)
    cum = history.iloc[start_idx:end_idx].add(1).prod(skipna=True) - 1
    cum = cum.fillna(0.0)
    # winsorize at 10/90 percentile
    lo, hi = cum.quantile(0.10), cum.quantile(0.90)
    cum = cum.clip(lo, hi)
    return cum  # already in "annual-ish" units over a 12-month window


def bl_equilibrium_returns(
    covariance: pd.DataFrame,
    assets: list[str],
    risk_aversion: float = MV_RISK_AVERSION,
    rf: float = RF_ANNUAL,
) -> pd.Series:
    """Black-Litterman implied returns using BM2 as the equilibrium reference.

    π = rf + λ * Σ * w_BM2

    BM2 is the natural equilibrium because it is the IPS policy portfolio —
    it has 0% Bitcoin, preventing the sample-mean from extrapolating Bitcoin's
    bull-market history as a long-run expected return.
    """
    cov = covariance.loc[assets, assets].to_numpy(dtype=float)
    w_bm2 = pd.Series(BM2_WEIGHTS, dtype=float).reindex(assets).fillna(0.0)
    if w_bm2.sum() > 0:
        w_bm2 /= w_bm2.sum()
    else:
        w_bm2 = pd.Series(1.0 / len(assets), index=assets)
    lower, upper = bounds_for_assets(assets)
    w_ref = project_weights_to_feasible_set(w_bm2.values, lower, upper, assets)
    pi = rf + risk_aversion * (cov @ w_ref)
    return pd.Series(pi, index=assets)


def bl_with_stress_views(
    policy_weights: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float = 2.5,
    regime_label: str | None = None,
    stress_equity_shock_sigmas: float = 1.0,
    equity_assets: list[str] | None = None,
) -> pd.Series:
    """Black-Litterman prior with regime-conditional pessimistic equity views.

    When `regime_label == "stress"`, each equity asset's equilibrium prior is
    shifted down by `stress_equity_shock_sigmas` annualized standard deviations.
    Otherwise the unmodified equilibrium prior is returned.

    Citation:
    - Black & Litterman (1992), "Global Portfolio Optimization":
      https://www.jstor.org/stable/4479577
    - He & Litterman (1999), "The Intuition Behind Black-Litterman Model Portfolios":
      https://faculty.fuqua.duke.edu/~charvey/Teaching/BA453_2006/He_Litterman_Black-Litterman.pdf

    Point-in-time safety:
    - Safe. The function uses only the supplied policy weights, covariance, and
      regime label, all of which are causal at the decision date.
    """

    assets = covariance.index.tolist()
    cov = covariance.loc[assets, assets].to_numpy(dtype=float)
    cov = (cov + cov.T) / 2.0
    cov = cov + 1e-8 * np.eye(len(assets))
    aligned_policy_weights = policy_weights.reindex(assets).fillna(0.0).astype(float)
    if float(aligned_policy_weights.sum()) > 0.0:
        aligned_policy_weights /= float(aligned_policy_weights.sum())
    else:
        aligned_policy_weights[:] = 1.0 / len(assets)

    equilibrium = pd.Series(
        RF_ANNUAL + risk_aversion * (cov @ aligned_policy_weights.to_numpy(dtype=float)),
        index=assets,
        dtype=float,
    )
    if regime_label != "stress":
        return equilibrium

    out = equilibrium.copy()
    equity_universe = EQUITY_ASSETS if equity_assets is None else equity_assets
    for asset in equity_universe:
        if asset in out.index:
            out.loc[asset] -= stress_equity_shock_sigmas * float(np.sqrt(covariance.loc[asset, asset] + 1e-8))
    return out


def blended_expected_returns(
    returns: pd.DataFrame,
    as_of_date: pd.Timestamp,
    covariance: pd.DataFrame,
    assets: list[str],
) -> pd.Series:
    """BL equilibrium returns blended with 12-1 month momentum.

    μ = (1 - α) * π_BL  +  α * mom_signal

    The momentum signal tilts allocations toward recent winners while the BL
    prior keeps the portfolio anchored to the IPS policy allocation and prevents
    unrealistic expected returns for assets with short or extreme histories.
    """
    pi_bl = bl_equilibrium_returns(covariance, assets)
    mom = momentum_returns(returns, as_of_date, assets)
    # rescale momentum to the same magnitude as BL returns
    bl_scale = float(pi_bl.abs().mean())
    mom_scale = float(mom.abs().mean()) if float(mom.abs().mean()) > 0 else 1.0
    mom_rescaled = mom * (bl_scale / mom_scale)
    return (1 - MOMENTUM_BLEND) * pi_bl + MOMENTUM_BLEND * mom_rescaled


def solve_mean_variance(
    assets: list[str],
    covariance: pd.DataFrame,
    expected_returns: pd.Series,
    target_vol: float = IPS_TARGET_VOL,
    rf: float = RF_ANNUAL,
) -> pd.Series:
    """Tangency portfolio: maximize Sharpe ratio subject to IPS constraints.

    Formulation: maximize (w'μ - rf) / sqrt(w'Σw)
    Implemented via: minimize -w'(μ - rf) / sqrt(w'Σw)

    If the tangency portfolio exceeds the IPS vol target, we scale back to
    target_vol using a vol-constrained return-maximization fallback. This
    directly implements the IPS mandate: "highest return at the target risk level."
    """
    cov = covariance.loc[assets, assets].to_numpy(dtype=float)
    mu = expected_returns.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    excess = mu - rf
    lower, upper = bounds_for_assets(assets)
    x0 = project_weights_to_feasible_set(np.full(len(assets), 1.0 / len(assets)), lower, upper, assets)

    # Step 1: maximize Sharpe ratio (tangency portfolio)
    def neg_sharpe(w: np.ndarray) -> float:
        port_vol = float(np.sqrt(max(w @ cov @ w, DIAGONAL_FLOOR)))
        return -float(w @ excess) / port_vol

    tangency = _slsqp(neg_sharpe, x0, assets)
    tangency_vol = float(np.sqrt(tangency @ cov @ tangency))

    if tangency_vol <= target_vol + 1e-4:
        solution = tangency
    else:
        # Vol-constrained return maximization: find the efficient frontier point
        # at target_vol with the highest expected return.
        # Start from the min-variance portfolio — always feasible under a tighter
        # vol constraint because it is the lowest-variance point on the frontier.
        min_var_x0 = _slsqp(lambda w: float(w @ cov @ w), x0, assets)
        sum_to_one, inequalities = build_linear_constraints(assets)
        vol_constraint = {
            "type": "ineq",
            "fun": lambda w: target_vol ** 2 - float(w @ cov @ w),
        }
        result = minimize(
            fun=lambda w: -float(w @ mu),
            x0=min_var_x0,
            method="SLSQP",
            bounds=Bounds(lower.values, upper.values),
            constraints=[sum_to_one, *inequalities, vol_constraint],
            options={"maxiter": 2000, "ftol": 1e-12},
        )
        solution = result.x if result.success else tangency
        solution = project_weights_to_feasible_set(solution, lower, upper, assets)

    full = pd.Series(0.0, index=ALL_SAA)
    full.loc[assets] = solution
    return full


# ---------------------------------------------------------------------------
# Per-rebalance dispatch
# ---------------------------------------------------------------------------

def compute_weights_for_method(
    method: str,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    inception_dates: pd.Series,
) -> pd.Series:
    eligible = available_assets_on(rebalance_date, inception_dates)
    assets = [a for a in eligible if pd.notna(prices.loc[rebalance_date, a])]

    if method == "equal_weight":
        return solve_equal_weight(assets)

    covariance = estimate_covariance(returns, rebalance_date, assets)

    if method == "inverse_vol":
        return solve_inverse_vol(assets, covariance)
    if method == "min_variance":
        return solve_min_variance(assets, covariance)
    if method == "risk_parity":
        return solve_risk_parity(assets, covariance)
    if method == "max_diversification":
        return solve_max_diversification(assets, covariance)
    if method == "hrp":
        return solve_hierarchical_risk_parity(assets, covariance)
    if method == "mean_variance":
        mu = blended_expected_returns(returns, rebalance_date, covariance, assets)
        return solve_mean_variance(assets, covariance, mu)

    raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Build one portfolio
# ---------------------------------------------------------------------------

def build_saa_method(
    method: str,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    schedule: list[pd.Timestamp],
    inception_dates: pd.Series,
    end_date: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rebalance_targets = {
        d: compute_weights_for_method(method, prices, returns, d, inception_dates)
        for d in schedule
    }
    weights_df, returns_df = simulate_portfolio(
        returns=returns,
        rebalance_targets=rebalance_targets,
        start_date=schedule[0],
        end_date=pd.Timestamp(end_date),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_df.to_csv(output_dir / f"saa_{method}_weights.csv")
    returns_df.to_csv(output_dir / f"saa_{method}_returns.csv")
    return weights_df, returns_df


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

def cumulative_wealth(returns_series: pd.Series) -> pd.Series:
    return (1 + returns_series).cumprod()


def drawdown_series(returns_series: pd.Series) -> pd.Series:
    wealth = cumulative_wealth(returns_series)
    peak = wealth.cummax()
    return (wealth - peak) / peak


def performance_stats(returns_series: pd.Series, label: str) -> dict:
    ann_ret = float((1 + returns_series).prod() ** (252 / len(returns_series)) - 1)
    ann_vol = float(returns_series.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    dd = drawdown_series(returns_series)
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "Method": label,
        "Ann. Return": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "Calmar": calmar,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _shade_recessions(ax: plt.Axes) -> None:
    for start, end in RECESSIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color="#CCCCCC", alpha=0.35, zorder=0, linewidth=0)


def plot_cumulative_wealth(all_returns: dict[str, pd.Series], figures_dir: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(15, 7))
        _shade_recessions(ax)

        order = METHODS + [k for k in all_returns if k not in METHODS]
        for key in order:
            if key not in all_returns:
                continue
            ret = all_returns[key]
            wealth = cumulative_wealth(ret)
            is_bm = key.startswith("bm")
            ax.plot(wealth.index, wealth.values,
                    label=METHOD_LABELS[key],
                    color=METHOD_COLORS[key],
                    linestyle="--" if is_bm else "-",
                    linewidth=1.8 if is_bm else 2.5,
                    alpha=0.75 if is_bm else 1.0,
                    zorder=2 if is_bm else 3)
            # end-point label
            final_val = float(wealth.iloc[-1])
            final_date = wealth.index[-1]
            ax.annotate(f"{final_val:.1f}×",
                        xy=(final_date, final_val),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8,
                        color=METHOD_COLORS[key], fontweight="bold")

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}×"))
        ax.set_xlim(left=list(all_returns.values())[0].index[0])
        ax.set_title("Cumulative Wealth — All SAA Methods vs Benchmarks",
                     fontsize=15, fontweight="bold", pad=14)
        ax.set_subtitle = None
        ax.set_xlabel("", fontsize=0)
        ax.set_ylabel("Growth of $1 (log scale)", fontsize=11)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9,
                  ncol=2, columnspacing=1.0)

        # recession label
        ax.text(0.01, 0.01, "Shaded = US recessions (NBER)",
                transform=ax.transAxes, fontsize=7.5, color="#777777")

        fig.tight_layout()
        fig.savefig(figures_dir / "saa_cumulative_wealth.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_drawdowns(all_returns: dict[str, pd.Series], figures_dir: Path) -> None:
    keys = list(all_returns.keys())
    n = len(keys)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2),
                                 sharex=True, sharey=True)
        axes_flat = axes.flatten()

        for idx, key in enumerate(keys):
            ax = axes_flat[idx]
            ret = all_returns[key]
            dd = drawdown_series(ret) * 100
            color = METHOD_COLORS[key]

            _shade_recessions(ax)
            ax.fill_between(dd.index, dd.values, 0,
                            color=color, alpha=0.25, zorder=2)
            ax.plot(dd.index, dd.values,
                    color=color, linewidth=1.8, zorder=3)

            max_dd = float(dd.min())
            ax.axhline(max_dd, color=color, linewidth=0.8,
                       linestyle=":", alpha=0.7)
            ax.text(dd.index[len(dd) // 2], max_dd - 1.5,
                    f"Max: {max_dd:.1f}%",
                    ha="center", va="top", fontsize=8,
                    color=color, fontweight="bold")

            ax.set_title(METHOD_LABELS[key], fontsize=10, fontweight="bold",
                         color=color)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
            ax.set_ylim(top=2)

        for idx in range(n, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle("Drawdown by SAA Method", fontsize=14,
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        fig.savefig(figures_dir / "saa_drawdown.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_risk_return(stats: list[dict], figures_dir: Path) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 8))

        # Sharpe iso-curves
        vol_range = np.linspace(5, 26, 200)
        for sharpe_val in [0.4, 0.6, 0.8, 1.0]:
            ret_line = sharpe_val * vol_range
            ax.plot(vol_range, ret_line, color="#BBBBBB",
                    linewidth=0.9, linestyle="--", zorder=1)
            ax.text(vol_range[-1] + 0.2, ret_line[-1],
                    f"SR={sharpe_val:.1f}", fontsize=7.5,
                    color="#999999", va="center")

        # IPS 8% return target
        ax.axhline(8.0, color="#AAAAAA", linewidth=1.0,
                   linestyle=":", zorder=1)
        ax.text(5.2, 8.15, "IPS 8% target", fontsize=8, color="#999999")

        for s in stats:
            key = next(k for k, v in METHOD_LABELS.items() if v == s["Method"])
            vol = s["Ann. Vol"] * 100
            ret = s["Ann. Return"] * 100
            marker = "D" if key.startswith("bm") else "o"
            size = 180 if key.startswith("bm") else 220
            ax.scatter(vol, ret, color=METHOD_COLORS[key],
                       s=size, marker=marker, zorder=4,
                       edgecolors="white", linewidths=1.5)

            # smart annotation offset to avoid overlap
            xoff, yoff = 8, 6
            if vol > 20:
                xoff = -8
                ha = "right"
            else:
                ha = "left"
            ax.annotate(s["Method"],
                        xy=(vol, ret),
                        xytext=(xoff, yoff),
                        textcoords="offset points",
                        fontsize=9, ha=ha, va="bottom",
                        color=METHOD_COLORS[key], fontweight="bold")

        ax.set_xlim(5, 27)
        ax.set_ylim(3, 17)
        ax.set_title("Risk / Return — SAA Methods vs Benchmarks",
                     fontsize=14, fontweight="bold", pad=14)
        ax.set_xlabel("Annualised Volatility (%)", fontsize=11)
        ax.set_ylabel("Annualised Return (%)", fontsize=11)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))

        fig.tight_layout()
        fig.savefig(figures_dir / "saa_risk_return.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_summary_table(stats: list[dict], figures_dir: Path) -> None:
    df = pd.DataFrame(stats).set_index("Method")
    cols = list(df.columns)

    # which direction is "better" for each column
    higher_is_better = {
        "Ann. Return": True,
        "Ann. Vol":    False,
        "Sharpe":      True,
        "Max Drawdown":True,   # values are negative; -30% > -77%, so higher = less bad = best
        "Calmar":      True,
    }
    fmt = {
        "Ann. Return": "{:.2%}",
        "Ann. Vol":    "{:.2%}",
        "Sharpe":      "{:.2f}",
        "Max Drawdown":"{:.2%}",
        "Calmar":      "{:.2f}",
    }

    cell_text = []
    for _, row in df.iterrows():
        cell_text.append([fmt[c].format(row[c]) for c in cols])

    # build color matrix
    BEST  = "#C8F7C5"
    WORST = "#FADBD8"
    NEUT  = "#FFFFFF"
    cell_colors = [[NEUT] * len(cols) for _ in range(len(df))]
    for ci, col in enumerate(cols):
        vals = df[col].values.astype(float)
        best_idx  = int(np.argmax(vals) if higher_is_better[col] else np.argmin(vals))
        worst_idx = int(np.argmin(vals) if higher_is_better[col] else np.argmax(vals))
        cell_colors[best_idx][ci]  = BEST
        cell_colors[worst_idx][ci] = WORST

    with plt.rc_context({**STYLE, "axes.facecolor": "white",
                         "figure.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(13, len(df) * 0.62 + 1.6))
        ax.axis("off")

        tbl = ax.table(
            cellText=cell_text,
            cellColours=cell_colors,
            rowLabels=df.index.tolist(),
            colLabels=["Ann. Return", "Ann. Vol", "Sharpe", "Max DD", "Calmar"],
            cellLoc="center",
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.15, 2.0)

        # bold header row
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_text_props(fontweight="bold")
                cell.set_facecolor("#2C3E50")
                cell.set_text_props(color="white", fontweight="bold")
            if col == -1:  # row labels
                cell.set_text_props(fontweight="bold", ha="right")
                cell.set_facecolor("#ECF0F1")

        ax.set_title("SAA Methods — Performance Summary  (2000–2025)\n"
                     "Green = best in column   Red = worst in column",
                     fontsize=12, fontweight="bold", pad=18, loc="left")

        fig.tight_layout()
        fig.savefig(figures_dir / "saa_summary_table.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_annual_returns_heatmap(all_returns: dict[str, pd.Series], figures_dir: Path) -> None:
    annual: dict[str, pd.Series] = {}
    for key, ret in all_returns.items():
        yr = (1 + ret).resample("YE").prod() - 1
        annual[METHOD_LABELS[key]] = yr

    df = pd.DataFrame(annual).T
    df.columns = df.columns.year

    with plt.rc_context({**STYLE, "axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(18, 5))

        cmap = LinearSegmentedColormap.from_list(
            "rg", ["#C0392B", "#FFFFFF", "#27AE60"], N=256
        )
        vabs = float(df.abs().max().max()) * 100
        im = ax.imshow(df.values * 100, aspect="auto",
                       cmap=cmap, vmin=-vabs, vmax=vabs)

        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels(df.index, fontsize=10)

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                val = df.values[i, j] * 100
                txt_color = "black" if abs(val) < vabs * 0.55 else "white"
                ax.text(j, i, f"{val:.0f}%",
                        ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold")

        plt.colorbar(im, ax=ax, format=mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"),
                     fraction=0.02, pad=0.01)
        ax.set_title("Annual Returns Heatmap — SAA Methods vs Benchmarks",
                     fontsize=13, fontweight="bold", pad=12)
        fig.tight_layout()
        fig.savefig(figures_dir / "saa_annual_heatmap.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def build_all(
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pd.Series]:
    prices_raw, _ = load_asset_prices(PRICES_CSV)
    prices = prices_raw.loc[:, ALL_SAA].copy()
    returns = log_returns(prices)
    inception_dates = first_valid_dates(prices)
    schedule = build_rebalance_schedule(prices, start_date, end_date)

    print(f"Rebalance schedule: {len(schedule)} dates from {schedule[0].date()} to {schedule[-1].date()}")

    all_returns: dict[str, pd.Series] = {}

    for method in METHODS:
        print(f"  Building {METHOD_LABELS[method]} ...", end=" ", flush=True)
        _, ret_df = build_saa_method(
            method, prices, returns, schedule, inception_dates, end_date, output_dir
        )
        all_returns[method] = ret_df["portfolio_return"]
        print("done")

    # load pre-built benchmarks
    for bm in ("bm1", "bm2"):
        path = output_dir / f"{bm}_returns.csv"
        if path.exists():
            bm_ret = pd.read_csv(path, index_col=0, parse_dates=True)["portfolio_return"]
            # align to the same date range as the SAA portfolios
            start_ts = list(all_returns.values())[0].index[0]
            all_returns[bm] = bm_ret.loc[start_ts:]
        else:
            print(f"  WARNING: {path} not found — skipping {bm} in plots")

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    stats = []
    for key, ret in all_returns.items():
        stats.append(performance_stats(ret.dropna(), METHOD_LABELS[key]))

    print("Generating plots ...")
    plot_cumulative_wealth(all_returns, figures_dir)
    plot_drawdowns(all_returns, figures_dir)
    plot_risk_return(stats, figures_dir)
    plot_summary_table(stats, figures_dir)
    plot_annual_returns_heatmap(all_returns, figures_dir)
    print(f"5 figures saved to {figures_dir}")

    # print stats to console
    df_stats = pd.DataFrame(stats).set_index("Method")
    df_stats["Ann. Return"] = df_stats["Ann. Return"].map("{:.2%}".format)
    df_stats["Ann. Vol"]    = df_stats["Ann. Vol"].map("{:.2%}".format)
    df_stats["Sharpe"]      = df_stats["Sharpe"].map("{:.2f}".format)
    df_stats["Max Drawdown"]= df_stats["Max Drawdown"].map("{:.2%}".format)
    df_stats["Calmar"]      = df_stats["Calmar"].map("{:.2f}".format)
    print("\n" + df_stats.to_string())

    return all_returns


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all 6 SAA construction methods.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    build_all(start_date=args.start, end_date=args.end, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
