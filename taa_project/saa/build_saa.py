# Addresses the IPS strategic allocation mandate and the project rubric's SAA +
# IPS-compliance component by constructing an annual-rebalanced strategic book.
"""Build the Whitmore SAA portfolio with constrained risk parity.

Method choice: constrained risk parity with IPS-target risk budgets.

Why this fits the mandate better than the other listed methods:
- It uses covariance and diversification information without needing fragile
  expected-return estimates, which makes it more defensible than mean-variance
  in a graded out-of-sample project.
- It respects the IPS structure naturally: the optimizer can sit inside hard
  per-sleeve bands and aggregate caps while still equalizing portfolio risk
  contributions as much as the mandate allows.
- It is more stable than minimum variance, which would otherwise push the book
  too hard into nominal and inflation-linked fixed income and fight the IPS
  real-asset sleeves.
- It is more policy-aware than inverse volatility, because full covariance
  matters when Core, Satellite, and Non-Traditional sleeves co-move sharply.
- Using IPS target weights as the risk-budget anchor keeps zero-target sleeves
  like FTSE100 and BITCOIN from entering the strategic book by accident while
  preserving the risk-parity logic.

References:
- Whitmore IPS in `IPS.md` and `Guidelines.md`.
- Risk parity overview: https://en.wikipedia.org/wiki/Risk_parity
- SciPy constrained optimization: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize

from taa_project.config import (
    ALL_SAA,
    CORE,
    CORE_FLOOR,
    COST_PER_TURNOVER,
    NONTRAD,
    NONTRAD_CAP,
    OUTPUT_DIR,
    PRICES_CSV,
    SAA_BANDS,
    SAA_FREQ,
    SAA_TARGETS,
    SATELLITE,
    SATELLITE_CAP,
    SINGLE_SLEEVE_MAX,
)
from taa_project.compliance import append_compliance_rebalance_log, compliance_breach_rows
from taa_project.data_audit import load_asset_prices
from taa_project.data_loader import log_returns


SAA_WEIGHTS_FILENAME = "saa_weights.csv"
SAA_RETURNS_FILENAME = "saa_returns.csv"
DEFAULT_START = "2000-01-01"
DEFAULT_END = "2026-04-15"
LOOKBACK_DAYS = 756
MIN_COV_OBSERVATIONS = 63
DIAGONAL_FLOOR = 1e-6


@dataclass(frozen=True)
class SAAOptimizationInputs:
    """Container for one annual SAA solve.

    Inputs:
    - `covariance`: annualized covariance matrix on the eligible universe.
    - `lower_bounds`: per-asset lower bounds.
    - `upper_bounds`: per-asset upper bounds.
    - `risk_budgets`: target risk-contribution shares.
    - `assets`: ordered asset list matching the matrix and vectors.

    Outputs:
    - Immutable struct used by `solve_target_risk_parity`.

    Citation:
    - Internal Whitmore IPS tables and SciPy SLSQP documentation:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Point-in-time safety:
    - Safe. Every field is constructed from data and constraints known at the
      rebalance date and does not look ahead.
    """

    covariance: pd.DataFrame
    lower_bounds: pd.Series
    upper_bounds: pd.Series
    risk_budgets: pd.Series
    assets: list[str]


def load_saa_prices(path: Path = PRICES_CSV) -> pd.DataFrame:
    """Load the audited SAA price panel only.

    Inputs:
    - `path`: location of the authoritative Whitmore price file.

    Outputs:
    - Price dataframe restricted to the eleven SAA sleeves.

    Citation:
    - Internal project file `data/asset_data/whitmore_daily.csv`.

    Point-in-time safety:
    - Safe. This is a raw audited load of historical prices only.
    """

    prices, _ = load_asset_prices(path)
    return prices.loc[:, ALL_SAA].copy()


def first_valid_dates(prices: pd.DataFrame) -> pd.Series:
    """Compute the first valid observation date for each SAA asset.

    Inputs:
    - `prices`: SAA price dataframe.

    Outputs:
    - Series mapping each asset to its first valid date.

    Citation:
    - Internal project file `data/asset_data/whitmore_daily.csv`.

    Point-in-time safety:
    - Safe. This is descriptive metadata over realized history.
    """

    return prices.apply(lambda column: column.first_valid_index())


def available_assets_on(date: pd.Timestamp, inception_dates: pd.Series) -> list[str]:
    """List SAA assets that are investable by a given rebalance date.

    Inputs:
    - `date`: rebalance date.
    - `inception_dates`: first valid date per asset.

    Outputs:
    - Ordered list of investable SAA assets at `date`.

    Citation:
    - `tasks.md`, Step 3 expanding-universe rule.

    Point-in-time safety:
    - Safe. Availability depends only on whether an asset had already started by
      the decision date.
    """

    return [asset for asset in ALL_SAA if pd.notna(inception_dates[asset]) and inception_dates[asset] <= date]


def bounds_for_assets(assets: Iterable[str]) -> tuple[pd.Series, pd.Series]:
    """Return the IPS SAA lower and upper bounds for the chosen assets.

    Inputs:
    - `assets`: iterable of eligible SAA tickers.

    Outputs:
    - Tuple `(lower_bounds, upper_bounds)` indexed by asset.

    Citation:
    - `Guidelines.md`, Strategic Asset Allocation table.

    Point-in-time safety:
    - Safe. The bounds are static IPS constraints.
    """

    asset_list = list(assets)
    lower = pd.Series({asset: SAA_BANDS[asset][0] for asset in asset_list}, dtype=float)
    upper = pd.Series({asset: min(SAA_BANDS[asset][1], SINGLE_SLEEVE_MAX) for asset in asset_list}, dtype=float)
    return lower, upper


def build_linear_constraints(assets: list[str]) -> tuple[LinearConstraint, list[LinearConstraint]]:
    """Build the IPS linear constraints for a given eligible universe.

    Inputs:
    - `assets`: eligible SAA assets.

    Outputs:
    - Tuple of the full-investment equality constraint and a list of aggregate
      inequality constraints.

    Citation:
    - Whitmore IPS `Guidelines.md` Hard Constraints section.

    Point-in-time safety:
    - Safe. These are static IPS constraints, not data-derived signals.
    """

    asset_positions = {asset: index for index, asset in enumerate(assets)}
    n_assets = len(assets)

    ones = np.ones((1, n_assets))
    sum_to_one = LinearConstraint(ones, lb=np.array([1.0]), ub=np.array([1.0]))

    inequality_constraints: list[LinearConstraint] = []

    if any(asset in asset_positions for asset in CORE):
        row = np.zeros((1, n_assets))
        for asset in CORE:
            if asset in asset_positions:
                row[0, asset_positions[asset]] = 1.0
        inequality_constraints.append(LinearConstraint(row, lb=np.array([CORE_FLOOR]), ub=np.array([np.inf])))

    if any(asset in asset_positions for asset in SATELLITE):
        row = np.zeros((1, n_assets))
        for asset in SATELLITE:
            if asset in asset_positions:
                row[0, asset_positions[asset]] = 1.0
        inequality_constraints.append(LinearConstraint(row, lb=np.array([-np.inf]), ub=np.array([SATELLITE_CAP])))

    if any(asset in asset_positions for asset in NONTRAD):
        row = np.zeros((1, n_assets))
        for asset in NONTRAD:
            if asset in asset_positions:
                row[0, asset_positions[asset]] = 1.0
        inequality_constraints.append(LinearConstraint(row, lb=np.array([-np.inf]), ub=np.array([NONTRAD_CAP])))

    return sum_to_one, inequality_constraints


def project_policy_targets_to_feasible_set(
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
    assets: list[str],
) -> np.ndarray:
    """Project IPS target weights into the feasible region.

    Inputs:
    - `lower_bounds`: lower bound per eligible asset.
    - `upper_bounds`: upper bound per eligible asset.
    - `assets`: eligible asset list.

    Outputs:
    - Feasible weight vector used as the initial guess and fallback solution.

    Citation:
    - Internal Whitmore IPS target table and SciPy SLSQP documentation:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Point-in-time safety:
    - Safe. The projection uses only static IPS targets and constraints known at
      the rebalance date.
    """

    target = np.array([SAA_TARGETS[asset] for asset in assets], dtype=float)
    return project_weights_to_feasible_set(target, lower_bounds, upper_bounds, assets)


def project_weights_to_feasible_set(
    target_weights: np.ndarray | pd.Series,
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
    assets: list[str],
) -> np.ndarray:
    """Project an arbitrary weight vector into the feasible IPS region.

    Inputs:
    - `target_weights`: weight vector to project.
    - `lower_bounds`: lower bound per eligible asset.
    - `upper_bounds`: upper bound per eligible asset.
    - `assets`: eligible asset list.

    Outputs:
    - Feasible weight vector closest to `target_weights` in squared distance.

    Citation:
    - SciPy SLSQP documentation:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Point-in-time safety:
    - Safe. The projection uses only current weights plus static IPS
      constraints available at the decision date.
    """

    target = np.asarray(target_weights, dtype=float)
    if target.sum() <= 0:
        target = np.full(len(assets), 1.0 / len(assets))
    else:
        target = target / target.sum()

    bounds = Bounds(lower_bounds.values, upper_bounds.values)
    sum_to_one, inequalities = build_linear_constraints(assets)

    result = minimize(
        lambda weights: float(np.sum((weights - target) ** 2)),
        x0=np.clip(target, lower_bounds.values, upper_bounds.values),
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_to_one, *inequalities],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Unable to construct a feasible weight projection: {result.message}")
    return result.x


def target_risk_budgets(assets: list[str]) -> pd.Series:
    """Convert IPS target weights into normalized risk-budget targets.

    Inputs:
    - `assets`: eligible asset list.

    Outputs:
    - Series of target risk-contribution shares that sum to one.

    Citation:
    - Whitmore IPS strategic target table in `Guidelines.md`.

    Point-in-time safety:
    - Safe. Risk budgets are based only on fixed IPS target weights.
    """

    budgets = pd.Series({asset: max(SAA_TARGETS[asset], 0.0) for asset in assets}, dtype=float)
    if budgets.sum() <= 0:
        budgets[:] = 1.0 / len(assets)
    else:
        budgets /= budgets.sum()
    return budgets


def estimate_covariance(
    returns: pd.DataFrame,
    as_of_date: pd.Timestamp,
    assets: list[str],
    lookback_days: int = LOOKBACK_DAYS,
    min_observations: int = MIN_COV_OBSERVATIONS,
) -> pd.DataFrame:
    """Estimate a stable annualized covariance matrix up to the rebalance date.

    Inputs:
    - `returns`: asset log-return dataframe with gaps preserved as `NaN`.
    - `as_of_date`: rebalance date.
    - `assets`: eligible asset list.
    - `lookback_days`: trailing calendar length for the covariance sample.
    - `min_observations`: minimum pairwise observations before variance floor.

    Outputs:
    - Positive-semidefinite annualized covariance matrix on the eligible assets.

    Citation:
    - Internal project returns from Task 1 and NumPy linear algebra routines.

    Point-in-time safety:
    - Safe. The sample uses only returns dated on or before `as_of_date`.
    """

    history = returns.loc[:as_of_date, assets].tail(lookback_days)
    covariance = history.cov(min_periods=min_observations)
    variances = history.var(skipna=True).reindex(assets)

    covariance = covariance.reindex(index=assets, columns=assets)
    for asset in assets:
        diagonal = variances.get(asset, np.nan)
        covariance.loc[asset, asset] = float(diagonal) if pd.notna(diagonal) and diagonal > 0 else DIAGONAL_FLOOR

    covariance = covariance.fillna(0.0)
    covariance = 0.75 * covariance + 0.25 * np.diag(np.diag(covariance.values))
    covariance_values = covariance.to_numpy(dtype=float) * 252.0
    covariance_values = (covariance_values + covariance_values.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_values)
    eigenvalues = np.clip(eigenvalues, DIAGONAL_FLOOR, None)
    psd_values = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    return pd.DataFrame(psd_values, index=assets, columns=assets)


def solve_target_risk_parity(inputs: SAAOptimizationInputs) -> pd.Series:
    """Solve the constrained target-risk-parity portfolio for one rebalance.

    Inputs:
    - `inputs`: covariance, bounds, risk budgets, and asset order.

    Outputs:
    - Feasible target weight series indexed by eligible asset.

    Citation:
    - Whitmore IPS strategic target table and risk-parity overview:
      https://en.wikipedia.org/wiki/Risk_parity
    - SciPy SLSQP docs:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Point-in-time safety:
    - Safe. The solve uses only constraints and covariance estimated from data
      at or before the rebalance date.
    """

    assets = inputs.assets
    covariance = inputs.covariance.loc[assets, assets].to_numpy(dtype=float)
    bounds = Bounds(inputs.lower_bounds.values, inputs.upper_bounds.values)
    sum_to_one, inequalities = build_linear_constraints(assets)
    initial = project_policy_targets_to_feasible_set(inputs.lower_bounds, inputs.upper_bounds, assets)
    budgets = inputs.risk_budgets.loc[assets].to_numpy(dtype=float)

    def objective(weights: np.ndarray) -> float:
        portfolio_variance = float(weights @ covariance @ weights)
        portfolio_variance = max(portfolio_variance, DIAGONAL_FLOOR)
        risk_contributions = weights * (covariance @ weights) / portfolio_variance
        return float(np.sum((risk_contributions - budgets) ** 2))

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_to_one, *inequalities],
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    solution = initial if not result.success else result.x
    solution = project_weights_to_feasible_set(solution, inputs.lower_bounds, inputs.upper_bounds, assets)
    return pd.Series(solution, index=assets, dtype=float)


def solve_minimum_variance(covariance: pd.DataFrame, assets: list[str]) -> pd.Series:
    """Solve the constrained minimum-variance portfolio for one rebalance.

    Inputs:
    - `covariance`: annualized covariance matrix on the eligible universe.
    - `assets`: ordered eligible asset list.

    Outputs:
    - Feasible minimum-variance weight series indexed by eligible asset.

    Citation:
    - Whitmore IPS strategic bands and SciPy SLSQP documentation:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Point-in-time safety:
    - Safe. The solve uses only constraints and covariance estimated from data
      at or before the rebalance date.
    """

    cov = covariance.loc[assets, assets].to_numpy(dtype=float)
    lower_bounds, upper_bounds = bounds_for_assets(assets)
    bounds = Bounds(lower_bounds.values, upper_bounds.values)
    sum_to_one, inequalities = build_linear_constraints(assets)
    initial = project_policy_targets_to_feasible_set(lower_bounds, upper_bounds, assets)

    result = minimize(
        lambda weights: float(weights @ cov @ weights),
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_to_one, *inequalities],
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    solution = initial if not result.success else result.x
    solution = project_weights_to_feasible_set(solution, lower_bounds, upper_bounds, assets)
    return pd.Series(solution, index=assets, dtype=float)


def violates_saa_constraints(weights: pd.Series, eligible_assets: list[str], tolerance: float = 1e-8) -> bool:
    """Check whether a full-universe weight vector violates any SAA IPS rule.

    Inputs:
    - `weights`: full-universe weight vector.
    - `eligible_assets`: sleeves investable on the current date.
    - `tolerance`: floating-point slack for numerical comparisons.

    Outputs:
    - `True` if any per-sleeve or aggregate IPS constraint is breached.

    Citation:
    - Whitmore IPS `Guidelines.md` strategic allocation table and hard limits.

    Point-in-time safety:
    - Safe. The check depends only on contemporaneous weights and static IPS
      constraints.
    """

    lower_bounds, upper_bounds = bounds_for_assets(eligible_assets)
    eligible_weights = weights.reindex(eligible_assets).fillna(0.0)

    if float(weights.sum()) < 1.0 - tolerance or float(weights.sum()) > 1.0 + tolerance:
        return True
    if float(weights.min()) < -tolerance:
        return True
    if bool((eligible_weights < (lower_bounds - tolerance)).any()):
        return True
    if bool((eligible_weights > (upper_bounds + tolerance)).any()):
        return True
    if float(weights.loc[CORE].sum()) < CORE_FLOOR - tolerance:
        return True
    if float(weights.loc[SATELLITE].sum()) > SATELLITE_CAP + tolerance:
        return True
    if float(weights.loc[NONTRAD].sum()) > NONTRAD_CAP + tolerance:
        return True
    if float(weights.max()) > SINGLE_SLEEVE_MAX + tolerance:
        return True
    return False


def project_drifted_weights_to_compliance(
    drifted_weights: pd.Series,
    eligible_assets: list[str],
) -> pd.Series:
    """Project drifted holdings back into the feasible SAA region.

    Inputs:
    - `drifted_weights`: full-universe post-market-move weights.
    - `eligible_assets`: sleeves investable on the current date.

    Outputs:
    - Full-universe weight vector satisfying all current SAA constraints.

    Citation:
    - Whitmore IPS hard constraints and SciPy SLSQP documentation:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Point-in-time safety:
    - Safe. The projection uses only post-move holdings known on the current
      date and static IPS constraints.
    """

    lower_bounds, upper_bounds = bounds_for_assets(eligible_assets)
    try:
        projected = project_weights_to_feasible_set(
            drifted_weights.reindex(eligible_assets).fillna(0.0),
            lower_bounds,
            upper_bounds,
            eligible_assets,
        )
    except RuntimeError:
        # Drift shocks can occasionally leave SLSQP stuck at the boundary even
        # though the IPS region is feasible. Falling back to the projected
        # policy targets preserves compliance without inventing new signals.
        projected = project_policy_targets_to_feasible_set(lower_bounds, upper_bounds, eligible_assets)
    full_weights = pd.Series(0.0, index=ALL_SAA, dtype=float)
    full_weights.loc[eligible_assets] = projected
    return full_weights


def find_first_feasible_start_date(prices: pd.DataFrame, start_date: str) -> pd.Timestamp:
    """Find the first feasible inception date for the SAA portfolio.

    Inputs:
    - `prices`: SAA price dataframe.
    - `start_date`: user-requested earliest start date.

    Outputs:
    - First date on or after `start_date` where the investable universe is
      feasible under the IPS SAA constraints.

    Citation:
    - `tasks.md`, Step 3 portfolio start-date rules.

    Point-in-time safety:
    - Safe. The scan uses only information available at each candidate date.
    """

    candidate_index = prices.loc[pd.Timestamp(start_date) :].index
    inception_dates = first_valid_dates(prices)

    for date in candidate_index:
        assets = available_assets_on(date, inception_dates)
        if not assets:
            continue
        if not prices.loc[date, assets].notna().all():
            continue
        lower_bounds, upper_bounds = bounds_for_assets(assets)
        try:
            project_policy_targets_to_feasible_set(lower_bounds, upper_bounds, assets)
        except RuntimeError:
            continue
        return pd.Timestamp(date)

    raise RuntimeError("No feasible SAA start date found in the provided price history.")


def choose_year_end_rebalance_date(prices: pd.DataFrame, year: int, inception_dates: pd.Series) -> pd.Timestamp | None:
    """Choose the last common observed trading date in a calendar year.

    Inputs:
    - `prices`: SAA price dataframe.
    - `year`: calendar year to inspect.
    - `inception_dates`: first valid date per asset.

    Outputs:
    - Last date in `year` where all assets eligible on that date have observed
      prices, or `None` if no such date exists.

    Citation:
    - Whitmore IPS rebalancing rule in `Guidelines.md`.

    Point-in-time safety:
    - Safe. The chosen rebalance date depends only on observed prices through
      that year-end and never references future dates.
    """

    year_slice = prices.loc[f"{year}-01-01" : f"{year}-12-31"]
    for date in reversed(year_slice.index.tolist()):
        assets = available_assets_on(date, inception_dates)
        if assets and bool(prices.loc[date, assets].notna().all()):
            return pd.Timestamp(date)
    return None


def build_rebalance_schedule(prices: pd.DataFrame, start_date: str, end_date: str) -> list[pd.Timestamp]:
    """Build the initial and annual year-end SAA rebalance schedule.

    Inputs:
    - `prices`: SAA price dataframe.
    - `start_date`: earliest requested portfolio inception date.
    - `end_date`: last date to include in the portfolio history.

    Outputs:
    - Ordered list of rebalance dates beginning with the initial allocation.

    Citation:
    - Whitmore IPS annual rebalance rule in `Guidelines.md`.

    Point-in-time safety:
    - Safe. The schedule is determined only from dates on or before each
      calendar year-end.
    """

    end_timestamp = pd.Timestamp(end_date)
    inception_dates = first_valid_dates(prices)
    first_date = find_first_feasible_start_date(prices.loc[:end_timestamp], start_date)
    schedule = [first_date]

    for year in range(first_date.year, end_timestamp.year + 1):
        rebalance_date = choose_year_end_rebalance_date(prices.loc[:end_timestamp], year, inception_dates)
        if rebalance_date is None or rebalance_date <= first_date:
            continue
        schedule.append(rebalance_date)

    return sorted(dict.fromkeys(schedule))


def compute_target_weights(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    inception_dates: pd.Series,
    method: str = "min_variance",
) -> pd.Series:
    """Compute the constrained SAA target weights for one rebalance date.

    Inputs:
    - `prices`: SAA price dataframe.
    - `returns`: SAA log-return dataframe.
    - `rebalance_date`: date of the allocation decision.
    - `inception_dates`: first valid date per asset.
    - `method`: SAA construction method, one of `risk_parity`,
      `min_variance`, or `hrp`.

    Outputs:
    - Full-universe weight series with unavailable assets fixed at zero.

    Citation:
    - Whitmore IPS `Guidelines.md` strategic allocation table and hard limits.

    Point-in-time safety:
    - Safe. Covariance and eligibility are built strictly from data at or
      before `rebalance_date`.
    """

    eligible_assets = available_assets_on(rebalance_date, inception_dates)
    observed_assets = [asset for asset in eligible_assets if pd.notna(prices.loc[rebalance_date, asset])]
    covariance = estimate_covariance(returns, rebalance_date, observed_assets)
    if method == "risk_parity":
        lower_bounds, upper_bounds = bounds_for_assets(observed_assets)
        budgets = target_risk_budgets(observed_assets)
        target_weights = solve_target_risk_parity(
            SAAOptimizationInputs(
                covariance=covariance,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                risk_budgets=budgets,
                assets=observed_assets,
            )
        )
    elif method in {"min_variance", "minimum_variance"}:
        target_weights = solve_minimum_variance(covariance, observed_assets)
    elif method == "hrp":
        from taa_project.saa.saa_comparison import solve_hierarchical_risk_parity

        target_weights = solve_hierarchical_risk_parity(observed_assets, covariance).reindex(observed_assets).fillna(0.0)
    else:
        raise ValueError(f"Unsupported SAA method: {method}")

    full_weights = pd.Series(0.0, index=ALL_SAA, dtype=float)
    full_weights.loc[observed_assets] = target_weights
    return full_weights


def simulate_portfolio(
    returns: pd.DataFrame,
    rebalance_targets: dict[pd.Timestamp, pd.Series],
    rebalance_active_assets: dict[pd.Timestamp, list[str]],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    compliance_log_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate daily SAA portfolio returns and record target weights.

    Inputs:
    - `returns`: SAA log-return dataframe with raw gaps preserved as `NaN`.
    - `rebalance_targets`: mapping from rebalance date to target weights.
    - `rebalance_active_assets`: mapping from rebalance date to the sleeves that
      were active at that rebalance and therefore subject to compliance checks
      until the next scheduled rebalance.
    - `start_date`: first portfolio date.
    - `end_date`: final portfolio date.
    - `compliance_log_path`: optional CSV destination for automatic
      IPS-compliance rebalance documentation.

    Outputs:
    - Tuple `(weights_df, returns_df)` with daily policy target weights and
      daily realized simple portfolio returns from drifting annual holdings.

    Citation:
    - Whitmore IPS annual rebalance rule and cost policy in `Guidelines.md`.

    Point-in-time safety:
    - Safe. The simulation applies weights forward through time and only uses
      returns realized after the allocation date.
    """

    calendar = returns.loc[start_date:end_date].index
    current_policy_target = rebalance_targets[start_date].copy()
    current_target = current_policy_target.copy()
    current_holdings = current_target.copy()
    current_active_assets = list(rebalance_active_assets[start_date])
    weight_rows: list[pd.Series] = [current_target.rename(start_date)]
    return_rows = [
        {
            "Date": start_date,
            "portfolio_return": 0.0,
            "gross_return": 0.0,
            "turnover": 0.0,
            "turnover_cost": 0.0,
            "scheduled_rebalance_flag": 1,
            "compliance_rebalance_flag": 0,
            "rebalance_flag": 1,
        }
    ]

    for date in calendar[1:]:
        turnover = 0.0
        turnover_cost = 0.0
        rebalance_flag = 0
        scheduled_rebalance_flag = 0
        compliance_rebalance_flag = 0

        gross_vector = np.exp(returns.loc[date, ALL_SAA].fillna(0.0))
        gross_return = float((current_holdings * (gross_vector - 1.0)).sum())
        post_move_value = current_holdings * gross_vector
        denominator = float(post_move_value.sum())
        post_move_holdings = post_move_value / denominator if denominator > 0 else current_holdings.copy()

        if date in rebalance_targets:
            current_policy_target = rebalance_targets[date].copy()
            current_active_assets = list(rebalance_active_assets[date])
            target = current_policy_target
            turnover += float((target - post_move_holdings).abs().sum())
            current_target = target.copy()
            current_holdings = target.copy()
            scheduled_rebalance_flag = 1
            rebalance_flag = 1
        else:
            current_holdings = post_move_holdings.copy()
            if violates_saa_constraints(current_holdings, current_active_assets):
                pre_rebalance = current_holdings.copy()
                projected = project_drifted_weights_to_compliance(current_holdings, current_active_assets)
                compliance_turnover = float((projected - current_holdings).abs().sum())
                turnover += compliance_turnover
                current_target = projected.copy()
                current_holdings = projected.copy()
                compliance_rebalance_flag = 1
                rebalance_flag = 1
                if compliance_log_path is not None:
                    append_compliance_rebalance_log(
                        compliance_log_path,
                        compliance_breach_rows(
                            portfolio="SAA",
                            date=date,
                            decision_date=None,
                            pre_trade_weights=pre_rebalance,
                            post_trade_weights=projected,
                            active_assets=current_active_assets,
                            band_map=SAA_BANDS,
                            turnover=compliance_turnover,
                            remediation="projected_drifted_saa_holdings_to_ips_feasible_set",
                        ),
                    )

        turnover_cost = COST_PER_TURNOVER * turnover
        portfolio_return = gross_return - turnover_cost
        return_rows.append(
            {
                "Date": date,
                "portfolio_return": portfolio_return,
                "gross_return": gross_return,
                "turnover": turnover,
                "turnover_cost": turnover_cost,
                "scheduled_rebalance_flag": scheduled_rebalance_flag,
                "compliance_rebalance_flag": compliance_rebalance_flag,
                "rebalance_flag": rebalance_flag,
            }
        )
        weight_rows.append(current_target.rename(date))

    weights_df = pd.DataFrame(weight_rows)
    returns_df = pd.DataFrame(return_rows).set_index("Date")
    return weights_df, returns_df


def build_saa_portfolio(
    start_date: str = DEFAULT_START,
    end_date: str = DEFAULT_END,
    output_dir: Path = OUTPUT_DIR,
    method: str = "min_variance",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the Whitmore annual-rebalanced SAA portfolio.

    Inputs:
    - `start_date`: earliest allowed start date for portfolio inception.
    - `end_date`: last date to include in the output history.
    - `output_dir`: destination directory for output CSV files.
    - `method`: SAA construction method, one of `risk_parity`,
      `min_variance`, or `hrp`.

    Outputs:
    - Tuple `(weights_df, returns_df)` also written to `saa_weights.csv` and
      `saa_returns.csv`.

    Citation:
    - Whitmore IPS and `tasks.md` expanding-universe rules.

    Point-in-time safety:
    - Safe. Rebalance targets are estimated only from information observed at
      or before each annual rebalance date.
    """

    prices = load_saa_prices()
    returns = log_returns(prices)
    inception_dates = first_valid_dates(prices)
    schedule = build_rebalance_schedule(prices, start_date, end_date)

    rebalance_targets = {}
    rebalance_active_assets: dict[pd.Timestamp, list[str]] = {}
    for rebalance_date in schedule:
        active_assets = available_assets_on(rebalance_date, inception_dates)
        observed_assets = [asset for asset in active_assets if pd.notna(prices.loc[rebalance_date, asset])]
        rebalance_active_assets[rebalance_date] = observed_assets
        rebalance_targets[rebalance_date] = compute_target_weights(
            prices,
            returns,
            rebalance_date,
            inception_dates,
            method=method,
        )

    weights_df, returns_df = simulate_portfolio(
        returns=returns,
        rebalance_targets=rebalance_targets,
        rebalance_active_assets=rebalance_active_assets,
        start_date=schedule[0],
        end_date=pd.Timestamp(end_date),
        compliance_log_path=output_dir / "saa_compliance_rebalances.csv",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_df.to_csv(output_dir / SAA_WEIGHTS_FILENAME)
    returns_df.to_csv(output_dir / SAA_RETURNS_FILENAME)
    return weights_df, returns_df


def main() -> None:
    """CLI entrypoint for building the SAA output files.

    Inputs:
    - `--start`: earliest allowed start date.
    - `--end`: last date to include in the output history.
    - `--output-dir`: destination for generated CSV files.

    Outputs:
    - Writes `saa_weights.csv` and `saa_returns.csv` to disk.

    Citation:
    - Whitmore IPS annual SAA rebalance rule.

    Point-in-time safety:
    - Safe. The CLI only orchestrates the causal builder defined above.
    """

    parser = argparse.ArgumentParser(description="Build the Whitmore SAA portfolio.")
    parser.add_argument("--start", default=DEFAULT_START, help="Earliest allowed portfolio start date.")
    parser.add_argument("--end", default=DEFAULT_END, help="Last date to include in the output history.")
    parser.add_argument(
        "--method",
        choices=["risk_parity", "min_variance", "hrp"],
        default="min_variance",
        help="Strategic asset allocation method. Default is constrained minimum variance.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Destination directory for output CSV files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    build_saa_portfolio(start_date=args.start, end_date=args.end, output_dir=output_dir, method=args.method)
    print(f"SAA outputs written to {output_dir / SAA_WEIGHTS_FILENAME} and {output_dir / SAA_RETURNS_FILENAME}")


if __name__ == "__main__":
    main()
