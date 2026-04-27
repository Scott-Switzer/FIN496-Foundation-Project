"""Nested sleeve risk budgeting for the Whitmore monthly TAA overlay.

References:
- CalPERS (2018), "Asset Allocation Process Review":
  https://www.calpers.ca.gov/docs/board-agendas/201802/invest/item07a-01_a.pdf
- Rockafellar & Uryasev (2000), Conditional Value-at-Risk:
  http://www.ise.ufl.edu/uryasev/files/2011/11/CVaR1_JOR.pdf
- ECOS solver:
  https://github.com/embotech/ecos

Point-in-time safety:
- Safe when the caller supplies expected returns, covariance, availability, and
  optional return scenarios computed strictly from data observed on or before
  the current decision date.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize

from taa_project.config import (
    ALL_SAA,
    CORE,
    CORE_FLOOR,
    COST_PER_TURNOVER,
    NONTRAD,
    NONTRAD_CAP,
    SATELLITE,
    SATELLITE_CAP,
    SINGLE_SLEEVE_MAX,
    TAA_BANDS,
)
from taa_project.memory import guard_process_memory
from taa_project.optimizer.cvxpy_opt import (
    BREACH_LOG_PATH,
    DEFAULT_RISK_AVERSION,
    DEFAULT_VOL_SLACK_PENALTY,
    DIAGONAL_FLOOR,
    OptimizationResult,
    _append_breach_log,
    _build_cvar_scenarios,
    _fallback_weights,
    _finalize_result,
    _solve_cvar_taa,
    cp,
)

if TYPE_CHECKING:
    from taa_project.optimizer.cvxpy_opt import EnsembleConfig


RENORMALIZATION_FILENAME = "nested_renormalizations.csv"
SLEEVE_NAMES = ("core", "satellite", "nontraditional")
SLEEVE_ASSETS = {
    "core": CORE,
    "satellite": SATELLITE,
    "nontraditional": NONTRAD,
}


@dataclass(frozen=True)
class NestedRiskConfig:
    """Configuration for sleeve-by-sleeve risk budgeting."""

    core_vol_target: float = 0.06
    satellite_vol_target: float = 0.10
    nontraditional_vol_target: float = 0.15
    sleeve_weights: tuple[float, float, float] = (0.55, 0.35, 0.10)
    use_cvar_per_sleeve: bool = False
    cvar_alpha: float = 0.95
    cvar_budget_by_sleeve: tuple[float, float, float] = (0.015, 0.025, 0.040)

    def __post_init__(self) -> None:
        if len(self.sleeve_weights) != 3:
            raise ValueError("sleeve_weights must contain exactly three entries.")
        if len(self.cvar_budget_by_sleeve) != 3:
            raise ValueError("cvar_budget_by_sleeve must contain exactly three entries.")
        if not np.isclose(sum(self.sleeve_weights), 1.0, atol=1e-8):
            raise ValueError("sleeve_weights must sum to 1.0.")
        if any(weight < 0.0 for weight in self.sleeve_weights):
            raise ValueError("sleeve_weights must be non-negative.")
        if not 0.5 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie strictly between 0.5 and 1.0.")
        if any(target <= 0.0 for target in (self.core_vol_target, self.satellite_vol_target, self.nontraditional_vol_target)):
            raise ValueError("All sleeve volatility targets must be strictly positive.")
        if any(budget <= 0.0 or budget >= 0.5 for budget in self.cvar_budget_by_sleeve):
            raise ValueError("Each sleeve CVaR budget must lie strictly between 0.0 and 0.5.")


def _normalized_sleeve_bounds(assets: list[str], sleeve_weight: float) -> tuple[pd.Series, pd.Series]:
    lower = pd.Series({asset: TAA_BANDS[asset][0] / sleeve_weight for asset in assets}, dtype=float)
    upper = pd.Series(
        {asset: min(TAA_BANDS[asset][1], SINGLE_SLEEVE_MAX) / sleeve_weight for asset in assets},
        dtype=float,
    ).clip(upper=1.0)
    if bool((lower > upper + 1e-10).any()):
        raise ValueError("Nested sleeve bounds are infeasible after translating the outer TAA bands.")
    return lower, upper


def _feasible_start(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    weights = lower.astype(float).copy()
    residual = 1.0 - float(weights.sum())
    if residual <= 0.0:
        return weights / float(weights.sum())

    room = upper - weights
    room_total = float(room.clip(min=0.0).sum())
    if room_total <= 0.0:
        return weights / float(weights.sum())
    weights += room.clip(min=0.0) / room_total * residual
    return np.clip(weights, lower, upper)


def _minimum_feasible_volatility(sigma: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    x0 = _feasible_start(lower, upper)
    result = minimize(
        lambda weights: float(weights @ sigma @ weights),
        x0=x0,
        method="SLSQP",
        bounds=Bounds(lower, upper),
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        return 0.0
    return float(np.sqrt(max(float(result.fun), 0.0)))


def _active_sleeve_weights(
    base_weights: tuple[float, float, float],
    sleeve_assets: dict[str, list[str]],
) -> tuple[tuple[float, float, float], bool]:
    weights = np.asarray(base_weights, dtype=float)
    active = np.asarray([len(sleeve_assets[name]) > 0 for name in SLEEVE_NAMES], dtype=bool)
    if not active[0]:
        raise ValueError("Nested risk requires at least one available Core asset to preserve the IPS core floor.")

    adjusted = np.where(active, weights, 0.0)
    adjusted /= float(adjusted.sum())
    changed = not np.allclose(adjusted, weights, atol=1e-10)

    if adjusted[0] < CORE_FLOOR:
        deficit = CORE_FLOOR - adjusted[0]
        donor_indices = [index for index in range(1, len(adjusted)) if active[index] and adjusted[index] > 0.0]
        donor_total = float(adjusted[donor_indices].sum()) if donor_indices else 0.0
        if donor_total <= deficit + 1e-12:
            raise ValueError("Unable to preserve the IPS Core floor after sleeve redistribution.")
        for index in donor_indices:
            transfer = deficit * (adjusted[index] / donor_total)
            adjusted[index] -= transfer
        adjusted[0] = CORE_FLOOR
        changed = True

    adjusted /= float(adjusted.sum())
    return tuple(float(weight) for weight in adjusted), changed


def _append_renormalization_row(
    as_of_date: pd.Timestamp | None,
    missing_sleeves: list[str],
    original_weights: tuple[float, float, float],
    adjusted_weights: tuple[float, float, float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "date": pd.Timestamp(as_of_date).isoformat() if as_of_date is not None else "",
                "missing_sleeves": ",".join(missing_sleeves),
                "original_core": float(original_weights[0]),
                "original_satellite": float(original_weights[1]),
                "original_nontraditional": float(original_weights[2]),
                "adjusted_core": float(adjusted_weights[0]),
                "adjusted_satellite": float(adjusted_weights[1]),
                "adjusted_nontraditional": float(adjusted_weights[2]),
            }
        ]
    )
    if output_path.exists():
        existing = pd.read_csv(
            output_path,
            dtype={
                "date": "string",
                "missing_sleeves": "string",
                "original_core": "float32",
                "original_satellite": "float32",
                "original_nontraditional": "float32",
                "adjusted_core": "float32",
                "adjusted_satellite": "float32",
                "adjusted_nontraditional": "float32",
            },
        )
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(output_path, index=False)


def _assert_outer_constraints(weights: pd.Series, available: pd.Series, tolerance: float = 1e-8) -> None:
    if not np.isclose(float(weights.sum()), 1.0, atol=1e-6):
        raise ValueError("Nested-risk blend does not sum to one.")
    if float(weights.loc[CORE].sum()) < CORE_FLOOR - tolerance:
        raise ValueError("Nested-risk blend violates the Core floor.")
    if float(weights.loc[SATELLITE].sum()) > SATELLITE_CAP + tolerance:
        raise ValueError("Nested-risk blend violates the Satellite cap.")
    if float(weights.loc[NONTRAD].sum()) > NONTRAD_CAP + tolerance:
        raise ValueError("Nested-risk blend violates the Non-Traditional cap.")
    if float(weights.max()) > SINGLE_SLEEVE_MAX + tolerance:
        raise ValueError("Nested-risk blend violates the single-sleeve cap.")

    availability_mask = available.reindex(ALL_SAA).fillna(0.0).astype(bool)
    for asset in ALL_SAA:
        weight = float(weights.get(asset, 0.0))
        if not availability_mask.get(asset, False):
            if abs(weight) > tolerance:
                raise ValueError(f"Nested-risk blend assigned non-zero weight to unavailable asset {asset}.")
            continue
        lower, upper = TAA_BANDS[asset]
        capped_upper = min(upper, SINGLE_SLEEVE_MAX)
        if weight < lower - tolerance:
            raise ValueError(f"Nested-risk blend violates lower TAA band for {asset}.")
        if weight > capped_upper + tolerance:
            raise ValueError(f"Nested-risk blend violates upper TAA band for {asset}.")


def _solve_vol_sleeve(
    assets: list[str],
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    previous_weights: pd.Series,
    sleeve_weight: float,
    vol_target: float,
) -> pd.Series:
    if cp is None:
        raise RuntimeError("cvxpy is not installed; nested risk optimization is unavailable.")

    mu = expected_returns.reindex(assets).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    sigma = cov_matrix.reindex(index=assets, columns=assets).fillna(0.0).to_numpy(dtype=float)
    sigma = (sigma + sigma.T) / 2.0
    np.fill_diagonal(sigma, np.maximum(np.diag(sigma), DIAGONAL_FLOOR))
    lower_bounds, upper_bounds = _normalized_sleeve_bounds(assets, sleeve_weight)
    lower_vector = lower_bounds.to_numpy(dtype=float)
    upper_vector = upper_bounds.to_numpy(dtype=float)
    effective_vol_target = max(
        float(vol_target),
        _minimum_feasible_volatility(sigma, lower_vector, upper_vector) + 1e-8,
    )

    prev_sleeve = previous_weights.reindex(assets).fillna(0.0)
    prev_total = float(prev_sleeve.sum())
    prev = (
        prev_sleeve.to_numpy(dtype=float) / prev_total
        if prev_total > 0.0
        else np.full(len(assets), 1.0 / len(assets), dtype=float)
    )

    guard_process_memory(f"nested_vol:{','.join(assets)}:before_build")
    weights = cp.Variable(len(assets), nonneg=True)
    turnover = cp.norm(weights - prev, 1)
    constraints = [
        cp.sum(weights) == 1.0,
        weights >= lower_vector,
        weights <= upper_vector,
        cp.quad_form(weights, cp.psd_wrap(sigma)) <= effective_vol_target**2,
    ]
    objective = cp.Maximize(
        mu @ weights
        - DEFAULT_RISK_AVERSION * cp.quad_form(weights, cp.psd_wrap(sigma))
        - COST_PER_TURNOVER * sleeve_weight * turnover
    )
    prob = cp.Problem(objective, constraints)
    try:
        guard_process_memory(f"nested_vol:{','.join(assets)}:before_solve")
        prob.solve(solver=cp.ECOS, verbose=False)
        guard_process_memory(f"nested_vol:{','.join(assets)}:after_solve")
        if weights.value is None:
            raise RuntimeError(f"Nested sleeve returned no solution. Problem status: {prob.status}")
        out = pd.Series(np.asarray(weights.value, dtype=float), index=assets, dtype=float)
    finally:
        del weights, turnover, constraints, objective, prob
        gc.collect()
        guard_process_memory(f"nested_vol:{','.join(assets)}:after_cleanup")
    return out


def solve_nested_taa(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    available: pd.Series,
    previous_weights: pd.Series,
    config: NestedRiskConfig,
    ensemble_config: EnsembleConfig,
    asset_log_returns: pd.DataFrame | None = None,
    as_of_date: pd.Timestamp | None = None,
    breach_log_path: Path = BREACH_LOG_PATH,
) -> OptimizationResult:
    """Solve per-sleeve TAA optimizations, then blend them at strategic weights."""

    availability_mask = available.reindex(ALL_SAA).fillna(0.0).astype(bool)
    sleeve_assets = {
        name: [asset for asset in assets if availability_mask.get(asset, False)]
        for name, assets in SLEEVE_ASSETS.items()
    }

    try:
        active_sleeve_weights, changed = _active_sleeve_weights(config.sleeve_weights, sleeve_assets)
        if changed:
            _append_renormalization_row(
                as_of_date=as_of_date,
                missing_sleeves=[name for name in SLEEVE_NAMES if not sleeve_assets[name]],
                original_weights=config.sleeve_weights,
                adjusted_weights=active_sleeve_weights,
                output_path=breach_log_path.parent / RENORMALIZATION_FILENAME,
            )

        use_cvar = config.use_cvar_per_sleeve or ensemble_config.optimizer_mode == "cvar"
        cvar_alpha = ensemble_config.cvar_alpha if ensemble_config.optimizer_mode == "cvar" else config.cvar_alpha
        cvar_budgets = {
            "core": ensemble_config.cvar_budget if ensemble_config.optimizer_mode == "cvar" else config.cvar_budget_by_sleeve[0],
            "satellite": ensemble_config.cvar_budget if ensemble_config.optimizer_mode == "cvar" else config.cvar_budget_by_sleeve[1],
            "nontraditional": ensemble_config.cvar_budget if ensemble_config.optimizer_mode == "cvar" else config.cvar_budget_by_sleeve[2],
        }
        vol_targets = {
            "core": config.core_vol_target,
            "satellite": config.satellite_vol_target,
            "nontraditional": config.nontraditional_vol_target,
        }

        blended = pd.Series(0.0, index=ALL_SAA, dtype=float)
        for sleeve_name, sleeve_weight in zip(SLEEVE_NAMES, active_sleeve_weights, strict=True):
            assets = sleeve_assets[sleeve_name]
            if sleeve_weight <= 0.0 or not assets:
                continue

            if use_cvar:
                if asset_log_returns is None:
                    raise ValueError("Nested CVaR mode requires strictly causal asset_log_returns.")
                scenario_matrix = _build_cvar_scenarios(
                    asset_log_returns=asset_log_returns,
                    decision_date=pd.Timestamp(as_of_date),
                    universe=assets,
                    lookback_days=ensemble_config.cvar_lookback_days,
                )
                lower_bounds, upper_bounds = _normalized_sleeve_bounds(assets, sleeve_weight)
                prev_sleeve = previous_weights.reindex(assets).fillna(0.0)
                prev_total = float(prev_sleeve.sum())
                prev = (
                    prev_sleeve.to_numpy(dtype=float) / prev_total
                    if prev_total > 0.0
                    else np.full(len(assets), 1.0 / len(assets), dtype=float)
                )
                sleeve_solution = pd.Series(
                    _solve_cvar_taa(
                        universe=assets,
                        expected_returns=expected_returns.reindex(assets).replace([np.inf, -np.inf], np.nan).fillna(0.0),
                        scenario_returns=scenario_matrix,
                        previous_weights=prev,
                        lower_bounds=lower_bounds,
                        upper_bounds=upper_bounds,
                        core_idx=[],
                        sat_idx=[],
                        nontrad_idx=[],
                        cvar_alpha=cvar_alpha,
                        cvar_budget=cvar_budgets[sleeve_name],
                        turnover_penalty=COST_PER_TURNOVER * sleeve_weight,
                    ),
                    index=assets,
                    dtype=float,
                )
                del scenario_matrix
                gc.collect()
            else:
                sleeve_solution = _solve_vol_sleeve(
                    assets=assets,
                    expected_returns=expected_returns,
                    cov_matrix=cov_matrix,
                    previous_weights=previous_weights,
                    sleeve_weight=sleeve_weight,
                    vol_target=vol_targets[sleeve_name],
                )

            blended.loc[assets] = sleeve_weight * sleeve_solution
            del sleeve_solution
            gc.collect()
            guard_process_memory(f"nested:{sleeve_name}:after_blend")

        _assert_outer_constraints(blended, available)
        return _finalize_result(
            blended,
            "taa_monthly",
            "optimal",
            previous_weights,
            cov_matrix,
            COST_PER_TURNOVER,
            False,
        )
    except Exception as exc:
        message = f"Nested TAA solve failed: {exc}. Falling back to the last feasible weights."
        _append_breach_log("taa_monthly", as_of_date, "nested_fallback", message, breach_log_path)
        fallback = _fallback_weights("taa_monthly", previous_weights, available)
        return _finalize_result(
            fallback,
            "taa_monthly",
            "fallback",
            previous_weights,
            cov_matrix,
            COST_PER_TURNOVER,
            True,
            message,
        )
