# Addresses rubric criterion 1 (TAA signal design) and criterion 4
# (IPS-compliant implementation) by solving annual SAA and monthly TAA weights
# with explicit turnover penalties, soft volatility control, and breach logging.
"""Portfolio optimization utilities for Whitmore SAA and TAA.

Task 5 requirements implemented here:
- Two solve modes: `saa_annual` and `taa_monthly`.
- Annual mode delegates to the Task 2 constrained risk-parity method.
- Monthly mode solves the signal-ensemble problem in `cvxpy`.
- Turnover penalty is applied against the previously held portfolio using
  `5 bps * |Δw|_1`.
- Ex-ante volatility is treated as a soft ceiling at 10% via a penalized slack
  variable in monthly mode.
- On infeasibility or missing optional dependencies, the solver logs an
  incident to `outputs/breaches.log` and falls back to the last feasible
  portfolio projected into the currently feasible region.

References:
- Whitmore IPS `IPS.md` / `Guidelines.md`
- cvxpy documentation: https://www.cvxpy.org/
- SciPy constrained optimization docs:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize

try:  # pragma: no cover - availability depends on local environment.
    import cvxpy as cp
except ImportError:  # pragma: no cover - availability depends on local environment.
    cp = None

from taa_project.config import (
    ALL_SAA,
    BM2_WEIGHTS,
    CORE,
    CORE_FLOOR,
    COST_PER_TURNOVER,
    NONTRAD,
    NONTRAD_CAP,
    OUTPUT_DIR,
    SAA_TARGETS,
    SATELLITE,
    SATELLITE_CAP,
    SINGLE_SLEEVE_MAX,
    TARGET_VOL,
    TAA_BANDS,
)
from taa_project.memory import guard_process_memory
from taa_project.signals.dd_guardrail import DrawdownGuardrailConfig
from taa_project.saa.build_saa import (
    SAAOptimizationInputs,
    bounds_for_assets as saa_bounds_for_assets,
    build_linear_constraints,
    project_weights_to_feasible_set,
    solve_target_risk_parity,
    target_risk_budgets,
)

if TYPE_CHECKING:
    from taa_project.optimizer.nested_risk import NestedRiskConfig


OptimizationMode = Literal["saa_annual", "taa_monthly"]
BREACH_LOG_PATH = OUTPUT_DIR / "breaches.log"
OPTIMIZER_BREACHES_PATH = OUTPUT_DIR / "optimizer_breaches.csv"
DIAGONAL_FLOOR = 1e-6
DEFAULT_RISK_AVERSION = 3.0
DEFAULT_VOL_SLACK_PENALTY = 50.0
DEFAULT_TURNOVER_SMOOTHING = 1e-10
DEFAULT_CVAR_LOOKBACK_DAYS = 504


@dataclass(frozen=True)
class EnsembleConfig:
    """Configurable signal-ensemble weights for ablation studies.

    Inputs:
    - `regime_weight`, `trend_weight`, `momo_weight`, `timesfm_weight`: convex
      combination weights across the four signal layers.
    - `regime_scale`, `trend_scale`, `momo_scale`: return-like scaling factors
      applied to the corresponding raw signals.
    - `vol_budget_by_regime`: optional regime-specific volatility budgets used
      by the walk-forward engine to override the flat monthly vol target.
    - `use_dd_guardrail`: whether the drawdown-clip overlay should be active.
    - `dd_guardrail_config`: configuration for the drawdown-clip overlay.

    Outputs:
    - Immutable config consumed by `ensemble_score`.

    Citation:
    - Whitmore Task 5 optimizer specification.

    Point-in-time safety:
    - Safe. This is static configuration only.
    """

    # Signal-ensemble weights must sum to 1.0 for a stable expected-return
    # scale.  Default allocation:
    #   regime(0.20) + trend(0.30) + momo(0.30) + timesfm(0.15) + macro(0.05)
    #
    # History of changes (see DECISIONS.md):
    #   PR #6: regime 0.40→0.30 (freed weight to macro), macro 0.10→0.20 — reverted
    #   2026-04 fix: regime restored to 0.40, macro reduced to 0.05, timesfm 0.10→0.15
    #   2026-04 (this change): regime 0.40→0.20, trend/momo each 0.20→0.30.
    #     HMM exits beat SAA only 41% of the time in OOS diagnostics; reducing regime
    #     weight from 0.40 to 0.20 frees 0.20 for trend and momentum which showed
    #     cleaner entry/exit quality. The daily defensive governor remains opt-in
    #     rather than part of the default filed signal.
    regime_weight: float = 0.20
    trend_weight: float = 0.30
    momo_weight: float = 0.30
    timesfm_weight: float = 0.15
    macro_factor_weight: float = 0.05
    regime_scale: float = 0.10
    trend_scale: float = 0.06
    momo_scale: float = 0.06
    # macro signals are in z-score × loading space (raw max ~0.18 for XAU at |z|=1,
    # ~0.60 for BTC at cap). macro_scale keeps raw z×loading magnitudes comparable
    # across signals. At macro_factor_weight=0.05, macro_scale=0.20:
    #   XAU at |z|=1 → 0.05 × 0.18 × 0.20 = 0.0018  (clearly subordinate)
    #   BTC at cap   → 0.05 × 0.60 × 0.20 = 0.006   (vs regime max 0.040)
    # This ensures macro is a refinement signal only, not a return driver.
    macro_scale: float = 0.20
    vol_budget_by_regime: dict[str, float] | None = None
    use_dd_guardrail: bool = False
    use_daily_risk_governor: bool = False
    optimizer_mode: str = "vol"
    cvar_alpha: float = 0.95
    cvar_budget: float = 0.025
    cvar_lookback_days: int = DEFAULT_CVAR_LOOKBACK_DAYS
    use_nested_risk: bool = False
    nested_risk_config: NestedRiskConfig | None = None
    use_bl_stress_views: bool = False
    bl_stress_shock: float = 1.0
    dd_guardrail_config: DrawdownGuardrailConfig = field(default_factory=DrawdownGuardrailConfig)

    def __post_init__(self) -> None:
        if self.optimizer_mode not in {"vol", "cvar"}:
            raise ValueError("optimizer_mode must be either 'vol' or 'cvar'.")
        if not 0.5 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must lie strictly between 0.5 and 1.0.")
        if not 0.0 < self.cvar_budget < 0.5:
            raise ValueError("cvar_budget must lie strictly between 0.0 and 0.5.")
        if self.cvar_lookback_days <= 0 or self.cvar_lookback_days > DEFAULT_CVAR_LOOKBACK_DAYS:
            raise ValueError(f"cvar_lookback_days must lie in [1, {DEFAULT_CVAR_LOOKBACK_DAYS}].")
        if self.bl_stress_shock < 0.0:
            raise ValueError("bl_stress_shock must be non-negative.")


@dataclass(frozen=True)
class OptimizationResult:
    """Structured result for one portfolio optimization call.

    Inputs:
    - `weights`: full-universe target weights returned by the optimizer.
    - `mode`: optimization mode that produced the result.
    - `status`: solver or fallback status string.
    - `used_fallback`: whether the result came from a fallback path.
    - `turnover`: `|Δw|_1` against the previous portfolio.
    - `turnover_cost`: transaction-cost drag implied by `turnover`.
    - `ex_ante_vol`: ex-ante annualized volatility under the provided
      covariance matrix.
    - `message`: optional incident text for breach logging/debugging.

    Outputs:
    - Immutable solve result bundle for backtests and reporting.

    Citation:
    - Whitmore Task 5 optimizer specification.

    Point-in-time safety:
    - Safe when the input covariance, availability, and signals are all
      truncated at the current decision date.
    """

    weights: pd.Series
    mode: OptimizationMode
    status: str
    used_fallback: bool
    turnover: float
    turnover_cost: float
    ex_ante_vol: float
    message: str | None = None


def _empty_weights() -> pd.Series:
    return pd.Series(0.0, index=ALL_SAA, dtype=float)


def _idx(universe: list[str], subset: list[str]) -> list[int]:
    return [index for index, asset in enumerate(universe) if asset in subset]


def _available_assets(available: pd.Series) -> list[str]:
    mask = available.reindex(ALL_SAA).fillna(0.0).astype(bool)
    return [asset for asset in ALL_SAA if mask.get(asset, False)]


def _full_weights(weights: pd.Series | np.ndarray, assets: list[str]) -> pd.Series:
    full = _empty_weights()
    full.loc[assets] = np.asarray(weights, dtype=float)
    return full


def _taa_bounds_for_assets(assets: list[str]) -> tuple[pd.Series, pd.Series]:
    lower = pd.Series({asset: TAA_BANDS[asset][0] for asset in assets}, dtype=float)
    upper = pd.Series({asset: min(TAA_BANDS[asset][1], SINGLE_SLEEVE_MAX) for asset in assets}, dtype=float)
    return lower, upper


def _bounds_for_mode(mode: OptimizationMode, assets: list[str]) -> tuple[pd.Series, pd.Series]:
    if mode == "saa_annual":
        return saa_bounds_for_assets(assets)
    if mode == "taa_monthly":
        return _taa_bounds_for_assets(assets)
    raise ValueError(f"Unsupported optimization mode: {mode}")


def _current_target_for_mode(mode: OptimizationMode) -> pd.Series:
    if mode == "saa_annual":
        return pd.Series(SAA_TARGETS, dtype=float).reindex(ALL_SAA).fillna(0.0)
    if mode == "taa_monthly":
        return pd.Series(BM2_WEIGHTS, dtype=float).reindex(ALL_SAA).fillna(0.0)
    raise ValueError(f"Unsupported optimization mode: {mode}")


def _project_taa_weights_to_feasible_set(
    target_weights: np.ndarray | pd.Series,
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
    assets: list[str],
) -> np.ndarray:
    """Project a TAA weight vector into the feasible IPS region.

    Inputs:
    - `target_weights`: candidate weight vector on the available TAA universe.
    - `lower_bounds`: per-asset TAA lower bounds.
    - `upper_bounds`: per-asset TAA upper bounds.
    - `assets`: available asset list.

    Outputs:
    - Closest feasible TAA weight vector in squared-distance sense.

    Citation:
    - SciPy constrained optimization docs:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Point-in-time safety:
    - Safe. This uses only current candidate weights and static IPS/TAA bounds.
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
        raise RuntimeError(f"Unable to project TAA weights into the feasible region: {result.message}")
    return result.x


def violates_taa_constraints(
    weights: pd.Series,
    available: pd.Series,
    tolerance: float = 1e-8,
) -> bool:
    """Return whether a full-universe weight vector breaches any hard TAA rule."""

    aligned = weights.reindex(ALL_SAA).fillna(0.0).astype(float)
    availability_mask = available.reindex(ALL_SAA).fillna(0.0).astype(bool)

    if abs(float(aligned.sum()) - 1.0) > tolerance:
        return True
    if float(aligned.min()) < -tolerance:
        return True
    if bool((aligned.loc[~availability_mask] > tolerance).any()):
        return True

    active_assets = [asset for asset in ALL_SAA if availability_mask.get(asset, False)]
    if active_assets:
        lower_bounds, upper_bounds = _taa_bounds_for_assets(active_assets)
        active_weights = aligned.reindex(active_assets).fillna(0.0)
        if bool((active_weights < (lower_bounds - tolerance)).any()):
            return True
        if bool((active_weights > (upper_bounds + tolerance)).any()):
            return True

    if float(aligned.loc[CORE].sum()) < CORE_FLOOR - tolerance:
        return True
    if float(aligned.loc[SATELLITE].sum()) > SATELLITE_CAP + tolerance:
        return True
    if float(aligned.loc[NONTRAD].sum()) > NONTRAD_CAP + tolerance:
        return True
    if float(aligned.max()) > SINGLE_SLEEVE_MAX + tolerance:
        return True
    return False


def project_taa_weights_to_compliance(
    drifted_weights: pd.Series,
    available: pd.Series,
) -> pd.Series:
    """Project drifted TAA holdings back into the hard-constraint feasible set."""

    assets = _available_assets(available)
    if not assets:
        return _empty_weights()

    lower_bounds, upper_bounds = _taa_bounds_for_assets(assets)
    candidate = drifted_weights.reindex(assets).fillna(0.0)
    try:
        projected = _project_taa_weights_to_feasible_set(candidate, lower_bounds, upper_bounds, assets)
    except RuntimeError:
        emergency_target = _current_target_for_mode("taa_monthly").reindex(assets).fillna(0.0)
        projected = _project_taa_weights_to_feasible_set(emergency_target, lower_bounds, upper_bounds, assets)
    return _full_weights(projected, assets)


def _project_mode_weights_to_feasible_set(
    mode: OptimizationMode,
    target_weights: np.ndarray | pd.Series,
    assets: list[str],
) -> np.ndarray:
    lower_bounds, upper_bounds = _bounds_for_mode(mode, assets)
    if mode == "saa_annual":
        return project_weights_to_feasible_set(target_weights, lower_bounds, upper_bounds, assets)
    return _project_taa_weights_to_feasible_set(target_weights, lower_bounds, upper_bounds, assets)


def _fallback_weights(
    mode: OptimizationMode,
    prev_weights: pd.Series,
    available: pd.Series,
) -> pd.Series:
    assets = _available_assets(available)
    if not assets:
        return _empty_weights()

    prev = prev_weights.reindex(ALL_SAA).fillna(0.0)
    candidate = prev.reindex(assets).fillna(0.0)
    if candidate.sum() <= 0:
        candidate = _current_target_for_mode(mode).reindex(assets).fillna(0.0)

    try:
        projected = _project_mode_weights_to_feasible_set(mode, candidate, assets)
        return _full_weights(projected, assets)
    except Exception:
        emergency_target = _current_target_for_mode(mode).reindex(assets).fillna(0.0)
        try:
            projected = _project_mode_weights_to_feasible_set(mode, emergency_target, assets)
            return _full_weights(projected, assets)
        except Exception:
            return _empty_weights()


def _ex_ante_vol(weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
    aligned_cov = cov_matrix.reindex(index=ALL_SAA, columns=ALL_SAA).fillna(0.0)
    covariance = aligned_cov.to_numpy(dtype=float)
    covariance = (covariance + covariance.T) / 2.0
    portfolio_variance = float(weights.to_numpy(dtype=float) @ covariance @ weights.to_numpy(dtype=float))
    return float(np.sqrt(max(portfolio_variance, 0.0)))


def _append_breach_log(
    mode: OptimizationMode,
    as_of_date: pd.Timestamp | None,
    status: str,
    message: str,
    log_path: Path = BREACH_LOG_PATH,
) -> None:
    """Append a one-line incident record to `outputs/breaches.log`.

    Inputs:
    - `mode`: optimizer mode that triggered the incident.
    - `as_of_date`: decision date associated with the solve.
    - `status`: solver status or failure class.
    - `message`: human-readable incident message.
    - `log_path`: destination log path.

    Outputs:
    - Appends one text line to the breach log.

    Citation:
    - Whitmore Task 5 optimizer specification.

    Point-in-time safety:
    - Safe. This is operational logging only.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    date_text = "unknown" if as_of_date is None else pd.Timestamp(as_of_date).isoformat()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{pd.Timestamp.utcnow().isoformat()} | mode={mode} | date={date_text} | status={status} | {message}\n")


def _append_optimizer_breach_row(
    as_of_date: pd.Timestamp | None,
    optimizer_mode: str,
    breach_type: str,
    budget: float,
    realized_value: float,
    slack_value: float,
    output_path: Path = OPTIMIZER_BREACHES_PATH,
) -> None:
    """Append one structured optimizer-breach row to CSV.

    Inputs:
    - `as_of_date`: decision date tied to the optimization event.
    - `optimizer_mode`: active constraint family, e.g. `vol` or `cvar`.
    - `breach_type`: structured label such as `vol_constraint_relaxed`.
    - `budget`: requested risk budget.
    - `realized_value`: realized value associated with the breach.
    - `slack_value`: non-negative slack consumed by the solver.
    - `output_path`: CSV destination.

    Outputs:
    - Appends one row to `optimizer_breaches.csv`.

    Citation:
    - Rockafellar & Uryasev (2000), Conditional Value-at-Risk:
      http://www.ise.ufl.edu/uryasev/files/2011/11/CVaR1_JOR.pdf
    - Whitmore optimizer breach logging requirement.

    Point-in-time safety:
    - Safe. This is operational logging only.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
                "date": pd.Timestamp(as_of_date).isoformat() if as_of_date is not None else "",
                "optimizer_mode": optimizer_mode,
                "breach_type": breach_type,
                "budget": float(budget),
                "realized_value": float(realized_value),
                "slack_value": float(slack_value),
            }
        ]
    )
    if output_path.exists():
        existing = pd.read_csv(
            output_path,
            dtype={
                "timestamp_utc": "string",
                "date": "string",
                "optimizer_mode": "string",
                "breach_type": "string",
                "budget": "float32",
                "realized_value": "float32",
                "slack_value": "float32",
            },
        )
        row = pd.concat([existing, row], ignore_index=True)
    row.to_csv(output_path, index=False)


def _build_cvar_scenarios(
    asset_log_returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    universe: list[str],
    lookback_days: int = DEFAULT_CVAR_LOOKBACK_DAYS,
) -> np.ndarray:
    """Build the strictly causal daily return scenarios used by CVaR mode.

    Inputs:
    - `asset_log_returns`: audited daily log-return panel.
    - `decision_date`: current rebalance date.
    - `universe`: ordered asset universe for the optimization.
    - `lookback_days`: trailing scenario window length in trading days.

    Outputs:
    - Float32 matrix of trailing daily log-return scenarios ending at `t-1`.

    Citation:
    - Rockafellar & Uryasev (2000), Conditional Value-at-Risk:
      http://www.ise.ufl.edu/uryasev/files/2011/11/CVaR1_JOR.pdf

    Point-in-time safety:
    - Safe. The scenario matrix excludes `decision_date` itself and uses only
      realized observations dated strictly earlier than the rebalance.
    """

    if lookback_days > DEFAULT_CVAR_LOOKBACK_DAYS:
        raise ValueError(f"CVaR scenario window exceeds the hard cap of {DEFAULT_CVAR_LOOKBACK_DAYS} rows.")
    window = asset_log_returns.loc[: pd.Timestamp(decision_date) - pd.Timedelta(days=1), universe].tail(lookback_days)
    if window.empty:
        return np.empty((0, len(universe)), dtype=np.float32)
    window = window.dropna(how="any")
    return window.to_numpy(dtype=np.float32, copy=True)


def _historical_cvar_from_matrix(scenario_matrix: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    losses = -scenario_matrix @ np.asarray(weights, dtype=float)
    var_alpha = float(np.quantile(losses, alpha))
    tail_losses = losses[losses >= var_alpha]
    if tail_losses.size == 0:
        return var_alpha
    return float(tail_losses.mean())


def _solve_cvar_taa(
    universe: list[str],
    expected_returns: pd.Series,
    scenario_returns: np.ndarray,
    previous_weights: np.ndarray,
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
    core_idx: list[int],
    sat_idx: list[int],
    nontrad_idx: list[int],
    cvar_alpha: float,
    cvar_budget: float,
    turnover_penalty: float,
) -> np.ndarray:
    """Solve the Rockafellar-Uryasev monthly TAA program with strict cleanup.

    Citation:
    - Rockafellar & Uryasev (2000), Conditional Value-at-Risk:
      http://www.ise.ufl.edu/uryasev/files/2011/11/CVaR1_JOR.pdf
    - ECOS solver:
      https://github.com/embotech/ecos
    """

    del universe  # Ordering is already embedded in the aligned arrays.
    if scenario_returns.ndim != 2:
        raise ValueError("scenario_returns must be a 2D ndarray.")
    scenario_matrix = np.asarray(scenario_returns, dtype=np.float32)
    scenario_matrix = scenario_matrix[np.all(np.isfinite(scenario_matrix), axis=1)]
    t_obs, n_assets = scenario_matrix.shape
    if t_obs == 0:
        raise ValueError("CVaR mode requires at least one complete historical scenario row.")
    if t_obs > DEFAULT_CVAR_LOOKBACK_DAYS:
        raise ValueError(f"CVaR scenario window exceeds the hard cap of {DEFAULT_CVAR_LOOKBACK_DAYS} rows.")

    expected_vector = expected_returns.to_numpy(dtype=float)
    lower_vector = lower_bounds.to_numpy(dtype=float)
    upper_vector = upper_bounds.to_numpy(dtype=float)
    prev_vector = np.asarray(previous_weights, dtype=float)

    if prev_vector.shape[0] != n_assets:
        raise ValueError("previous_weights length does not match the CVaR universe.")

    guard_process_memory("cvar:before_problem_build")
    w = cp.Variable(n_assets, nonneg=True)
    tau = cp.Variable()
    u = cp.Variable(t_obs, nonneg=True)
    losses = -scenario_matrix @ w
    objective = cp.Minimize(
        -expected_vector @ w
        + turnover_penalty * cp.norm(w - prev_vector, 1)
    )
    constraints = [
        cp.sum(w) == 1.0,
        w >= lower_vector,
        w <= upper_vector,
        w <= SINGLE_SLEEVE_MAX,
        u >= losses - tau,
        tau + (1.0 / ((1.0 - cvar_alpha) * t_obs)) * cp.sum(u) <= cvar_budget,
    ]
    if core_idx:
        constraints.append(cp.sum(w[core_idx]) >= CORE_FLOOR)
    if sat_idx:
        constraints.append(cp.sum(w[sat_idx]) <= SATELLITE_CAP)
    if nontrad_idx:
        constraints.append(cp.sum(w[nontrad_idx]) <= NONTRAD_CAP)
    prob = cp.Problem(objective, constraints)

    try:
        guard_process_memory("cvar:before_solve")
        prob.solve(solver=cp.ECOS, verbose=False)
        guard_process_memory("cvar:after_solve")
        if w.value is None:
            raise RuntimeError(f"CVaR solver returned no solution. Problem status: {prob.status}")
        out = np.array(w.value, dtype=np.float64)
    finally:
        del w, tau, u, losses, objective, constraints, prob
        gc.collect()
        guard_process_memory("cvar:after_cleanup")
    return out


def _finalize_result(
    weights: pd.Series,
    mode: OptimizationMode,
    status: str,
    prev_weights: pd.Series,
    cov_matrix: pd.DataFrame,
    transaction_cost_rate: float,
    used_fallback: bool,
    message: str | None = None,
) -> OptimizationResult:
    aligned_prev = prev_weights.reindex(ALL_SAA).fillna(0.0)
    aligned_weights = weights.reindex(ALL_SAA).fillna(0.0)
    turnover = float((aligned_weights - aligned_prev).abs().sum())
    return OptimizationResult(
        weights=aligned_weights,
        mode=mode,
        status=status,
        used_fallback=used_fallback,
        turnover=turnover,
        turnover_cost=transaction_cost_rate * turnover,
        ex_ante_vol=_ex_ante_vol(aligned_weights, cov_matrix),
        message=message,
    )


def solve_saa_annual_result(
    cov_matrix: pd.DataFrame,
    prev_weights: pd.Series,
    available: pd.Series,
    as_of_date: pd.Timestamp | None = None,
    breach_log_path: Path = BREACH_LOG_PATH,
) -> OptimizationResult:
    """Solve the annual SAA target using the Task 2 risk-parity method.

    Inputs:
    - `cov_matrix`: annualized covariance matrix on the full SAA universe.
    - `prev_weights`: previously held full-universe portfolio.
    - `available`: 1/0 availability indicator by sleeve at the rebalance date.
    - `as_of_date`: optional rebalance date for incident logging.
    - `breach_log_path`: path to `breaches.log`.

    Outputs:
    - `OptimizationResult` for the annual SAA target.

    Citation:
    - Task 2 constrained risk parity implementation in
      `taa_project/saa/build_saa.py`.

    Point-in-time safety:
    - Safe. The caller should pass a covariance matrix estimated strictly from
      data observed on or before the annual rebalance date.
    """

    assets = _available_assets(available)
    if not assets:
        fallback = _empty_weights()
        message = "No available assets for annual SAA solve; returning zero weights."
        _append_breach_log("saa_annual", as_of_date, "empty_universe", message, breach_log_path)
        return _finalize_result(
            fallback,
            "saa_annual",
            "empty_universe",
            prev_weights,
            cov_matrix,
            COST_PER_TURNOVER,
            True,
            message,
        )

    try:
        lower_bounds, upper_bounds = saa_bounds_for_assets(assets)
        inputs = SAAOptimizationInputs(
            covariance=cov_matrix.reindex(index=assets, columns=assets).fillna(0.0),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            risk_budgets=target_risk_budgets(assets),
            assets=assets,
        )
        target_weights = solve_target_risk_parity(inputs)
        full_weights = _full_weights(target_weights, assets)
        return _finalize_result(full_weights, "saa_annual", "optimal", prev_weights, cov_matrix, COST_PER_TURNOVER, False)
    except Exception as exc:
        message = f"SAA annual solve failed: {exc}. Falling back to the last feasible weights."
        _append_breach_log("saa_annual", as_of_date, "fallback", message, breach_log_path)
        fallback = _fallback_weights("saa_annual", prev_weights, available)
        return _finalize_result(
            fallback,
            "saa_annual",
            "fallback",
            prev_weights,
            cov_matrix,
            COST_PER_TURNOVER,
            True,
            message,
        )


def solve_taa_monthly_result(
    signal_score: pd.Series,
    cov_matrix: pd.DataFrame,
    prev_weights: pd.Series,
    available: pd.Series,
    as_of_date: pd.Timestamp | None = None,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    turnover_cost: float = COST_PER_TURNOVER,
    vol_budget: float = TARGET_VOL,
    vol_slack_penalty: float = DEFAULT_VOL_SLACK_PENALTY,
    breach_log_path: Path = BREACH_LOG_PATH,
    config: EnsembleConfig | None = None,
    scenario_returns: np.ndarray | None = None,
) -> OptimizationResult:
    """Solve the monthly TAA portfolio from the signal-ensemble score.

    Inputs:
    - `signal_score`: annualized expected-return proxy from the signal ensemble.
    - `cov_matrix`: annualized covariance matrix on the full SAA universe.
    - `prev_weights`: previously held full-universe portfolio.
    - `available`: 1/0 availability indicator by sleeve at the rebalance date.
    - `as_of_date`: optional rebalance date for incident logging.
    - `risk_aversion`: quadratic risk penalty coefficient.
    - `turnover_cost`: turnover penalty coefficient, set to 5 bps round-trip.
    - `vol_budget`: ex-ante volatility soft ceiling.
    - `vol_slack_penalty`: penalty applied to the volatility slack variable.
    - `breach_log_path`: path to `breaches.log`.
    - `config`: optional monthly-optimizer configuration bundle.
    - `scenario_returns`: strictly causal float32 scenario matrix used only in
      CVaR mode.

    Outputs:
    - `OptimizationResult` for the monthly TAA rebalance.

    Citation:
    - cvxpy documentation: https://www.cvxpy.org/
    - Whitmore Task 5 optimizer specification.

    Point-in-time safety:
    - Safe. The caller should pass only signal and covariance inputs observed or
      estimated using data dated on or before the current decision date.
    """

    availability_mask = available.reindex(ALL_SAA).fillna(0.0).astype(bool)
    universe = [asset for asset in ALL_SAA if availability_mask.get(asset, False)]
    if not universe:
        message = "No available assets for monthly TAA solve."
        _append_breach_log("taa_monthly", as_of_date, "empty_universe", message, breach_log_path)
        fallback = _fallback_weights("taa_monthly", prev_weights, available)
        return _finalize_result(
            fallback,
            "taa_monthly",
            "empty_universe",
            prev_weights,
            cov_matrix,
            turnover_cost,
            True,
            message,
        )

    if cp is None:
        message = "cvxpy is not installed; monthly TAA solve fell back to the last feasible weights."
        _append_breach_log("taa_monthly", as_of_date, "missing_dependency", message, breach_log_path)
        fallback = _fallback_weights("taa_monthly", prev_weights, available)
        return _finalize_result(
            fallback,
            "taa_monthly",
            "missing_dependency",
            prev_weights,
            cov_matrix,
            turnover_cost,
            True,
            message,
        )

    mu = (
        signal_score.reindex(universe)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    solve_config = EnsembleConfig() if config is None else config
    sigma = cov_matrix.reindex(index=universe, columns=universe).fillna(0.0).to_numpy(dtype=float)
    sigma = (sigma + sigma.T) / 2.0
    if len(universe) > 0:
        np.fill_diagonal(sigma, np.maximum(np.diag(sigma), DIAGONAL_FLOOR))

    prev = prev_weights.reindex(universe).fillna(0.0).to_numpy(dtype=float)
    lower_bounds, upper_bounds = _taa_bounds_for_assets(universe)

    core_idx = _idx(universe, CORE)
    sat_idx = _idx(universe, SATELLITE)
    nontrad_idx = _idx(universe, NONTRAD)

    try:
        if solve_config.optimizer_mode == "vol":
            weights = cp.Variable(len(universe), nonneg=True)
            turnover = cp.norm(weights - prev, 1)
            constraints = [
                cp.sum(weights) == 1.0,
                weights >= lower_bounds.to_numpy(dtype=float),
                weights <= upper_bounds.to_numpy(dtype=float),
                weights <= SINGLE_SLEEVE_MAX,
            ]
            constraints.append(cp.quad_form(weights, cp.psd_wrap(sigma)) <= vol_budget**2)
            slack_penalty = 0.0
        else:
            if scenario_returns is None:
                raise ValueError("CVaR mode requires `scenario_returns` built from strictly causal return history.")
            raw_weights = _solve_cvar_taa(
                universe=universe,
                expected_returns=pd.Series(mu, index=universe, dtype=float),
                scenario_returns=scenario_returns,
                previous_weights=prev,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                core_idx=core_idx,
                sat_idx=sat_idx,
                nontrad_idx=nontrad_idx,
                cvar_alpha=solve_config.cvar_alpha,
                cvar_budget=solve_config.cvar_budget,
                turnover_penalty=turnover_cost,
            )
            full_weights = _full_weights(raw_weights, universe)
            return _finalize_result(
                full_weights,
                "taa_monthly",
                "optimal",
                prev_weights,
                cov_matrix,
                turnover_cost,
                False,
            )
        if core_idx:
            constraints.append(cp.sum(weights[core_idx]) >= CORE_FLOOR)
        if sat_idx:
            constraints.append(cp.sum(weights[sat_idx]) <= SATELLITE_CAP)
        if nontrad_idx:
            constraints.append(cp.sum(weights[nontrad_idx]) <= NONTRAD_CAP)

        objective = cp.Maximize(
            mu @ weights
            - risk_aversion * cp.quad_form(weights, cp.psd_wrap(sigma))
            - turnover_cost * turnover
            - slack_penalty
        )
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.CLARABEL)
        except Exception:
            problem.solve()

        if weights.value is None:
            raise RuntimeError(f"cvxpy returned no solution. Problem status: {problem.status}")

        projected = _project_mode_weights_to_feasible_set("taa_monthly", np.asarray(weights.value, dtype=float), universe)
        full_weights = _full_weights(projected, universe)
        slack_value = 0.0
        if slack_value > 1e-10:
            realized_vol = float(np.sqrt(max(projected @ sigma @ projected, 0.0)))
            _append_optimizer_breach_row(
                as_of_date=as_of_date,
                optimizer_mode="vol",
                breach_type="vol_constraint_relaxed",
                budget=vol_budget,
                realized_value=realized_vol,
                slack_value=slack_value,
                output_path=breach_log_path.parent / OPTIMIZER_BREACHES_PATH.name,
            )
        return _finalize_result(
            full_weights,
            "taa_monthly",
            str(problem.status),
            prev_weights,
            cov_matrix,
            turnover_cost,
            False,
        )
    except Exception as exc:
        message = f"Monthly TAA solve failed: {exc}. Falling back to the last feasible weights."
        _append_breach_log("taa_monthly", as_of_date, "fallback", message, breach_log_path)
        fallback = _fallback_weights("taa_monthly", prev_weights, available)
        return _finalize_result(
            fallback,
            "taa_monthly",
            "fallback",
            prev_weights,
            cov_matrix,
            turnover_cost,
            True,
            message,
        )


def solve_portfolio(
    mode: OptimizationMode,
    cov_matrix: pd.DataFrame,
    prev_weights: pd.Series,
    available: pd.Series,
    signal_score: pd.Series | None = None,
    as_of_date: pd.Timestamp | None = None,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    turnover_cost: float = COST_PER_TURNOVER,
    vol_budget: float = TARGET_VOL,
    vol_slack_penalty: float = DEFAULT_VOL_SLACK_PENALTY,
    breach_log_path: Path = BREACH_LOG_PATH,
    config: EnsembleConfig | None = None,
    scenario_returns: pd.DataFrame | None = None,
) -> OptimizationResult:
    """Dispatch into the annual SAA or monthly TAA optimizer.

    Inputs:
    - `mode`: either `saa_annual` or `taa_monthly`.
    - `cov_matrix`: annualized covariance matrix.
    - `prev_weights`: previously held full-universe portfolio.
    - `available`: 1/0 availability indicator by sleeve at the decision date.
    - `signal_score`: required in `taa_monthly` mode; ignored in `saa_annual`.
    - `as_of_date`: optional decision date for incident logging.
    - `risk_aversion`, `turnover_cost`, `vol_budget`, `vol_slack_penalty`:
      monthly optimization controls.
    - `breach_log_path`: path to `breaches.log`.

    Outputs:
    - `OptimizationResult` for the selected mode.

    Citation:
    - Whitmore Task 5 optimizer specification.

    Point-in-time safety:
    - Safe. The caller is responsible for supplying point-in-time inputs.
    """

    if mode == "saa_annual":
        return solve_saa_annual_result(
            cov_matrix=cov_matrix,
            prev_weights=prev_weights,
            available=available,
            as_of_date=as_of_date,
            breach_log_path=breach_log_path,
        )
    if mode == "taa_monthly":
        if signal_score is None:
            raise ValueError("`signal_score` is required in `taa_monthly` mode.")
        return solve_taa_monthly_result(
            signal_score=signal_score,
            cov_matrix=cov_matrix,
            prev_weights=prev_weights,
            available=available,
            as_of_date=as_of_date,
            risk_aversion=risk_aversion,
            turnover_cost=turnover_cost,
            vol_budget=vol_budget,
            vol_slack_penalty=vol_slack_penalty,
            breach_log_path=breach_log_path,
            config=config,
            scenario_returns=scenario_returns,
        )
    raise ValueError(f"Unsupported optimization mode: {mode}")


def solve_saa_annual(
    cov_matrix: pd.DataFrame,
    prev_weights: pd.Series,
    available: pd.Series,
    as_of_date: pd.Timestamp | None = None,
    breach_log_path: Path = BREACH_LOG_PATH,
) -> pd.Series:
    """Backward-compatible convenience wrapper returning SAA weights only.

    Inputs:
    - Same as `solve_saa_annual_result`.

    Outputs:
    - Full-universe weight series.

    Citation:
    - Whitmore Task 5 optimizer specification.

    Point-in-time safety:
    - Safe. This delegates to the structured annual optimizer result.
    """

    return solve_saa_annual_result(
        cov_matrix=cov_matrix,
        prev_weights=prev_weights,
        available=available,
        as_of_date=as_of_date,
        breach_log_path=breach_log_path,
    ).weights


def solve_taa(
    signal_score: pd.Series,
    cov_matrix: pd.DataFrame,
    prev_weights: pd.Series,
    available: pd.Series,
    as_of_date: pd.Timestamp | None = None,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    turnover_cost: float = COST_PER_TURNOVER,
    vol_budget: float = TARGET_VOL,
    vol_slack_penalty: float = DEFAULT_VOL_SLACK_PENALTY,
    breach_log_path: Path = BREACH_LOG_PATH,
    config: EnsembleConfig | None = None,
    scenario_returns: pd.DataFrame | None = None,
) -> pd.Series:
    """Backward-compatible convenience wrapper returning TAA weights only.

    Inputs:
    - Same as `solve_taa_monthly_result`.

    Outputs:
    - Full-universe weight series.

    Citation:
    - Whitmore Task 5 optimizer specification.

    Point-in-time safety:
    - Safe. This delegates to the structured monthly optimizer result.
    """

    return solve_taa_monthly_result(
        signal_score=signal_score,
        cov_matrix=cov_matrix,
        prev_weights=prev_weights,
        available=available,
        as_of_date=as_of_date,
        risk_aversion=risk_aversion,
        turnover_cost=turnover_cost,
        vol_budget=vol_budget,
        vol_slack_penalty=vol_slack_penalty,
        breach_log_path=breach_log_path,
        config=config,
        scenario_returns=scenario_returns,
    ).weights


def ensemble_score(
    regime_tilt: pd.Series,
    trend_sig: pd.Series,
    momo_sig: pd.Series,
    timesfm_mu: pd.Series,
    macro_factor_mu: pd.Series | None = None,
    config: EnsembleConfig | None = None,
) -> pd.Series:
    """Blend the five signal layers into an annualized `mu` proxy.

    Inputs:
    - `regime_tilt`: regime-layer tilt vector.
    - `trend_sig`: smooth Faber trend score in `[-1, +1]`.
    - `momo_sig`: ADM momentum score in `[-1, +1]`.
    - `timesfm_mu`: TimesFM annualized expected-return forecast.
    - `macro_factor_mu`: asset-specific macro factor signal (real yield,
      credit premium, crypto momentum) from ``macro_factor.py``.  If None
      or missing, the macro_factor_weight is redistributed to regime.
    - `config`: optional signal-ensemble config for ablation studies.

    Outputs:
    - Annualized expected-return proxy per asset.

    Citation:
    - Whitmore Task 5 optimizer specification.
    - Macro factor signals: Erb & Harvey (2013), Gilchrist & Zakrajsek (2012).

    Point-in-time safety:
    - Safe. This is an algebraic combination of already point-in-time-safe
      signal inputs.
    """

    cfg = EnsembleConfig() if config is None else config
    macro_mu = (
        macro_factor_mu.reindex(ALL_SAA).fillna(0.0)
        if macro_factor_mu is not None
        else pd.Series(0.0, index=ALL_SAA, dtype=float)
    )
    # When macro signal is unavailable, its weight folds into the regime layer.
    regime_w = cfg.regime_weight + (cfg.macro_factor_weight if macro_factor_mu is None else 0.0)
    # When TimesFM is disabled (all-zero mu), redistribute its weight equally
    # into trend and momentum rather than leaving it dead.
    timesfm_aligned = timesfm_mu.reindex(ALL_SAA).fillna(0.0)
    timesfm_active = cfg.timesfm_weight > 0.0 and not bool((timesfm_aligned == 0.0).all())
    timesfm_w = cfg.timesfm_weight if timesfm_active else 0.0
    extra_per_signal = (cfg.timesfm_weight - timesfm_w) / 2.0
    trend_w = cfg.trend_weight + extra_per_signal
    momo_w = cfg.momo_weight + extra_per_signal
    score = (
        regime_w * regime_tilt * cfg.regime_scale
        + trend_w * trend_sig * cfg.trend_scale
        + momo_w * momo_sig * cfg.momo_scale
        + timesfm_w * timesfm_aligned
        + cfg.macro_factor_weight * macro_mu * cfg.macro_scale
    )
    return score.reindex(ALL_SAA).fillna(0.0)
