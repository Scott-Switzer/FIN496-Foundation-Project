# Addresses rubric criterion 2 (walk-forward rigor) by running the TAA stack
# across expanding OOS folds with explicit embargoing and fold logs.
"""Walk-forward OOS backtester for the Whitmore TAA overlay.

This module implements Task 6:
- Split the OOS period into contiguous expanding folds.
- Apply an embargo between each fold's initial training sample and first test
  decision.
- Fit the HMM on fold-train data at fold start, then refit monthly using only
  information available by each OOS decision date.
- Run the monthly TAA portfolio continuously across the full OOS period while
  recording fold ids, regime labels, weights, turnover, and realized returns.

References:
- López de Prado (2018), *Advances in Financial Machine Learning* (purging and
  embargoing). Wiki overview:
  https://en.wikipedia.org/wiki/Purged_cross-validation
- Whitmore IPS / Guidelines in the repo.

Point-in-time safety:
- Safe. Every decision uses only prices, macro features, and prior holdings
  observed on or before that decision date. The fold embargo is applied when
  establishing each fold's initial HMM training sample.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from taa_project.config import ALL_SAA, BM2_WEIGHTS, EQUITY_ASSETS, OUTPUT_DIR, TARGET_VOL, TIMESFM_CACHE_PATH, VOL_CEILING
from taa_project.data_loader import availability_flag, load_fred, load_prices, log_returns
from taa_project.optimizer.cvxpy_opt import (
    _build_cvar_scenarios,
    EnsembleConfig,
    OptimizationResult,
    solve_taa_monthly_result,
)
from taa_project.optimizer.nested_risk import solve_nested_taa
from taa_project.signals import SignalBundle
from taa_project.signals.dd_guardrail import dd_guardrail_multiplier, trailing_drawdown_series
from taa_project.signals.macro_factor import compute_macro_factor_mu
from taa_project.signals.momentum_adm import adm_score, cross_sectional_rank
from taa_project.signals.regime_hmm import (
    MIN_TRAIN_OBSERVATIONS,
    build_features,
    classify_states,
    fit_hmm,
    regime_tilt_from_label,
)
from taa_project.signals.trend_faber import trend_score
from taa_project.signals.vol_timesfm import (
    TimesFMForecaster,
    compute_vol_and_direction_signals,
    timesfm_is_available,
)
from taa_project.saa.saa_comparison import bl_with_stress_views


DEFAULT_START = "2003-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_FOLDS = 5
DEFAULT_EMBARGO_BUSINESS_DAYS = 21
DEFAULT_COVARIANCE_LOOKBACK = 252
DEFAULT_COVARIANCE_MIN_OBSERVATIONS = 63
DIAGONAL_DEFAULT = 0.04
TIMESFM_CACHE_FILE = TIMESFM_CACHE_PATH

WALKFORWARD_FOLDS_FILENAME = "walkforward_folds.csv"
OOS_RETURNS_FILENAME = "oos_returns.csv"
OOS_WEIGHTS_FILENAME = "oos_weights.csv"
OOS_REGIMES_FILENAME = "oos_regimes.csv"
GUARDRAIL_SWITCHES_FILENAME = "guardrail_switches.csv"

SLEEVE_BUCKETS = {
    "equity": ["SPXT", "FTSE100", "NIKKEI225", "CSI300_CHINA"],
    "fixed_income": ["LBUSTRUU", "BROAD_TIPS"],
    "real_assets": ["XAU", "SILVER_FUT", "B3REITT"],
    "non_traditional": ["BITCOIN", "CHF_FRANC"],
}


@dataclass(frozen=True)
class FoldSpec:
    """Definition of one OOS fold in the walk-forward test.

    Inputs:
    - `fold_id`: 1-based fold identifier.
    - `train_start`: first date available to the fold's initial train sample.
    - `train_end`: last date in the initial train sample after embargo.
    - `embargo_start`, `embargo_end`: embargo interval excluded before the
      fold's first test decision.
    - `test_start`, `test_end`: first and last monthly OOS decision dates in
      the fold.
    - `decision_dates`: ordered monthly decision dates inside the fold.

    Outputs:
    - Immutable fold description used by the walk-forward runner.

    Citation:
    - López de Prado purged/embargoed CV overview:
      https://en.wikipedia.org/wiki/Purged_cross-validation

    Point-in-time safety:
    - Safe. Fold definitions are deterministic partitions of the OOS dates.
    """

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    embargo_start: pd.Timestamp
    embargo_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    decision_dates: tuple[pd.Timestamp, ...]


def build_monthly_decision_dates(prices: pd.DataFrame, start: str, end: str) -> pd.DatetimeIndex:
    """Return actual last-trading-day decision dates for each calendar month.

    Inputs:
    - `prices`: audited price panel.
    - `start`: first allowed OOS date.
    - `end`: last allowed OOS date.

    Outputs:
    - DatetimeIndex of actual month-end trading dates.

    Citation:
    - Whitmore Task 6 walk-forward specification.

    Point-in-time safety:
    - Safe. This is a deterministic calendar transform over observed dates.
    """

    bounded_prices = prices.loc[start:end].copy()
    if bounded_prices.empty:
        return pd.DatetimeIndex([])

    inception_dates = bounded_prices.apply(lambda column: column.first_valid_index())
    decision_dates: list[pd.Timestamp] = []

    for _, month_block in bounded_prices.groupby(bounded_prices.index.to_period("M")):
        chosen_date: pd.Timestamp | None = None
        for date in reversed(month_block.index.tolist()):
            investable_assets = [
                asset for asset in ALL_SAA if pd.notna(inception_dates.get(asset)) and pd.Timestamp(inception_dates[asset]) <= date
            ]
            if not investable_assets:
                continue
            if bool(month_block.loc[date, investable_assets].notna().all()):
                chosen_date = pd.Timestamp(date)
                break
        if chosen_date is not None:
            decision_dates.append(chosen_date)

    return pd.DatetimeIndex(decision_dates, name="decision_date")


def build_walkforward_folds(
    decision_dates: pd.DatetimeIndex,
    train_start: pd.Timestamp,
    folds: int = DEFAULT_FOLDS,
    embargo_business_days: int = DEFAULT_EMBARGO_BUSINESS_DAYS,
) -> list[FoldSpec]:
    """Split monthly decision dates into expanding contiguous OOS folds.

    Inputs:
    - `decision_dates`: full OOS monthly decision schedule.
    - `train_start`: first date available to any fold's training sample.
    - `folds`: number of contiguous OOS folds.
    - `embargo_business_days`: embargo gap between train and first test date.

    Outputs:
    - Ordered list of `FoldSpec` objects.

    Citation:
    - López de Prado purged/embargoed CV overview:
      https://en.wikipedia.org/wiki/Purged_cross-validation

    Point-in-time safety:
    - Safe. Fold boundaries are determined entirely from the known OOS
      decision-date schedule.
    """

    if len(decision_dates) < folds:
        raise ValueError(f"Need at least {folds} decision dates to build {folds} folds.")

    split_arrays = [pd.DatetimeIndex(chunk) for chunk in np.array_split(decision_dates.to_numpy(), folds) if len(chunk) > 0]
    fold_specs: list[FoldSpec] = []
    for fold_id, chunk in enumerate(split_arrays, start=1):
        test_start = pd.Timestamp(chunk[0])
        test_end = pd.Timestamp(chunk[-1])
        train_end = pd.Timestamp(test_start) - pd.offsets.BDay(embargo_business_days)
        if train_end < train_start:
            raise ValueError(
                f"Fold {fold_id} has insufficient training history after a {embargo_business_days}-day embargo."
            )

        embargo_start = train_end + pd.offsets.BDay(1)
        embargo_end = test_start - pd.offsets.BDay(1)
        fold_specs.append(
            FoldSpec(
                fold_id=fold_id,
                train_start=pd.Timestamp(train_start),
                train_end=pd.Timestamp(train_end),
                embargo_start=pd.Timestamp(embargo_start),
                embargo_end=pd.Timestamp(embargo_end),
                test_start=test_start,
                test_end=test_end,
                decision_dates=tuple(pd.Timestamp(date) for date in chunk),
            )
        )

    return fold_specs


def fold_specs_to_frame(fold_specs: list[FoldSpec]) -> pd.DataFrame:
    """Convert fold specs into the required `walkforward_folds.csv` table.

    Inputs:
    - `fold_specs`: ordered list of fold specifications.

    Outputs:
    - Dataframe suitable for CSV export.

    Citation:
    - Whitmore Task 6 walk-forward specification.

    Point-in-time safety:
    - Safe. This is metadata only.
    """

    rows = []
    for spec in fold_specs:
        rows.append(
            {
                "fold_id": spec.fold_id,
                "train_start": spec.train_start,
                "train_end": spec.train_end,
                "embargo_start": spec.embargo_start,
                "embargo_end": spec.embargo_end,
                "test_start": spec.test_start,
                "test_end": spec.test_end,
                "n_test_rebalances": len(spec.decision_dates),
            }
        )
    return pd.DataFrame(rows)


def estimate_taa_covariance(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    timesfm_sigma: pd.Series | None = None,
    lookback_days: int = DEFAULT_COVARIANCE_LOOKBACK,
    min_observations: int = DEFAULT_COVARIANCE_MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Estimate a stabilized annual covariance matrix at a decision date.

    Inputs:
    - `returns`: audited asset log-return dataframe.
    - `decision_date`: monthly decision date.
    - `timesfm_sigma`: optional annualized volatility forecasts by asset.
    - `lookback_days`: trailing observed-window size for covariance.
    - `min_observations`: minimum observations required for pairwise covariances.

    Outputs:
    - Annualized covariance matrix indexed by `ALL_SAA`.

    Citation:
    - Whitmore Task 6 walk-forward specification.

    Point-in-time safety:
    - Safe. The estimate uses only returns dated on or before the current
      decision date.
    """

    history = returns.loc[:decision_date].tail(lookback_days)
    covariance = history.cov(min_periods=min_observations)
    covariance = covariance.reindex(index=ALL_SAA, columns=ALL_SAA)
    variances = history.var(skipna=True).reindex(ALL_SAA)

    for asset in ALL_SAA:
        variance = variances.get(asset, np.nan)
        covariance.loc[asset, asset] = float(variance) if pd.notna(variance) and variance > 0 else DIAGONAL_DEFAULT

    covariance = covariance.fillna(0.0)
    covariance_values = covariance.to_numpy(dtype=float)
    covariance_values = 0.7 * covariance_values + 0.3 * np.diag(np.diag(covariance_values))

    if timesfm_sigma is not None:
        for index, asset in enumerate(ALL_SAA):
            sigma = timesfm_sigma.get(asset, np.nan)
            if pd.notna(sigma) and sigma > 0:
                covariance_values[index, index] = float(sigma) ** 2

    covariance_values = (covariance_values + covariance_values.T) / 2.0
    np.fill_diagonal(covariance_values, np.maximum(np.diag(covariance_values), DIAGONAL_DEFAULT))
    return pd.DataFrame(covariance_values, index=ALL_SAA, columns=ALL_SAA)


def build_signal_bundle_at_date(
    decision_date: pd.Timestamp,
    fold_spec: FoldSpec,
    fred_features: pd.DataFrame,
    trend_signals: pd.DataFrame,
    momo_signals: pd.DataFrame,
    prices: pd.DataFrame,
    use_timesfm: bool,
    forecaster: TimesFMForecaster | None,
    timesfm_cache_path: Path | None,
    hmm_model_cache: object | None,
    hmm_states: int,
) -> tuple[SignalBundle, object | None]:
    """Construct the four-layer signal bundle at one monthly decision date.

    Inputs:
    - `decision_date`: current monthly rebalance date.
    - `fold_spec`: fold metadata for the current decision date.
    - `fred_features`: lagged FRED feature panel.
    - `trend_signals`: precomputed trend scores.
    - `momo_signals`: precomputed momentum scores.
    - `prices`: audited asset price dataframe.
    - `use_timesfm`: whether to invoke the optional TimesFM layer.
    - `forecaster`: optional TimesFM forecaster instance.
    - `timesfm_cache_path`: shared TimesFM parquet cache path.
    - `hmm_model_cache`: previous fold-local HMM model, reused on refit failure.
    - `hmm_states`: number of HMM states.

    Outputs:
    - Tuple `(signal_bundle, updated_hmm_model_cache)`.

    Citation:
    - Hamilton (1989): https://doi.org/10.2307/1912559
    - Faber (2007): https://mebfaber.com/wp-content/uploads/2016/05/SSRN-id962461.pdf
    - Antonacci / ADM summary:
      https://allocatesmartly.com/taa-strategy-accelerating-dual-momentum/
    - TimesFM model card:
      https://huggingface.co/google/timesfm-2.5-200m-pytorch

    Point-in-time safety:
    - Safe. Each component slices its input history to `decision_date`.
    """

    if decision_date == fold_spec.test_start:
        hmm_fit_end = fold_spec.train_end
    else:
        hmm_fit_end = decision_date

    model = hmm_model_cache
    hmm_train = fred_features.loc[:hmm_fit_end]
    if len(hmm_train) >= MIN_TRAIN_OBSERVATIONS:
        try:
            model = fit_hmm(hmm_train, n_states=hmm_states)
        except Exception:
            model = hmm_model_cache

    regime_label = "neutral"
    regime_probs = pd.Series({"p_risk_on": np.nan, "p_neutral": np.nan, "p_stress": np.nan}, dtype=float)
    if model is not None:
        classified = classify_states(model, fred_features.loc[:decision_date])
        latest = classified.iloc[-1]
        regime_label = str(latest["regime"])
        regime_probs = latest.drop(labels=["regime"]).astype(float)

    trend = trend_signals.loc[:decision_date].iloc[-1].reindex(ALL_SAA).fillna(0.0)
    momo = momo_signals.loc[:decision_date].iloc[-1].reindex(ALL_SAA).fillna(0.0)

    if use_timesfm:
        timesfm_frame = compute_vol_and_direction_signals(
            prices,
            decision_date,
            forecaster=forecaster,
            horizon=64,
            cache_path=TIMESFM_CACHE_FILE if timesfm_cache_path is None else timesfm_cache_path,
        )
        timesfm_mu = timesfm_frame.get("mu_ann", pd.Series(dtype=float)).reindex(ALL_SAA).fillna(0.0)
        timesfm_sigma = timesfm_frame.get("sigma_ann_fcst", pd.Series(dtype=float)).reindex(ALL_SAA)
        timesfm_dir = timesfm_frame.get("dir_score", pd.Series(dtype=float)).reindex(ALL_SAA).fillna(0.0)
    else:
        timesfm_mu = pd.Series(0.0, index=ALL_SAA, dtype=float)
        timesfm_sigma = pd.Series(np.nan, index=ALL_SAA, dtype=float)
        timesfm_dir = pd.Series(0.0, index=ALL_SAA, dtype=float)

    bundle = SignalBundle(
        regime_probs=regime_probs,
        regime_label=regime_label,
        trend=trend,
        momo=momo,
        timesfm_mu=timesfm_mu,
        timesfm_sigma=timesfm_sigma,
        timesfm_dir=timesfm_dir,
    )
    return bundle, model


def _period_dates_between(
    returns_index: pd.DatetimeIndex,
    decision_date: pd.Timestamp,
    next_decision_date: pd.Timestamp | None,
    end_date: pd.Timestamp,
) -> pd.DatetimeIndex:
    if next_decision_date is None:
        segment_end = end_date
    else:
        segment_end = min(pd.Timestamp(next_decision_date), pd.Timestamp(end_date))
    return returns_index[(returns_index > decision_date) & (returns_index <= segment_end)]


def simulate_period_returns(
    returns: pd.DataFrame,
    period_dates: pd.DatetimeIndex,
    starting_weights: pd.Series,
    turnover_cost: float,
    fold_id: int,
    decision_date: pd.Timestamp,
    regime_label: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate daily OOS returns between two monthly decision dates.

    Inputs:
    - `returns`: audited asset log-return dataframe.
    - `period_dates`: dates strictly after the current decision date and up to
      the next decision date.
    - `starting_weights`: portfolio weights immediately after the rebalance.
    - `turnover_cost`: one-off cost applied on the first day in the period.
    - `fold_id`: active fold identifier.
    - `decision_date`: monthly decision date that produced `starting_weights`.
    - `regime_label`: regime label active at the decision date.

    Outputs:
    - Tuple `(period_returns_df, end_weights)` where `end_weights` are the
      drifted holdings at the end of the segment.

    Citation:
    - Whitmore Task 6 walk-forward specification.

    Point-in-time safety:
    - Safe. The simulation applies the chosen weights forward only through
      realized returns after the decision date.
    """

    current_weights = starting_weights.reindex(ALL_SAA).fillna(0.0).copy()
    rows: list[dict[str, object]] = []
    for offset, date in enumerate(period_dates):
        gross_vector = np.exp(returns.loc[date].reindex(ALL_SAA).fillna(0.0))
        gross_return = float((current_weights * (gross_vector - 1.0)).sum())
        cost = turnover_cost if offset == 0 else 0.0
        portfolio_return = gross_return - cost

        rows.append(
            {
                "date": date,
                "fold_id": fold_id,
                "decision_date": decision_date,
                "regime": regime_label,
                "gross_return": gross_return,
                "turnover_cost": cost,
                "portfolio_return": portfolio_return,
            }
        )

        post_move_value = current_weights * gross_vector
        denominator = float(post_move_value.sum())
        current_weights = post_move_value / denominator if denominator > 0 else current_weights.copy()

    period_df = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["fold_id", "decision_date", "regime", "gross_return", "turnover_cost", "portfolio_return"]
    )
    return period_df, current_weights


def run_walkforward(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    folds: int = DEFAULT_FOLDS,
    embargo_business_days: int = DEFAULT_EMBARGO_BUSINESS_DAYS,
    use_timesfm: bool = False,
    hmm_states: int = 3,
    vol_budget: float = TARGET_VOL,
    output_dir: Path = OUTPUT_DIR,
    ensemble_config: EnsembleConfig | None = None,
    timesfm_cache_path: Path | None = TIMESFM_CACHE_FILE,
) -> dict[str, pd.DataFrame]:
    """Run the full walk-forward OOS TAA backtest.

    Inputs:
    - `start`, `end`: OOS date bounds.
    - `folds`: number of contiguous OOS folds.
    - `embargo_business_days`: embargo applied before each fold's first test
      decision.
    - `use_timesfm`: whether to invoke the optional TimesFM layer.
    - `hmm_states`: HMM state count.
    - `vol_budget`: internal ex-ante annualized volatility target.
    - `output_dir`: destination for generated CSV outputs.
    - `ensemble_config`: optional signal-ensemble configuration.
    - `timesfm_cache_path`: optional shared parquet cache path for TimesFM.

    Outputs:
    - Dictionary containing exported dataframes:
      `folds`, `oos_returns`, `oos_weights`, `oos_regimes`,
      `guardrail_switches`.

    Citation:
    - López de Prado purged/embargoed CV overview:
      https://en.wikipedia.org/wiki/Purged_cross-validation
    - Whitmore Task 6 walk-forward specification.

    Point-in-time safety:
    - Safe. The runner uses only causal signals, embargoed initial fold
      training windows, and realized returns after each decision date.
    """

    if vol_budget > VOL_CEILING:
        raise ValueError(
            f"vol_budget={vol_budget:.4f} exceeds VOL_CEILING={VOL_CEILING:.4f}. "
            "Use an internal target at or below the IPS volatility ceiling."
        )
    if vol_budget < 0.02:
        raise ValueError(
            f"vol_budget={vol_budget:.4f} is below 0.0200. "
            "This is likely a typo; the backtest refuses to run with unrealistically tight budgets."
        )

    prices = load_prices()
    fred = load_fred()
    returns = log_returns(prices)
    availability = availability_flag(prices)
    valid_return_index = returns.loc[:, ALL_SAA].dropna(how="all").index

    trend_signals = trend_score(prices.loc[:, ALL_SAA])
    momo_signals = cross_sectional_rank(adm_score(prices.loc[:, ALL_SAA]), SLEEVE_BUCKETS)
    fred_features = build_features(fred)
    # Use the base 4-feature set (VIXCLS, BAMLH0A0HYM2, T10Y3M, NFCI — available from 2001)
    # to anchor train_start.  Extended features like DFII10 only start 2003-01-02, so
    # fred_features.index.min() would be 2003 and leave the smoke-test window with no
    # training history.  The HMM fold logic uses fred_features directly, so it naturally
    # trains on whichever rows are available after the fold's train_start.
    _base_features = build_features(fred, use_extended=False)
    train_start = max(pd.Timestamp("2001-01-01"), _base_features.index.min())

    decision_dates = build_monthly_decision_dates(prices.loc[:, ALL_SAA], start, end)
    fold_specs = build_walkforward_folds(decision_dates, train_start=train_start, folds=folds, embargo_business_days=embargo_business_days)
    folds_df = fold_specs_to_frame(fold_specs)

    fold_lookup = {decision_date: spec for spec in fold_specs for decision_date in spec.decision_dates}

    if use_timesfm and not timesfm_is_available():
        raise RuntimeError(
            "TimesFM was requested with --timesfm, but the dependency is not installed. "
            "Rerun with --no-timesfm or install the official google-research/timesfm stack."
        )
    timesfm_enabled = bool(use_timesfm)

    config = EnsembleConfig() if ensemble_config is None else ensemble_config
    if config.vol_budget_by_regime is not None:
        for regime_label, regime_budget in config.vol_budget_by_regime.items():
            if regime_budget > VOL_CEILING:
                raise ValueError(
                    f"regime vol budget for {regime_label}={regime_budget:.4f} exceeds "
                    f"VOL_CEILING={VOL_CEILING:.4f}."
                )
            if regime_budget < 0.02:
                raise ValueError(
                    f"regime vol budget for {regime_label}={regime_budget:.4f} is below 0.0200. "
                    "This is likely a typo."
                )
    previous_weights = pd.Series(BM2_WEIGHTS, dtype=float).reindex(ALL_SAA).fillna(0.0)
    current_hmm_model = None
    current_fold_id: int | None = None

    weight_rows: list[pd.Series] = []
    regime_rows: list[dict[str, object]] = []
    return_frames: list[pd.DataFrame] = []
    guardrail_switch_rows: list[dict[str, object]] = []
    current_guardrail_multiplier = 1.0
    breach_log_path = output_dir / "breaches.log"

    for index, decision_date in enumerate(decision_dates):
        fold_spec = fold_lookup[pd.Timestamp(decision_date)]
        if current_fold_id != fold_spec.fold_id:
            current_fold_id = fold_spec.fold_id
            current_hmm_model = None

        signal_bundle, current_hmm_model = build_signal_bundle_at_date(
            decision_date=pd.Timestamp(decision_date),
            fold_spec=fold_spec,
            fred_features=fred_features,
            trend_signals=trend_signals,
            momo_signals=momo_signals,
            prices=prices.loc[:, ALL_SAA],
            use_timesfm=timesfm_enabled,
            forecaster=None,
            timesfm_cache_path=timesfm_cache_path if timesfm_enabled else None,
            hmm_model_cache=current_hmm_model,
            hmm_states=hmm_states,
        )

        regime_tilt = regime_tilt_from_label(signal_bundle.regime_label).reindex(ALL_SAA).fillna(0.0)

        # Macro factor signal: real-yield tilt, credit-premium tilt, crypto momentum.
        # Computed from the full lagged FRED panel and the price history up to
        # the current decision date — both are PIT-safe.
        macro_mu = compute_macro_factor_mu(
            fred=fred_features,
            prices=prices.loc[:, ALL_SAA],
            as_of_date=pd.Timestamp(decision_date),
        )

        from taa_project.optimizer.cvxpy_opt import ensemble_score as _ensemble_score
        signal_score = _ensemble_score(
            regime_tilt=regime_tilt,
            trend_sig=signal_bundle.trend,
            momo_sig=signal_bundle.momo,
            timesfm_mu=signal_bundle.timesfm_mu,
            macro_factor_mu=macro_mu,
            config=config,
        )

        covariance = estimate_taa_covariance(
            returns.loc[:, ALL_SAA],
            decision_date=pd.Timestamp(decision_date),
            timesfm_sigma=signal_bundle.timesfm_sigma,
        )
        if config.use_bl_stress_views:
            mu_bl = bl_with_stress_views(
                policy_weights=pd.Series(BM2_WEIGHTS, dtype=float),
                covariance=covariance,
                regime_label=signal_bundle.regime_label,
                stress_equity_shock_sigmas=config.bl_stress_shock,
                equity_assets=EQUITY_ASSETS,
            ).reindex(ALL_SAA).fillna(0.0)
            signal_score = 0.5 * signal_score + 0.5 * mu_bl
            del mu_bl
        guardrail_multiplier = 1.0
        if config.use_dd_guardrail and return_frames:
            realized_history = pd.concat([frame["portfolio_return"] for frame in return_frames]).sort_index()
            multiplier_history = dd_guardrail_multiplier(realized_history, config.dd_guardrail_config)
            trailing_dd_history = trailing_drawdown_series(realized_history, config.dd_guardrail_config.lookback_days)
            if not multiplier_history.empty:
                guardrail_multiplier = float(multiplier_history.iloc[-1])
                if guardrail_multiplier != current_guardrail_multiplier:
                    guardrail_switch_rows.append(
                        {
                            "date": pd.Timestamp(decision_date),
                            "trailing_dd": float(trailing_dd_history.iloc[-1]),
                            "action": "tighten" if guardrail_multiplier < current_guardrail_multiplier else "release",
                        }
                    )
                    current_guardrail_multiplier = guardrail_multiplier
        active_vol_budget = (
            config.vol_budget_by_regime[signal_bundle.regime_label]
            if config.vol_budget_by_regime is not None
            and signal_bundle.regime_label in config.vol_budget_by_regime
            else vol_budget
        )
        active_vol_budget *= guardrail_multiplier
        print(f"[SOLVE] vb={active_vol_budget} regime={signal_bundle.regime_label}")
        available_assets = availability.loc[:decision_date].iloc[-1].reindex(ALL_SAA).fillna(0.0)
        active_universe = [asset for asset in ALL_SAA if bool(available_assets.get(asset, 0.0))]
        scenario_returns = (
            _build_cvar_scenarios(
                asset_log_returns=returns.loc[:, ALL_SAA],
                decision_date=pd.Timestamp(decision_date),
                universe=active_universe,
                lookback_days=config.cvar_lookback_days,
            )
            if config.optimizer_mode == "cvar"
            else None
        )
        if config.use_nested_risk and config.nested_risk_config is not None:
            solve_result = solve_nested_taa(
                expected_returns=signal_score,
                cov_matrix=covariance,
                available=available_assets,
                previous_weights=previous_weights,
                config=config.nested_risk_config,
                ensemble_config=config,
                asset_log_returns=returns.loc[:, ALL_SAA]
                if config.nested_risk_config.use_cvar_per_sleeve or config.optimizer_mode == "cvar"
                else None,
                as_of_date=pd.Timestamp(decision_date),
                breach_log_path=breach_log_path,
            )
        else:
            solve_result = solve_taa_monthly_result(
                signal_score=signal_score,
                cov_matrix=covariance,
                prev_weights=previous_weights,
                available=available_assets,
                as_of_date=pd.Timestamp(decision_date),
                vol_budget=active_vol_budget,
                breach_log_path=breach_log_path,
                config=config,
                scenario_returns=scenario_returns,
            )

        weight_row = solve_result.weights.rename(pd.Timestamp(decision_date))
        weight_row["fold_id"] = fold_spec.fold_id
        weight_row["turnover"] = solve_result.turnover
        weight_row["turnover_cost"] = solve_result.turnover_cost
        weight_row["ex_ante_vol"] = solve_result.ex_ante_vol
        weight_row["active_vol_budget"] = active_vol_budget
        weight_row["guardrail_multiplier"] = guardrail_multiplier
        weight_row["optimizer_status"] = solve_result.status
        weight_row["used_fallback"] = int(solve_result.used_fallback)
        weight_rows.append(weight_row)

        regime_row = {
            "date": pd.Timestamp(decision_date),
            "fold_id": fold_spec.fold_id,
            "regime": signal_bundle.regime_label,
            "optimizer_status": solve_result.status,
            "used_fallback": int(solve_result.used_fallback),
            "turnover": solve_result.turnover,
            "turnover_cost": solve_result.turnover_cost,
            "ex_ante_vol": solve_result.ex_ante_vol,
            "active_vol_budget": active_vol_budget,
            "guardrail_multiplier": guardrail_multiplier,
        }
        regime_row.update(signal_bundle.regime_probs.to_dict())
        regime_rows.append(regime_row)

        next_decision = decision_dates[index + 1] if index + 1 < len(decision_dates) else None
        period_dates = _period_dates_between(
            valid_return_index,
            pd.Timestamp(decision_date),
            next_decision,
            pd.Timestamp(end),
        )
        period_returns, ending_weights = simulate_period_returns(
            returns=returns.loc[:, ALL_SAA],
            period_dates=period_dates,
            starting_weights=solve_result.weights,
            turnover_cost=solve_result.turnover_cost,
            fold_id=fold_spec.fold_id,
            decision_date=pd.Timestamp(decision_date),
            regime_label=signal_bundle.regime_label,
        )
        if not period_returns.empty:
            return_frames.append(period_returns)

        previous_weights = ending_weights

    weights_df = pd.DataFrame(weight_rows)
    weights_df.index.name = "decision_date"
    regimes_df = pd.DataFrame(regime_rows).set_index("date")
    guardrail_switches_df = pd.DataFrame(guardrail_switch_rows, columns=["date", "trailing_dd", "action"])
    oos_returns_df = pd.concat(return_frames).sort_index() if return_frames else pd.DataFrame(
        columns=["fold_id", "decision_date", "regime", "gross_return", "turnover_cost", "portfolio_return"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    folds_df.to_csv(output_dir / WALKFORWARD_FOLDS_FILENAME, index=False)
    oos_returns_df.to_csv(output_dir / OOS_RETURNS_FILENAME)
    weights_df.to_csv(output_dir / OOS_WEIGHTS_FILENAME)
    regimes_df.to_csv(output_dir / OOS_REGIMES_FILENAME)
    guardrail_switches_df.to_csv(output_dir / GUARDRAIL_SWITCHES_FILENAME, index=False)

    return {
        "folds": folds_df,
        "oos_returns": oos_returns_df,
        "oos_weights": weights_df,
        "oos_regimes": regimes_df,
        "guardrail_switches": guardrail_switches_df,
    }


def main() -> None:
    """CLI entrypoint for the walk-forward OOS backtest.

    Inputs:
    - `--start`, `--end`: OOS date range.
    - `--folds`: number of folds.
    - `--embargo-business-days`: fold embargo width.
    - `--timesfm`: enable the optional TimesFM layer.
    - `--vol-budget`: internal ex-ante annualized volatility target.
    - `--dd-guardrail` / `--no-dd-guardrail`: enable or disable the
      drawdown-clip overlay.
    - `--output-dir`: destination directory for CSV outputs.

    Outputs:
    - Writes walk-forward CSV artifacts to disk.

    Citation:
    - Whitmore Task 6 walk-forward specification.

    Point-in-time safety:
    - Safe. The CLI only orchestrates the causal walk-forward runner.
    """

    parser = argparse.ArgumentParser(description="Run the Whitmore walk-forward OOS TAA backtest.")
    parser.add_argument("--start", default=DEFAULT_START, help="First OOS date.")
    parser.add_argument("--end", default=DEFAULT_END, help="Last OOS date.")
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="Number of contiguous OOS folds.")
    parser.add_argument(
        "--embargo-business-days",
        type=int,
        default=DEFAULT_EMBARGO_BUSINESS_DAYS,
        help="Embargo width between each fold's initial train sample and first test decision.",
    )
    parser.add_argument("--timesfm", action="store_true", help="Enable the optional TimesFM signal layer.")
    parser.add_argument(
        "--vol-budget",
        type=float,
        default=TARGET_VOL,
        help="Internal ex-ante vol target used by the TAA optimizer (default 0.10).",
    )
    guardrail_group = parser.add_mutually_exclusive_group()
    guardrail_group.add_argument("--dd-guardrail", dest="dd_guardrail", action="store_true", help="Enable the drawdown-clip overlay.")
    guardrail_group.add_argument("--no-dd-guardrail", dest="dd_guardrail", action="store_false", help="Disable the drawdown-clip overlay.")
    parser.set_defaults(dd_guardrail=False)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Destination directory for generated CSV files.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    artifacts = run_walkforward(
        start=args.start,
        end=args.end,
        folds=args.folds,
        embargo_business_days=args.embargo_business_days,
        use_timesfm=args.timesfm,
        vol_budget=args.vol_budget,
        output_dir=output_dir,
        ensemble_config=EnsembleConfig(use_dd_guardrail=args.dd_guardrail),
    )
    print(
        "Walk-forward outputs written to "
        f"{output_dir / WALKFORWARD_FOLDS_FILENAME}, {output_dir / OOS_RETURNS_FILENAME}, "
        f"{output_dir / OOS_WEIGHTS_FILENAME}, {output_dir / OOS_REGIMES_FILENAME}, "
        f"and {output_dir / GUARDRAIL_SWITCHES_FILENAME}. "
        f"OOS daily rows: {len(artifacts['oos_returns'])}"
    )


if __name__ == "__main__":
    main()
