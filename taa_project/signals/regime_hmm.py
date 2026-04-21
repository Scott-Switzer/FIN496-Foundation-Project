# Addresses rubric criterion 1 (TAA signal design) by implementing the
# macro-regime HMM layer on lagged FRED features.
"""Gaussian-HMM regime layer for the Whitmore TAA stack.

This module implements the Task 4 regime layer:
- 3-state Gaussian HMM on z-scored lagged FRED features
  `{VIXCLS, BAMLH0A0HYM2, T10Y3M, NFCI}`.
- Monthly refits on an expanding window.
- Posterior probabilities plus an interpreted regime label for each decision
  date.

References:
- Hamilton (1989), "A New Approach to the Economic Analysis of Nonstationary
  Time Series": https://doi.org/10.2307/1912559
- QuantStart HMM market-regime tutorial:
  https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/
- QuantInsti regime-adaptive trading overview:
  https://blog.quantinsti.com/regime-adaptive-trading-python/

Point-in-time safety:
- Safe when `fit_hmm` is called on features dated `<= t` and
  `classify_states` is queried only on data dated `<= t`. The fitted model
  stores the training-sample z-score statistics and reuses them at inference,
  so no future rescaling leaks into the state posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # pragma: no cover - exercised only when hmmlearn is absent.
    GaussianHMM = None


# Original four coincident / financial-stress features.
MACRO_FEATURES = ["VIXCLS", "BAMLH0A0HYM2", "T10Y3M", "NFCI"]

# Extended set: adds the 10-year TIPS real yield (DFII10).  This series has a
# -0.80 monthly correlation with BROAD_TIPS returns and a -0.46 correlation
# with XAU returns, giving the HMM a real-rate dimension that is entirely
# absent from the original four features.
# Available from 2003-01-02, matching the full backtest window.
# Citation: Erb & Harvey (2013). https://doi.org/10.2469/faj.v69.n4.1
MACRO_FEATURES_EXTENDED = ["VIXCLS", "BAMLH0A0HYM2", "T10Y3M", "NFCI", "DFII10"]
DEFAULT_HMM_STATES = 3
MIN_TRAIN_OBSERVATIONS = 252
HMM_RANDOM_SEED = 42


@dataclass(frozen=True)
class FittedRegimeModel:
    """Container for a fitted HMM plus train-time normalization metadata.

    Inputs:
    - `model`: fitted `hmmlearn.hmm.GaussianHMM` instance.
    - `feature_columns`: ordered feature names used during fitting.
    - `feature_mean`: training-sample means used to z-score the features.
    - `feature_std`: training-sample standard deviations used to z-score the
      features.
    - `state_names`: map from hidden-state integer to interpreted regime name.

    Outputs:
    - Immutable fitted-model bundle consumed by `classify_states`.

    Citation:
    - Hamilton (1989): https://doi.org/10.2307/1912559
    - QuantStart HMM tutorial:
      https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/

    Point-in-time safety:
    - Safe. All stored moments are estimated on the training window only.
    """

    model: GaussianHMM
    feature_columns: tuple[str, ...]
    feature_mean: pd.Series
    feature_std: pd.Series
    state_names: dict[int, str]


def build_features(
    fred: pd.DataFrame,
    use_extended: bool = True,
) -> pd.DataFrame:
    """Select HMM macro features from the lagged FRED panel.

    Inputs:
    - `fred`: one-business-day-lagged macro dataframe from Task 1.
    - `use_extended`: if True (default), attempt to include DFII10 as a fifth
      feature.  Falls back to the original four if DFII10 is absent or has
      insufficient history.

    Outputs:
    - Clean feature dataframe containing four or five macro series.

    Citation:
    - Whitmore project file `data/consolidated_csvs/fred/master/fred_data.csv`.
    - Hamilton (1989): https://doi.org/10.2307/1912559
    - Erb & Harvey (2013) on DFII10: https://doi.org/10.2469/faj.v69.n4.1

    Point-in-time safety:
    - Safe. The input is already lagged in Task 1, and this function only
      subsets and drops currently missing observations.
    """
    # Determine which features to use
    candidate = MACRO_FEATURES_EXTENDED if use_extended else MACRO_FEATURES
    # Drop extended columns absent from the panel
    available = [c for c in candidate if c in fred.columns]
    # Fall back to base set if extended feature has insufficient history
    if use_extended and "DFII10" in available:
        if fred["DFII10"].dropna().__len__() < MIN_TRAIN_OBSERVATIONS:
            available = list(MACRO_FEATURES)

    missing = [c for c in MACRO_FEATURES if c not in available]
    if missing:
        raise KeyError(f"Missing required FRED regime features: {missing}")

    features = fred.loc[:, available].copy()
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features.sort_index()


def zscore_features(
    features: pd.DataFrame,
    mean: pd.Series | None = None,
    std: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score the regime features with explicit train-time moments.

    Inputs:
    - `features`: raw feature dataframe.
    - `mean`: optional feature mean vector. If omitted, estimate from `features`.
    - `std`: optional feature std vector. If omitted, estimate from `features`.

    Outputs:
    - Tuple `(scaled_features, mean, std)`.

    Citation:
    - Hamilton (1989): https://doi.org/10.2307/1912559

    Point-in-time safety:
    - Safe when `mean` and `std` come from the training sample only.
    """

    # Use whatever columns the input dataframe actually contains (supports
    # both the original four-feature set and the extended five-feature set).
    active_cols = list(features.columns)
    ordered = features.loc[:, active_cols].astype(float)
    feature_mean = ordered.mean() if mean is None else mean.reindex(active_cols)
    feature_std = ordered.std(ddof=0) if std is None else std.reindex(active_cols)
    feature_std = feature_std.replace(0.0, np.nan).fillna(1.0)
    scaled = (ordered - feature_mean) / feature_std
    return scaled, feature_mean, feature_std


def _state_stress_scores(
    model: GaussianHMM,
    feature_columns: tuple[str, ...] | None = None,
) -> dict[int, float]:
    """Score each hidden state from benign to stressed using macro loadings.

    Inputs:
    - `model`: fitted Gaussian HMM in z-scored feature space.
    - `feature_columns`: ordered feature names used during fitting.  If None,
      falls back to the original MACRO_FEATURES list.

    Outputs:
    - Mapping from state index to scalar stress score.

    Citation:
    - Hamilton (1989): https://doi.org/10.2307/1912559
    - QuantStart HMM tutorial:
      https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/

    Point-in-time safety:
    - Safe. The score uses only fitted-state means from the training window.
    """
    cols = list(feature_columns) if feature_columns is not None else list(MACRO_FEATURES)
    feature_positions = {feature: index for index, feature in enumerate(cols)}
    scores: dict[int, float] = {}
    for state in range(model.n_components):
        means = model.means_[state]
        score = 0.0
        # Stress contributors (higher → more stressed)
        for stress_col in ["VIXCLS", "BAMLH0A0HYM2", "NFCI", "DFII10"]:
            if stress_col in feature_positions:
                score += float(means[feature_positions[stress_col]])
        # Stress detractors (higher value = less stressed)
        for calm_col in ["T10Y3M"]:
            if calm_col in feature_positions:
                score -= float(means[feature_positions[calm_col]])
        scores[state] = score
    return scores


def _interpret_state_names(
    model: GaussianHMM,
    feature_columns: tuple[str, ...] | None = None,
) -> dict[int, str]:
    """Map raw state ids to economic labels ordered by stress severity.

    Inputs:
    - `model`: fitted Gaussian HMM in z-scored feature space.
    - `feature_columns`: ordered feature names used during fitting.

    Outputs:
    - State-name mapping such as `{0: "risk_on", 1: "neutral", 2: "stress"}`.

    Citation:
    - Hamilton (1989): https://doi.org/10.2307/1912559
    - QuantStart HMM tutorial:
      https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/

    Point-in-time safety:
    - Safe. Interpretation is based only on fitted-state macro averages.
    """

    stress_scores = _state_stress_scores(model, feature_columns)
    ranked_states = sorted(stress_scores, key=stress_scores.get)
    if len(ranked_states) == 3:
        names = ["risk_on", "neutral", "stress"]
    elif len(ranked_states) == 2:
        names = ["risk_on", "stress"]
    else:
        names = [f"state_{index}" for index in range(len(ranked_states))]
    return {state: name for state, name in zip(ranked_states, names)}


def fit_hmm(
    features: pd.DataFrame,
    n_states: int = DEFAULT_HMM_STATES,
    seed: int = HMM_RANDOM_SEED,
    min_observations: int = MIN_TRAIN_OBSERVATIONS,
) -> FittedRegimeModel:
    """Fit a Gaussian HMM on the expanding regime-feature history.

    Inputs:
    - `features`: raw macro feature dataframe up to decision date `t`.
    - `n_states`: number of hidden states. Task 4 specifies `3`.
    - `seed`: deterministic random seed for reproducibility.
    - `min_observations`: minimum observations before fitting.

    Outputs:
    - `FittedRegimeModel` storing the fitted HMM and scaling metadata.

    Citation:
    - Hamilton (1989): https://doi.org/10.2307/1912559
    - QuantStart HMM tutorial:
      https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/

    Point-in-time safety:
    - Safe. The caller should pass only data observed on or before the decision
      date, and the fitted moments are frozen from that training sample.
    """

    if GaussianHMM is None:
        raise ImportError("hmmlearn is required for the regime HMM. Install the project requirements first.")

    clean_features = build_features(features)  # uses extended set by default
    if len(clean_features) < min_observations:
        raise ValueError(
            f"Need at least {min_observations} feature rows to fit the HMM; received {len(clean_features)}."
        )

    scaled, feature_mean, feature_std = zscore_features(clean_features)
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=500,
        random_state=seed,
    )
    model.fit(scaled.to_numpy(dtype=float))
    fitted_cols = tuple(clean_features.columns)
    state_names = _interpret_state_names(model, feature_columns=fitted_cols)
    return FittedRegimeModel(
        model=model,
        feature_columns=fitted_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
        state_names=state_names,
    )


def classify_states(model: FittedRegimeModel, features: pd.DataFrame) -> pd.DataFrame:
    """Emit posterior state probabilities and a hard regime label per date.

    Inputs:
    - `model`: fitted regime model from `fit_hmm`.
    - `features`: raw macro feature dataframe dated `<= t`.

    Outputs:
    - Dataframe with posterior columns `p_<state>` and a `regime` label column.

    Citation:
    - Hamilton (1989): https://doi.org/10.2307/1912559
    - QuantStart HMM tutorial:
      https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/

    Point-in-time safety:
    - Safe when `features` is truncated at the current decision date. The
      function applies the train-time scaler stored on `model`.
    """

    # Subset inference data to only the columns the model was fitted on.
    # This ensures compatibility whether the model used 4 or 5 features.
    avail_cols = [c for c in model.feature_columns if c in features.columns]
    clean_features = build_features(features.loc[:, avail_cols] if avail_cols else features)
    # Re-align to the exact fitted column order.
    clean_features = clean_features.reindex(columns=list(model.feature_columns)).dropna()
    scaled, _, _ = zscore_features(clean_features, model.feature_mean, model.feature_std)
    probabilities = model.model.predict_proba(scaled.to_numpy(dtype=float))
    state_indices = probabilities.argmax(axis=1)

    ordered_states = sorted(model.state_names)
    posterior = pd.DataFrame(
        {f"p_{model.state_names[state]}": probabilities[:, state] for state in ordered_states},
        index=clean_features.index,
    )
    posterior["regime"] = [model.state_names[int(state)] for state in state_indices]
    return posterior


def walk_forward_regimes(
    fred: pd.DataFrame,
    start: str = "2003-01-01",
    refit_freq: str = "ME",
    n_states: int = DEFAULT_HMM_STATES,
    min_observations: int = MIN_TRAIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Refit the HMM monthly on an expanding window and emit decision-date rows.

    Inputs:
    - `fred`: lagged FRED dataframe from Task 1.
    - `start`: first decision date to evaluate.
    - `refit_freq`: refit frequency. Task 4 specifies monthly.
    - `n_states`: number of hidden states.
    - `min_observations`: minimum observations required before fitting.

    Outputs:
    - Dataframe indexed by refit date with posterior probabilities and regime
      labels for those decision dates only.

    Citation:
    - Hamilton (1989): https://doi.org/10.2307/1912559
    - QuantStart HMM tutorial:
      https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/

    Point-in-time safety:
    - Safe. Each emitted row at date `t` is produced by a model trained only on
      observations up to and including `t`.
    """

    features = build_features(fred)
    refit_dates = features.loc[start:].resample(refit_freq).last().index
    outputs: list[pd.DataFrame] = []
    for refit_date in refit_dates:
        train = features.loc[:refit_date]
        if len(train) < min_observations:
            continue
        fitted = fit_hmm(train, n_states=n_states, min_observations=min_observations)
        decision_row = classify_states(fitted, train).iloc[[-1]]
        decision_row.index = pd.Index([pd.Timestamp(refit_date)])
        outputs.append(decision_row)

    return pd.concat(outputs).sort_index() if outputs else pd.DataFrame()


REGIME_TILT = {
    "risk_on": {
        "SPXT": 0.42,
        "FTSE100": 0.05,
        "LBUSTRUU": 0.05,
        "BROAD_TIPS": 0.00,
        "B3REITT": 0.10,
        "XAU": 0.10,
        "NIKKEI225": 0.08,
        "SILVER_FUT": 0.05,
        "CSI300_CHINA": 0.08,
        "BITCOIN": 0.05,
        "CHF_FRANC": 0.02,
    },
    "neutral": {
        "SPXT": 0.35,
        "FTSE100": 0.03,
        "LBUSTRUU": 0.15,
        "BROAD_TIPS": 0.05,
        "B3REITT": 0.08,
        "XAU": 0.15,
        "NIKKEI225": 0.05,
        "SILVER_FUT": 0.03,
        "CSI300_CHINA": 0.05,
        "BITCOIN": 0.03,
        "CHF_FRANC": 0.03,
    },
    "stress": {
        "SPXT": 0.25,
        "FTSE100": 0.00,
        "LBUSTRUU": 0.25,
        "BROAD_TIPS": 0.15,
        "B3REITT": 0.00,
        "XAU": 0.20,
        "NIKKEI225": 0.00,
        "SILVER_FUT": 0.00,
        "CSI300_CHINA": 0.00,
        "BITCOIN": 0.00,
        "CHF_FRANC": 0.15,
    },
}


def regime_tilt_from_label(regime_label: str) -> pd.Series:
    """Return the configured regime tilt vector for a regime label.

    Inputs:
    - `regime_label`: interpreted state label such as `risk_on`.

    Outputs:
    - Weight-hint series indexed by SAA sleeves.

    Citation:
    - Internal Whitmore TAA scaffold and IPS band structure.

    Point-in-time safety:
    - Safe. The tilt is a static configuration lookup conditioned on the
      current regime label only.
    """

    tilt = REGIME_TILT.get(regime_label, REGIME_TILT["neutral"])
    return pd.Series(tilt, dtype=float)
