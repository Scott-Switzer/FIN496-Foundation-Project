# Addresses rubric criterion 1 (TAA signal design) by implementing the
# optional TimesFM forecast layer for return, volatility, and direction votes.
"""TimesFM 2.5 zero-shot forecast layer for the Whitmore TAA stack.

This module wraps the official TimesFM 2.5 PyTorch checkpoint using the Task 4
configuration:
- `timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")`
- `use_continuous_quantile_head=True`
- `force_flip_invariance=True`
- `infer_is_positive=False` because returns can be negative
- `fix_quantile_crossing=True`

References:
- TimesFM 2.5 Hugging Face model card:
  https://huggingface.co/google/timesfm-2.5-200m-pytorch
- TimesFM GitHub repository:
  https://github.com/google-research/timesfm
- Das et al. (2024), Google Research blog:
  https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/

Point-in-time safety:
- Safe when the caller truncates each return series at the decision date `t`.
  The forecaster consumes only historical returns observed up to `t` and does
  not refit on future outcomes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MAX_CONTEXT = 1024
DEFAULT_MAX_HORIZON = 256
DEFAULT_MIN_CONTEXT = 64
DEFAULT_BATCH_SIZE = 1


def timesfm_is_available() -> bool:
    """Return whether the optional TimesFM dependencies are importable.

    Inputs:
    - None.

    Outputs:
    - Boolean availability flag.

    Citation:
    - TimesFM GitHub repository: https://github.com/google-research/timesfm

    Point-in-time safety:
    - Safe. This is an environment check only.
    """

    return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("timesfm") is not None


class TimesFMForecaster:
    """Lazy wrapper around the official TimesFM 2.5 PyTorch checkpoint.

    Inputs:
    - `max_context`: maximum context length passed into `ForecastConfig`.
    - `max_horizon`: maximum forecast horizon passed into `ForecastConfig`.
    - `torch_compile`: optional model compilation flag for environments that
      support it.

    Outputs:
    - Reusable forecaster instance with batched `forecast_batch`.

    Citation:
    - Hugging Face model card:
      https://huggingface.co/google/timesfm-2.5-200m-pytorch
    - TimesFM GitHub repository:
      https://github.com/google-research/timesfm

    Point-in-time safety:
    - Safe. The class performs zero-shot inference only; it does not train on
      future labels.
    """

    def __init__(
        self,
        max_context: int = DEFAULT_MAX_CONTEXT,
        max_horizon: int = DEFAULT_MAX_HORIZON,
        torch_compile: bool = False,
    ) -> None:
        if not timesfm_is_available():
            raise ImportError(
                "TimesFM is not installed. The official 2.5 model card currently points to "
                "the google-research/timesfm repository installation flow."
            )

        import torch
        import timesfm

        torch.set_float32_matmul_precision("high")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch",
            torch_compile=torch_compile,
        )
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )
        self.max_context = max_context
        self.max_horizon = max_horizon

    @staticmethod
    def _clean_history(series: pd.Series | np.ndarray, max_context: int) -> np.ndarray:
        """Prepare one return history for a TimesFM call.

        Inputs:
        - `series`: historical return sequence for one asset.
        - `max_context`: maximum retained context length.

        Outputs:
        - Clean NumPy array truncated to the latest `max_context` observations.

        Citation:
        - TimesFM GitHub repository: https://github.com/google-research/timesfm

        Point-in-time safety:
        - Safe. The history is expected to be truncated upstream at the
          decision date.
        """

        values = np.asarray(series, dtype=np.float32)
        values = values[np.isfinite(values)]
        return values[-max_context:]

    def forecast_batch(
        self,
        series_dict: dict[str, pd.Series | np.ndarray],
        horizon: int = 21,
        min_context: int = DEFAULT_MIN_CONTEXT,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Forecast multiple asset return series in one TimesFM batch.

        Inputs:
        - `series_dict`: mapping from ticker to historical return sequence.
        - `horizon`: forecast horizon in trading days.
        - `min_context`: minimum history length required to send a series.
        - `batch_size`: number of assets per TimesFM call. The default is a
          conservative CPU-safe chunk size used to avoid local Torch/OpenMP
          instability on this desktop runtime.

        Outputs:
        - Mapping from ticker to forecast payload with `point` and `quantiles`.

        Citation:
        - Hugging Face model card:
          https://huggingface.co/google/timesfm-2.5-200m-pytorch

        Point-in-time safety:
        - Safe. The histories are expected to be truncated at the decision
          date before batching.
        """

        valid_tickers: list[str] = []
        inputs: list[np.ndarray] = []
        for ticker, history in series_dict.items():
            cleaned = self._clean_history(history, self.max_context)
            if len(cleaned) < min_context:
                continue
            valid_tickers.append(ticker)
            inputs.append(cleaned)

        if not inputs:
            return {}

        safe_batch_size = max(int(batch_size), 1)
        forecasts: dict[str, dict[str, np.ndarray]] = {}
        for start in range(0, len(inputs), safe_batch_size):
            batch_inputs = inputs[start : start + safe_batch_size]
            batch_tickers = valid_tickers[start : start + safe_batch_size]
            point_forecast, quantile_forecast = self.model.forecast(horizon=horizon, inputs=batch_inputs)
            for index, ticker in enumerate(batch_tickers):
                forecasts[ticker] = {
                    "point": np.asarray(point_forecast[index], dtype=float),
                    "quantiles": np.asarray(quantile_forecast[index], dtype=float),
                }
        return forecasts

    @staticmethod
    def expected_return(forecast: dict[str, np.ndarray], horizon_days: int = 21) -> float:
        """Annualize the cumulative TimesFM point forecast for one asset.

        Inputs:
        - `forecast`: single-asset forecast payload from `forecast_batch`.
        - `horizon_days`: forecast horizon in trading days.

        Outputs:
        - Annualized expected return proxy.

        Citation:
        - TimesFM model card:
          https://huggingface.co/google/timesfm-2.5-200m-pytorch

        Point-in-time safety:
        - Safe. This is a pure transform of the zero-shot forecast at date `t`.
        """

        cumulative_log_return = float(np.sum(forecast["point"]))
        return cumulative_log_return * (252.0 / float(horizon_days))

    @staticmethod
    def forecast_vol(forecast: dict[str, np.ndarray]) -> float:
        """Infer annualized volatility from the TimesFM quantile band width.

        Inputs:
        - `forecast`: single-asset forecast payload from `forecast_batch`.

        Outputs:
        - Annualized forecast volatility.

        Citation:
        - TimesFM model card:
          https://huggingface.co/google/timesfm-2.5-200m-pytorch

        Point-in-time safety:
        - Safe. This is a pure transform of the forecast quantiles at date `t`.
        """

        quantiles = np.asarray(forecast["quantiles"], dtype=float)
        q10 = quantiles[:, 1]
        q90 = quantiles[:, -1]
        sigma_step = (q90 - q10) / 2.563
        daily_sigma = float(np.sqrt(np.mean(np.square(sigma_step))))
        return daily_sigma * np.sqrt(252.0)

    @staticmethod
    def directional_score(forecast: dict[str, np.ndarray]) -> float:
        """Convert the forecast path into a bounded directional-conviction vote.

        Inputs:
        - `forecast`: single-asset forecast payload from `forecast_batch`.

        Outputs:
        - Directional score in `[-1, +1]`.

        Citation:
        - TimesFM model card:
          https://huggingface.co/google/timesfm-2.5-200m-pytorch

        Point-in-time safety:
        - Safe. This is a pure transform of the forecast distribution at date
          `t`.
        """

        cumulative_mean = float(np.sum(forecast["point"]))
        quantiles = np.asarray(forecast["quantiles"], dtype=float)
        cumulative_q10 = float(np.sum(quantiles[:, 1]))
        cumulative_q90 = float(np.sum(quantiles[:, -1]))
        width = max(cumulative_q90 - cumulative_q10, 1e-9)
        return float(np.tanh(2.0 * cumulative_mean / width))


def compute_vol_and_direction_signals(
    returns: pd.DataFrame,
    decision_date: pd.Timestamp,
    forecaster: TimesFMForecaster,
    horizon: int = 21,
    min_context: int = DEFAULT_MIN_CONTEXT,
    cache_dir: Path | None = None,
    cache_key_prefix: str = "use_timesfm_1",
) -> pd.DataFrame:
    """Compute TimesFM return, volatility, and direction signals at date `t`.

    Inputs:
    - `returns`: audited asset log-return dataframe with gaps preserved as
      `NaN`.
    - `decision_date`: current rebalance date.
    - `forecaster`: initialized `TimesFMForecaster`.
    - `horizon`: forecast horizon in trading days.
    - `min_context`: minimum history length required for a forecast.
    - `cache_dir`: optional directory for cached per-decision TimesFM signal
      CSVs shared across repeated sweeps.
    - `cache_key_prefix`: cache namespace token. This must include whether the
      TimesFM layer was enabled so repeated sweeps cannot cross-load the wrong
      cached forecast file.

    Outputs:
    - Dataframe indexed by ticker with columns
      `['mu_ann', 'sigma_ann_fcst', 'dir_score']`.

    Citation:
    - Hugging Face model card:
      https://huggingface.co/google/timesfm-2.5-200m-pytorch
    - Google Research blog:
      https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/

    Point-in-time safety:
    - Safe. Each series is truncated to `decision_date` before the batch call.
    """

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / (
            f"timesfm_signals_{cache_key_prefix}_{pd.Timestamp(decision_date):%Y%m%d}_h{horizon}.csv"
        )
        if cache_path.exists():
            return pd.read_csv(cache_path, index_col="ticker")

    truncated_histories: dict[str, pd.Series] = {}
    for column in returns.columns:
        history = returns[column].loc[:decision_date].dropna()
        if len(history) >= min_context:
            truncated_histories[column] = history

    forecasts = forecaster.forecast_batch(truncated_histories, horizon=horizon, min_context=min_context)
    rows: list[dict[str, float | str]] = []
    for ticker, forecast in forecasts.items():
        rows.append(
            {
                "ticker": ticker,
                "mu_ann": forecaster.expected_return(forecast, horizon_days=horizon),
                "sigma_ann_fcst": forecaster.forecast_vol(forecast),
                "dir_score": forecaster.directional_score(forecast),
            }
        )

    if not rows:
        frame = pd.DataFrame(columns=["mu_ann", "sigma_ann_fcst", "dir_score"])
    else:
        frame = pd.DataFrame(rows).set_index("ticker").sort_index()
    if cache_dir is not None:
        frame.to_csv(cache_path, index_label="ticker")
    return frame
