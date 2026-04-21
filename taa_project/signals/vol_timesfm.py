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
- Arrow parquet filtering:
  https://arrow.apache.org/docs/python/parquet.html#filtering

Point-in-time safety:
- Safe when the caller truncates each asset price history at the decision date
  `t`. The forecaster consumes only historical prices observed up to `t`.
"""

from __future__ import annotations

import gc
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from taa_project.config import TIMESFM_CACHE_PATH
from taa_project.memory import guard_process_memory


DEFAULT_MAX_CONTEXT = 1024
DEFAULT_MAX_HORIZON = 256
DEFAULT_MIN_CONTEXT = 64
DEFAULT_BATCH_SIZE = 1
DEFAULT_MODEL_VERSION = "google/timesfm-2.5-200m-pytorch"
FORECAST_QUANTILE_LEVELS = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90], dtype=np.float32)
CACHE_QUANTILE_LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float32)
CACHE_QUANTILE_COLUMNS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")
TIMESFM_CACHE_SCHEMA = pa.schema(
    [
        ("cache_key", pa.string()),
        ("asset", pa.string()),
        ("decision_date", pa.date32()),
        ("context_length", pa.int32()),
        ("horizon", pa.int32()),
        ("model_version", pa.string()),
        ("q05", pa.float32()),
        ("q10", pa.float32()),
        ("q25", pa.float32()),
        ("q50", pa.float32()),
        ("q75", pa.float32()),
        ("q90", pa.float32()),
        ("q95", pa.float32()),
        ("created_at_utc", pa.timestamp("us", tz="UTC")),
    ]
)


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


def _decision_date_iso(decision_date: pd.Timestamp) -> str:
    return pd.Timestamp(decision_date).normalize().date().isoformat()


def _series_to_price_history(series: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    values = values[values > 0.0]
    return values


def _price_history_to_log_returns(
    series: pd.Series | np.ndarray,
    max_context: int,
) -> np.ndarray:
    prices = _series_to_price_history(series)
    if prices.size < 2:
        return np.empty(0, dtype=np.float32)
    log_rets = np.diff(np.log(prices))
    log_rets = log_rets[np.isfinite(log_rets)]
    return np.asarray(log_rets[-max_context:], dtype=np.float32)


def _interpolate_quantile(level: float, levels: np.ndarray, values: np.ndarray) -> float:
    if level <= float(levels[0]):
        slope = float(values[1] - values[0]) / float(levels[1] - levels[0])
        return float(values[0] + slope * (level - float(levels[0])))
    if level >= float(levels[-1]):
        slope = float(values[-1] - values[-2]) / float(levels[-1] - levels[-2])
        return float(values[-1] + slope * (level - float(levels[-1])))
    return float(np.interp(level, levels.astype(float), values.astype(float)))


def _forecast_to_cache_quantiles(
    forecast: dict[str, np.ndarray],
    horizon_days: int,
) -> dict[str, float]:
    quantiles = np.asarray(forecast["quantiles"], dtype=np.float32)
    if quantiles.ndim != 2 or quantiles.shape[1] < 10:
        raise ValueError("TimesFM quantile forecast must have shape (horizon, >=10).")

    annualized_quantiles = np.sum(quantiles[:, 1:10], axis=0, dtype=np.float64) * (252.0 / float(horizon_days))
    outputs: dict[str, float] = {}
    for level, column in zip(CACHE_QUANTILE_LEVELS, CACHE_QUANTILE_COLUMNS, strict=True):
        outputs[column] = _interpolate_quantile(float(level), FORECAST_QUANTILE_LEVELS, annualized_quantiles)
    return outputs


def _build_cache_row_table(
    cache_key: str,
    asset: str,
    decision_date: pd.Timestamp,
    context_length: int,
    horizon: int,
    model_version: str,
    quantiles: dict[str, float],
) -> pa.Table:
    row = {
        "cache_key": [cache_key],
        "asset": [asset],
        "decision_date": [pd.Timestamp(decision_date).normalize().date()],
        "context_length": [np.int32(context_length)],
        "horizon": [np.int32(horizon)],
        "model_version": [model_version],
        "q05": [np.float32(quantiles["q05"])],
        "q10": [np.float32(quantiles["q10"])],
        "q25": [np.float32(quantiles["q25"])],
        "q50": [np.float32(quantiles["q50"])],
        "q75": [np.float32(quantiles["q75"])],
        "q90": [np.float32(quantiles["q90"])],
        "q95": [np.float32(quantiles["q95"])],
        "created_at_utc": [pd.Timestamp.utcnow()],
    }
    return pa.Table.from_pydict(row, schema=TIMESFM_CACHE_SCHEMA)


def _read_cache_row(cache_path: Path, cache_key: str) -> dict[str, float] | None:
    if not cache_path.exists():
        return None
    table = pq.read_table(
        cache_path,
        columns=list(CACHE_QUANTILE_COLUMNS),
        filters=[("cache_key", "=", cache_key)],
    )
    if table.num_rows == 0:
        return None
    values = table.to_pydict()
    return {column: float(values[column][0]) for column in CACHE_QUANTILE_COLUMNS}


def _append_cache_row(
    cache_path: Path,
    cache_key: str,
    asset: str,
    decision_date: pd.Timestamp,
    context_length: int,
    horizon: int,
    model_version: str,
    quantiles: dict[str, float],
) -> None:
    """Persist one TimesFM cache row.

    The cache stays in a single parquet file. Because the file is expected to
    remain small, the miss path rewrites the full file after appending one row.
    Cache hits still use predicate pushdown via `_read_cache_row` and never
    load the entire parquet.
    """

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = _build_cache_row_table(
        cache_key=cache_key,
        asset=asset,
        decision_date=decision_date,
        context_length=context_length,
        horizon=horizon,
        model_version=model_version,
        quantiles=quantiles,
    )
    if not cache_path.exists():
        pq.write_table(new_row, cache_path, compression="zstd")
        return
    existing = pq.read_table(cache_path)
    combined = pa.concat_tables([existing, new_row], promote_options="none")
    pq.write_table(combined, cache_path, compression="zstd")


def _quantile_band_to_sigma(quantiles: dict[str, float]) -> float:
    """Infer annualized sigma from the cached q10/q90 band."""

    band = max(float(quantiles["q90"]) - float(quantiles["q10"]), 1e-9)
    return band / 2.5631031311


def _quantile_band_to_direction(quantiles: dict[str, float]) -> float:
    width = max(float(quantiles["q90"]) - float(quantiles["q10"]), 1e-9)
    return float(np.tanh(2.0 * float(quantiles["q50"]) / width))


class TimesFMForecaster:
    """Lazy wrapper around the official TimesFM 2.5 PyTorch checkpoint.

    Inputs:
    - `max_context`: maximum context length passed into `ForecastConfig`.
    - `max_horizon`: maximum forecast horizon passed into `ForecastConfig`.
    - `model_version`: Hugging Face model identifier.
    - `torch_compile`: optional model compilation flag for environments that
      support it.

    Outputs:
    - Reusable forecaster instance with per-asset `forecast_quantiles`.

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
        model_version: str = DEFAULT_MODEL_VERSION,
        torch_compile: bool = False,
    ) -> None:
        if not timesfm_is_available():
            raise ImportError(
                "TimesFM is not installed. The official 2.5 model card currently points to "
                "the google-research/timesfm repository installation flow."
            )

        import torch
        import timesfm

        guard_process_memory("timesfm:init:before_load")
        torch.set_float32_matmul_precision("high")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            model_version,
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
        self.model_version = model_version
        guard_process_memory("timesfm:init:after_load")

    @staticmethod
    def _clean_history(series: pd.Series | np.ndarray, max_context: int) -> np.ndarray:
        """Prepare one log-return history for a TimesFM call."""

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
        """Forecast multiple asset return series in one TimesFM batch."""

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
        guard_process_memory("timesfm:batch:before_forecast")
        for start in range(0, len(inputs), safe_batch_size):
            batch_inputs = inputs[start : start + safe_batch_size]
            batch_tickers = valid_tickers[start : start + safe_batch_size]
            point_forecast, quantile_forecast = self.model.forecast(horizon=horizon, inputs=batch_inputs)
            for index, ticker in enumerate(batch_tickers):
                forecasts[ticker] = {
                    "point": np.asarray(point_forecast[index], dtype=np.float32),
                    "quantiles": np.asarray(quantile_forecast[index], dtype=np.float32),
                }
        guard_process_memory("timesfm:batch:after_forecast")
        return forecasts

    def forecast_quantiles(
        self,
        price_history: pd.Series,
        horizon: int = 64,
        context_length: int = DEFAULT_MAX_CONTEXT,
        min_context: int = DEFAULT_MIN_CONTEXT,
    ) -> dict[str, float]:
        """Forecast one asset and return cacheable annualized quantiles."""

        log_returns = _price_history_to_log_returns(price_history, min(context_length, self.max_context))
        if log_returns.size < min_context:
            raise ValueError(
                f"TimesFM requires at least {min_context} clean log returns; received {log_returns.size}."
            )
        guard_process_memory("timesfm:single:before_forecast")
        point_forecast, quantile_forecast = self.model.forecast(horizon=horizon, inputs=[log_returns])
        guard_process_memory("timesfm:single:after_forecast")
        forecast = {
            "point": np.asarray(point_forecast[0], dtype=np.float32),
            "quantiles": np.asarray(quantile_forecast[0], dtype=np.float32),
        }
        return _forecast_to_cache_quantiles(forecast, horizon_days=horizon)

    @staticmethod
    def expected_return(forecast: dict[str, np.ndarray], horizon_days: int = 21) -> float:
        """Annualize the cumulative TimesFM point forecast for one asset."""

        cumulative_log_return = float(np.sum(forecast["point"]))
        return cumulative_log_return * (252.0 / float(horizon_days))

    @staticmethod
    def forecast_vol(forecast: dict[str, np.ndarray]) -> float:
        """Infer annualized volatility from the TimesFM quantile band width."""

        quantiles = np.asarray(forecast["quantiles"], dtype=float)
        q10 = quantiles[:, 1]
        q90 = quantiles[:, -1]
        sigma_step = (q90 - q10) / 2.5631031311
        daily_sigma = float(np.sqrt(np.mean(np.square(sigma_step))))
        return daily_sigma * np.sqrt(252.0)

    @staticmethod
    def directional_score(forecast: dict[str, np.ndarray]) -> float:
        """Convert the forecast path into a bounded directional-conviction vote."""

        cumulative_mean = float(np.sum(forecast["point"]))
        quantiles = np.asarray(forecast["quantiles"], dtype=float)
        cumulative_q10 = float(np.sum(quantiles[:, 1]))
        cumulative_q90 = float(np.sum(quantiles[:, -1]))
        width = max(cumulative_q90 - cumulative_q10, 1e-9)
        return float(np.tanh(2.0 * cumulative_mean / width))


def get_or_compute_timesfm_quantiles(
    asset: str,
    decision_date: pd.Timestamp,
    price_history: pd.Series,
    forecaster: TimesFMForecaster | None = None,
    context_length: int = DEFAULT_MAX_CONTEXT,
    horizon: int = 64,
    model_version: str = DEFAULT_MODEL_VERSION,
    cache_path: Path = TIMESFM_CACHE_PATH,
) -> dict[str, float]:
    """Return q05..q95 quantiles for one `(asset, decision_date)` pair.

    Memory:
    - Cache hits use parquet predicate pushdown and never load the full file.
    - Cache misses compute one asset at a time.
    """

    cache_key = (
        f"{asset}|{_decision_date_iso(pd.Timestamp(decision_date))}|{context_length}|{horizon}|"
        f"{model_version}|timesfm"
    )
    cached = _read_cache_row(cache_path, cache_key)
    if cached is not None:
        return cached
    if forecaster is None:
        raise RuntimeError(
            f"Cache miss for {cache_key} but no forecaster provided. "
            "Caller must construct a forecaster once and pass it for cache misses."
        )

    quantiles = forecaster.forecast_quantiles(
        price_history=price_history,
        horizon=horizon,
        context_length=context_length,
    )
    _append_cache_row(
        cache_path=cache_path,
        cache_key=cache_key,
        asset=asset,
        decision_date=pd.Timestamp(decision_date),
        context_length=context_length,
        horizon=horizon,
        model_version=model_version,
        quantiles=quantiles,
    )
    return quantiles


def compute_vol_and_direction_signals(
    prices: pd.DataFrame,
    decision_date: pd.Timestamp,
    forecaster: TimesFMForecaster | None = None,
    horizon: int = 64,
    min_context: int = DEFAULT_MIN_CONTEXT,
    context_length: int = DEFAULT_MAX_CONTEXT,
    model_version: str = DEFAULT_MODEL_VERSION,
    cache_path: Path = TIMESFM_CACHE_PATH,
) -> pd.DataFrame:
    """Compute TimesFM return, volatility, and direction signals at date `t`.

    Inputs:
    - `prices`: audited asset price dataframe with gaps preserved as `NaN`.
    - `decision_date`: current rebalance date.
    - `forecaster`: optional pre-built TimesFM forecaster. When omitted, the
      function instantiates the model lazily on the first cache miss only.
    - `horizon`: forecast horizon in trading days.
    - `min_context`: minimum log-return history length required for a forecast.
    - `context_length`: retained history length sent into the model.
    - `model_version`: Hugging Face model identifier.
    - `cache_path`: shared parquet cache path.

    Outputs:
    - Dataframe indexed by ticker with columns
      `['mu_ann', 'sigma_ann_fcst', 'dir_score']`.

    Citation:
    - Hugging Face model card:
      https://huggingface.co/google/timesfm-2.5-200m-pytorch
    - Arrow parquet filtering:
      https://arrow.apache.org/docs/python/parquet.html#filtering

    Point-in-time safety:
    - Safe. Each series is truncated to `decision_date` before forecasting.
    """

    rows: list[dict[str, float | str]] = []
    owned_forecaster = forecaster

    try:
        for asset in prices.columns:
            price_history = prices.loc[:decision_date, asset].dropna()
            if price_history.shape[0] < min_context + 1:
                continue
            try:
                quantiles = get_or_compute_timesfm_quantiles(
                    asset=asset,
                    decision_date=pd.Timestamp(decision_date),
                    price_history=price_history,
                    forecaster=owned_forecaster,
                    context_length=context_length,
                    horizon=horizon,
                    model_version=model_version,
                    cache_path=cache_path,
                )
            except RuntimeError:
                if owned_forecaster is None:
                    owned_forecaster = TimesFMForecaster(
                        max_context=context_length,
                        max_horizon=horizon,
                        model_version=model_version,
                    )
                quantiles = get_or_compute_timesfm_quantiles(
                    asset=asset,
                    decision_date=pd.Timestamp(decision_date),
                    price_history=price_history,
                    forecaster=owned_forecaster,
                    context_length=context_length,
                    horizon=horizon,
                    model_version=model_version,
                    cache_path=cache_path,
                )

            rows.append(
                {
                    "ticker": asset,
                    "mu_ann": float(quantiles["q50"]),
                    "sigma_ann_fcst": _quantile_band_to_sigma(quantiles),
                    "dir_score": _quantile_band_to_direction(quantiles),
                }
            )
    finally:
        if forecaster is None and owned_forecaster is not None:
            del owned_forecaster
            gc.collect()
            guard_process_memory("timesfm:after_release")

    if not rows:
        return pd.DataFrame(columns=["mu_ann", "sigma_ann_fcst", "dir_score"])
    return pd.DataFrame(rows).set_index("ticker").sort_index()
