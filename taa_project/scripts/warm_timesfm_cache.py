"""Warm the shared TimesFM parquet cache across the monthly decision grid.

References:
- TimesFM model card: https://huggingface.co/google/timesfm-2.5-200m-pytorch
- Arrow parquet filtering:
  https://arrow.apache.org/docs/python/parquet.html#filtering

Point-in-time safety:
- Safe. Every cache row is built from the asset history available on or before
  its decision date.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import torch as _torch  # noqa: F401  # Pin the OpenMP runtime before sklearn / hmmlearn imports.
except ImportError:
    _torch = None

import pandas as pd

from taa_project.backtest.walkforward import build_monthly_decision_dates
from taa_project.config import ALL_SAA, OUTPUT_DIR, TIMESFM_CACHE_PATH
from taa_project.data_loader import load_prices
from taa_project.memory import guard_process_memory
from taa_project.signals.vol_timesfm import (
    DEFAULT_MAX_CONTEXT,
    DEFAULT_MIN_CONTEXT,
    DEFAULT_MODEL_VERSION,
    TimesFMForecaster,
    get_or_compute_timesfm_quantiles,
    timesfm_is_available,
)


def warm_timesfm_cache(
    start: str = "2003-01-01",
    end: str = "2025-12-31",
    folds: int = 5,
    context_length: int = DEFAULT_MAX_CONTEXT,
    horizon: int = 64,
    model_version: str = DEFAULT_MODEL_VERSION,
    cache_path: Path = TIMESFM_CACHE_PATH,
) -> Path:
    """Populate the shared TimesFM cache across the monthly decision grid.

    Inputs:
    - `start`, `end`: decision-date bounds.
    - `folds`: accepted for CLI parity with the sweep command; the cache itself
      is independent of fold count because it is keyed only by decision date.
    - `context_length`: model context length used in the cache key.
    - `horizon`: forecast horizon used in the cache key.
    - `model_version`: Hugging Face model identifier used in the cache key.
    - `cache_path`: parquet cache destination.

    Outputs:
    - Path to the warmed parquet file.

    Citation:
    - TimesFM model card: https://huggingface.co/google/timesfm-2.5-200m-pytorch

    Point-in-time safety:
    - Safe. The script truncates every asset history at each decision date.
    """

    del folds  # Decision dates do not depend on fold partitioning.
    if not timesfm_is_available():
        raise RuntimeError(
            "TimesFM is not installed. Warm the cache from the configured runtime that has the TimesFM dependency."
        )

    prices = load_prices().loc[:, ALL_SAA]
    decision_dates = build_monthly_decision_dates(prices, start, end)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    guard_process_memory("warm_timesfm_cache:start")

    for asset in ALL_SAA:
        forecaster: TimesFMForecaster | None = None
        try:
            asset_prices = prices.loc[:, asset]
            for decision_date in decision_dates:
                history = asset_prices.loc[:decision_date].dropna()
                if history.shape[0] < DEFAULT_MIN_CONTEXT + 1:
                    continue
                try:
                    get_or_compute_timesfm_quantiles(
                        asset=asset,
                        decision_date=pd.Timestamp(decision_date),
                        price_history=history,
                        forecaster=forecaster,
                        context_length=context_length,
                        horizon=horizon,
                        model_version=model_version,
                        cache_path=cache_path,
                    )
                except RuntimeError:
                    if forecaster is None:
                        forecaster = TimesFMForecaster(
                            max_context=context_length,
                            max_horizon=horizon,
                            model_version=model_version,
                        )
                    get_or_compute_timesfm_quantiles(
                        asset=asset,
                        decision_date=pd.Timestamp(decision_date),
                        price_history=history,
                        forecaster=forecaster,
                        context_length=context_length,
                        horizon=horizon,
                        model_version=model_version,
                        cache_path=cache_path,
                    )
                guard_process_memory(f"warm_timesfm_cache:{asset}:{pd.Timestamp(decision_date).date()}")
        finally:
            if forecaster is not None:
                del forecaster
                gc.collect()
                guard_process_memory(f"warm_timesfm_cache:{asset}:after_release")

    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm the shared TimesFM parquet cache.")
    parser.add_argument("--start", default="2003-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--context-length", type=int, default=DEFAULT_MAX_CONTEXT)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--cache-path", default=str(TIMESFM_CACHE_PATH))
    args = parser.parse_args()

    cache_path = warm_timesfm_cache(
        start=args.start,
        end=args.end,
        folds=args.folds,
        context_length=args.context_length,
        horizon=args.horizon,
        model_version=args.model_version,
        cache_path=Path(args.cache_path),
    )
    print(f"Warmed TimesFM cache at {cache_path}")


if __name__ == "__main__":
    main()
