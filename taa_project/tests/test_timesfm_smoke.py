"""Task 1 smoke test for the optional TimesFM runtime."""

from __future__ import annotations

import numpy as np
import pytest

from taa_project.signals.vol_timesfm import TimesFMForecaster, timesfm_is_available


def test_timesfm_smoke_forecast_shape() -> None:
    if not timesfm_is_available():
        pytest.skip("timesfm not installed")

    import torch

    torch.set_num_threads(1)
    history = np.random.default_rng(42).normal(0.0, 0.01, 512).astype(np.float32)
    forecaster = TimesFMForecaster(max_context=512, max_horizon=64, torch_compile=False)
    _, quantile_forecast = forecaster.model.forecast(inputs=[history], horizon=64)

    assert np.asarray(quantile_forecast).shape == (1, 64, 10)
