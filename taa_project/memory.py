"""Shared process-memory guardrails for the Whitmore pipeline.

References:
- psutil process-memory API:
  https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info

Point-in-time safety:
- Safe. This module inspects the current process only and does not touch data.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
import psutil

from taa_project.config import MAX_PROCESS_RSS_GB, MEMORY_BREACH_LOG


def current_rss_gb() -> float:
    """Return the current process RSS in gigabytes."""

    return float(psutil.Process().memory_info().rss) / 1024**3


def guard_process_memory(
    step: str,
    limit_gb: float = MAX_PROCESS_RSS_GB,
    log_path: Path = MEMORY_BREACH_LOG,
) -> float:
    """Raise `MemoryError` if RSS exceeds the configured budget.

    Inputs:
    - `step`: human-readable step label written into the breach log.
    - `limit_gb`: maximum allowed resident set size in gigabytes.
    - `log_path`: destination log file for breach rows.

    Outputs:
    - Current RSS in gigabytes when within budget.

    Citation:
    - psutil process-memory API:
      https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info

    Point-in-time safety:
    - Safe. This is operational monitoring only.
    """

    rss_gb = current_rss_gb()
    if rss_gb <= limit_gb:
        return rss_gb

    gc.collect()
    rss_gb = current_rss_gb()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{pd.Timestamp.utcnow().isoformat()} | step={step} | rss_gb={rss_gb:.3f} | limit_gb={limit_gb:.3f}\n"
        )
    raise MemoryError(
        f"Process RSS exceeded the {limit_gb:.2f} GB budget at step '{step}': {rss_gb:.3f} GB"
    )
