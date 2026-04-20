# Provides lightweight pandas helpers shared across data, analysis, and
# reporting modules without introducing circular imports.
"""Small pandas/numpy helpers for Whitmore.

References:
- NumPy indexing documentation: https://numpy.org/doc/stable/
- pandas object construction documentation: https://pandas.pydata.org/

Point-in-time safety:
- Safe. These helpers perform deterministic array transformations and do not
  change what information is visible at decision time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_propagate(obj: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Carry the last observed value forward without using pandas fill methods.

    Inputs:
    - `obj`: pandas Series or DataFrame.

    Outputs:
    - Object of the same shape/type with each missing row replaced by the most
      recent observed value in that column, while leading missing values remain
      missing.

    Citation:
    - NumPy advanced indexing documentation: https://numpy.org/doc/stable/

    Point-in-time safety:
    - Safe when used only to align already-observed data onto a denser calendar.
      It never looks ahead; each row sees only prior observed values.
    """

    is_series = isinstance(obj, pd.Series)
    frame = obj.to_frame() if is_series else obj.copy()
    values = frame.to_numpy(copy=True, dtype=object)
    missing = pd.isna(values)
    row_numbers = np.arange(values.shape[0], dtype=int)[:, None]
    last_seen = np.where(~missing, row_numbers, -1)
    np.maximum.accumulate(last_seen, axis=0, out=last_seen)
    col_numbers = np.broadcast_to(np.arange(values.shape[1], dtype=int), values.shape)

    propagated = np.empty_like(values, dtype=object)
    propagated[:] = np.nan
    valid = last_seen >= 0
    propagated[valid] = values[last_seen[valid], col_numbers[valid]]

    result = pd.DataFrame(propagated, index=frame.index, columns=frame.columns)
    for column in frame.columns:
        try:
            result[column] = result[column].astype(frame[column].dtype, copy=False)
        except (TypeError, ValueError):
            continue
    if is_series:
        return result.iloc[:, 0].rename(obj.name)
    return result
