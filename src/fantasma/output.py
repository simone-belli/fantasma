"""Turning simulated states into Almanac-format DataFrames.

Kept apart from the simulation so OHLC bars and volume can be added later.

Almanac's ``load()`` returns long frames indexed by (``date`` or ``ts``, ``symbol``); the wide
form, one column per symbol, is ``load(...)["close"].unstack("symbol")``. Fantasma's per-path
frame is exactly that wide form: index ``ts`` (UTC) or ``date`` (naive), columns named
``symbol``. :func:`to_long` gives the long form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import TimeGrid
from .process import Process

SOURCE = "fantasma"


def price_frame(process: Process, states: np.ndarray, grid: TimeGrid) -> pd.DataFrame:
    """Close prices of one path from its states ``(n_rows, n_state)``."""
    prices = np.exp(process.observe(states))
    columns = pd.Index(list(process.asset_names), name="symbol", dtype="string")
    return pd.DataFrame(prices, index=grid.index, columns=columns, dtype="float64")


def hidden_frame(process: Process, states: np.ndarray, grid: TimeGrid) -> pd.DataFrame:
    """Hidden state of one path, on the same index as its prices."""
    columns = pd.Index(list(process.hidden_states), name="state", dtype="string")
    return pd.DataFrame(process.hidden(states), index=grid.index, columns=columns, dtype="float64")


def to_long(prices: pd.DataFrame, source: str = SOURCE) -> pd.DataFrame:
    """A wide price frame in Almanac's long layout: indexed by (``ts``|``date``, ``symbol``),
    with ``close`` and ``source`` columns, sorted like ``almanac_data.load``."""
    key = prices.index.name or "ts"
    long = prices.rename_axis(index=key, columns="symbol").stack().rename("close").to_frame()
    long["source"] = pd.Series(source, index=long.index, dtype="string")
    long.index = long.index.set_levels(long.index.levels[1].astype("string"), level="symbol")
    return long.sort_index()
