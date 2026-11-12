"""Turning simulated states into DataFrames in Almanac's layout.

Kept apart from the simulation so OHLC bars and volume can be added later.

A simulated path is a frame of **log returns** (see :mod:`fantasma.transforms`): index ``ts``
(UTC) or ``date`` (naive) as in Almanac, one row per step labelled at its end, columns named
``symbol``. ``fm.to_prices`` turns it into close prices, the wide form of Almanac's ``load()``,
``load(...)["close"].unstack("symbol")``; :func:`to_long` gives Almanac's long form of those.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import TimeGrid
from .process import Process
from .transforms import PRICE, QUANTITY, log_return_attrs

SOURCE = "fantasma"


def log_return_frame(process: Process, states: np.ndarray, grid: TimeGrid) -> pd.DataFrame:
    """Log returns of one path from its states ``(n_steps + 1, n_state)``.

    Row t is the return over the step ending at ``grid.index[t + 1]``; ``attrs`` record the
    start timestamp and the starting log prices.
    """
    log_prices = process.observe(states)
    columns = pd.Index(list(process.asset_names), name="symbol", dtype="string")
    frame = pd.DataFrame(
        np.diff(log_prices, axis=0), index=grid.index[1:], columns=columns, dtype="float64"
    )
    frame.attrs.update(
        log_return_attrs(grid.index[0], dict(zip(columns, log_prices[0], strict=True)))
    )
    return frame


def hidden_frame(process: Process, states: np.ndarray, grid: TimeGrid) -> pd.DataFrame:
    """Hidden state of one path, on the full grid: one row more than its log returns, so
    ``hidden[t - 1]`` is the state that drove the return at row t."""
    columns = pd.Index(list(process.hidden_states), name="state", dtype="string")
    return pd.DataFrame(process.hidden(states), index=grid.index, columns=columns, dtype="float64")


def to_long(prices: pd.DataFrame, source: str = SOURCE) -> pd.DataFrame:
    """A wide price frame in Almanac's long layout: indexed by (``ts``|``date``, ``symbol``),
    with ``close`` and ``source`` columns, sorted like ``almanac_data.load``.

    It takes prices: convert simulated log returns with ``fm.to_prices`` first.
    """
    held = prices.attrs.get(QUANTITY)
    if held is not None and held != PRICE:
        raise ValueError(f"to_long takes prices, got {held!r}; use fm.to_prices first")
    key = prices.index.name or "ts"
    long = prices.rename_axis(index=key, columns="symbol").stack().rename("close").to_frame()
    long["source"] = pd.Series(source, index=long.index, dtype="string")
    long.index = long.index.set_levels(long.index.levels[1].astype("string"), level="symbol")
    return long.sort_index()
