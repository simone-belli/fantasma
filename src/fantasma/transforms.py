"""Fantasma's contract, and helpers to get other quantities from it.

**Every simulation produces log returns** (log price differences): one row per step, labelled
at the end of the step, one column per asset. A path of ``n_steps`` steps has ``n_steps``
rows; the start instant is not a row. The frame's ``attrs`` record where the path starts:

- ``attrs["quantity"] = "log_return"``
- ``attrs["start"]``: the start timestamp
- ``attrs["initial_log_price"]``: ``{column: log price at the start}``

The helpers turn log returns into other quantities:

====================  =============================================  =================
helper                gives                                          rows
====================  =============================================  =================
:func:`to_prices`     ``P[t] = P0 * exp(r[1] + ... + r[t])``         start + one/step
:func:`to_log_prices` ``log P[t] = log P0 + r[1] + ... + r[t]``      start + one/step
:func:`to_returns`    ``P[t] / P[t-1] - 1 = expm1(r[t])``            one per step
:func:`from_prices`   log returns from prices (e.g. Almanac data)    one per step
====================  =============================================  =================

The level helpers take the start and the starting price from ``attrs`` by default; pass
``start=`` / ``initial_price=`` to override them, or when the frame does not carry them (pandas
drops ``attrs`` after some operations, and driven models have no natural starting level).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

QUANTITY = "quantity"
START = "start"
INITIAL_LOG_PRICE = "initial_log_price"

PRICE = "price"
LOG_PRICE = "log_price"
RETURN = "return"
LOG_RETURN = "log_return"


def per_column(name: str, value, columns: pd.Index, non_negative: bool = False) -> np.ndarray:
    """A scalar, a sequence with one value per column, or a Series/dict keyed by column."""
    n = len(columns)
    if isinstance(value, Mapping):
        value = pd.Series(value)
    if isinstance(value, pd.Series):
        if set(value.index) != set(columns) or len(value) != n:
            raise ValueError(f"{name}: keys {list(value.index)} must be the columns")
        value = value.reindex(columns).to_numpy()
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric, got {value!r}") from None
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    elif arr.shape != (n,):
        raise ValueError(f"{name} must be a scalar or have one value per column ({n})")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if non_negative and np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    return arr


def log_return_attrs(start, initial_log_price: Mapping | None) -> dict:
    """The ``attrs`` of a log-return frame."""
    attrs = {QUANTITY: LOG_RETURN}
    if start is not None:
        attrs[START] = pd.Timestamp(start)
    if initial_log_price is not None:
        attrs[INITIAL_LOG_PRICE] = {k: float(v) for k, v in initial_log_price.items()}
    return attrs


def _expect(frame, quantity: str, hint: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(frame).__name__}")
    held = frame.attrs.get(QUANTITY)
    if held is not None and held != quantity:
        raise ValueError(f"expected {quantity}, got a frame holding {held!r}; {hint}")


def _labelled(values, index, columns, quantity: str) -> pd.DataFrame:
    frame = pd.DataFrame(values, index=index, columns=columns, dtype="float64")
    frame.attrs[QUANTITY] = quantity
    return frame


def to_log_prices(log_returns: pd.DataFrame, initial_price=None, start=None) -> pd.DataFrame:
    """Log prices from log returns, with the start as an extra first row.

    ``initial_price`` is the price at the start (not its log): a scalar, one value per column,
    or a Series/dict keyed by column. ``start`` is the start timestamp. Both default to the
    frame's ``attrs``. NaN in a return propagates to every later row.
    """
    _expect(log_returns, LOG_RETURN, "to_log_prices takes log returns")
    columns, index = log_returns.columns, log_returns.index
    if initial_price is None:
        recorded = log_returns.attrs.get(INITIAL_LOG_PRICE)
        if recorded is None:
            raise ValueError(
                "initial_price is needed: the frame does not record its starting price"
            )
        try:
            log0 = np.array([recorded[c] for c in columns], dtype=float)
        except KeyError:
            raise ValueError(
                "the recorded starting prices do not match the columns; pass initial_price"
            ) from None
    else:
        p0 = per_column("initial_price", initial_price, columns)
        if np.any(p0 <= 0):
            raise ValueError("initial_price must be positive")
        log0 = np.log(p0)
    if start is None:
        start = log_returns.attrs.get(START)
        if start is None:
            raise ValueError("start is needed: the frame does not record its start timestamp")
    start = pd.Timestamp(start)
    if isinstance(index, pd.DatetimeIndex):
        if (start.tz is None) != (index.tz is None):
            raise ValueError(f"start {start} and the index must both have, or lack, a timezone")
        if len(index) and start >= index[0]:
            raise ValueError(f"start {start} must be before the first row {index[0]}")
    levels = np.vstack([log0, log0 + np.cumsum(log_returns.to_numpy(dtype=float), axis=0)])
    return _labelled(levels, index.insert(0, start), columns, LOG_PRICE)


def to_prices(log_returns: pd.DataFrame, initial_price=None, start=None) -> pd.DataFrame:
    """Prices from log returns, with the start as an extra first row (see
    :func:`to_log_prices` for the arguments)."""
    _expect(log_returns, LOG_RETURN, "to_prices takes log returns")
    log_prices = to_log_prices(log_returns, initial_price, start)
    return _labelled(np.exp(log_prices.to_numpy()), log_prices.index, log_prices.columns, PRICE)


def to_returns(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Relative returns ``P[t] / P[t-1] - 1``, on the same rows as the log returns."""
    _expect(log_returns, LOG_RETURN, "to_returns takes log returns")
    values = np.expm1(log_returns.to_numpy(dtype=float))
    return _labelled(values, log_returns.index, log_returns.columns, RETURN)


def from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns from prices, e.g. real data to feed a driven model.

    The first row becomes the start: it is dropped from the rows and recorded in ``attrs``
    with its prices, so ``to_prices(from_prices(p))`` gives ``p`` back.
    """
    _expect(prices, PRICE, "from_prices takes prices")
    if len(prices) < 2:
        raise ValueError("prices need at least two rows")
    values = prices.to_numpy(dtype=float)
    if np.any(values[~np.isnan(values)] <= 0):
        raise ValueError("prices must be positive to take logs")
    if np.isnan(values[0]).any():
        raise ValueError("the first row of prices is the start and must be complete")
    log_p = np.log(values)
    frame = pd.DataFrame(
        np.diff(log_p, axis=0), index=prices.index[1:], columns=prices.columns, dtype="float64"
    )
    frame.attrs.update(
        log_return_attrs(prices.index[0], dict(zip(prices.columns, log_p[0], strict=True)))
    )
    return frame
