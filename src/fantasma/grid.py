"""The time grid: maps abstract simulation steps to timestamps and year fractions.

Two grids exist. :class:`RegularGrid` is evenly spaced continuous time, with no sessions,
holidays or overnight gaps. :class:`IndexGrid` runs on given timestamps, gaps included: it is
the grid when data sources are among a model's sources. The simulator only asks a grid for its
``index`` and the per-step ``dts``, so a session calendar can be added later as another
``TimeGrid``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# Almanac's frequency names (``almanac_data.schema``), plus its older ``daily`` spelling.
ALMANAC_FREQUENCIES = {
    "1m": pd.Timedelta(minutes=1),
    "10m": pd.Timedelta(minutes=10),
    "1h": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1),
    "daily": pd.Timedelta(days=1),
}

DAYS_PER_YEAR = 365.25  # a continuous grid runs every calendar day


class TimeGrid(ABC):
    @property
    @abstractmethod
    def index(self) -> pd.DatetimeIndex:
        """One timestamp per row: the start, then one after each step."""

    @property
    @abstractmethod
    def dts(self) -> np.ndarray:
        """Year fraction of each step, shape ``(n_steps,)``."""

    @property
    def n_steps(self) -> int:
        return len(self.index) - 1


def parse_freq(freq) -> pd.Timedelta:
    """A step length from an Almanac name (``"10m"``, ``"1d"``) or any pandas offset/timedelta."""
    if isinstance(freq, str) and freq in ALMANAC_FREQUENCIES:
        return ALMANAC_FREQUENCIES[freq]
    try:
        step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    except (TypeError, ValueError):
        try:
            step = pd.Timedelta(freq)
        except (TypeError, ValueError):
            raise ValueError(f"unknown frequency {freq!r}") from None
    if step <= pd.Timedelta(0):
        raise ValueError(f"frequency must be positive, got {freq!r}")
    return step


class RegularGrid(TimeGrid):
    """Evenly spaced timestamps from ``start``, ``n_steps`` steps long or running to ``end``.

    Index conventions follow Almanac: intraday steps give a UTC ``ts`` index (naive starts are
    taken as UTC); steps of a whole number of days give a naive ``date`` index, normalised to
    midnight. ``dt`` is the step length over a year of ``days_per_year`` days.
    """

    def __init__(
        self,
        start,
        freq,
        n_steps: int | None = None,
        end=None,
        days_per_year: float = DAYS_PER_YEAR,
    ):
        if (n_steps is None) == (end is None):
            raise ValueError("give exactly one of n_steps and end")
        if days_per_year <= 0:
            raise ValueError("days_per_year must be positive")
        self.step = parse_freq(freq)
        self.daily = self.step % pd.Timedelta(days=1) == pd.Timedelta(0)
        start = pd.Timestamp(start)
        if self.daily:
            start = (start.tz_convert("UTC").tz_localize(None) if start.tz else start).normalize()
        else:
            start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
        if end is not None:
            end = pd.Timestamp(end)
            if self.daily:
                end = end.tz_convert("UTC").tz_localize(None) if end.tz else end
            else:
                end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
            if end < start:
                raise ValueError("end is before start")
            n_steps = int((end - start) // self.step)
        if not isinstance(n_steps, (int, np.integer)) or isinstance(n_steps, bool) or n_steps < 1:
            raise ValueError(f"n_steps must be a positive integer, got {n_steps!r}")
        self.start = start
        self._n = int(n_steps)
        self.days_per_year = float(days_per_year)
        self.dt = self.step / pd.Timedelta(days=self.days_per_year)

    @property
    def index(self) -> pd.DatetimeIndex:
        idx = pd.date_range(self.start, periods=self._n + 1, freq=self.step)
        if self.daily:
            return pd.DatetimeIndex(idx.astype("datetime64[ns]"), name="date")
        return pd.DatetimeIndex(idx.astype("datetime64[ns, UTC]"), name="ts")

    @property
    def dts(self) -> np.ndarray:
        return np.full(self._n, self.dt)


class IndexGrid(TimeGrid):
    """A grid on given timestamps, e.g. the index of real data, gaps included.

    ``index`` holds the end of each step (the rows of a log-return frame); the grid's own
    index is ``start`` followed by them, and each ``dt`` is the gap to the previous instant
    over a year of ``days_per_year`` days. Without ``start``, the first gap is assumed to
    repeat before the first row: ``start = index[0] - (index[1] - index[0])``.
    """

    def __init__(self, index, start=None, days_per_year: float = DAYS_PER_YEAR):
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("an IndexGrid needs a DatetimeIndex")
        if len(index) < 1 or (start is None and len(index) < 2):
            raise ValueError("an IndexGrid needs at least two instants, counting the start")
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("the index must be strictly increasing")
        if days_per_year <= 0:
            raise ValueError("days_per_year must be positive")
        start = index[0] - (index[1] - index[0]) if start is None else pd.Timestamp(start)
        if (start.tz is None) != (index.tz is None):
            raise ValueError(f"start {start} and the index must both have, or lack, a timezone")
        if start >= index[0]:
            raise ValueError(f"start {start} must be before the first row {index[0]}")
        self.start = start
        self._index = index.insert(0, start)
        self.days_per_year = float(days_per_year)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self._index

    @property
    def dts(self) -> np.ndarray:
        # Timedelta arithmetic, not asi8: the index may be in s/ms/us resolution, not ns.
        gaps = self._index[1:] - self._index[:-1]
        return np.asarray(gaps / pd.Timedelta(days=self.days_per_year), dtype=float)
