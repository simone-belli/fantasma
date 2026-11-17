"""Type aliases shared across modules (for annotations only)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# Anything ``pd.Timestamp`` accepts as an instant (``datetime`` and ``Timestamp`` are ``date``s).
type TimestampLike = str | dt.date | np.datetime64 | pd.Timestamp

# A step length: an Almanac name (``"10m"``), any pandas offset alias, offset or timedelta.
type FreqLike = str | pd.DateOffset | pd.Timedelta | dt.timedelta | np.timedelta64
