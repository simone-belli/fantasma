import numpy as np
import pandas as pd
import pytest

import fantasma as fm


@pytest.mark.parametrize(
    "freq, minutes",
    [("1m", 1), ("10m", 10), ("1h", 60), ("1d", 1440), ("daily", 1440), ("10min", 10), ("2h", 120)],
)
def test_dt_from_frequency(freq, minutes):
    grid = fm.RegularGrid("2024-01-01", freq, n_steps=3)
    assert grid.dt == pytest.approx(minutes / (365.25 * 1440))
    np.testing.assert_allclose(grid.dts, grid.dt)


def test_end_date_gives_inclusive_index():
    grid = fm.RegularGrid("2024-01-01", "1h", end="2024-01-02")
    assert grid.n_steps == 24
    assert grid.index[-1] == pd.Timestamp("2024-01-02", tz="UTC")


def test_timezone_start_converted_to_utc():
    grid = fm.RegularGrid(pd.Timestamp("2024-01-01 09:00", tz="Europe/Rome"), "10m", n_steps=1)
    assert grid.index[0] == pd.Timestamp("2024-01-01 08:00", tz="UTC")


def test_days_per_year_override():
    (a,) = fm.simulate(fm.RandomWalk(0.2), start="2024-01-01", freq="1d", n_steps=3, seed=0)
    grid = fm.RegularGrid("2024-01-01", "1d", n_steps=3, days_per_year=252)
    assert grid.dt == pytest.approx(1 / 252)
    (b,) = fm.simulate_on(fm.RandomWalk(0.2), grid, seed=0)
    assert not np.allclose(a.to_numpy()[1:], b.to_numpy()[1:])
