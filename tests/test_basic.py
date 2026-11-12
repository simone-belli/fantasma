"""Checks required of every process (SPEC section 8, basic checks)."""

import numpy as np
import pandas as pd
import pytest
from conftest import ASSETS, HIDDEN

import fantasma as fm

RUN = dict(start="2024-01-01", freq="10m", n_steps=500)


def test_same_seed_same_output(process):
    a = fm.simulate(process, seed=7, n_paths=3, return_hidden=True, **RUN)
    b = fm.simulate(process, seed=7, n_paths=3, return_hidden=True, **RUN)
    for (pa, ha), (pb, hb) in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(pa, pb)
        pd.testing.assert_frame_equal(ha, hb)


def test_different_seeds_differ(process):
    a = fm.simulate(process, seed=1, **RUN)
    b = fm.simulate(process, seed=2, **RUN)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_paths_independent_and_stable(process):
    paths = fm.simulate(process, seed=3, n_paths=4, **RUN)
    assert not np.allclose(paths[0].to_numpy(), paths[1].to_numpy())
    # Asking for more paths leaves the first ones unchanged.
    first = fm.simulate(process, seed=3, n_paths=1, **RUN)
    pd.testing.assert_frame_equal(first, paths[0])


def test_shape_index_and_names(name, process):
    paths = fm.simulate(process, seed=0, n_paths=2, return_hidden=True, **RUN)
    assert len(paths) == 2
    log_returns, hidden = paths[0]
    start = pd.Timestamp("2024-01-01", tz="UTC")
    # One row per step, labelled at the end of the step: the start is not a row.
    assert log_returns.shape == (500, len(ASSETS[name]))
    assert list(log_returns.columns) == ASSETS[name]
    assert log_returns.columns.name == "symbol"
    assert log_returns.index.name == "ts"
    assert str(log_returns.index.dtype) == "datetime64[ns, UTC]"
    assert log_returns.index[0] == start + pd.Timedelta(minutes=10)
    assert (log_returns.index.to_series().diff().dropna() == pd.Timedelta(minutes=10)).all()
    assert (log_returns.dtypes == "float64").all()
    assert log_returns.attrs["quantity"] == "log_return"
    assert log_returns.attrs["start"] == start
    assert set(log_returns.attrs["initial_log_price"]) == set(ASSETS[name])
    # Hidden state keeps the full grid, start included.
    assert list(hidden.columns) == HIDDEN[name]
    assert hidden.index[0] == start and hidden.index[1:].equals(log_returns.index)


def test_daily_index(process):
    log_returns = fm.simulate(
        process, start="2024-01-01 15:30", freq="1d", end="2024-03-01", seed=0
    )
    assert log_returns.index.name == "date"
    assert str(log_returns.index.dtype) == "datetime64[ns]"
    assert log_returns.attrs["start"] == pd.Timestamp("2024-01-01")
    assert log_returns.index[0] == pd.Timestamp("2024-01-02")
    assert log_returns.index[-1] == pd.Timestamp("2024-03-01")


def test_finite_and_not_exploding(process):
    log_returns, hidden = fm.simulate(
        process, start="2024-01-01", freq="1d", n_steps=3650, seed=11, return_hidden=True
    )
    prices = fm.to_prices(log_returns)
    assert np.isfinite(log_returns.to_numpy()).all() and np.isfinite(hidden.to_numpy()).all()
    assert (prices > 0).all().all()
    assert (prices.max() < 1e5).all() and (prices.min() > 1e-1).all()


def test_long_format_matches_almanac(process):
    log_returns = fm.simulate(process, seed=0, **RUN)
    with pytest.raises(ValueError, match="to_prices"):
        fm.to_long(log_returns)
    prices = fm.to_prices(log_returns)
    long = fm.to_long(prices)
    assert list(long.index.names) == ["ts", "symbol"]
    assert list(long.columns) == ["close", "source"]
    assert long.index.is_monotonic_increasing
    assert (long["source"] == "fantasma").all()
    pd.testing.assert_frame_equal(
        long["close"].unstack("symbol"), prices, check_names=False, check_flags=False
    )


@pytest.mark.parametrize(
    "make",
    [
        lambda: fm.RandomWalk(sigma=-0.1),
        lambda: fm.RandomWalk(sigma=float("nan")),
        lambda: fm.RandomWalk(sigma=0.1, initial_price=0),
        lambda: fm.RandomWalk(sigma=0.1, asset=""),
        lambda: fm.DriftPlusNoise(mu="x", sigma=0.1),
        lambda: fm.DriftPlusNoise(mu=float("inf"), sigma=0.1),
        lambda: fm.OrnsteinUhlenbeck(theta=0, half_life=0, sigma=0.1),
        lambda: fm.OrnsteinUhlenbeck(theta=0, half_life=-1, sigma=0.1),
        lambda: fm.OrnsteinUhlenbeck(theta=0, half_life=1, sigma=-0.1),
        lambda: fm.Cointegrated([[1], [2]], 0.1, 0.1, 0.1, ["A"]),  # too few names
        lambda: fm.Cointegrated([[1], [2]], 0.1, 0.1, 0.1, ["A", "A"]),  # duplicate names
        lambda: fm.Cointegrated([[1, 2], [2, 4], [3, 6]], 0.1, 0.1, 0.1, list("ABC")),  # rank
        lambda: fm.Cointegrated([[1, 0], [0, 1]], 0.1, 0.1, 0.1, list("AB")),  # K == N
        lambda: fm.Cointegrated([[1], [2]], [0.1, -1], 0.1, 0.1, list("AB")),
        lambda: fm.Cointegrated([[1], [2]], [0.1, 0.1, 0.1], 0.1, 0.1, list("AB")),
        lambda: fm.Cointegrated([[1], [2]], 0.1, -0.1, 0.1, list("AB")),
        lambda: fm.StochasticDrift(sigma=0.2, drift_mean=0, drift_half_life=0, drift_sigma=0.1),
        lambda: fm.StochasticDrift(sigma=-1, drift_mean=0, drift_half_life=1, drift_sigma=0.1),
    ],
)
def test_parameter_validation(make):
    with pytest.raises(ValueError):
        make()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(start="2024-01-01", freq="10m"),  # neither n_steps nor end
        dict(start="2024-01-01", freq="10m", n_steps=10, end="2024-02-01"),  # both
        dict(start="2024-01-01", freq="10m", n_steps=0),
        dict(start="2024-01-01", freq="10m", n_steps=1.5),
        dict(start="2024-02-01", freq="1d", end="2024-01-01"),
        dict(start="2024-01-01", freq="bogus", n_steps=10),
        dict(start="2024-01-01", freq="10m", n_steps=10, n_paths=0),
    ],
)
def test_run_validation(kwargs):
    with pytest.raises(ValueError):
        fm.simulate(fm.RandomWalk(0.1), seed=0, **kwargs)


def test_seed_required():
    with pytest.raises(ValueError):
        fm.simulate(fm.RandomWalk(0.1), start="2024-01-01", freq="1d", n_steps=5, seed=None)


def test_single_path_is_returned_itself():
    one = fm.simulate(fm.RandomWalk(0.2), seed=3, **RUN)  # n_paths=1 is the default
    assert isinstance(one, pd.DataFrame)
    log_returns, hidden = fm.simulate(fm.RandomWalk(0.2), seed=3, return_hidden=True, **RUN)
    assert isinstance(log_returns, pd.DataFrame) and isinstance(hidden, pd.DataFrame)
    many = fm.simulate(fm.RandomWalk(0.2), seed=3, n_paths=2, **RUN)
    assert isinstance(many, list) and len(many) == 2
    pd.testing.assert_frame_equal(one, many[0])
    grid = fm.RegularGrid(RUN["start"], RUN["freq"], n_steps=RUN["n_steps"])
    pd.testing.assert_frame_equal(fm.simulate_on(fm.RandomWalk(0.2), grid, seed=3), one)
    assert len(fm.simulate_on(fm.RandomWalk(0.2), grid, seed=3, n_paths=2)) == 2
