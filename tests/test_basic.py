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
    (a,) = fm.simulate(process, seed=1, **RUN)
    (b,) = fm.simulate(process, seed=2, **RUN)
    assert not np.allclose(a.to_numpy()[1:], b.to_numpy()[1:])


def test_paths_independent_and_stable(process):
    paths = fm.simulate(process, seed=3, n_paths=4, **RUN)
    assert not np.allclose(paths[0].to_numpy()[1:], paths[1].to_numpy()[1:])
    # Asking for more paths leaves the first ones unchanged.
    (first,) = fm.simulate(process, seed=3, n_paths=1, **RUN)
    pd.testing.assert_frame_equal(first, paths[0])


def test_shape_index_and_names(name, process):
    paths = fm.simulate(process, seed=0, n_paths=2, return_hidden=True, **RUN)
    assert len(paths) == 2
    prices, hidden = paths[0]
    assert prices.shape == (501, len(ASSETS[name]))
    assert list(prices.columns) == ASSETS[name]
    assert prices.columns.name == "symbol"
    assert prices.index.name == "ts"
    assert str(prices.index.dtype) == "datetime64[ns, UTC]"
    assert prices.index[0] == pd.Timestamp("2024-01-01", tz="UTC")
    assert (prices.index.to_series().diff().dropna() == pd.Timedelta(minutes=10)).all()
    assert (prices.dtypes == "float64").all()
    assert list(hidden.columns) == HIDDEN[name]
    assert hidden.index.equals(prices.index)


def test_daily_index(process):
    (prices,) = fm.simulate(process, start="2024-01-01 15:30", freq="1d", end="2024-03-01", seed=0)
    assert prices.index.name == "date"
    assert str(prices.index.dtype) == "datetime64[ns]"
    assert prices.index[0] == pd.Timestamp("2024-01-01")
    assert prices.index[-1] == pd.Timestamp("2024-03-01")


def test_finite_and_not_exploding(process):
    ((prices, hidden),) = fm.simulate(
        process, start="2024-01-01", freq="1d", n_steps=3650, seed=11, return_hidden=True
    )
    assert np.isfinite(prices.to_numpy()).all() and np.isfinite(hidden.to_numpy()).all()
    assert (prices > 0).all().all()
    assert (prices.max() < 1e5).all() and (prices.min() > 1e-1).all()


def test_long_format_matches_almanac(process):
    (prices,) = fm.simulate(process, seed=0, **RUN)
    long = fm.to_long(prices)
    assert list(long.index.names) == ["ts", "symbol"]
    assert list(long.columns) == ["close", "source"]
    assert long.index.is_monotonic_increasing
    assert (long["source"] == "fantasma").all()
    pd.testing.assert_frame_equal(long["close"].unstack("symbol"), prices, check_names=False)


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
