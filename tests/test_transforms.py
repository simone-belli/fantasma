"""The log-return contract and the helpers that turn log returns into other quantities."""

import numpy as np
import pandas as pd
import pytest

import fantasma as fm

RUN = dict(start="2024-01-01", freq="10m", n_steps=300)


def test_to_prices_rebuilds_the_true_price_path(process):
    """``to_prices`` of the simulated log returns is exactly ``exp`` of the model's log prices,
    start row included: for OU too, whose dynamics depend on the level."""
    log_returns = fm.simulate(process, seed=4, **RUN)
    grid = fm.RegularGrid(**RUN)
    (states,) = fm.simulate_states(process, grid, seed=4)
    truth = np.exp(process.observe(states))

    prices = fm.to_prices(log_returns)
    assert prices.index.equals(grid.index)
    assert prices.index.name == "ts" and prices.columns.equals(log_returns.columns)
    np.testing.assert_allclose(prices.to_numpy(), truth, rtol=1e-12)
    assert prices.attrs["quantity"] == "price"

    log_prices = fm.to_log_prices(log_returns)
    np.testing.assert_allclose(log_prices.to_numpy(), process.observe(states), atol=1e-12)
    assert log_prices.attrs["quantity"] == "log_price"


def test_to_returns(process):
    log_returns = fm.simulate(process, seed=4, **RUN)
    returns = fm.to_returns(log_returns)
    prices = fm.to_prices(log_returns)
    assert returns.index.equals(log_returns.index)
    assert returns.attrs["quantity"] == "return"
    np.testing.assert_allclose(returns, prices.pct_change().iloc[1:], rtol=1e-9, atol=1e-14)


def test_from_prices_round_trip(process):
    log_returns = fm.simulate(process, seed=4, **RUN)
    back = fm.from_prices(fm.to_prices(log_returns))
    np.testing.assert_allclose(back.to_numpy(), log_returns.to_numpy(), atol=1e-12)
    assert back.index.equals(log_returns.index)
    assert back.attrs["quantity"] == "log_return"
    assert back.attrs["start"] == log_returns.attrs["start"]
    for col, value in log_returns.attrs["initial_log_price"].items():
        assert back.attrs["initial_log_price"][col] == pytest.approx(value, abs=1e-12)


def test_from_prices_accepts_unlabelled_real_data():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"], name="date")
    prices = pd.DataFrame({"ES": [100.0, 101.0, 99.0]}, index=idx)
    log_returns = fm.from_prices(prices)
    np.testing.assert_allclose(log_returns["ES"], np.diff(np.log([100.0, 101.0, 99.0])))
    pd.testing.assert_frame_equal(fm.to_prices(log_returns), prices, check_flags=False)


@pytest.mark.parametrize(
    "initial_price, expected",
    [
        (50.0, [50.0, 50.0]),
        ([10.0, 20.0], [10.0, 20.0]),
        (pd.Series({"Y": 20.0, "X": 10.0}), [10.0, 20.0]),
        ({"X": 10.0, "Y": 20.0}, [10.0, 20.0]),
    ],
)
def test_explicit_initial_price_overrides_attrs(initial_price, expected):
    model = fm.Composite({c: fm.RandomWalk(0.2, asset=c) for c in ("X", "Y")})
    log_returns = fm.simulate(model, seed=0, **RUN)
    prices = fm.to_prices(log_returns, initial_price=initial_price)
    np.testing.assert_allclose(prices.iloc[0], expected)
    default = fm.to_prices(log_returns)  # started at 100
    np.testing.assert_allclose(prices / prices.iloc[0], default / default.iloc[0])


def test_explicit_start():
    log_returns = fm.simulate(fm.RandomWalk(0.2), seed=0, **RUN)
    earlier = pd.Timestamp("2023-12-31 23:00", tz="UTC")
    assert fm.to_prices(log_returns, start=earlier).index[0] == earlier


def test_missing_attrs_need_arguments():
    log_returns = fm.simulate(fm.RandomWalk(0.2), seed=0, **RUN)
    plain = log_returns.copy()
    plain.attrs.clear()
    with pytest.raises(ValueError, match="initial_price is needed"):
        fm.to_prices(plain)
    with pytest.raises(ValueError, match="start is needed"):
        fm.to_prices(plain, initial_price=100.0)
    prices = fm.to_prices(plain, initial_price=100.0, start=log_returns.attrs["start"])
    pd.testing.assert_frame_equal(prices, fm.to_prices(log_returns), check_flags=False)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(initial_price=0.0), "positive"),
        (dict(initial_price=[1.0, 2.0]), "one value per column"),
        (dict(initial_price={"Z": 1.0}), "must be the columns"),
        (dict(start="2024-06-01"), "timezone"),
        (dict(start=pd.Timestamp("2024-06-01", tz="UTC")), "before the first row"),
    ],
)
def test_bad_arguments(kwargs, match):
    log_returns = fm.simulate(fm.RandomWalk(0.2), seed=0, **RUN)
    with pytest.raises(ValueError, match=match):
        fm.to_prices(log_returns, **kwargs)


def test_mislabelled_frames_are_refused():
    log_returns = fm.simulate(fm.RandomWalk(0.2), seed=0, **RUN)
    prices = fm.to_prices(log_returns)
    for helper in (fm.to_prices, fm.to_log_prices, fm.to_returns):
        with pytest.raises(ValueError, match="takes log returns"):
            helper(prices)
    with pytest.raises(ValueError, match="takes prices"):
        fm.from_prices(log_returns)


def test_from_prices_validation():
    idx = pd.date_range("2024-01-01", periods=3, name="date")
    with pytest.raises(ValueError, match="positive"):
        fm.from_prices(pd.DataFrame({"A": [1.0, -1.0, 2.0]}, index=idx))
    with pytest.raises(ValueError, match="first row"):
        fm.from_prices(pd.DataFrame({"A": [np.nan, 1.0, 2.0]}, index=idx))
    with pytest.raises(ValueError, match="two rows"):
        fm.from_prices(pd.DataFrame({"A": [1.0]}, index=idx[:1]))


def test_nan_propagates_to_later_levels():
    log_returns = fm.simulate(fm.RandomWalk(0.2), seed=0, **RUN)
    log_returns.iloc[10, 0] = np.nan
    prices = fm.to_prices(log_returns)
    assert prices.iloc[:11].notna().all().all()  # start + the first 10 returns
    assert prices.iloc[11:].isna().all().all()
