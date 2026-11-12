"""LinearCombination: a model built from sources (models, data, other combinations)."""

import numpy as np
import pandas as pd
import pytest

import fantasma as fm
from fantasma.simulate import step_states

GRID = dict(start="2024-01-01", freq="1d", n_steps=300)
DT = 1 / 365.25


def mkt():
    return fm.RandomWalk(0.2, asset="MKT")


def data(name="SPY", n_steps=300, seed=9, freq="1d"):
    """A log-return Series, as real data converted with fm.from_prices would be."""
    (r,) = fm.simulate(
        fm.RandomWalk(0.15, asset=name), start="2024-01-01", freq=freq, n_steps=n_steps, seed=seed
    )
    return r[name]


def run(combos, **kw):
    return fm.simulate(combos, return_hidden=True, **kw)[0]


# --- the formula ------------------------------------------------------------------------------


def test_models_only_is_exact():
    m, mom = mkt(), fm.StochasticDrift(0.1, 0.05, 2.0, 0.3, asset="MOM")
    a = fm.LinearCombination([m, mom], weights=[1.2, 0.5], asset="A")
    res = run(a, seed=1, **GRID)
    sources = res.sources["MKT"].output["MKT"], res.sources["MOM"].output["MOM"]
    np.testing.assert_allclose(res.output["A"], 1.2 * sources[0] + 0.5 * sources[1])


def test_data_only_is_exact_with_drift_on_uneven_index():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-08", "2024-01-09"], name="date")
    x = pd.Series([0.01, -0.02, 0.03, 0.0], index=idx, name="X")
    combo = fm.LinearCombination([x], weights=[2.0], asset="Y", constant=0.5)
    res = run(combo, seed=0)
    days = np.array([1, 1, 5, 1])  # the first row repeats the first gap
    np.testing.assert_allclose(res.output["Y"], 2.0 * x + 0.5 * days * DT)
    assert res.output.index.equals(idx)
    assert res.output.attrs["start"] == pd.Timestamp("2024-01-01")


def test_mixed_models_and_data_share_the_data_index():
    spy = data(freq="1d")
    gappy = spy.iloc[::3]  # an uneven index: every third day
    m = mkt()
    combo = fm.LinearCombination([m, gappy], weights=[1.0, -0.3], asset="A")
    res = run(combo, seed=2)
    assert res.output.index.equals(gappy.index)
    assert res.sources["MKT"].output.index.equals(gappy.index)
    np.testing.assert_allclose(
        res.output["A"], res.sources["MKT"].output["MKT"] - 0.3 * gappy.to_numpy()
    )
    # The model was simulated on the data's gaps: its start is the data's recorded start.
    assert res.sources["MKT"].hidden.index[0] == spy.attrs["start"]


def test_noise_scale_and_custom_noise():
    m = mkt()
    combo = fm.LinearCombination([m], weights=[1.0], sigma=0.3, asset="A")
    res = run(combo, seed=4, start="2000-01-01", freq="1d", n_steps=20_000)
    resid = res.output["A"] - res.sources["MKT"].output["MKT"]
    sigma_hat = resid.std(ddof=0) / np.sqrt(DT)
    assert abs(sigma_hat - 0.3) < 4 * 0.3 / np.sqrt(2 * len(resid))

    class Zero(fm.Noise):
        def sample(self, rng, n_steps, n_shocks):
            return np.zeros((n_steps, n_shocks))

    quiet = run(combo, seed=4, noise=Zero(), **GRID)
    np.testing.assert_allclose(quiet.output["A"], quiet.sources["MKT"].output["MKT"])


# --- sharing realisations ----------------------------------------------------------------------


def test_same_model_object_is_simulated_once():
    m = mkt()
    a = fm.LinearCombination([m], weights=[1.2], asset="A")
    b = fm.LinearCombination([m], weights=[0.8], asset="B")
    res = run([a, b], seed=1, **GRID)
    assert list(res.output.columns) == ["A", "B"]
    np.testing.assert_allclose(res.output["A"] / 1.2, res.output["B"] / 0.8)


def test_equal_but_distinct_models_are_independent():
    a = fm.LinearCombination([mkt()], weights=[1.0], asset="A")
    b = fm.LinearCombination([fm.RandomWalk(0.2, asset="MKT2")], weights=[1.0], asset="B")
    res = run([a, b], seed=1, **GRID)
    assert not np.allclose(res.output["A"], res.output["B"])


def test_assets_of_one_multi_asset_model_share_one_realisation():
    coint = fm.Cointegrated([[1.0], [0.5], [2.0]], 0.1, 0.2, 0.05, ["X", "Y", "Z"])
    a = fm.LinearCombination([(coint, "X")], weights=[1.0], asset="A")
    b = fm.LinearCombination([(coint, "Y")], weights=[1.0], asset="B")
    res = run([a, b], seed=5, **GRID)
    assert res.sources["X"] is res.sources["Y"]  # the whole model's realisation
    # It is coint simulated alone with its child seed: the first child of path 0.
    child = np.random.SeedSequence(5).spawn(1)[0].spawn(3)[0]
    grid = fm.RegularGrid(**GRID)
    (states,) = step_states(coint, grid, [child])
    np.testing.assert_allclose(res.sources["X"].output, np.diff(coint.observe(states), axis=0))
    np.testing.assert_allclose(res.output["A"], res.sources["X"].output["X"])
    np.testing.assert_allclose(res.output["B"], res.sources["X"].output["Y"])


def test_nested_combinations():
    m = mkt()
    sector = fm.LinearCombination([m], weights=[1.0], sigma=0.1, asset="SECTOR")
    stock = fm.LinearCombination([sector, m], weights=[0.9, 0.2], asset="STOCK")
    both = run([sector, stock], seed=3, **GRID)
    alone = run(stock, seed=3, **GRID)
    assert isinstance(alone.sources["SECTOR"], fm.PathResult)
    sector_r = alone.sources["SECTOR"].output["SECTOR"]
    np.testing.assert_allclose(
        alone.output["STOCK"], 0.9 * sector_r + 0.2 * alone.sources["MKT"].output["MKT"]
    )
    # The nested combination shares MKT with the outer one.
    assert alone.sources["SECTOR"].sources["MKT"] is alone.sources["MKT"]
    assert list(both.output.columns) == ["SECTOR", "STOCK"]


# --- the result -------------------------------------------------------------------------------


def test_path_result_and_plain_output():
    m, mom = mkt(), fm.StochasticDrift(0.1, 0.05, 2.0, 0.3, asset="MOM")
    spy = data()
    combo = fm.LinearCombination([m, mom, spy], weights=[1.0, 0.5, -0.2], sigma=0.1, asset="A")
    res = run(combo, seed=0)
    assert list(res.sources) == ["MKT", "MOM", "SPY"]
    assert isinstance(res.sources["MOM"], fm.SourceRealisation)
    hidden = res.sources["MOM"].hidden
    assert list(hidden.columns) == ["drift.mu"] and len(hidden) == len(spy) + 1
    assert res.sources["SPY"].output["SPY"].equals(spy)
    assert res.sources["SPY"].hidden.shape[1] == 0

    (plain,) = fm.simulate(combo, seed=0)
    pd.testing.assert_frame_equal(plain, res.output)
    assert plain.attrs["quantity"] == "log_return"


def test_to_prices_starts_at_initial_price():
    combo = fm.LinearCombination([mkt()], weights=[1.0], asset="A", initial_price=50.0)
    (out,) = fm.simulate(combo, seed=0, **GRID)
    prices = fm.to_prices(out)
    assert prices.index[0] == pd.Timestamp("2024-01-01")
    np.testing.assert_allclose(prices.iloc[0], 50.0)


def test_reproducible_and_path_stable():
    combo = fm.LinearCombination([mkt(), data()], weights=[1.0, 0.5], sigma=0.2, asset="A")
    a = fm.simulate(combo, seed=7, n_paths=3)
    b = fm.simulate(combo, seed=7, n_paths=3)
    for x, y in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(x, y)
    assert not np.allclose(a[0], a[1])
    (other,) = fm.simulate(combo, seed=8)
    assert not np.allclose(a[0], other)
    (first,) = fm.simulate(combo, seed=7, n_paths=1)
    pd.testing.assert_frame_equal(first, a[0])


# --- validation -------------------------------------------------------------------------------


def _coint():
    return fm.Cointegrated([[1.0], [0.5]], 0.1, 0.2, 0.05, ["X", "Y"])


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: fm.LinearCombination([], weights=[], asset="A"), "non-empty"),
        (lambda: fm.LinearCombination([mkt()], weights=[1.0, 2.0], asset="A"), "one number"),
        (lambda: fm.LinearCombination([mkt()], weights=[np.nan], asset="A"), "finite"),
        (lambda: fm.LinearCombination([mkt()], weights=[1.0], sigma=-1, asset="A"), "non-neg"),
        (lambda: fm.LinearCombination([mkt()], weights=[1.0], asset=""), "asset"),
        (lambda: fm.LinearCombination([mkt()], [1.0], asset="A", initial_price=0), "positive"),
        (lambda: fm.LinearCombination([_coint()], weights=[1.0], asset="A"), "select one"),
        (lambda: fm.LinearCombination([(_coint(), "Q")], weights=[1.0], asset="A"), "not an asset"),
        (lambda: fm.LinearCombination([mkt(), mkt()], [1.0, 1.0], asset="A"), "distinct"),
        (lambda: fm.LinearCombination([data().rename(None)], [1.0], asset="A"), "needs a name"),
        (
            lambda: fm.LinearCombination([fm.to_prices(data().to_frame())], [1.0], asset="A"),
            "from_prices",
        ),
        (
            lambda: fm.LinearCombination(
                [data("SPY"), data("QQQ").iloc[1:]], [1.0, 1.0], asset="A"
            ),
            "different indexes",
        ),
        (
            lambda: fm.LinearCombination([data().reset_index(drop=True)], [1.0], asset="A"),
            "DatetimeIndex",
        ),
        (lambda: fm.LinearCombination(["x"], [1.0], asset="A"), "must be a model"),
    ],
)
def test_validation_at_creation(make, match):
    with pytest.raises((ValueError, TypeError), match=match):
        make()


def test_validation_at_simulation():
    with_data = fm.LinearCombination([data()], weights=[1.0], asset="A")
    with pytest.raises(ValueError, match="do not pass"):
        fm.simulate(with_data, seed=0, **GRID)
    models_only = fm.LinearCombination([mkt()], weights=[1.0], asset="A")
    with pytest.raises(ValueError, match="give start, freq"):
        fm.simulate(models_only, seed=0)
    same_asset = fm.LinearCombination([mkt()], weights=[1.0], asset="A")
    with pytest.raises(ValueError, match="distinct"):
        fm.simulate([models_only, same_asset], seed=0, **GRID)
    # Two different models both named MKT would make res.sources["MKT"] ambiguous.
    other = fm.LinearCombination([mkt()], weights=[1.0], asset="B")
    with pytest.raises(ValueError, match="two different sources"):
        fm.simulate([models_only, other], seed=0, **GRID)
    with pytest.raises(ValueError, match="seed"):
        fm.simulate(models_only, seed=None, **GRID)


def test_plain_models_still_need_a_grid():
    with pytest.raises(ValueError, match="start and freq"):
        fm.simulate(mkt(), seed=0, n_steps=5)
