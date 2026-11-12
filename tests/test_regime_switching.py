"""RegimeSwitching: one of N regime sources at a time, the regime following a Markov chain."""

import numpy as np
import pandas as pd
import pytest

import fantasma as fm

GRID = dict(start="2024-01-01", freq="1d", n_steps=500)
Z = 4.0
P2 = [[0.9, 0.1], [0.2, 0.8]]


def mkt():
    return fm.RandomWalk(0.2, asset="MKT")


def data(name="SPY", n_steps=500, seed=9):
    """A log-return Series, as real data converted with fm.from_prices would be."""
    r = fm.simulate(
        fm.RandomWalk(0.15, asset=name), start="2024-01-01", freq="1d", n_steps=n_steps, seed=seed
    )
    return r[name]


def returns(res, name):
    """The log returns of source ``name`` in a PathResult."""
    return res.sources[name].output[name].to_numpy()


def run(node, **kw):
    return fm.simulate(node, return_hidden=True, **kw)


# --- the output follows the active regime -----------------------------------------------------


def test_output_is_the_active_regimes_return():
    m = mkt()
    calm = fm.LinearCombination([m], weights=[0.8], sigma=0.05, asset="CALM")
    wild = fm.LinearCombination([m], weights=[1.5], sigma=0.3, asset="WILD")
    node = fm.RegimeSwitching([calm, wild], P2, asset="STOCK")
    res = run(node, seed=1, **GRID)
    regime = res.regimes["STOCK"].to_numpy()
    assert set(np.unique(regime)) == {0, 1}
    expected = np.choose(regime, [returns(res, "CALM"), returns(res, "WILD")])
    np.testing.assert_array_equal(res.output["STOCK"].to_numpy(), expected)


def test_models_and_data_as_regimes():
    spy = data()
    mom = fm.StochasticDrift(0.1, 0.05, 2.0, 0.3, asset="MOM")
    node = fm.RegimeSwitching([mom, spy], P2, asset="X")
    res = run(node, seed=2)  # the grid is the data's index
    assert res.output.index.equals(spy.index)
    regime = res.regimes["X"].to_numpy()
    expected = np.choose(regime, [returns(res, "MOM"), spy.to_numpy()])
    np.testing.assert_array_equal(res.output["X"].to_numpy(), expected)
    assert list(res.sources["MOM"].hidden.columns) == ["drift.mu"]


def test_three_regimes_and_one_asset_of_a_multi_asset_model():
    coint = fm.Cointegrated([[1.0], [0.5], [2.0]], 0.1, 0.2, 0.05, ["X", "Y", "Z"])
    P3 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    node = fm.RegimeSwitching([(coint, "X"), (coint, "Y"), (coint, "Z")], P3, asset="R")
    res = run(node, seed=3, **GRID)
    out = res.sources["X"].output  # one realisation of the whole model
    regime = res.regimes["R"].to_numpy()
    assert set(np.unique(regime)) == {0, 1, 2}
    expected = out.to_numpy()[np.arange(len(regime)), regime]
    np.testing.assert_array_equal(res.output["R"].to_numpy(), expected)


# --- the chain --------------------------------------------------------------------------------


def test_fixed_initial_regime_and_absorbing_chain():
    node = fm.RegimeSwitching(
        [mkt(), fm.RandomWalk(0.5, asset="B")], [[1.0, 0.0], [0.0, 1.0]], initial=1, asset="R"
    )
    res = run(node, seed=0, **GRID)
    assert (res.regimes["R"] == 1).all()
    np.testing.assert_array_equal(res.output["R"], returns(res, "B"))


def test_initial_distribution():
    node = fm.RegimeSwitching(
        [mkt(), fm.RandomWalk(0.5, asset="B")], P2, initial=[0.0, 1.0], asset="R"
    )
    assert run(node, seed=0, **GRID).regimes["R"].iloc[0] == 1


def test_initial_distribution_default_is_stationary():
    chain = fm.MarkovChain(P2)
    np.testing.assert_allclose(chain.initial_distribution, [2 / 3, 1 / 3])
    # Started from it, the chain stays there: the first-row regime has that distribution.
    rng = np.random.default_rng(0)
    first = np.array([chain.sample(rng, 2)[0] for _ in range(20_000)])
    p = 1 / 3
    assert abs(first.mean() - p) < Z * np.sqrt(p * (1 - p) / len(first))


def test_regimes_do_not_depend_on_the_noise_type():
    class Zero(fm.Noise):
        def sample(self, rng, n_steps, n_shocks):
            return np.zeros((n_steps, n_shocks))

    node = fm.RegimeSwitching([mkt(), fm.RandomWalk(0.5, asset="B")], P2, asset="R")
    a = run(node, seed=6, **GRID)
    b = run(node, seed=6, noise=Zero(), **GRID)
    pd.testing.assert_frame_equal(a.regimes, b.regimes)


# --- sharing and nesting ----------------------------------------------------------------------


def test_a_model_shared_by_regimes_is_simulated_once():
    m = mkt()
    calm = fm.LinearCombination([m], weights=[1.0], asset="CALM")
    wild = fm.LinearCombination([m], weights=[2.0], asset="WILD")
    res = run(fm.RegimeSwitching([calm, wild], P2, asset="R"), seed=1, **GRID)
    assert res.sources["CALM"].sources["MKT"] is res.sources["WILD"].sources["MKT"]
    np.testing.assert_allclose(returns(res, "WILD"), 2 * returns(res, "CALM"))


def test_regime_switching_inside_a_linear_combination_and_the_reverse():
    m = mkt()
    inner = fm.RegimeSwitching([m, fm.RandomWalk(0.5, asset="B")], P2, asset="INNER")
    outer = fm.LinearCombination([inner, m], weights=[0.5, 1.0], asset="OUTER")
    res = run(outer, seed=4, **GRID)
    assert isinstance(res.sources["INNER"], fm.PathResult)
    assert list(res.sources["INNER"].regimes.columns) == ["INNER"]
    assert res.regimes.shape[1] == 0  # OUTER is not regime switching
    np.testing.assert_allclose(
        res.output["OUTER"], 0.5 * returns(res, "INNER") + returns(res, "MKT")
    )
    top = fm.RegimeSwitching([outer, m], P2, asset="TOP")
    res2 = run(top, seed=4, **GRID)
    assert isinstance(res2.sources["OUTER"].sources["INNER"], fm.PathResult)


def test_simulated_together_with_a_combination():
    m = mkt()
    reg = fm.RegimeSwitching([m, fm.RandomWalk(0.5, asset="B")], P2, asset="R")
    lin = fm.LinearCombination([m], weights=[1.0], asset="L")
    res = run([reg, lin], seed=5, **GRID)
    assert list(res.output.columns) == ["R", "L"]
    assert list(res.regimes.columns) == ["R"]
    assert res.regimes.index.equals(res.output.index)


# --- the result -------------------------------------------------------------------------------


def test_plain_output_single_path_and_prices():
    node = fm.RegimeSwitching(
        [mkt(), fm.RandomWalk(0.5, asset="B")], P2, asset="R", initial_price=50.0
    )
    plain = fm.simulate(node, seed=0, **GRID)
    assert isinstance(plain, pd.DataFrame) and plain.attrs["quantity"] == "log_return"
    pd.testing.assert_frame_equal(plain, run(node, seed=0, **GRID).output)
    many = fm.simulate(node, seed=0, n_paths=3, **GRID)
    assert isinstance(many, list) and len(many) == 3
    pd.testing.assert_frame_equal(plain, many[0])
    np.testing.assert_allclose(fm.to_prices(plain).iloc[0], 50.0)


# --- validation -------------------------------------------------------------------------------


def _coint():
    return fm.Cointegrated([[1.0], [0.5]], 0.1, 0.2, 0.05, ["X", "Y"])


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: fm.RegimeSwitching([], [[1.0]], asset="R"), "non-empty"),
        (lambda: fm.RegimeSwitching([mkt()], P2, asset="R"), "for 2 regimes, got 1"),
        (lambda: fm.RegimeSwitching([mkt(), mkt()], P2, asset="R"), "distinct"),
        (lambda: fm.RegimeSwitching([_coint(), mkt()], P2, asset="R"), "select one"),
        (lambda: fm.RegimeSwitching([mkt()], [[1.0]], asset=""), "asset"),
        (lambda: fm.RegimeSwitching([mkt()], [[1.0]], asset="R", initial_price=0), "positive"),
        (lambda: fm.MarkovChain([[0.5, 0.4], [0.5, 0.5]]), "sum to 1"),
        (lambda: fm.MarkovChain([[1.2, -0.2], [0.5, 0.5]]), "non-negative"),
        (lambda: fm.MarkovChain([[1.0, 0.0]]), "square"),
        (lambda: fm.MarkovChain([[0.5, 0.5], [0.5, 0.5]], initial=2), "initial regime"),
        (lambda: fm.MarkovChain([[0.5, 0.5], [0.5, 0.5]], initial=[0.3, 0.3]), "summing to 1"),
        (lambda: fm.MarkovChain([[0.5, 0.5], [0.5, 0.5]], initial=[1.0]), "summing to 1"),
        (lambda: fm.MarkovChain([[1.0, 0.0], [0.0, 1.0]]), "stationary"),
    ],
)
def test_validation(make, match):
    with pytest.raises((ValueError, TypeError), match=match):
        make()


def test_models_only_need_a_grid():
    node = fm.RegimeSwitching([mkt()], [[1.0]], asset="R")
    with pytest.raises(ValueError, match="give start, freq"):
        fm.simulate(node, seed=0)
