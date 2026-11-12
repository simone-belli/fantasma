"""Driven models: LinearCombination and RegimeSwitching on given log-return series."""

import numpy as np
import pandas as pd
import pytest

import fantasma as fm

Z = 4.0
COLS = ["X", "Y"]


def make_inputs(n_inputs=2, n_steps=200, freq="1d", seed=0):
    """Log-return frames with identical index and columns: random walks per column."""
    model = fm.Composite({c: fm.RandomWalk(0.2, asset=c) for c in COLS})
    paths = fm.simulate(
        model, start="2024-01-01", freq=freq, n_steps=n_steps, seed=seed, n_paths=n_inputs
    )
    return paths


def two_regimes(sigma=0.1):
    return fm.RegimeSwitching(
        [
            fm.LinearCombination([1.0, 0.0], constant=0.0, sigma=sigma),
            fm.LinearCombination([0.0, 1.0], constant=10.0, sigma=2 * sigma),
        ],
        transition_matrix=[[0.9, 0.1], [0.2, 0.8]],
    )


MODELS = {
    "linear": lambda: fm.LinearCombination([0.5, [1.0, -1.0]], constant=1.0, sigma=0.2),
    "regime": two_regimes,
}


@pytest.fixture(params=sorted(MODELS))
def model(request):
    return MODELS[request.param]()


# --- structure and reproducibility ------------------------------------------------------


@pytest.mark.parametrize("freq", ["10m", "1d"])
def test_output_has_input_index_and_columns(model, freq):
    inputs = make_inputs(freq=freq)
    outs = fm.simulate_driven(model, inputs, seed=1, n_paths=3, return_hidden=True)
    assert len(outs) == 3
    for out, hidden in outs:
        assert out.index.equals(inputs[0].index) and out.index.name == inputs[0].index.name
        assert out.columns.equals(inputs[0].columns)
        assert out.columns.name == inputs[0].columns.name
        assert (out.dtypes == "float64").all()
        assert np.isfinite(out.to_numpy()).all()
        assert hidden.index.equals(out.index)
        assert list(hidden.columns) == list(model.hidden_states)


def test_arbitrary_columns_and_index_preserved():
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-08"], name="when")
    cols = pd.Index(["a", "b", "c"], name="things")
    x = pd.DataFrame(np.arange(9.0).reshape(3, 3), index=idx, columns=cols)
    (out,) = fm.simulate_driven(fm.LinearCombination([2.0], constant=0.0), [x], seed=0)
    pd.testing.assert_frame_equal(out, 2.0 * x, check_flags=False)


def test_same_seed_same_output(model):
    inputs = make_inputs()
    a = fm.simulate_driven(model, inputs, seed=5, n_paths=2, return_hidden=True)
    b = fm.simulate_driven(model, inputs, seed=5, n_paths=2, return_hidden=True)
    for (oa, ha), (ob, hb) in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(oa, ob)
        pd.testing.assert_frame_equal(ha, hb)


def test_different_seeds_and_paths_differ(model):
    inputs = make_inputs()
    a, b = fm.simulate_driven(model, inputs, seed=1, n_paths=2)
    (c,) = fm.simulate_driven(model, inputs, seed=2)
    assert not np.allclose(a, b) and not np.allclose(a, c)
    (first,) = fm.simulate_driven(model, inputs, seed=1)
    pd.testing.assert_frame_equal(first, a)


# --- the linear combination ---------------------------------------------------------------


def test_noise_free_combination_is_exact():
    x1, x2 = make_inputs()
    model = fm.LinearCombination([0.5, [1.0, -2.0]], constant=[3.0, 4.0], sigma=0.0)
    (out,) = fm.simulate_driven(model, [x1, x2], seed=0)
    dt = 1 / 365.25  # the constant is an annualised drift
    expected = pd.DataFrame(
        {
            "X": 3.0 * dt + 0.5 * x1["X"] + 1.0 * x2["X"],
            "Y": 4.0 * dt + 0.5 * x1["Y"] - 2.0 * x2["Y"],
        }
    )
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy())


def test_series_parameters_are_matched_by_column():
    x1, x2 = make_inputs()
    by_series = fm.LinearCombination(
        [pd.Series({"Y": 2.0, "X": 1.0}), 0.0], constant=pd.Series({"Y": 0.0, "X": 5.0})
    )
    by_list = fm.LinearCombination([[1.0, 2.0], 0.0], constant=[5.0, 0.0])
    pd.testing.assert_frame_equal(
        fm.simulate_driven(by_series, [x1, x2], seed=0)[0],
        fm.simulate_driven(by_list, [x1, x2], seed=0)[0],
    )


def test_noise_scales_with_sqrt_dt_and_sigma():
    x1, x2 = make_inputs(n_steps=20_000)
    model = fm.LinearCombination([1.0, 0.0], sigma=[0.1, 0.4])
    (out,) = fm.simulate_driven(model, [x1, x2], seed=3)
    resid = (out - x1).to_numpy()
    sd = resid.std(axis=0, ddof=0) / np.sqrt(1 / 365.25)
    se = np.array([0.1, 0.4]) / np.sqrt(2 * len(resid))
    assert np.all(np.abs(sd - [0.1, 0.4]) < Z * se)


def test_uneven_index_scales_noise_by_elapsed_time():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-06"])
    x = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=idx)

    class Ones(fm.Noise):
        def sample(self, rng, n_steps, n_shocks):
            return np.ones((n_steps, n_shocks))

    (out,) = fm.simulate_driven(fm.LinearCombination([1.0], sigma=1.0), [x], seed=0, noise=Ones())
    days = np.array([1, 1, 4])  # the first row uses the first gap
    np.testing.assert_allclose(out["A"], np.sqrt(days / 365.25))


def test_custom_noise_is_used(model):
    class Zero(fm.Noise):
        def sample(self, rng, n_steps, n_shocks):
            return np.zeros((n_steps, n_shocks))

    inputs = make_inputs()
    (noisy,) = fm.simulate_driven(model, inputs, seed=0)
    (quiet,) = fm.simulate_driven(model, inputs, seed=0, noise=Zero())
    assert not np.allclose(noisy, quiet)


def test_nan_inputs_propagate():
    x1, x2 = make_inputs()
    x1.iloc[10, 0] = np.nan
    (out,) = fm.simulate_driven(fm.LinearCombination([1.0, 1.0], sigma=0.1), [x1, x2], seed=0)
    assert np.isnan(out.iloc[10, 0]) and out.drop(out.index[10]).notna().all().all()


# --- regime switching -----------------------------------------------------------------------


def test_single_regime_equals_linear_combination():
    inputs = make_inputs()
    lin = fm.LinearCombination([0.5, 1.5], constant=1.0, sigma=0.3)
    reg = fm.RegimeSwitching([lin], transition_matrix=[[1.0]])
    pd.testing.assert_frame_equal(
        fm.simulate_driven(lin, inputs, seed=4)[0], fm.simulate_driven(reg, inputs, seed=4)[0]
    )


def test_output_follows_the_regime():
    x1, x2 = make_inputs()
    ((out, hidden),) = fm.simulate_driven(
        two_regimes(sigma=0.0), [x1, x2], seed=2, return_hidden=True
    )
    regime = hidden["regime"].to_numpy()
    assert set(np.unique(regime)) == {0, 1}
    expected = np.where(regime[:, None] == 0, x1.to_numpy(), 10.0 / 365.25 + x2.to_numpy())
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_regimes_do_not_depend_on_noise_type():
    class Zero(fm.Noise):
        def sample(self, rng, n_steps, n_shocks):
            return np.zeros((n_steps, n_shocks))

    inputs = make_inputs()
    ((_, a),) = fm.simulate_driven(two_regimes(), inputs, seed=6, return_hidden=True)
    ((_, b),) = fm.simulate_driven(two_regimes(), inputs, seed=6, return_hidden=True, noise=Zero())
    pd.testing.assert_frame_equal(a, b)


def test_fixed_initial_regime_and_absorbing_state():
    model = fm.RegimeSwitching(
        [fm.LinearCombination([1.0]), fm.LinearCombination([2.0])],
        transition_matrix=[[1.0, 0.0], [0.0, 1.0]],
        initial=1,
    )
    (x, _) = make_inputs()
    ((_, hidden),) = fm.simulate_driven(model, [x], seed=0, return_hidden=True)
    assert (hidden["regime"] == 1).all()


def test_initial_distribution_default_is_stationary():
    chain = fm.MarkovChain([[0.9, 0.1], [0.2, 0.8]])
    np.testing.assert_allclose(chain.initial_distribution, [2 / 3, 1 / 3])
    # Started from it, the chain stays there: the first-row regime has that distribution.
    rng = np.random.default_rng(0)
    first = np.array([chain.sample(rng, 2)[0] for _ in range(20_000)])
    p = 1 / 3
    assert abs(first.mean() - p) < Z * np.sqrt(p * (1 - p) / len(first))


# --- validation -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: fm.LinearCombination([]),
        lambda: fm.LinearCombination(1.0),
        lambda: fm.LinearCombination(["a"]),
        lambda: fm.LinearCombination([float("nan")]),
        lambda: fm.LinearCombination([[[1.0]]]),
        lambda: fm.LinearCombination([1.0], sigma=-0.1),
        lambda: fm.LinearCombination([1.0], constant=float("inf")),
        lambda: fm.MarkovChain([[0.5, 0.4], [0.5, 0.5]]),  # row does not sum to 1
        lambda: fm.MarkovChain([[1.2, -0.2], [0.5, 0.5]]),  # negative probability
        lambda: fm.MarkovChain([[1.0, 0.0]]),  # not square
        lambda: fm.MarkovChain([[0.5, 0.5], [0.5, 0.5]], initial=2),
        lambda: fm.MarkovChain([[0.5, 0.5], [0.5, 0.5]], initial=[0.3, 0.3]),
        lambda: fm.MarkovChain([[1.0, 0.0], [0.0, 1.0]]),  # no unique stationary distribution
        lambda: fm.RegimeSwitching([], [[1.0]]),
        lambda: fm.RegimeSwitching([fm.LinearCombination([1.0])], [[0.5, 0.5], [0.5, 0.5]]),
        lambda: fm.RegimeSwitching(
            [fm.LinearCombination([1.0]), fm.LinearCombination([1.0, 1.0])],
            [[0.5, 0.5], [0.5, 0.5]],
        ),
    ],
)
def test_parameter_validation(make):
    with pytest.raises((ValueError, TypeError)):
        make()


def _shifted_index(x):
    y = x.copy()
    y.index = y.index + pd.Timedelta(days=1)
    return y


@pytest.mark.parametrize(
    "build, model",
    [
        (lambda x1, x2: [x1], fm.LinearCombination([1.0, 1.0])),  # wrong number of inputs
        (lambda x1, x2: [], fm.LinearCombination([1.0])),
        (lambda x1, x2: x1, fm.LinearCombination([1.0])),  # a frame, not a list
        (lambda x1, x2: [x1, x2[["Y", "X"]]], fm.LinearCombination([1.0, 1.0])),  # columns
        (lambda x1, x2: [x1, _shifted_index(x2)], fm.LinearCombination([1.0, 1.0])),
        (lambda x1, x2: [x1.reset_index(drop=True)], fm.LinearCombination([1.0])),
        (lambda x1, x2: [x1.iloc[:1]], fm.LinearCombination([1.0])),  # one row: no dt
        (lambda x1, x2: [x1.iloc[::-1]], fm.LinearCombination([1.0])),  # decreasing index
        (lambda x1, x2: [x1.assign(X="a")], fm.LinearCombination([1.0])),  # not numeric
        (lambda x1, x2: [x1.replace(x1.iloc[0, 0], np.inf)], fm.LinearCombination([1.0])),
        (lambda x1, x2: [x1], fm.LinearCombination([[1.0, 2.0, 3.0]])),  # 3 values, 2 columns
        (lambda x1, x2: [x1], fm.LinearCombination([pd.Series({"X": 1.0, "Z": 1.0})])),
    ],
)
def test_input_validation(build, model):
    x1, x2 = make_inputs()
    with pytest.raises((ValueError, TypeError)):
        fm.simulate_driven(model, build(x1, x2), seed=0)


def test_run_validation():
    inputs = make_inputs()
    m = fm.LinearCombination([1.0, 1.0])
    with pytest.raises(ValueError):
        fm.simulate_driven(m, inputs, seed=None)
    with pytest.raises(ValueError):
        fm.simulate_driven(m, inputs, seed=0, n_paths=0)
    with pytest.raises(TypeError):
        fm.simulate_driven(fm.RandomWalk(0.1), inputs, seed=0)


# --- the log-return contract ------------------------------------------------------------------


def test_uneven_index_scales_drift_by_elapsed_time():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-06"])
    x = pd.DataFrame({"A": [0.0, 0.0, 0.0]}, index=idx)
    (out,) = fm.simulate_driven(fm.LinearCombination([1.0], constant=2.0), [x], seed=0)
    days = np.array([1, 1, 4])  # the first row uses the first gap
    np.testing.assert_allclose(out["A"], 2.0 * days / 365.25)


def test_output_is_labelled_and_keeps_the_inputs_start(model):
    inputs = make_inputs()
    (out,) = fm.simulate_driven(model, inputs, seed=0)
    assert out.attrs["quantity"] == "log_return"
    assert out.attrs["start"] == inputs[0].attrs["start"]
    assert "initial_log_price" not in out.attrs  # no natural starting price...
    with pytest.raises(ValueError, match="initial_price"):
        fm.to_prices(out)
    prices = fm.to_prices(out, initial_price=100.0)  # ...so it is given here
    assert prices.index[0] == inputs[0].attrs["start"]
    np.testing.assert_allclose(prices.iloc[0], 100.0)


def test_start_dropped_when_inputs_disagree():
    a, _ = make_inputs()
    b = a.copy()
    b.attrs["start"] = pd.Timestamp("2023-12-31")
    (out,) = fm.simulate_driven(fm.LinearCombination([1.0, 1.0]), [a, b], seed=0)
    assert "start" not in out.attrs


def test_prices_are_refused_as_inputs():
    prices = [fm.to_prices(x) for x in make_inputs()]
    with pytest.raises(ValueError, match="from_prices"):
        fm.simulate_driven(fm.LinearCombination([1.0, 1.0]), prices, seed=0)
    # Converted back to log returns they are accepted.
    inputs = [fm.from_prices(p) for p in prices]
    fm.simulate_driven(fm.LinearCombination([1.0, 1.0]), inputs, seed=0)
