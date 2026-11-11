"""The generic composition mechanism."""

import numpy as np
import pytest

import fantasma as fm


def test_stochastic_drift_equals_hand_wired_composite():
    built = fm.StochasticDrift(sigma=0.2, drift_mean=0.05, drift_half_life=0.5, drift_sigma=0.1)
    drift = fm.OrnsteinUhlenbeck(0.05, 0.5, 0.1, state_name="mu")
    price = fm.DriftPlusNoise(0.05, 0.2, asset="ASSET")
    wired = fm.Composite({"drift": drift, "price": price}, links={"price.mu": "drift.mu"})
    run = dict(start="2024-01-01", freq="1h", n_steps=200, seed=5)
    np.testing.assert_array_equal(fm.simulate(built, **run)[0], fm.simulate(wired, **run)[0])


def test_driven_drift_moves_the_price():
    # With zero price noise, log-price increments are exactly drift * dt.
    model = fm.StochasticDrift(sigma=0.0, drift_mean=0.3, drift_half_life=0.2, drift_sigma=0.5)
    ((prices, hidden),) = fm.simulate(
        model, start="2024-01-01", freq="1d", n_steps=100, seed=1, return_hidden=True
    )
    dt = 1 / 365.25
    np.testing.assert_allclose(
        np.diff(np.log(prices["ASSET"].to_numpy())), hidden["drift.mu"].to_numpy()[:-1] * dt
    )


def test_nested_composites_and_multiple_assets():
    a = fm.StochasticDrift(0.2, 0.0, 1.0, 0.1, asset="A")
    b = fm.OrnsteinUhlenbeck(4.6, 0.1, 0.2, asset="B")
    model = fm.Composite({"a": a, "b": b})
    ((prices, hidden),) = fm.simulate(
        model, start="2024-01-01", freq="1d", n_steps=10, seed=0, return_hidden=True
    )
    assert list(prices.columns) == ["A", "B"]
    assert list(hidden.columns) == ["a.drift.mu"]
    assert model.n_shocks == 3


def test_ou_level_drives_ou_mean():
    level = fm.RandomWalk(0.1, asset="L")  # observable log price used as a moving target
    follower = fm.OrnsteinUhlenbeck(4.6, 0.01, 0.0, asset="F", x0=np.log(100))
    model = fm.Composite({"l": level, "f": follower}, links={"f.theta": "l.log_price"})
    (prices,) = fm.simulate(model, start="2024-01-01", freq="1d", n_steps=50, seed=0)
    # Without noise, F reverts towards the level L read at the start of each step.
    lf, ll = np.log(prices["F"].to_numpy()), np.log(prices["L"].to_numpy())
    decay = np.exp(-np.log(2) / 0.01 / 365.25)
    np.testing.assert_allclose(lf[1:], ll[:-1] + (lf[:-1] - ll[:-1]) * decay)


@pytest.mark.parametrize(
    "links",
    [
        {"price.nope": "drift.mu"},  # no such input
        {"price.mu": "drift.nope"},  # no such state
        {"missing.mu": "drift.mu"},  # no such component
        {"price": "drift.mu"},  # malformed target
    ],
)
def test_bad_links_rejected(links):
    drift = fm.OrnsteinUhlenbeck(0.0, 1.0, 0.1, state_name="mu")
    price = fm.DriftPlusNoise(0.0, 0.2)
    with pytest.raises(ValueError):
        fm.Composite({"drift": drift, "price": price}, links=links)


def test_duplicate_assets_rejected():
    with pytest.raises(ValueError):
        fm.Composite({"a": fm.RandomWalk(0.1, asset="X"), "b": fm.RandomWalk(0.1, asset="X")})


def test_custom_noise_is_used():
    class Zero(fm.Noise):
        def sample(self, rng, n_steps, n_shocks):
            return np.zeros((n_steps, n_shocks))

    (prices,) = fm.simulate(
        fm.DriftPlusNoise(0.1, 0.5), start="2024-01-01", freq="1d", n_steps=10, seed=0, noise=Zero()
    )
    np.testing.assert_allclose(np.diff(np.log(prices["ASSET"])), 0.1 / 365.25)
