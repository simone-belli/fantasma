import pytest

import fantasma as fm

# One factory per v1 process, with reasonable parameters.
PROCESSES = {
    "random_walk": lambda: fm.RandomWalk(sigma=0.2, asset="RW"),
    "drift_plus_noise": lambda: fm.DriftPlusNoise(mu=0.1, sigma=0.2, asset="DN"),
    "ou": lambda: fm.OrnsteinUhlenbeck(theta=4.6, half_life=0.1, sigma=0.3, asset="OU"),
    "cointegrated": lambda: fm.Cointegrated(
        loadings=[[1.0], [0.5], [2.0]],
        spread_half_lives=[0.05, 0.1, 0.2],
        factor_sigmas=0.2,
        spread_sigmas=0.1,
        assets=["X", "Y", "Z"],
    ),
    "stochastic_drift": lambda: fm.StochasticDrift(
        sigma=0.2, drift_mean=0.05, drift_half_life=0.5, drift_sigma=0.1, asset="SD"
    ),
}

ASSETS = {
    "random_walk": ["RW"],
    "drift_plus_noise": ["DN"],
    "ou": ["OU"],
    "cointegrated": ["X", "Y", "Z"],
    "stochastic_drift": ["SD"],
}

HIDDEN = {
    "random_walk": [],
    "drift_plus_noise": [],
    "ou": [],
    "cointegrated": ["factor_0", "spread_X", "spread_Y", "spread_Z"],
    "stochastic_drift": ["drift.mu"],
}


@pytest.fixture(params=sorted(PROCESSES))
def name(request):
    return request.param


@pytest.fixture
def process(name):
    return PROCESSES[name]()
