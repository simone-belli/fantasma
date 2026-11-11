"""Random walk and drift plus noise, in log price."""

from __future__ import annotations

import numpy as np

from ..process import (
    Inputs,
    Process,
    check_assets,
    check_finite,
    check_non_negative,
    check_positive,
)


class DriftPlusNoise(Process):
    """Log price with constant drift ``mu`` and volatility ``sigma`` (both annualised).

    ``log p[t+dt] = log p[t] + mu * dt + sigma * sqrt(dt) * z``. ``mu`` is the drift of the
    *log* price (no Itô correction). ``mu`` and ``sigma`` are drivable inputs, so another process
    can supply a time-varying drift or volatility when composed.
    """

    inputs = ("mu", "sigma")

    def __init__(self, mu: float, sigma: float, asset: str = "ASSET", initial_price: float = 100.0):
        self.params = {
            "mu": check_finite("mu", mu),
            "sigma": check_non_negative("sigma", sigma),
            "initial_price": check_positive("initial_price", initial_price),
        }
        (name,) = check_assets(asset, 1)
        self.asset_names = (name,)
        self.state_names = ("log_price",)
        self.observable_states = ("log_price",)
        self.n_shocks = 1

    def initial_state(self) -> np.ndarray:
        return np.array([np.log(self.params["initial_price"])])

    def step(self, state: np.ndarray, shocks: np.ndarray, dt: float, inputs: Inputs) -> np.ndarray:
        mu = self.param("mu", inputs)
        sigma = self.param("sigma", inputs)
        return state + (mu * dt + sigma * np.sqrt(dt) * shocks[:, 0])[:, None]


class RandomWalk(DriftPlusNoise):
    """Driftless Gaussian random walk in log price with volatility ``sigma``: the null case."""

    def __init__(self, sigma: float, asset: str = "ASSET", initial_price: float = 100.0):
        super().__init__(0.0, sigma, asset, initial_price)
        del self.params["mu"]

    def param(self, name: str, inputs: Inputs):
        return 0.0 if name == "mu" and name not in inputs else super().param(name, inputs)
