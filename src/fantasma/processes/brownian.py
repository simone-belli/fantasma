"""Brownian motion in log price, with an optional drift."""

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


class Brownian(Process):
    """Arithmetic Brownian motion in log price: ``dlog p = mu dt + sigma dW``.

    ``log p[t+dt] = log p[t] + mu * dt + sigma * sqrt(dt) * z``, with ``sigma`` and ``mu``
    annualised. ``mu`` defaults to 0, the driftless random walk: the null scenario, with
    nothing to find. ``mu`` is the drift of the *log* price (no Itô correction). ``mu`` and
    ``sigma`` are drivable inputs, so another process can supply a time-varying drift or
    volatility when composed (see ``StochasticDrift``).
    """

    inputs = ("mu", "sigma")

    def __init__(
        self, sigma: float, mu: float = 0.0, asset: str = "ASSET", initial_price: float = 100.0
    ):
        self.params = {
            "sigma": check_non_negative("sigma", sigma),
            "mu": check_finite("mu", mu),
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
