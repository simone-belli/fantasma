"""Ornstein-Uhlenbeck (mean-reverting) process, with the exact discretisation."""

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


def half_life_to_kappa(half_life):
    """Speed of mean reversion ``kappa = ln 2 / half_life`` (half-life in years)."""
    return np.log(2.0) / half_life


def ou_step(x, theta, kappa, sigma, dt, z):
    """Exact OU transition over ``dt``: mean ``theta + (x - theta) e^{-kappa dt}``."""
    decay = np.exp(-kappa * dt)
    sd = sigma * np.sqrt(-np.expm1(-2.0 * kappa * dt) / (2.0 * kappa))
    return theta + (x - theta) * decay + sd * z


class OrnsteinUhlenbeck(Process):
    """Mean-reverting ``dx = kappa (theta - x) dt + sigma dW``, ``kappa = ln 2 / half_life``.

    With an ``asset`` name, ``x`` is that asset's log price (so ``theta`` is the mean *log*
    price) and the default start is ``theta``. Without one, ``x`` is a hidden state named
    ``state_name``, for driving another process (e.g. a stochastic drift). ``half_life`` is in
    years. ``theta`` is a drivable input.
    """

    inputs = ("theta",)

    def __init__(
        self,
        theta: float,
        half_life: float,
        sigma: float,
        asset: str | None = None,
        x0: float | None = None,
        state_name: str = "x",
    ):
        self.params = {
            "theta": check_finite("theta", theta),
            "half_life": check_positive("half_life", half_life),
            "sigma": check_non_negative("sigma", sigma),
            "x0": check_finite("x0", theta if x0 is None else x0),
        }
        self.kappa = float(half_life_to_kappa(self.params["half_life"]))
        if asset is None:
            if not isinstance(state_name, str) or not state_name or "." in state_name:
                raise ValueError(f"state_name must be non-empty and dot-free, got {state_name!r}")
            self.state_names = (state_name,)
            self.observable_states = ()
            self.asset_names = ()
        else:
            (name,) = check_assets(asset, 1)
            self.state_names = ("log_price",)
            self.observable_states = ("log_price",)
            self.asset_names = (name,)
        self.n_shocks = 1

    def initial_state(self) -> np.ndarray:
        return np.array([self.params["x0"]])

    def step(self, state: np.ndarray, shocks: np.ndarray, dt: float, inputs: Inputs) -> np.ndarray:
        theta = self.param("theta", inputs)
        x = ou_step(state[:, 0], theta, self.kappa, self.params["sigma"], dt, shocks[:, 0])
        return x[:, None]
