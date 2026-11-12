"""Log price whose drift follows an OU process: the reference composite model."""

from __future__ import annotations

from ..composite import Composite
from .brownian import Brownian
from .ou import OrnsteinUhlenbeck


class StochasticDrift(Composite):
    """Momentum-like two-factor model, built by composing two processes.

    ``drift``: ``OrnsteinUhlenbeck(drift_mean, drift_half_life, drift_sigma)`` (hidden, state
    ``drift.mu``); ``price``: ``Brownian`` whose ``mu`` is wired to the drift, i.e.
    ``log p[t+dt] = log p[t] + mu[t] dt + sigma sqrt(dt) z``. All parameters annualised.
    """

    def __init__(
        self,
        sigma: float,
        drift_mean: float,
        drift_half_life: float,
        drift_sigma: float,
        asset: str = "ASSET",
        initial_price: float = 100.0,
        initial_drift: float | None = None,
    ):
        drift = OrnsteinUhlenbeck(
            drift_mean, drift_half_life, drift_sigma, x0=initial_drift, state_name="mu"
        )
        price = Brownian(sigma, mu=drift_mean, asset=asset, initial_price=initial_price)
        super().__init__({"drift": drift, "price": price}, links={"price.mu": "drift.mu"})
