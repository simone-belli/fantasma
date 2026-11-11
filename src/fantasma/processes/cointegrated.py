"""N assets driven by K common random-walk factors plus stationary OU spreads."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..process import Inputs, Process, check_array, check_assets
from .ou import half_life_to_kappa, ou_step


class Cointegrated(Process):
    """Cointegrated system: ``log p = log p0 + loadings @ f + s``.

    - ``f`` (K) are independent driftless random walks with vols ``factor_sigmas``, starting at 0;
    - ``s`` (N) are independent OU spreads around 0 with ``spread_half_lives`` (years) and vols
      ``spread_sigmas``, starting at 0 (exact discretisation).

    ``loadings`` is ``(N, K)`` with full column rank and ``K < N``, so the system has exactly
    ``N - K`` cointegrating relations: the vectors ``b`` with ``b @ loadings = 0`` make
    ``b @ log p`` stationary. Hidden state: ``factor_<k>`` and ``spread_<asset>``.
    Per-asset arguments accept a scalar, applied to every asset.
    """

    def __init__(
        self,
        loadings,
        spread_half_lives,
        factor_sigmas,
        spread_sigmas,
        assets: Sequence[str],
        initial_prices=100.0,
    ):
        B = check_array("loadings", loadings)
        if B.ndim == 1:
            B = B[:, None]
        if B.ndim != 2:
            raise ValueError(f"loadings must be an (N, K) matrix, got shape {B.shape}")
        n, k = B.shape
        if k < 1 or k >= n:
            raise ValueError(f"need 1 <= K < N factors, got N={n}, K={k}")
        if np.linalg.matrix_rank(B) != k:
            raise ValueError("loadings must have full column rank")
        self.asset_names = check_assets(assets, n)
        self.params = {
            "loadings": B,
            "spread_half_lives": check_array("spread_half_lives", spread_half_lives, (n,), True),
            "factor_sigmas": check_array("factor_sigmas", factor_sigmas, (k,)),
            "spread_sigmas": check_array("spread_sigmas", spread_sigmas, (n,)),
            "initial_prices": check_array("initial_prices", initial_prices, (n,), True),
        }
        if np.any(self.params["factor_sigmas"] < 0) or np.any(self.params["spread_sigmas"] < 0):
            raise ValueError("volatilities must be non-negative")
        self.kappas = half_life_to_kappa(self.params["spread_half_lives"])
        self.n_assets, self.n_factors = n, k
        self.state_names = tuple(f"factor_{i}" for i in range(k)) + tuple(
            f"spread_{a}" for a in self.asset_names
        )
        self.observable_states = ()
        self.n_shocks = k + n

    @property
    def hidden_states(self) -> tuple[str, ...]:
        return self.state_names

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.n_factors + self.n_assets)

    def step(self, state: np.ndarray, shocks: np.ndarray, dt: float, inputs: Inputs) -> np.ndarray:
        k = self.n_factors
        f = state[:, :k] + self.params["factor_sigmas"] * np.sqrt(dt) * shocks[:, :k]
        s = ou_step(state[:, k:], 0.0, self.kappas, self.params["spread_sigmas"], dt, shocks[:, k:])
        return np.concatenate([f, s], axis=1)

    def observe(self, state: np.ndarray) -> np.ndarray:
        k = self.n_factors
        f, s = state[..., :k], state[..., k:]
        return np.log(self.params["initial_prices"]) + f @ self.params["loadings"].T + s

    def hidden(self, state: np.ndarray) -> np.ndarray:
        return state
