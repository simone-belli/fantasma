"""Discrete-time Markov chains for regime states."""

from __future__ import annotations

import numpy as np

_TOL = 1e-10


class MarkovChain:
    """A Markov chain over regimes ``0 .. n-1`` that moves once per time step.

    ``transition_matrix[i, j]`` is the probability of moving from regime ``i`` to regime ``j``
    in one step; rows must sum to 1. ``initial`` is the regime at the first row: an ``int``
    for a fixed regime, a probability vector to draw it, or ``None`` (default) for the
    stationary distribution of the chain.
    """

    def __init__(self, transition_matrix, initial=None):
        try:
            P = np.asarray(transition_matrix, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("transition_matrix must be numeric") from None
        if P.ndim != 2 or P.shape[0] != P.shape[1] or P.shape[0] < 1:
            raise ValueError(f"transition_matrix must be square, got shape {P.shape}")
        if not np.all(np.isfinite(P)) or np.any(P < 0):
            raise ValueError("transition probabilities must be finite and non-negative")
        if not np.allclose(P.sum(axis=1), 1.0, atol=_TOL, rtol=0):
            raise ValueError(f"transition_matrix rows must sum to 1, got {P.sum(axis=1)}")
        self.transition_matrix = P / P.sum(axis=1, keepdims=True)
        self.n_regimes = P.shape[0]
        self.initial = initial
        self.initial_distribution = self._initial_distribution(initial)

    def _initial_distribution(self, initial) -> np.ndarray:
        n = self.n_regimes
        if initial is None:
            return self.stationary_distribution()
        if isinstance(initial, (int, np.integer)) and not isinstance(initial, bool):
            if not 0 <= initial < n:
                raise ValueError(f"initial regime must be in 0..{n - 1}, got {initial}")
            dist = np.zeros(n)
            dist[initial] = 1.0
            return dist
        try:
            dist = np.asarray(initial, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                f"initial must be a regime or a distribution, got {initial!r}"
            ) from None
        if dist.shape != (n,) or np.any(dist < 0) or not np.isclose(dist.sum(), 1.0, atol=_TOL):
            raise ValueError(f"initial distribution must be {n} probabilities summing to 1")
        return dist / dist.sum()

    def stationary_distribution(self) -> np.ndarray:
        """The distribution ``pi`` with ``pi @ P = pi``; raises if it is not unique."""
        n = self.n_regimes
        A = np.vstack([self.transition_matrix.T - np.eye(n), np.ones(n)])
        if np.linalg.matrix_rank(A[:-1], tol=1e-9) != n - 1:
            raise ValueError("the chain has no unique stationary distribution; pass initial")
        b = np.zeros(n + 1)
        b[-1] = 1.0
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        pi = np.clip(pi, 0.0, None)
        return pi / pi.sum()

    def sample(self, rng: np.random.Generator, n_rows: int) -> np.ndarray:
        """A regime path of ``n_rows`` integers: the initial regime, then one move per row."""
        u = rng.random(n_rows)
        cum = np.cumsum(self.transition_matrix, axis=1)
        cum[:, -1] = 1.0  # guard against rounding: every uniform falls in some regime
        regimes = np.empty(n_rows, dtype=np.int64)
        regimes[0] = np.searchsorted(np.cumsum(self.initial_distribution), u[0], side="right")
        regimes[0] = min(regimes[0], self.n_regimes - 1)
        for t in range(1, n_rows):
            regimes[t] = np.searchsorted(cum[regimes[t - 1]], u[t], side="right")
        return regimes

    def __repr__(self) -> str:
        return f"MarkovChain({self.transition_matrix.tolist()!r}, initial={self.initial!r})"
