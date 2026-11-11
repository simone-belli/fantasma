"""Shock generation, kept apart from the processes that consume the shocks.

A noise model draws the whole ``(n_steps, n_shocks)`` block of shocks for one path from that
path's own generator. Processes only ever see standardised shocks, so fat-tailed or
GARCH-type noise can be added later as another ``Noise`` without touching any process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Noise(ABC):
    @abstractmethod
    def sample(self, rng: np.random.Generator, n_steps: int, n_shocks: int) -> np.ndarray:
        """Shocks for one path, shape ``(n_steps, n_shocks)``, zero mean and unit variance."""


class GaussianNoise(Noise):
    """Independent standard normal shocks (the v1 default)."""

    def sample(self, rng: np.random.Generator, n_steps: int, n_shocks: int) -> np.ndarray:
        return rng.standard_normal((n_steps, n_shocks))

    def __repr__(self) -> str:
        return "GaussianNoise()"
