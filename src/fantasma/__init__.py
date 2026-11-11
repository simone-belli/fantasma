"""Fantasma: synthetic price paths with known dynamics, in the Almanac data format."""

from .composite import Composite
from .driven import DrivenModel, LinearCombination, RegimeSwitching, simulate_driven
from .grid import RegularGrid, TimeGrid
from .noise import GaussianNoise, Noise
from .output import to_long
from .process import Process
from .processes import (
    Cointegrated,
    DriftPlusNoise,
    OrnsteinUhlenbeck,
    RandomWalk,
    StochasticDrift,
    half_life_to_kappa,
)
from .regimes import MarkovChain
from .simulate import simulate, simulate_on, simulate_states

__version__ = "0.1.0"

__all__ = [
    "Cointegrated",
    "Composite",
    "DrivenModel",
    "DriftPlusNoise",
    "GaussianNoise",
    "LinearCombination",
    "MarkovChain",
    "Noise",
    "OrnsteinUhlenbeck",
    "Process",
    "RandomWalk",
    "RegimeSwitching",
    "RegularGrid",
    "StochasticDrift",
    "TimeGrid",
    "half_life_to_kappa",
    "simulate",
    "simulate_driven",
    "simulate_on",
    "simulate_states",
    "to_long",
]
