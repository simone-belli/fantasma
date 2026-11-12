"""Fantasma: synthetic price paths with known dynamics, in the Almanac data format."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

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
from .transforms import from_prices, to_log_prices, to_prices, to_returns

# The version is defined once, in pyproject.toml, and read back from the installed metadata.
# tests/test_version.py fails if the two disagree (e.g. a bump without `poetry install`).
try:
    __version__ = _version("fantasma")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

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
    "from_prices",
    "half_life_to_kappa",
    "simulate",
    "simulate_driven",
    "simulate_on",
    "simulate_states",
    "to_log_prices",
    "to_long",
    "to_prices",
    "to_returns",
]
