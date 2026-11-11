"""The ``Process`` interface every stochastic model implements.

A process owns a state vector (e.g. log price, drift, spread). The framework holds the state of
all paths as an array of shape ``(n_paths, n_state)`` and calls :meth:`Process.step` once per
time step with a matching array of standard shocks of shape ``(n_paths, n_shocks)``. Step
functions are written with NumPy array operations, so the same code advances one path or many.

Parameters are given in annualised units; ``dt`` is a year fraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import numpy as np

Inputs = Mapping[str, np.ndarray]


class Process(ABC):
    """Base class of all models.

    Subclasses set, in ``__init__``:

    - ``params``: the parameter values by name (validated before being stored);
    - ``state_names``: the names of the state variables, in state-vector order;
    - ``observable_states``: the states that are log prices, one per asset, in asset order;
    - ``asset_names``: the user-supplied names of the observable prices;
    - ``n_shocks``: independent standard shocks consumed per step;

    and implement :meth:`initial_state` and :meth:`step`. ``inputs`` lists the parameters that
    another process may drive when composed (see :class:`fantasma.Composite`); read them in
    ``step`` with :meth:`param`, which returns the driven value when there is one.

    A process whose prices are not simply some of its states (e.g. sums of factors) overrides
    :meth:`observe` and :attr:`hidden_states`.
    """

    params: dict[str, object]
    state_names: tuple[str, ...]
    observable_states: tuple[str, ...]
    asset_names: tuple[str, ...]
    n_shocks: int
    inputs: tuple[str, ...] = ()

    @property
    def n_state(self) -> int:
        return len(self.state_names)

    @property
    def hidden_states(self) -> tuple[str, ...]:
        """State variables returned as ground truth: every state that is not a log price."""
        return tuple(s for s in self.state_names if s not in self.observable_states)

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """The state at time zero, shape ``(n_state,)``."""

    @abstractmethod
    def step(self, state: np.ndarray, shocks: np.ndarray, dt: float, inputs: Inputs) -> np.ndarray:
        """Advance ``state`` (``(n_paths, n_state)``) by ``dt`` years; return the new state."""

    def observe(self, state: np.ndarray) -> np.ndarray:
        """Log prices, shape ``(..., n_assets)``, from a state array ``(..., n_state)``."""
        return state[..., self._indices(self.observable_states)]

    def hidden(self, state: np.ndarray) -> np.ndarray:
        """Hidden state values, shape ``(..., len(hidden_states))``."""
        return state[..., self._indices(self.hidden_states)]

    def param(self, name: str, inputs: Inputs):
        """Parameter ``name``, or the value driving it in this step if it is a wired input."""
        return inputs[name] if name in inputs else self.params[name]

    def _indices(self, names: Sequence[str]) -> list[int]:
        return [self.state_names.index(n) for n in names]

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


# --- validation helpers -------------------------------------------------------------------


def check_assets(assets: str | Sequence[str], n: int | None = None) -> tuple[str, ...]:
    """Asset names as a tuple of distinct non-empty strings (``n`` of them if given)."""
    names = (assets,) if isinstance(assets, str) else tuple(assets)
    if not names:
        raise ValueError("at least one asset name is required")
    for a in names:
        if not isinstance(a, str) or not a:
            raise ValueError(f"asset names must be non-empty strings, got {a!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"asset names must be distinct, got {list(names)}")
    if n is not None and len(names) != n:
        raise ValueError(f"expected {n} asset names, got {len(names)}")
    return names


def check_finite(name: str, value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def check_positive(name: str, value) -> float:
    v = check_finite(name, value)
    if v <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return v


def check_non_negative(name: str, value) -> float:
    v = check_finite(name, value)
    if v < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")
    return v


def check_array(name: str, value, shape: tuple[int, ...] | None = None, positive=False):
    """A finite float array, broadcast to ``shape`` if given (scalars allowed)."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric, got {value!r}") from None
    if shape is not None:
        try:
            arr = np.broadcast_to(arr, shape).copy()
        except ValueError:
            raise ValueError(f"{name} must have shape {shape}, got {arr.shape}") from None
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(arr <= 0):
        raise ValueError(f"{name} must be positive")
    return arr
