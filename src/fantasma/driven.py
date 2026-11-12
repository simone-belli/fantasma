"""Models driven by given input series rather than by their own dynamics.

Like every Fantasma model, driven models work in **log returns** (see
:mod:`fantasma.transforms`). A driven model takes a list of log-return DataFrames, all with the
same index and columns (paths from other Fantasma models, or real prices converted with
``fm.from_prices``), and returns log returns with that same index and columns. Run one with
:func:`simulate_driven`.

- :class:`LinearCombination`, a factor model on log returns:
  ``r[t] = c * dt[t] + sum_i w_i * r_i[t] + sigma * sqrt(dt[t]) * eps[t]``.
- :class:`RegimeSwitching`: one ``LinearCombination`` per regime, the regime following a
  :class:`~fantasma.regimes.MarkovChain`.

``c`` is an annualised drift and ``sigma`` an annualised volatility. ``eps`` comes from the
``Noise`` passed to :func:`simulate_driven` (standard Gaussian by default), independent across
rows and columns. ``dt[t]`` is the time from the previous row to row ``t`` in years (for the
first row, the time to the second), so uneven indexes, e.g. with weekend gaps, scale the drift
and noise by the time elapsed.

The output has no natural starting price, so ``fm.to_prices`` needs ``initial_price=`` for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .grid import DAYS_PER_YEAR
from .noise import GaussianNoise, Noise
from .regimes import MarkovChain
from .transforms import LOG_RETURN, QUANTITY, START, log_return_attrs, per_column


class DrivenModel(ABC):
    """A model whose output is a function of given input series, regimes and noise."""

    hidden_states: tuple[str, ...] = ()

    @property
    @abstractmethod
    def n_inputs(self) -> int:
        """The number of input series the model combines."""

    @abstractmethod
    def generate(
        self,
        inputs: np.ndarray,
        columns: pd.Index,
        dt: np.ndarray,
        rng_regime: np.random.Generator,
        rng_noise: np.random.Generator,
        noise: Noise,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Output ``(n_rows, n_columns)`` and hidden state ``(n_rows, len(hidden_states))``
        from ``inputs`` of shape ``(n_inputs, n_rows, n_columns)``."""


def _check_scalar_or_vector(name: str, value, non_negative: bool = False):
    """Validate what can be checked before the columns are known."""
    if isinstance(value, pd.Series):
        per_column(name, value, value.index, non_negative)
        return value
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric, got {value!r}") from None
    if arr.ndim > 1:
        raise ValueError(f"{name} must be a scalar or one value per column")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    if non_negative and np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative")
    return value


class LinearCombination(DrivenModel):
    """Factor model on log returns:
    ``r[t] = constant * dt[t] + sum_i weights[i] * r_i[t] + sigma * sqrt(dt[t]) * eps[t]``.

    ``constant`` is an annualised drift and ``sigma`` an annualised volatility.

    ``weights`` has one entry per input series; each entry, and ``constant`` and ``sigma``, is
    a scalar (same for every column), a sequence with one value per column, or a Series keyed
    by column name. ``sigma`` is annualised. NaN in an input gives NaN in the output there.
    """

    def __init__(self, weights: Sequence, constant=0.0, sigma=0.0):
        if isinstance(weights, (str, bytes)) or not isinstance(weights, Sequence) or not weights:
            raise ValueError("weights must be a non-empty list, one entry per input series")
        self.weights = [_check_scalar_or_vector(f"weights[{i}]", w) for i, w in enumerate(weights)]
        self.constant = _check_scalar_or_vector("constant", constant)
        self.sigma = _check_scalar_or_vector("sigma", sigma, non_negative=True)

    @property
    def n_inputs(self) -> int:
        return len(self.weights)

    def resolve(self, columns: pd.Index) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weights ``(n_inputs, n_columns)``, constant and sigma ``(n_columns,)``."""
        W = np.stack([per_column(f"weights[{i}]", w, columns) for i, w in enumerate(self.weights)])
        c = per_column("constant", self.constant, columns)
        s = per_column("sigma", self.sigma, columns, non_negative=True)
        return W, c, s

    def combine(self, inputs, columns, dt, eps) -> np.ndarray:
        W, c, s = self.resolve(columns)
        drift = c * dt[:, None]
        return drift + np.einsum("ic,itc->tc", W, inputs) + s * np.sqrt(dt)[:, None] * eps

    def generate(self, inputs, columns, dt, rng_regime, rng_noise, noise):
        eps = noise.sample(rng_noise, inputs.shape[1], inputs.shape[2])
        return self.combine(inputs, columns, dt, eps), np.empty((inputs.shape[1], 0))

    def __repr__(self) -> str:
        return (
            f"LinearCombination(weights={self.weights!r}, constant={self.constant!r}, "
            f"sigma={self.sigma!r})"
        )


class RegimeSwitching(DrivenModel):
    """A linear combination whose weights, constant and sigma depend on a Markov regime.

    ``regimes`` holds one :class:`LinearCombination` per regime (all over the same number of
    inputs); ``transition_matrix`` and ``initial`` define the :class:`MarkovChain`, which moves
    once per row and is shared by all columns. The regime path is returned as hidden state
    ``regime``. The noise ``eps`` is drawn once and scaled by the current regime's sigma.
    """

    hidden_states = ("regime",)

    def __init__(self, regimes: Sequence[LinearCombination], transition_matrix, initial=None):
        if isinstance(regimes, LinearCombination) or not regimes:
            raise ValueError("regimes must be a non-empty list of LinearCombination")
        for i, r in enumerate(regimes):
            if not isinstance(r, LinearCombination):
                raise TypeError(f"regimes[{i}] is not a LinearCombination: {r!r}")
        if len({r.n_inputs for r in regimes}) != 1:
            raise ValueError("every regime must combine the same number of input series")
        self.regimes = list(regimes)
        self.chain = MarkovChain(transition_matrix, initial)
        if self.chain.n_regimes != len(self.regimes):
            raise ValueError(
                f"transition_matrix is for {self.chain.n_regimes} regimes, "
                f"got {len(self.regimes)} regime models"
            )

    @property
    def n_inputs(self) -> int:
        return self.regimes[0].n_inputs

    def generate(self, inputs, columns, dt, rng_regime, rng_noise, noise):
        n_rows = inputs.shape[1]
        regime = self.chain.sample(rng_regime, n_rows)
        eps = noise.sample(rng_noise, n_rows, inputs.shape[2])
        outputs = np.stack([r.combine(inputs, columns, dt, eps) for r in self.regimes])
        out = np.take_along_axis(outputs, regime[None, :, None], axis=0)[0]
        return out, regime[:, None]

    def __repr__(self) -> str:
        return f"RegimeSwitching({self.regimes!r}, {self.chain!r})"


def _check_inputs(inputs) -> tuple[pd.Index, pd.DatetimeIndex, np.ndarray]:
    if isinstance(inputs, pd.DataFrame) or not isinstance(inputs, Sequence) or not inputs:
        raise ValueError("inputs must be a non-empty list of DataFrames")
    first = inputs[0]
    for i, df in enumerate(inputs):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"inputs[{i}] is not a DataFrame")
        if not df.columns.equals(first.columns):
            raise ValueError(f"inputs[{i}] columns differ from inputs[0]")
        if not df.index.equals(first.index):
            raise ValueError(f"inputs[{i}] index differs from inputs[0]")
        held = df.attrs.get(QUANTITY)
        if held is not None and held != LOG_RETURN:
            raise ValueError(
                f"inputs[{i}] holds {held!r}; driven models take log returns "
                "(convert prices with fm.from_prices)"
            )
    if first.shape[1] == 0:
        raise ValueError("inputs have no columns")
    if not isinstance(first.index, pd.DatetimeIndex):
        raise ValueError("inputs must have a DatetimeIndex, to measure dt")
    if len(first.index) < 2:
        raise ValueError("inputs need at least two rows, to measure dt")
    if not first.index.is_monotonic_increasing or not first.index.is_unique:
        raise ValueError("the input index must be strictly increasing")
    try:
        values = np.stack([df.to_numpy(dtype=float) for df in inputs])
    except (TypeError, ValueError):
        raise ValueError("inputs must be numeric") from None
    if np.isinf(values).any():
        raise ValueError("inputs must not contain infinities")
    return first.columns, first.index, values


def simulate_driven(
    model: DrivenModel,
    inputs: Sequence[pd.DataFrame],
    *,
    seed: int,
    n_paths: int = 1,
    return_hidden: bool = False,
    noise: Noise | None = None,
    days_per_year: float = DAYS_PER_YEAR,
) -> list[pd.DataFrame] | list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Run a driven model on log-return ``inputs`` for ``n_paths`` independent noise and regime
    paths.

    Returns a list with one DataFrame of log returns per path, with exactly the inputs' index
    and columns. When every input records the same start timestamp, the output records it too;
    it has no starting price, so give ``initial_price=`` to ``fm.to_prices``.
    With ``return_hidden``, each element is ``(output, hidden)``, ``hidden`` holding the hidden
    state (e.g. ``regime``) on the same index. Path ``i`` uses child ``i`` of
    ``SeedSequence(seed)``, split into a regime stream and a noise stream, so the regimes do not
    depend on the noise type.
    """
    if not isinstance(model, DrivenModel):
        raise TypeError(f"expected a driven model, got {model!r}")
    if seed is None:
        raise ValueError("a seed is required, so every run is reproducible")
    if not isinstance(n_paths, (int, np.integer)) or isinstance(n_paths, bool) or n_paths < 1:
        raise ValueError(f"n_paths must be a positive integer, got {n_paths!r}")
    if days_per_year <= 0:
        raise ValueError("days_per_year must be positive")
    columns, index, values = _check_inputs(inputs)
    if values.shape[0] != model.n_inputs:
        raise ValueError(f"the model combines {model.n_inputs} inputs, got {values.shape[0]}")
    model_noise = noise or GaussianNoise()

    # Timedelta arithmetic, not asi8: the index may be in s/ms/us resolution, not ns.
    gaps = np.asarray((index[1:] - index[:-1]) / pd.Timedelta(days=days_per_year), dtype=float)
    dt = np.concatenate([gaps[:1], gaps])
    starts = {df.attrs.get(START) for df in inputs}
    attrs = log_return_attrs(starts.pop() if len(starts) == 1 else None, None)
    hidden_columns = pd.Index(list(model.hidden_states), name="state", dtype="string")

    results = []
    for path in np.random.SeedSequence(seed).spawn(n_paths):
        regime_seq, noise_seq = path.spawn(2)
        out, hidden = model.generate(
            values,
            columns,
            dt,
            np.random.default_rng(regime_seq),
            np.random.default_rng(noise_seq),
            model_noise,
        )
        frame = pd.DataFrame(out, index=index, columns=columns, dtype="float64")
        frame.attrs.update(attrs)
        if return_hidden:
            results.append((frame, pd.DataFrame(hidden, index=index, columns=hidden_columns)))
        else:
            results.append(frame)
    return results
