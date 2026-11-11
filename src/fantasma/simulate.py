"""The simulation driver: seeding, the time loop over all paths, and output."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import RegularGrid, TimeGrid
from .noise import GaussianNoise, Noise
from .output import hidden_frame, price_frame
from .process import Process


def simulate_states(
    process: Process, grid: TimeGrid, seed: int, n_paths: int = 1, noise: Noise | None = None
) -> np.ndarray:
    """Raw state paths, shape ``(n_paths, n_steps + 1, n_state)``.

    Path ``i`` draws its shocks from child ``i`` of ``SeedSequence(seed)``, so a path depends
    only on the seed and its position: asking for more paths does not change the first ones.
    """
    if not isinstance(process, Process):
        raise TypeError(f"expected a Process, got {process!r}")
    if not isinstance(n_paths, (int, np.integer)) or isinstance(n_paths, bool) or n_paths < 1:
        raise ValueError(f"n_paths must be a positive integer, got {n_paths!r}")
    if seed is None:
        raise ValueError("a seed is required, so every run is reproducible")
    noise = noise or GaussianNoise()
    dts = grid.dts
    n_steps = len(dts)

    children = np.random.SeedSequence(seed).spawn(n_paths)
    shocks = np.stack(
        [noise.sample(np.random.default_rng(c), n_steps, process.n_shocks) for c in children],
        axis=1,
    )  # (n_steps, n_paths, n_shocks)

    states = np.empty((n_steps + 1, n_paths, process.n_state))
    states[0] = process.initial_state()
    for t in range(n_steps):
        states[t + 1] = process.step(states[t], shocks[t], float(dts[t]), {})
    return states.transpose(1, 0, 2)


def simulate(
    process: Process,
    *,
    start,
    freq,
    seed: int,
    n_steps: int | None = None,
    end=None,
    n_paths: int = 1,
    return_hidden: bool = False,
    noise: Noise | None = None,
    days_per_year: float | None = None,
) -> list[pd.DataFrame] | list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Simulate ``n_paths`` independent paths of ``process`` on a regular grid.

    Returns a list with one wide close-price DataFrame per path (index ``ts`` in UTC for
    intraday ``freq``, ``date`` for daily; one column per asset). With ``return_hidden``, each
    element is a ``(prices, hidden)`` pair, ``hidden`` holding the ground-truth state on the
    same index. Rows are the start plus one per step, so there are ``n_steps + 1``.

    ``freq`` is an Almanac frequency (``"1m"``, ``"10m"``, ``"1h"``, ``"1d"``) or any pandas
    offset. Give either ``n_steps`` or ``end``.
    """
    kwargs = {} if days_per_year is None else {"days_per_year": days_per_year}
    grid = RegularGrid(start, freq, n_steps=n_steps, end=end, **kwargs)
    return simulate_on(
        process, grid, seed=seed, n_paths=n_paths, return_hidden=return_hidden, noise=noise
    )


def simulate_on(
    process: Process,
    grid: TimeGrid,
    *,
    seed: int,
    n_paths: int = 1,
    return_hidden: bool = False,
    noise: Noise | None = None,
):
    """Like :func:`simulate`, on a prebuilt :class:`TimeGrid`."""
    paths = simulate_states(process, grid, seed, n_paths, noise)
    prices = [price_frame(process, p, grid) for p in paths]
    if not return_hidden:
        return prices
    return [(px, hidden_frame(process, p, grid)) for px, p in zip(prices, paths, strict=True)]
