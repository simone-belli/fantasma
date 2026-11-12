"""The simulation driver: seeding, the time loop over all paths, and output."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import RegularGrid, TimeGrid
from .noise import GaussianNoise, Noise
from .output import hidden_frame, log_return_frame
from .process import Process


def one_or_list(results: list, n_paths: int):
    """The single path itself when ``n_paths`` is 1, else the list of paths."""
    return results[0] if n_paths == 1 else results


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
    children = np.random.SeedSequence(seed).spawn(n_paths)
    return step_states(process, grid, children, noise)


def step_states(
    process: Process,
    grid: TimeGrid,
    children: list[np.random.SeedSequence],
    noise: Noise | None = None,
) -> np.ndarray:
    """State paths ``(len(children), n_steps + 1, n_state)``, path ``i`` drawing its shocks
    from ``children[i]``. All paths step together, vectorised."""
    noise = noise or GaussianNoise()
    dts = grid.dts
    n_steps = len(dts)
    n_paths = len(children)
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
    process,
    *,
    start=None,
    freq=None,
    seed: int,
    n_steps: int | None = None,
    end=None,
    n_paths: int = 1,
    return_hidden: bool = False,
    noise: Noise | None = None,
    days_per_year: float | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame] | list:
    """Simulate ``n_paths`` independent paths of ``process`` on a regular grid.

    With ``n_paths=1`` (the default) the single path is returned itself; with more, a list
    with one element per path. Each path is a DataFrame of **log returns** (index ``ts`` in UTC for
    intraday ``freq``, ``date`` for daily; one column per asset): ``n_steps`` rows, row t the
    return over the step ending at t. ``attrs`` record the start timestamp and starting
    prices, so ``fm.to_prices(frame)`` gives the price path (``n_steps + 1`` rows, start
    included); see :mod:`fantasma.transforms`.

    With ``return_hidden``, each path is a ``(log_returns, hidden)`` pair, ``hidden`` holding
    the ground-truth state on the full grid (``n_steps + 1`` rows, start included).

    ``freq`` is an Almanac frequency (``"1m"``, ``"10m"``, ``"1h"``, ``"1d"``) or any pandas
    offset. Give either ``n_steps`` or ``end``.

    ``process`` may also be a :class:`~fantasma.combination.LinearCombination` or a list of
    them: the result is then one log-return frame per path with a column per combination, or
    with ``return_hidden`` a :class:`~fantasma.combination.PathResult` giving access to every
    source's realisation. With data among the sources, the grid is the data's index and
    ``start``/``freq``/``n_steps``/``end`` must not be given (see
    :mod:`fantasma.combination`).
    """
    from .combination import LinearCombination, simulate_combinations

    if isinstance(process, LinearCombination) or (
        isinstance(process, (list, tuple))
        and process
        and all(isinstance(p, LinearCombination) for p in process)
    ):
        paths = simulate_combinations(
            process,
            seed=seed,
            n_paths=n_paths,
            return_hidden=return_hidden,
            noise=noise,
            days_per_year=days_per_year,
            start=start,
            freq=freq,
            n_steps=n_steps,
            end=end,
        )
        return one_or_list(paths, n_paths)
    if start is None or freq is None:
        raise ValueError("give start and freq (and n_steps or end) to simulate a model")
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
    """Like :func:`simulate`, on a prebuilt :class:`TimeGrid` (the single path itself when
    ``n_paths`` is 1, else a list)."""
    paths = simulate_states(process, grid, seed, n_paths, noise)
    frames = [log_return_frame(process, p, grid) for p in paths]
    if return_hidden:
        frames = [(f, hidden_frame(process, p, grid)) for f, p in zip(frames, paths, strict=True)]
    return one_or_list(frames, n_paths)
