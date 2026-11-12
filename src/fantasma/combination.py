"""Linear combinations of sources: models the framework simulates, or given data.

A :class:`LinearCombination` produces one log-return series:

    r[t] = constant * dt[t] + sum_i weights[i] * r_i[t] + sigma * sqrt(dt[t]) * eps[t]

where ``r_i`` are the log returns of its sources. A source is

- a one-asset model (any :class:`~fantasma.process.Process`), simulated by the framework;
- ``(model, "X")``: asset ``X`` of a multi-asset model;
- data: a log-return ``Series`` (named) or one-column ``DataFrame``, e.g. real prices
  converted with ``fm.from_prices``;
- another ``LinearCombination``, so combinations nest.

Run combinations with ``fm.simulate(combination_or_list, ...)``. Within a path, each model
object is simulated **once**, however many combinations use it or however many of its assets
are selected: two combinations loading on the same market model see the same market path.
With data among the sources, the grid is the data's index (models are simulated on it, gaps
included); with models only, it comes from ``start``/``freq``/``n_steps``.

With ``return_hidden=True`` each path is a :class:`PathResult`: ``.output`` (the
combinations' log returns, one column each) and ``.sources[name]``, the realisation of each
source (its log returns and hidden state).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .grid import DAYS_PER_YEAR, IndexGrid, RegularGrid, TimeGrid
from .noise import GaussianNoise, Noise
from .output import hidden_frame, log_return_frame
from .process import Process
from .transforms import LOG_RETURN, QUANTITY, START, log_return_attrs


@dataclass(frozen=True)
class SourceRealisation:
    """One source in one path: its log returns (all of a model's assets) and hidden state on
    the full grid (start included; empty for data)."""

    output: pd.DataFrame
    hidden: pd.DataFrame


@dataclass(frozen=True)
class PathResult:
    """One path: the combinations' log returns, and every source's realisation by name.

    A nested combination appears in ``sources`` as its own ``PathResult``.
    """

    output: pd.DataFrame
    sources: dict[str, SourceRealisation | PathResult] = field(default_factory=dict)


@dataclass(frozen=True)
class _Source:
    kind: str  # "model", "data" or "combination"
    name: str
    model: Process | None = None
    column: str | None = None
    data: pd.Series | None = None
    combination: LinearCombination | None = None
    origin: int = 0  # id of the object the user passed, for data sources


def _finite(name: str, value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _check_data(name: str, series: pd.Series) -> pd.Series:
    held = series.attrs.get(QUANTITY)
    if held is not None and held != LOG_RETURN:
        raise ValueError(
            f"source {name!r} holds {held!r}; sources take log returns "
            "(convert prices with fm.from_prices)"
        )
    index = series.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"source {name!r} needs a DatetimeIndex")
    if len(index) < 1:
        raise ValueError(f"source {name!r} is empty")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError(f"source {name!r}: the index must be strictly increasing")
    try:
        values = series.to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"source {name!r} must be numeric") from None
    if np.isinf(values).any():
        raise ValueError(f"source {name!r} must not contain infinities")
    return series


def _as_source(i: int, src) -> _Source:
    if isinstance(src, LinearCombination):
        return _Source("combination", src.asset, combination=src)
    if isinstance(src, tuple):
        if len(src) != 2 or not isinstance(src[0], Process) or not isinstance(src[1], str):
            raise TypeError(f"sources[{i}]: a tuple must be (model, asset name), got {src!r}")
        model, column = src
        if column not in model.asset_names:
            raise ValueError(
                f"sources[{i}]: {column!r} is not an asset of the model; "
                f"assets: {list(model.asset_names)}"
            )
        return _Source("model", column, model=model, column=column)
    if isinstance(src, Process):
        if len(src.asset_names) != 1:
            raise ValueError(
                f"sources[{i}] has assets {list(src.asset_names)}; select one with "
                "(model, asset name)"
            )
        return _Source("model", src.asset_names[0], model=src, column=src.asset_names[0])
    if isinstance(src, pd.DataFrame):
        if src.shape[1] != 1:
            raise ValueError(f"sources[{i}]: a DataFrame source needs exactly one column")
        series = src.iloc[:, 0]
        series.attrs = dict(src.attrs)
        name = str(src.columns[0])
        return _Source("data", name, data=_check_data(name, series.rename(name)), origin=id(src))
    if isinstance(src, pd.Series):
        if src.name is None:
            raise ValueError(f"sources[{i}]: a Series source needs a name")
        name = str(src.name)
        return _Source("data", name, data=_check_data(name, src), origin=id(src))
    raise TypeError(
        f"sources[{i}] must be a model, (model, asset), a Series, a one-column DataFrame "
        f"or a LinearCombination; got {type(src).__name__}"
    )


class LinearCombination:
    """``r[t] = constant * dt[t] + sum_i weights[i] * r_i[t] + sigma * sqrt(dt[t]) * eps[t]``.

    ``sources``: models, ``(model, asset)`` pairs, log-return data, or other combinations (see
    the module docstring). ``weights``: one number per source. ``sigma``: annualised
    volatility of the noise. ``asset``: the output's name. ``constant``: an optional
    annualised drift. ``initial_price``: recorded with the output, so ``fm.to_prices`` starts
    there.
    """

    def __init__(
        self,
        sources: Sequence,
        weights: Sequence[float],
        sigma: float = 0.0,
        *,
        asset: str,
        constant: float = 0.0,
        initial_price: float = 100.0,
    ):
        if not isinstance(sources, (list, tuple)) or not sources:
            raise ValueError("sources must be a non-empty list")
        self.sources = [_as_source(i, s) for i, s in enumerate(sources)]
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"source names must be distinct, got {names}")
        if not isinstance(weights, (list, tuple, np.ndarray)) or len(weights) != len(sources):
            raise ValueError(f"weights must have one number per source ({len(sources)})")
        self.weights = [_finite(f"weights[{i}]", w) for i, w in enumerate(weights)]
        self.sigma = _finite("sigma", sigma)
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma!r}")
        self.constant = _finite("constant", constant)
        self.initial_price = _finite("initial_price", initial_price)
        if self.initial_price <= 0:
            raise ValueError(f"initial_price must be positive, got {initial_price!r}")
        if not isinstance(asset, str) or not asset:
            raise ValueError(f"asset must be a non-empty string, got {asset!r}")
        self.asset = asset
        self.data_index = _shared_index([self])  # raises if data indexes differ

    def __repr__(self) -> str:
        names = [s.name for s in self.sources]
        return (
            f"LinearCombination(sources={names}, weights={self.weights}, sigma={self.sigma}, "
            f"asset={self.asset!r}, constant={self.constant})"
        )


# --- simulation ---------------------------------------------------------------------------


def _walk(roots: Sequence[LinearCombination]):
    """Unique models and combinations (by identity) in depth-first order, and all data."""
    models: dict[int, Process] = {}
    combos: dict[int, LinearCombination] = {}
    data: list[_Source] = []

    def visit(combo: LinearCombination) -> None:
        if id(combo) in combos:
            return
        for s in combo.sources:
            if s.kind == "model":
                models.setdefault(id(s.model), s.model)
            elif s.kind == "data":
                data.append(s)
            else:
                visit(s.combination)
        combos[id(combo)] = combo

    for root in roots:
        visit(root)
    return list(models.values()), list(combos.values()), data


def _shared_index(roots) -> pd.DatetimeIndex | None:
    _, _, data = _walk(roots)
    if not data:
        return None
    first = data[0].data.index
    for s in data[1:]:
        if not s.data.index.equals(first):
            raise ValueError(f"data sources {data[0].name!r} and {s.name!r} have different indexes")
    return first


def _grid(roots, grid_args: dict, days_per_year: float | None) -> TimeGrid:
    _, _, data = _walk(roots)
    dpy = DAYS_PER_YEAR if days_per_year is None else days_per_year
    if data:
        given = sorted(k for k, v in grid_args.items() if v is not None)
        if given:
            raise ValueError(f"the grid comes from the data sources' index; do not pass {given}")
        index = _shared_index(roots)
        starts = {s.data.attrs.get(START) for s in data} - {None}
        start = starts.pop() if len(starts) == 1 else None
        return IndexGrid(index, start=start, days_per_year=dpy)
    if grid_args.get("start") is None or grid_args.get("freq") is None:
        raise ValueError("with only model sources, give start, freq and n_steps (or end)")
    return RegularGrid(
        grid_args["start"],
        grid_args["freq"],
        grid_args.get("n_steps"),
        grid_args.get("end"),
        days_per_year=dpy,
    )


def _name_map(combos: list[LinearCombination]) -> None:
    """Every source name must mean one thing across all combinations."""
    seen: dict[str, tuple] = {}
    for combo in combos:
        for s in combo.sources:
            key = {
                "model": ("model", id(s.model), s.column),
                "data": ("data", s.origin),
                "combination": ("combination", id(s.combination)),
            }[s.kind]
            if seen.setdefault(s.name, key) != key:
                raise ValueError(f"the source name {s.name!r} refers to two different sources")


def simulate_combinations(
    roots,
    *,
    seed: int,
    n_paths: int = 1,
    return_hidden: bool = False,
    noise: Noise | None = None,
    days_per_year: float | None = None,
    **grid_args,
):
    """Simulate one or more combinations; see :func:`fantasma.simulate`."""
    from .simulate import step_states  # imported here: simulate dispatches to this module

    roots = [roots] if isinstance(roots, LinearCombination) else list(roots)
    if not roots or not all(isinstance(r, LinearCombination) for r in roots):
        raise TypeError("expected a LinearCombination or a list of them")
    assets = [r.asset for r in roots]
    if len(set(assets)) != len(assets):
        raise ValueError(f"the combinations' assets must be distinct, got {assets}")
    if seed is None:
        raise ValueError("a seed is required, so every run is reproducible")
    if not isinstance(n_paths, (int, np.integer)) or isinstance(n_paths, bool) or n_paths < 1:
        raise ValueError(f"n_paths must be a positive integer, got {n_paths!r}")
    noise = noise or GaussianNoise()
    models, combos, _ = _walk(roots)
    _name_map(combos)
    grid = _grid(roots, grid_args, days_per_year)
    index, dts = grid.index[1:], grid.dts
    sqrt_dts = np.sqrt(dts)

    results = []
    for path in np.random.SeedSequence(seed).spawn(n_paths):
        children = path.spawn(len(models) + len(combos))
        realised: dict[int, SourceRealisation] = {}
        for model, child in zip(models, children, strict=False):
            (states,) = step_states(model, grid, [child], noise)
            realised[id(model)] = SourceRealisation(
                log_return_frame(model, states, grid), hidden_frame(model, states, grid)
            )
        outputs: dict[int, np.ndarray] = {}
        for combo, child in zip(combos, children[len(models) :], strict=True):
            eps = noise.sample(np.random.default_rng(child), len(dts), 1)[:, 0]
            total = combo.constant * dts + combo.sigma * sqrt_dts * eps
            for w, s in zip(combo.weights, combo.sources, strict=True):
                total = total + w * _source_values(s, realised, outputs)
            outputs[id(combo)] = total

        frame = _output_frame(roots, outputs, index, grid)
        if return_hidden:
            results.append(_path_result(roots, frame, realised, outputs, index, grid))
        else:
            results.append(frame)
    return results


def _source_values(s: _Source, realised, outputs) -> np.ndarray:
    if s.kind == "model":
        return realised[id(s.model)].output[s.column].to_numpy()
    if s.kind == "data":
        return s.data.to_numpy(dtype=float)
    return outputs[id(s.combination)]


def _output_frame(combos, outputs, index, grid) -> pd.DataFrame:
    columns = pd.Index([c.asset for c in combos], name="symbol", dtype="string")
    values = np.column_stack([outputs[id(c)] for c in combos])
    frame = pd.DataFrame(values, index=index, columns=columns, dtype="float64")
    initial = {c.asset: np.log(c.initial_price) for c in combos}
    frame.attrs.update(log_return_attrs(grid.index[0], initial))
    return frame


def _path_result(combos, frame, realised, outputs, index, grid) -> PathResult:
    empty_hidden = pd.DataFrame(
        index=grid.index, columns=pd.Index([], name="state", dtype="string"), dtype="float64"
    )
    sources: dict = {}
    for combo in combos:
        for s in combo.sources:
            if s.name in sources:
                continue
            if s.kind == "model":
                sources[s.name] = realised[id(s.model)]
            elif s.kind == "data":
                data = s.data.to_frame()
                data.attrs = dict(s.data.attrs)
                sources[s.name] = SourceRealisation(data, empty_hidden)
            else:
                sub = s.combination
                sub_frame = _output_frame([sub], outputs, index, grid)
                sources[s.name] = _path_result([sub], sub_frame, realised, outputs, index, grid)
    return PathResult(frame, sources)
