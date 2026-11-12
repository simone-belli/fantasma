"""Models built from sources: linear combinations and regime switching.

Both node types produce one log-return series from **sources**. A source is

- a one-asset model (any :class:`~fantasma.process.Process`), simulated by the framework;
- ``(model, "X")``: asset ``X`` of a multi-asset model;
- data: a log-return ``Series`` (named) or one-column ``DataFrame``, e.g. real prices
  converted with ``fm.from_prices``;
- another node, so nodes nest: a :class:`LinearCombination` or a :class:`RegimeSwitching`.

:class:`LinearCombination`::

    r[t] = constant * dt[t] + sum_i weights[i] * r_i[t] + sigma * sqrt(dt[t]) * eps[t]

:class:`RegimeSwitching`: ``r[t] = r_k[t]`` with ``k = regime[t]``, the regime following a
:class:`~fantasma.regimes.MarkovChain` that moves once per row. Every regime's source is
simulated over the whole grid (inactive regimes keep running); the output takes the active one.

Run nodes with ``fm.simulate(node_or_list, ...)``. Within a path, each model object is
simulated **once**, however many nodes use it or however many of its assets are selected. With
data among the sources, the grid is the data's index (models are simulated on it, gaps
included); with models only, it comes from ``start``/``freq``/``n_steps``.

With ``return_hidden=True`` each path is a :class:`PathResult`: ``.output`` (one log-return
column per node), ``.regimes`` (the regime path of each regime-switching node) and
``.sources[name]``, the realisation of each source (its log returns and hidden state).
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
from .regimes import MarkovChain
from .transforms import LOG_RETURN, QUANTITY, START, log_return_attrs


@dataclass(frozen=True)
class SourceRealisation:
    """One source in one path: its log returns (all of a model's assets) and hidden state on
    the full grid (start included; empty for data)."""

    output: pd.DataFrame
    hidden: pd.DataFrame


def _no_regimes() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index([], name="symbol", dtype="string"), dtype="int64")


@dataclass(frozen=True)
class PathResult:
    """One path: the nodes' log returns, their regime paths, and every source's realisation.

    ``regimes`` has one integer column per regime-switching node (indexed like ``output``);
    it is empty when there is none. A nested node appears in ``sources`` as its own
    ``PathResult``.
    """

    output: pd.DataFrame
    sources: dict[str, SourceRealisation | PathResult] = field(default_factory=dict)
    regimes: pd.DataFrame = field(default_factory=_no_regimes)


@dataclass(frozen=True)
class _Source:
    kind: str  # "model", "data" or "node"
    name: str
    model: Process | None = None
    column: str | None = None
    data: pd.Series | None = None
    node: _Node | None = None
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


def _as_source(label: str, i: int, src) -> _Source:
    where = f"{label}[{i}]"
    if isinstance(src, _Node):
        return _Source("node", src.asset, node=src)
    if isinstance(src, tuple):
        if len(src) != 2 or not isinstance(src[0], Process) or not isinstance(src[1], str):
            raise TypeError(f"{where}: a tuple must be (model, asset name), got {src!r}")
        model, column = src
        if column not in model.asset_names:
            raise ValueError(
                f"{where}: {column!r} is not an asset of the model; "
                f"assets: {list(model.asset_names)}"
            )
        return _Source("model", column, model=model, column=column)
    if isinstance(src, Process):
        if len(src.asset_names) != 1:
            raise ValueError(
                f"{where} has assets {list(src.asset_names)}; select one with (model, asset name)"
            )
        return _Source("model", src.asset_names[0], model=src, column=src.asset_names[0])
    if isinstance(src, pd.DataFrame):
        if src.shape[1] != 1:
            raise ValueError(f"{where}: a DataFrame source needs exactly one column")
        series = src.iloc[:, 0]
        series.attrs = dict(src.attrs)
        name = str(src.columns[0])
        return _Source("data", name, data=_check_data(name, series.rename(name)), origin=id(src))
    if isinstance(src, pd.Series):
        if src.name is None:
            raise ValueError(f"{where}: a Series source needs a name")
        name = str(src.name)
        return _Source("data", name, data=_check_data(name, src), origin=id(src))
    raise TypeError(
        f"{where} must be a model, (model, asset), a Series, a one-column DataFrame, "
        f"a LinearCombination or a RegimeSwitching; got {type(src).__name__}"
    )


class _Node:
    """A model built from sources, producing one log-return series named ``asset``."""

    sources: list[_Source]
    asset: str
    initial_price: float

    def _init_sources(self, label: str, sources, asset: str, initial_price: float) -> None:
        if not isinstance(sources, (list, tuple)) or not sources:
            raise ValueError(f"{label} must be a non-empty list")
        self.sources = [_as_source(label, i, s) for i, s in enumerate(sources)]
        names = [s.name for s in self.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"{label} names must be distinct, got {names}")
        self.initial_price = _finite("initial_price", initial_price)
        if self.initial_price <= 0:
            raise ValueError(f"initial_price must be positive, got {initial_price!r}")
        if not isinstance(asset, str) or not asset:
            raise ValueError(f"asset must be a non-empty string, got {asset!r}")
        self.asset = asset
        self.data_index = _shared_index([self])  # raises if data indexes differ

    def _evaluate(self, values, dts, rng, noise) -> tuple[np.ndarray, np.ndarray | None]:
        """Output log returns from the sources' ``values`` (one array per source), and the
        regime path for a regime-switching node."""
        raise NotImplementedError


class LinearCombination(_Node):
    """``r[t] = constant * dt[t] + sum_i weights[i] * r_i[t] + sigma * sqrt(dt[t]) * eps[t]``.

    ``sources``: models, ``(model, asset)`` pairs, log-return data, or other nodes (see the
    module docstring). ``weights``: one number per source. ``sigma``: annualised volatility of
    the noise. ``asset``: the output's name. ``constant``: an optional annualised drift.
    ``initial_price``: recorded with the output, so ``fm.to_prices`` starts there.
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
        self._init_sources("sources", sources, asset, initial_price)
        if not isinstance(weights, (list, tuple, np.ndarray)) or len(weights) != len(sources):
            raise ValueError(f"weights must have one number per source ({len(sources)})")
        self.weights = [_finite(f"weights[{i}]", w) for i, w in enumerate(weights)]
        self.sigma = _finite("sigma", sigma)
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma!r}")
        self.constant = _finite("constant", constant)

    def _evaluate(self, values, dts, rng, noise):
        eps = noise.sample(rng, len(dts), 1)[:, 0]
        total = self.constant * dts + self.sigma * np.sqrt(dts) * eps
        for w, v in zip(self.weights, values, strict=True):
            total = total + w * v
        return total, None

    def __repr__(self) -> str:
        names = [s.name for s in self.sources]
        return (
            f"LinearCombination(sources={names}, weights={self.weights}, sigma={self.sigma}, "
            f"asset={self.asset!r}, constant={self.constant})"
        )


class RegimeSwitching(_Node):
    """The log returns of one of N regimes at a time, the regime following a Markov chain.

    ``models``: the N regimes, each a source as for :class:`LinearCombination` (a model,
    ``(model, asset)``, log-return data, or another node). ``transition_matrix``: N×N,
    ``transition_matrix[i][j]`` the probability of moving from regime ``i`` to ``j`` per row.
    ``initial``: the distribution of the first row's regime (N probabilities), a fixed regime
    index, or ``None`` for the chain's stationary distribution. ``asset``: the output's name.
    ``initial_price``: recorded with the output, so ``fm.to_prices`` starts there.

    Every regime's source is simulated over the whole grid; ``r[t]`` is the return of the
    regime active at row ``t``. The regime path is returned in ``PathResult.regimes``.
    """

    def __init__(
        self,
        models: Sequence,
        transition_matrix,
        initial=None,
        *,
        asset: str,
        initial_price: float = 100.0,
    ):
        self._init_sources("models", models, asset, initial_price)
        self.chain = MarkovChain(transition_matrix, initial)
        if self.chain.n_regimes != len(self.sources):
            raise ValueError(
                f"transition_matrix is for {self.chain.n_regimes} regimes, "
                f"got {len(self.sources)} models"
            )

    @property
    def n_regimes(self) -> int:
        return len(self.sources)

    def _evaluate(self, values, dts, rng, noise):
        regime = self.chain.sample(rng, len(dts))
        return np.choose(regime, np.stack(values)), regime

    def __repr__(self) -> str:
        names = [s.name for s in self.sources]
        return (
            f"RegimeSwitching(models={names}, "
            f"transition_matrix={self.chain.transition_matrix.tolist()}, "
            f"initial={self.chain.initial!r}, asset={self.asset!r})"
        )


# --- simulation ---------------------------------------------------------------------------


def _walk(roots: Sequence[_Node]):
    """Unique models and nodes (by identity) in depth-first order, and all data."""
    models: dict[int, Process] = {}
    nodes: dict[int, _Node] = {}
    data: list[_Source] = []

    def visit(node: _Node) -> None:
        if id(node) in nodes:
            return
        for s in node.sources:
            if s.kind == "model":
                models.setdefault(id(s.model), s.model)
            elif s.kind == "data":
                data.append(s)
            else:
                visit(s.node)
        nodes[id(node)] = node

    for root in roots:
        visit(root)
    return list(models.values()), list(nodes.values()), data


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


def _name_map(nodes: list[_Node]) -> None:
    """Every source name must mean one thing across all nodes."""
    seen: dict[str, tuple] = {}
    for node in nodes:
        for s in node.sources:
            key = {
                "model": ("model", id(s.model), s.column),
                "data": ("data", s.origin),
                "node": ("node", id(s.node)),
            }[s.kind]
            if seen.setdefault(s.name, key) != key:
                raise ValueError(f"the source name {s.name!r} refers to two different sources")


def is_node(value) -> bool:
    return isinstance(value, _Node)


def simulate_nodes(
    roots,
    *,
    seed: int,
    n_paths: int = 1,
    return_hidden: bool = False,
    noise: Noise | None = None,
    days_per_year: float | None = None,
    **grid_args,
):
    """Simulate one or more nodes; returns a list of paths. See :func:`fantasma.simulate`."""
    from .simulate import step_states  # imported here: simulate dispatches to this module

    roots = [roots] if isinstance(roots, _Node) else list(roots)
    if not roots or not all(isinstance(r, _Node) for r in roots):
        raise TypeError("expected a LinearCombination or RegimeSwitching, or a list of them")
    assets = [r.asset for r in roots]
    if len(set(assets)) != len(assets):
        raise ValueError(f"the nodes' assets must be distinct, got {assets}")
    if seed is None:
        raise ValueError("a seed is required, so every run is reproducible")
    if not isinstance(n_paths, (int, np.integer)) or isinstance(n_paths, bool) or n_paths < 1:
        raise ValueError(f"n_paths must be a positive integer, got {n_paths!r}")
    noise = noise or GaussianNoise()
    models, nodes, _ = _walk(roots)
    _name_map(nodes)
    grid = _grid(roots, grid_args, days_per_year)
    index, dts = grid.index[1:], grid.dts

    results = []
    for path in np.random.SeedSequence(seed).spawn(n_paths):
        # One child per model, then one per node (its noise, or its regime chain).
        children = path.spawn(len(models) + len(nodes))
        realised: dict[int, SourceRealisation] = {}
        for model, child in zip(models, children, strict=False):
            (states,) = step_states(model, grid, [child], noise)
            realised[id(model)] = SourceRealisation(
                log_return_frame(model, states, grid), hidden_frame(model, states, grid)
            )
        outputs: dict[int, np.ndarray] = {}
        regimes: dict[int, np.ndarray] = {}
        for node, child in zip(nodes, children[len(models) :], strict=True):
            values = [_source_values(s, realised, outputs) for s in node.sources]
            out, regime = node._evaluate(values, dts, np.random.default_rng(child), noise)
            outputs[id(node)] = out
            if regime is not None:
                regimes[id(node)] = regime

        state = _PathState(realised, outputs, regimes, index, grid)
        frame = _output_frame(roots, state)
        results.append(_path_result(roots, frame, state) if return_hidden else frame)
    return results


@dataclass(frozen=True)
class _PathState:
    realised: dict
    outputs: dict
    regimes: dict
    index: pd.DatetimeIndex
    grid: TimeGrid


def _source_values(s: _Source, realised, outputs) -> np.ndarray:
    if s.kind == "model":
        return realised[id(s.model)].output[s.column].to_numpy()
    if s.kind == "data":
        return s.data.to_numpy(dtype=float)
    return outputs[id(s.node)]


def _output_frame(nodes, state: _PathState) -> pd.DataFrame:
    columns = pd.Index([n.asset for n in nodes], name="symbol", dtype="string")
    values = np.column_stack([state.outputs[id(n)] for n in nodes])
    frame = pd.DataFrame(values, index=state.index, columns=columns, dtype="float64")
    initial = {n.asset: np.log(n.initial_price) for n in nodes}
    frame.attrs.update(log_return_attrs(state.grid.index[0], initial))
    return frame


def _regimes_frame(nodes, state: _PathState) -> pd.DataFrame:
    switching = [n for n in nodes if id(n) in state.regimes]
    columns = pd.Index([n.asset for n in switching], name="symbol", dtype="string")
    if not switching:
        return pd.DataFrame(index=state.index, columns=columns, dtype="int64")
    values = np.column_stack([state.regimes[id(n)] for n in switching])
    return pd.DataFrame(values, index=state.index, columns=columns, dtype="int64")


def _path_result(nodes, frame, state: _PathState) -> PathResult:
    empty_hidden = pd.DataFrame(
        index=state.grid.index, columns=pd.Index([], name="state", dtype="string"), dtype="float64"
    )
    sources: dict = {}
    for node in nodes:
        for s in node.sources:
            if s.name in sources:
                continue
            if s.kind == "model":
                sources[s.name] = state.realised[id(s.model)]
            elif s.kind == "data":
                data = s.data.to_frame()
                data.attrs = dict(s.data.attrs)
                sources[s.name] = SourceRealisation(data, empty_hidden)
            else:
                sub_frame = _output_frame([s.node], state)
                sources[s.name] = _path_result([s.node], sub_frame, state)
    return PathResult(frame, sources, _regimes_frame(nodes, state))
