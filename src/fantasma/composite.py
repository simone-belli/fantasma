"""Generic composition: processes whose states drive other processes' parameters."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .process import Inputs, Process


class Composite(Process):
    """Several processes advanced together, with states of some feeding inputs of others.

    ``components`` maps a component name to a process. ``links`` maps ``"target.input"`` to
    ``"source.state"``: in each step the target process reads the source's state *at the start
    of the step* as the value of its drivable parameter ``input``. Because every link reads the
    pre-step state, the order of components does not matter and feedback loops are allowed.

    The composite is itself a :class:`Process`, so composites nest. Its states are named
    ``"<component>.<state>"``; its assets are the components' assets, in component order, and
    must be distinct. Shocks are split between components in component order.

    >>> drift = OrnsteinUhlenbeck(theta=0.05, half_life=0.5, sigma=0.1, state_name="drift")
    >>> price = Brownian(sigma=0.2, asset="A")
    >>> Composite({"A": drift, "A_price": price}, links={"A_price.mu": "A.drift"})
    """

    def __init__(self, components: Mapping[str, Process], links: Mapping[str, str] | None = None):
        if not components:
            raise ValueError("a composite needs at least one component")
        for name, proc in components.items():
            if not isinstance(name, str) or not name or "." in name:
                raise ValueError(f"component names must be non-empty and dot-free, got {name!r}")
            if not isinstance(proc, Process):
                raise TypeError(f"component {name!r} is not a Process: {proc!r}")
        self.components = dict(components)
        self.links = dict(links or {})

        self.state_names = tuple(
            f"{c}.{s}" for c, p in self.components.items() for s in p.state_names
        )
        self.observable_states = tuple(
            f"{c}.{s}" for c, p in self.components.items() for s in p.observable_states
        )
        assets = tuple(a for p in self.components.values() for a in p.asset_names)
        if len(set(assets)) != len(assets):
            raise ValueError(f"asset names across components must be distinct, got {list(assets)}")
        self.asset_names = assets
        self.n_shocks = sum(p.n_shocks for p in self.components.values())
        self.params = {"components": self.components, "links": self.links}

        # Slices into the composite state and shock vectors, per component.
        self._state_slices: dict[str, slice] = {}
        self._shock_slices: dict[str, slice] = {}
        s = k = 0
        for c, p in self.components.items():
            self._state_slices[c] = slice(s, s + p.n_state)
            self._shock_slices[c] = slice(k, k + p.n_shocks)
            s += p.n_state
            k += p.n_shocks

        # Resolved wiring: target component -> [(input name, index into composite state)].
        self._wiring: dict[str, list[tuple[str, int]]] = {c: [] for c in self.components}
        for target, source in self.links.items():
            tc, _, tin = target.partition(".")
            if tc not in self.components or not tin:
                raise ValueError(f"link target {target!r} is not '<component>.<input>'")
            if tin not in self.components[tc].inputs:
                known = list(self.components[tc].inputs)
                raise ValueError(f"{tc!r} has no drivable input {tin!r}; inputs: {known}")
            if source not in self.state_names:
                raise ValueError(
                    f"link source {source!r} is not a state; states: {list(self.state_names)}"
                )
            self._wiring[tc].append((tin, self.state_names.index(source)))

        # Inputs of the composite itself: unwired inputs of components, re-exposed by full name.
        wired = set(self.links)
        self.inputs = tuple(
            f"{c}.{i}"
            for c, p in self.components.items()
            for i in p.inputs
            if f"{c}.{i}" not in wired
        )

    @property
    def hidden_states(self) -> tuple[str, ...]:
        return tuple(f"{c}.{s}" for c, p in self.components.items() for s in p.hidden_states)

    def initial_state(self) -> np.ndarray:
        return np.concatenate([p.initial_state() for p in self.components.values()])

    def step(self, state: np.ndarray, shocks: np.ndarray, dt: float, inputs: Inputs) -> np.ndarray:
        new = np.empty_like(state)
        for c, p in self.components.items():
            local = {i: state[:, idx] for i, idx in self._wiring[c]}
            prefix = f"{c}."
            local.update({k[len(prefix) :]: v for k, v in inputs.items() if k.startswith(prefix)})
            sl = self._state_slices[c]
            new[:, sl] = p.step(state[:, sl], shocks[:, self._shock_slices[c]], dt, local)
        return new

    def observe(self, state: np.ndarray) -> np.ndarray:
        parts = [p.observe(state[..., self._state_slices[c]]) for c, p in self.components.items()]
        return np.concatenate(parts, axis=-1)

    def hidden(self, state: np.ndarray) -> np.ndarray:
        parts = [p.hidden(state[..., self._state_slices[c]]) for c, p in self.components.items()]
        return np.concatenate(parts, axis=-1)

    def __repr__(self) -> str:
        return f"Composite({self.components!r}, links={self.links!r})"
