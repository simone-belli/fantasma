# Fantasma

Synthetic price paths with known dynamics, for testing quantitative models on scenarios where
the true data-generating process and its parameters are known. Output uses the
[Almanac](https://github.com/simone-belli/almanac) index and column conventions, so synthetic
and real data are interchangeable downstream. See [SPEC.md](SPEC.md).

## What Fantasma generates

**Every simulation produces log returns** (log price differences), whatever the model:

- one row per step, labelled at the **end** of the step (the start instant is not a row), so
  `n_steps` steps give `n_steps` rows;
- one `float64` column per asset;
- `frame.attrs` records the start: `{"quantity": "log_return", "start": <timestamp>,
  "initial_log_price": {asset: log price}}`.

Helpers turn log returns into the other quantities:

| Helper | Gives | Rows |
| --- | --- | --- |
| `fm.to_prices(r, initial_price=None, start=None)` | `P[t] = P0 * exp(r[1] + ... + r[t])` | start + one per step |
| `fm.to_log_prices(r, initial_price=None, start=None)` | `log P0 + r[1] + ... + r[t]` | start + one per step |
| `fm.to_returns(r)` | relative returns `P[t] / P[t-1] - 1 = expm1(r[t])` | one per step |
| `fm.from_prices(prices)` | log returns from prices, e.g. real data as a combination source | one per step |

The level helpers take the start timestamp and starting prices from `attrs` by default, so
`fm.to_prices(r)` rebuilds the model's exact price path, start included. Pass `initial_price=`
(a scalar, one value per column, or a dict/Series keyed by column) or `start=` to override
them, or when a frame does not carry them: pandas drops `attrs` after some operations, and
`RegimeSwitching` output has no natural starting price. `fm.to_prices(fm.from_prices(p))` returns `p`.

## Install

Dependencies are managed with [Poetry](https://python-poetry.org/) (2.0+):

```bash
poetry install                   # from a clone: creates .venv from poetry.lock, with dev tools
eval $(poetry env activate)      # activate it (or prefix commands with `poetry run`)
```

From another Poetry project, add Fantasma as a dependency (the core needs only numpy and pandas):

```bash
poetry add git+https://github.com/simone-belli/fantasma.git    # or: poetry add --editable ../fantasma
```

Python 3.12+.

## Use

```python
import fantasma as fm

model = fm.StochasticDrift(sigma=0.2, drift_mean=0.05, drift_half_life=0.5, drift_sigma=0.1,
                           asset="MOM")
log_returns, hidden = fm.simulate(model, start="2024-01-01", freq="10m", n_steps=10_000,
                                  seed=42, return_hidden=True)
prices = fm.to_prices(log_returns)   # 10_001 rows, from the start
# hidden["drift.mu"] is the true drift

paths = fm.simulate(model, start="2024-01-01", freq="10m", n_steps=10_000, seed=42,
                    n_paths=100)     # a list of 100 log-return frames
```

- `simulate` returns one log-return DataFrame per path (or a `(log_returns, hidden)` pair with
  `return_hidden=True`). With `n_paths=1`, the default, that single path is returned itself;
  with more, a list of paths. `fm.simulate_on` behaves the same.
- Path `i` is drawn from child `i` of `SeedSequence(seed)`: identical inputs and seed give
  identical output, and asking for more paths leaves the first unchanged, so
  `fm.simulate(..., seed=s)` equals `fm.simulate(..., seed=s, n_paths=10)[0]`.
- `freq` is an Almanac frequency (`1m`, `10m`, `1h`, `1d`) or any pandas offset. Give either
  `n_steps` or `end`.
- Hidden state is on the full grid, start included (`n_steps + 1` rows): `hidden` at row
  `t - 1` is the state that drove the return at row `t`.
- All parameters are annualised; `dt` is the step length over a 365.25-day year (the grid is
  continuous: no sessions or gaps). Override with `days_per_year`.

### Output format

Frames follow Almanac's conventions, so `fm.to_prices(r)` is the wide form of Almanac's
`load()`, i.e. `load(...)["close"].unstack("symbol")`:

| | intraday (`1m`, `10m`, `1h`) | daily (`1d`) |
| --- | --- | --- |
| index | `ts`, `datetime64[ns, UTC]` | `date`, naive `datetime64[ns]` at midnight |
| columns | one `float64` per asset, columns named `symbol` | same |

`fm.to_long(fm.to_prices(r))` gives Almanac's long layout: indexed by (`ts`|`date`, `symbol`),
with `close` and `source` (`"fantasma"`) columns. It takes prices, and refuses a frame
labelled as log returns. The hidden-state frame has columns named `state`.

### Models (v1)

All models evolve log prices; their output is the log returns.

| Class | Dynamics | Hidden state |
| --- | --- | --- |
| `RandomWalk(sigma)` | `dlog p = sigma dW` | — |
| `DriftPlusNoise(mu, sigma)` | `dlog p = mu dt + sigma dW` | — |
| `OrnsteinUhlenbeck(theta, half_life, sigma, asset=...)` | `dx = kappa (theta - x) dt + sigma dW`, `kappa = ln 2 / half_life`, exact discretisation | — (or `x` when used without an asset) |
| `Cointegrated(loadings, spread_half_lives, factor_sigmas, spread_sigmas, assets)` | `log p = log p0 + loadings @ f + s`; `f` random walks, `s` OU spreads | `factor_<k>`, `spread_<asset>` |
| `StochasticDrift(sigma, drift_mean, drift_half_life, drift_sigma)` | drift plus noise whose drift is OU | `drift.mu` |

### Linear combinations

`LinearCombination(sources, weights, sigma=0.0, *, asset, constant=0.0, initial_price=100.0)`
builds one log-return series from **sources**:

`r[t] = constant * dt[t] + sum_i weights[i] * r_i[t] + sigma * sqrt(dt[t]) * eps[t]`

A source is any of:

| Source | Name in `res.sources` |
| --- | --- |
| a one-asset model, e.g. `fm.RandomWalk(0.2, asset="MKT")`, simulated by Fantasma | its asset |
| `(model, "X")`: asset `X` of a multi-asset model such as `Cointegrated` | `"X"` |
| data: a log-return `Series` (named) or one-column `DataFrame`, e.g. real prices via `fm.from_prices` | its name / column |
| another `LinearCombination` or a `RegimeSwitching`, so they nest | its `asset` |

```python
mkt = fm.RandomWalk(0.2, asset="MKT")
mom = fm.StochasticDrift(0.1, 0.05, 2.0, 0.3, asset="MOM")
a = fm.LinearCombination([mkt, mom], weights=[1.2, 0.5], sigma=0.1, asset="A")
b = fm.LinearCombination([mkt, spy], weights=[0.8, -0.3], sigma=0.2, asset="B")  # spy: data

res = fm.simulate([a, b], seed=1, return_hidden=True)
res.output                     # log returns, columns A and B
res.sources["MKT"].output      # the market path both A and B loaded on
res.sources["MOM"].hidden      # its true drift, on the full grid
fm.to_prices(res.output)       # prices from initial_price
```

- `weights` has one number per source; `sigma` (noise volatility) and `constant` (drift) are
  annualised. `eps` comes from the `noise=` passed to `fm.simulate` (standard Gaussian by
  default). Being returns, the noise accumulates in the output's price.
- **Shared realisations.** Within a path, each model object is simulated once, however many
  combinations use it, or however many of its assets are selected: `a` and `b` above see the
  same market path. Equal but distinct model objects are independent.
- **Grid.** With data among the sources, the grid is the data's index (all data must share
  it), and the models are simulated on it, gaps included; `dt[t]` is the time since the
  previous row (the first row's comes from the data's recorded start, else the first gap is
  repeated). Do not pass `start`/`freq`/`n_steps` then. With models only, give them as usual.
- **Result.** `fm.simulate(node_or_list, ...)` returns one log-return frame per path (the path
  itself with `n_paths=1`, else a list), with a column per combination. With
  `return_hidden=True`, each path is a `fm.PathResult`:
  `.output`, and `.sources[name]`, a `fm.SourceRealisation` with `.output` (the source's log
  returns, all of a model's assets) and `.hidden` (its hidden state); a nested combination's
  entry is its own `PathResult`. `.regimes` holds regime paths (see below).
- The output records `initial_price`, so `fm.to_prices` works directly.

### Regime switching

`RegimeSwitching(models, transition_matrix, initial=None, *, asset, initial_price=100.0)`
follows one of N regimes at a time:

`r[t] = r_k[t]` with `k = regime[t]`

```python
calm = fm.LinearCombination([mkt], weights=[0.8], sigma=0.05, asset="CALM")
wild = fm.LinearCombination([mkt, mom], weights=[1.5, 1.0], sigma=0.3, asset="WILD")
stock = fm.RegimeSwitching([calm, wild], [[0.98, 0.02], [0.05, 0.95]], asset="STOCK")

res = fm.simulate(stock, start="2024-01-01", freq="1d", n_steps=750, seed=1, return_hidden=True)
res.output["STOCK"]       # the active regime's log return, row by row
res.regimes["STOCK"]      # the regime path, 0..N-1
res.sources["WILD"]       # each regime's own realisation
```

- `models`: the N regimes, each a source as for `LinearCombination` (a model,
  `(model, "X")`, log-return data, or another `LinearCombination` / `RegimeSwitching`).
  Names must be distinct.
- `transition_matrix`: N×N, `transition_matrix[i][j]` the probability of moving from regime
  `i` to `j` per row (rows sum to 1). `initial`: the first row's regime distribution (N
  probabilities), a fixed regime index, or `None` for the chain's stationary distribution.
- **Inactive regimes keep running.** Every regime's source is simulated over the whole grid,
  and the output takes the active regime's return at each row. A level-dependent model (e.g.
  OU) therefore reverts towards its own level, not the output's.
- The regime moves once per output row. With `return_hidden=True`, `PathResult.regimes` holds
  the regime path of every regime-switching node among the simulated ones (one column each,
  indexed like `output`); a nested one's is in its own `PathResult`. The regime path and the
  noise use separate seeded streams.
- Everything else (sharing, grids, nesting, `initial_price`) works as for
  `LinearCombination`, and the two nest in each other.

## Notebooks

`notebooks/` has one notebook per model. Each starts from a parameters cell (model parameters,
seed, start, frequency, number of steps and paths; for combinations, the sources),
simulates log returns, plots the prices (via `fm.to_prices`) and any hidden state, and runs a
quick check of the result against the inputs. `01_random_walk.ipynb` shows all the helpers
side by side.

```bash
poetry install --with notebooks
poetry run jupyter lab notebooks/
```

| Notebook | Model |
| --- | --- |
| `01_random_walk.ipynb` | `RandomWalk` |
| `02_drift_plus_noise.ipynb` | `DriftPlusNoise` |
| `03_ornstein_uhlenbeck.ipynb` | `OrnsteinUhlenbeck` |
| `04_cointegrated.ipynb` | `Cointegrated` |
| `05_stochastic_drift.ipynb` | `StochasticDrift` |
| `06_linear_combination.ipynb` | `LinearCombination` |
| `07_regime_switching.ipynb` | `RegimeSwitching` |

Notebooks are committed without outputs; the test suite executes each one.

## Extending

### A new model is one class

```python
import numpy as np
from fantasma import Process
from fantasma.process import check_positive

class MyWalk(Process):
    inputs = ("sigma",)                        # parameters other processes may drive

    def __init__(self, sigma, asset="ASSET"):
        self.params = {"sigma": check_positive("sigma", sigma)}
        self.asset_names = (asset,)
        self.state_names = ("log_price",)
        self.observable_states = ("log_price",)  # log prices, one per asset
        self.n_shocks = 1

    def initial_state(self):
        return np.array([np.log(100.0)])

    def step(self, state, shocks, dt, inputs):     # state: (n_paths, n_state)
        sigma = self.param("sigma", inputs)
        return state + (sigma * np.sqrt(dt) * shocks[:, 0])[:, None]
```

The framework handles the time grid, seeding, shocks, naming and output (the log returns of
the observed log prices). A process whose prices
are not just some of its states overrides `observe` / `hidden` / `hidden_states` (see
`Cointegrated`).

### Composition

`Composite({"name": process, ...}, links={"target.input": "source.state"})` wires the state of
one process into a drivable parameter of another. Each link reads the source state at the start
of the step, so component order is irrelevant. Composites are processes, so they nest.
`StochasticDrift` is exactly:

```python
drift = fm.OrnsteinUhlenbeck(0.05, 0.5, 0.1, state_name="mu")
price = fm.DriftPlusNoise(0.05, 0.2, asset="MOM")
fm.Composite({"drift": drift, "price": price}, links={"price.mu": "drift.mu"})
```

### Seams for later

- Noise: subclass `fm.Noise` (`sample(rng, n_steps, n_shocks)`) and pass `noise=` for
  fat-tailed or GARCH shocks; processes are unchanged.
- Time grid: subclass `fm.TimeGrid` (`index`, per-step `dts`) for sessions and overnight gaps,
  and call `fm.simulate_on(process, grid, seed=...)`.
- Output: `fantasma.output` builds the log-return frames from simulated states, and
  `fantasma.transforms` holds the helpers; OHLC and volume go there.

## Tests

```bash
pytest                    # basic checks for every model + parameter-recovery tests
pytest -m "not recovery and not notebooks"  # basic checks only
```

Recovery tests compare estimates with the truth within 4 standard errors of the estimator's
expected error (analytic, delta-method, or across seeded paths), not fixed thresholds.

## Versioning

The version is set in one place, `version` in `pyproject.toml`; `fantasma.__version__` reads it
from the installed package metadata. To release, change it there (or `poetry version minor`) and
re-run `poetry install`. `tests/test_version.py` guards this: it fails if the installed version
disagrees with `pyproject.toml`, or if a hardcoded `__version__` appears in the sources.

## License

MIT, see [LICENSE](LICENSE).
