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
| `fm.from_prices(prices)` | log returns from prices, e.g. real data for driven models | one per step |

The level helpers take the start timestamp and starting prices from `attrs` by default, so
`fm.to_prices(r)` rebuilds the model's exact price path, start included. Pass `initial_price=`
(a scalar, one value per column, or a dict/Series keyed by column) or `start=` to override
them, or when a frame does not carry them: pandas drops `attrs` after some operations, and
driven models have no natural starting price. `fm.to_prices(fm.from_prices(p))` returns `p`.

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
paths = fm.simulate(model, start="2024-01-01", freq="10m", n_steps=10_000, seed=42,
                    n_paths=100, return_hidden=True)
log_returns, hidden = paths[0]       # hidden["drift.mu"] is the true drift
prices = fm.to_prices(log_returns)   # 10_001 rows, from the start
```

- `simulate` returns a list with one log-return DataFrame per path (or `(log_returns, hidden)`
  pairs with `return_hidden=True`). Path `i` is drawn from child `i` of `SeedSequence(seed)`:
  identical inputs and seed give identical output, and asking for more paths leaves the first
  unchanged.
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

### Driven models

These combine input series you give them instead of evolving their own state. Like everything
in Fantasma they work in log returns: `inputs` is a list of log-return DataFrames with
identical index (a `DatetimeIndex`) and columns, such as paths from the models above, or real
prices converted with `fm.from_prices` (e.g.
`fm.from_prices(load(...)["close"].unstack("symbol"))`). Frames labelled as anything other
than log returns are refused. The output is log returns with exactly the inputs' index and
columns.

| Class | Output, column by column | Hidden state |
| --- | --- | --- |
| `LinearCombination(weights, constant=0, sigma=0)` | factor model: `r[t] = constant * dt[t] + sum_i weights[i] * r_i[t] + sigma * sqrt(dt[t]) * eps[t]` | — |
| `RegimeSwitching(regimes, transition_matrix, initial=None)` | the `LinearCombination` of the current regime; the regime follows a Markov chain | `regime` |

```python
lin = fm.LinearCombination([1.0, [0.5, -0.5]], constant=0.1, sigma=0.05)
paths = fm.simulate_driven(lin, [r1, r2], seed=42, n_paths=10)   # r1, r2: log returns
prices = fm.to_prices(paths[0], initial_price=100.0)              # no natural start price

model = fm.RegimeSwitching(
    [fm.LinearCombination([1.0, 0.0], sigma=0.05), fm.LinearCombination([0.0, 1.0], sigma=0.2)],
    transition_matrix=[[0.98, 0.02], [0.05, 0.95]],
)
out, hidden = fm.simulate_driven(model, [r1, r2], seed=42, return_hidden=True)[0]
```

- `weights` has one entry per input series. Each weight, and `constant` and `sigma`, is a
  scalar (every column), a list with one value per column, or a Series keyed by column.
- `constant` is an annualised drift and `sigma` an annualised volatility. `eps` is independent
  across rows and columns, from the `noise=` passed to `simulate_driven` (standard Gaussian by
  default). Being returns, the noise accumulates in the output's price. `dt[t]` is the time
  since the previous row, so uneven indexes (weekend gaps) scale drift and noise by elapsed
  time. The first row uses the first gap.
- The output records the inputs' start timestamp when they agree, but has no starting price:
  give `initial_price=` to `fm.to_prices`.
- NaN in an input gives NaN in the output at that row and column.
- Regimes: `transition_matrix[i][j]` is the probability of moving from regime `i` to `j` per
  row; one regime path is shared by all columns. `initial` is a regime, a probability per
  regime, or `None` for the chain's stationary distribution. The regime path and the noise use
  separate seeded streams, so changing the noise type leaves the regimes unchanged.

## Notebooks

`notebooks/` has one notebook per model. Each starts from a parameters cell (model parameters,
seed, start, frequency, number of steps and paths; for driven models, the inputs too),
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
