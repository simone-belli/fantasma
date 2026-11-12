"""Parameter-recovery tests (SPEC section 8, optional per model).

Each test simulates seeded paths with known parameters, estimates them back with a standard
method, and checks the estimate lies within ``Z`` standard errors of the truth, where the
standard error is the method's expected estimation error (analytic, or across many paths).
"""

import numpy as np
import pytest

import fantasma as fm

sm = pytest.importorskip("statsmodels.api")
from statsmodels.tsa.vector_ar.vecm import coint_johansen  # noqa: E402

pytestmark = [
    pytest.mark.recovery,
    pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning"),
]

Z = 4.0  # tolerance in standard errors: a false failure about once in 16,000 runs
DT = 1 / 365.25


def daily(process, n_steps, seed, **kw):
    return fm.simulate(process, start="2000-01-01", freq="1d", n_steps=n_steps, seed=seed, **kw)


def test_random_walk_realised_vol():
    sigma, n = 0.2, 50_000
    (log_returns,) = daily(fm.RandomWalk(sigma), n, seed=10)
    r = log_returns["ASSET"].to_numpy()
    sigma_hat = np.sqrt(np.mean(r**2) / DT)  # known zero mean
    se = sigma / np.sqrt(2 * n)
    assert abs(sigma_hat - sigma) < Z * se


def test_drift_plus_noise_drift_and_vol():
    mu, sigma, n = 0.1, 0.2, 73_000  # 200 years, so the drift is measurable
    (log_returns,) = daily(fm.DriftPlusNoise(mu, sigma), n, seed=11)
    r = log_returns["ASSET"].to_numpy()
    mu_hat = r.mean() / DT
    sigma_hat = r.std(ddof=1) / np.sqrt(DT)
    assert abs(mu_hat - mu) < Z * sigma / np.sqrt(n * DT)
    assert abs(sigma_hat - sigma) < Z * sigma / np.sqrt(2 * n)


def test_ou_ar1_recovers_mean_and_half_life():
    theta, half_life, sigma, n = np.log(50.0), 0.1, 0.3, 50_000
    (log_returns,) = daily(fm.OrnsteinUhlenbeck(theta, half_life, sigma, asset="OU"), n, seed=12)
    x = fm.to_log_prices(log_returns)["OU"].to_numpy()
    fit = sm.OLS(x[1:], sm.add_constant(x[:-1])).fit()
    (a, b), cov = fit.params, fit.cov_params()

    hl_hat = -np.log(2) * DT / np.log(b)
    theta_hat = a / (1 - b)
    # Delta method for the standard errors of the transformed estimates.
    g_hl = np.array([0.0, np.log(2) * DT / (b * np.log(b) ** 2)])
    g_theta = np.array([1 / (1 - b), a / (1 - b) ** 2])
    assert abs(hl_hat - half_life) < Z * np.sqrt(g_hl @ cov @ g_hl)
    assert abs(theta_hat - theta) < Z * np.sqrt(g_theta @ cov @ g_theta)


def test_cointegrated_rank_and_loadings():
    loadings = np.array([[1.0], [0.5], [2.0]])
    model = fm.Cointegrated(
        loadings,
        spread_half_lives=[0.05, 0.1, 0.2],
        factor_sigmas=0.3,
        spread_sigmas=0.1,
        assets=["X", "Y", "Z"],
    )
    n_paths = 40
    paths = daily(model, 20_000, seed=13, n_paths=n_paths)

    ratios, ranks = [], []
    for log_returns in paths:
        # Measured from the start, the factors and spreads start at zero and have no drift, so
        # the test takes no deterministic terms: log prices relative to the start, from 1.0.
        logp = fm.to_log_prices(log_returns, initial_price=1.0).to_numpy()
        # Johansen trace test at 5%: the rank is the first r not rejected; truth is N - K = 2.
        jo = coint_johansen(logp, det_order=-1, k_ar_diff=1)
        accepted = jo.lr1 < jo.cvt[:, 1]
        ranks.append(int(np.argmax(accepted)) if accepted.any() else 3)
        # Loadings are identified up to scale: the common-trend direction is the null space of
        # the true-rank cointegrating vectors. Normalise it to the first asset's loading.
        beta = np.real(jo.evec[:, :2])
        trend = np.linalg.svd(beta.T)[2][-1]
        ratios.append(trend / trend[0])

    # A 5% test finds the wrong rank in about 5% of paths; allow for that, binomially.
    wrong = sum(r != 2 for r in ranks)
    assert wrong <= n_paths * 0.05 + Z * np.sqrt(n_paths * 0.05 * 0.95)

    ratios = np.array(ratios)
    truth = loadings[:, 0] / loadings[0, 0]
    se = ratios.std(axis=0, ddof=1) / np.sqrt(n_paths)
    assert np.all(np.abs(ratios.mean(axis=0) - truth) < Z * se + 1e-12)


def kalman_drift(r, dt, sigma, mean, half_life, drift_sigma, mu0):
    """Filtered drift and its variance, for returns ``r = mu dt + sigma sqrt(dt) eps`` with
    ``mu`` an exactly discretised OU process (the true model, known parameters)."""
    phi = np.exp(-np.log(2) / half_life * dt)
    q = drift_sigma**2 * (1 - phi**2) / (2 * np.log(2) / half_life)
    h, rvar = dt, sigma**2 * dt
    m, p = mu0, 0.0
    est, var = np.empty(len(r)), np.empty(len(r))
    for t, y in enumerate(r):
        # Return r[t] is driven by the drift at the start of step t.
        k = p * h / (h * h * p + rvar)
        m, p = m + k * (y - h * m), (1 - k * h) * p
        est[t], var[t] = m, p
        m, p = mean + phi * (m - mean), phi**2 * p + q
    return est, var


def test_stochastic_drift_kalman_tracks_hidden_drift():
    params = dict(sigma=0.1, drift_mean=0.05, drift_half_life=2.0, drift_sigma=0.3)
    model = fm.StochasticDrift(**params)
    ((log_returns, hidden),) = daily(model, 40_000, seed=14, return_hidden=True)
    r = log_returns["ASSET"].to_numpy()
    # Hidden state is on the full grid: return t is driven by the drift at the start of its step.
    true_mu = hidden["drift.mu"].to_numpy()[:-1]
    est, var = kalman_drift(
        r,
        DT,
        params["sigma"],
        params["drift_mean"],
        params["drift_half_life"],
        params["drift_sigma"],
        mu0=params["drift_mean"],
    )
    burn = 2_000
    err, var = (est - true_mu)[burn:], var[burn:]

    # Tracking: the filter explains most of the drift's variation...
    assert np.corrcoef(est[burn:], true_mu[burn:])[0, 1] > 0.8
    # ...and its errors are the size its own variance predicts. Errors are autocorrelated, so
    # the standard error of the mean squared standardised error is taken from batch means.
    z2 = err**2 / var
    batches = z2[: len(z2) // 40 * 40].reshape(40, -1).mean(axis=1)
    assert abs(z2.mean() - 1) < Z * batches.std(ddof=1) / np.sqrt(len(batches))


def driven_inputs(n_steps, seed):
    """Two log-return frames, each two columns of random walks."""
    model = fm.Composite({c: fm.RandomWalk(0.3, asset=c) for c in ("X", "Y")})
    return daily(model, n_steps, seed=seed, n_paths=2)


def test_linear_combination_ols_recovers_weights_constant_sigma():
    # Two outputs sharing one market source, plus their own second source.
    mkt = fm.RandomWalk(0.3, asset="MKT")
    mom = fm.StochasticDrift(0.2, 0.05, 1.0, 0.3, asset="MOM")
    ou = fm.OrnsteinUhlenbeck(np.log(100), 0.2, 0.3, asset="OU")
    combos = [
        fm.LinearCombination([mkt, mom], weights=[0.8, -0.5], sigma=0.05, asset="A", constant=0.2),
        fm.LinearCombination([mkt, ou], weights=[1.5, 0.3], sigma=0.1, asset="B", constant=-1.0),
    ]
    (res,) = fm.simulate(
        combos, start="2000-01-01", freq="1d", n_steps=20_000, seed=21, return_hidden=True
    )
    for combo in combos:
        X = sm.add_constant(
            np.column_stack([res.sources[s.name].output[s.name] for s in combo.sources])
        )
        fit = sm.OLS(res.output[combo.asset].to_numpy(), X).fit()
        # The constant is an annualised drift: the intercept is constant * dt.
        truth = np.array([combo.constant * DT, *combo.weights])
        assert np.all(np.abs(fit.params - truth) < Z * fit.bse)
        sigma_hat = np.sqrt(fit.mse_resid / DT)
        assert abs(sigma_hat - combo.sigma) < Z * combo.sigma / np.sqrt(2 * len(res.output))


def test_regime_switching_recovers_transitions_and_regime_parameters():
    P = np.array([[0.95, 0.04, 0.01], [0.05, 0.9, 0.05], [0.02, 0.08, 0.9]])
    regimes = [
        dict(weights=[1.0, 0.0], constant=0.0, sigma=0.05),
        dict(weights=[0.5, 0.5], constant=1.0, sigma=0.1),
        dict(weights=[-1.0, 2.0], constant=-0.5, sigma=0.2),
    ]
    x1, x2 = driven_inputs(30_000, seed=22)
    model = fm.RegimeSwitching(regimes, transition_matrix=P)
    ((out, hidden),) = fm.simulate_driven(model, [x1, x2], seed=23, return_hidden=True)
    r = hidden["regime"].to_numpy()

    # Transition probabilities: counts from each regime are multinomial.
    counts = np.zeros((3, 3))
    np.add.at(counts, (r[:-1], r[1:]), 1)
    n_from = counts.sum(axis=1, keepdims=True)
    P_hat = counts / n_from
    assert np.all(np.abs(P_hat - P) < Z * np.sqrt(P * (1 - P) / n_from) + 1e-12)

    # Given the true regimes, OLS within each regime recovers that regime's combination.
    for k, reg in enumerate(regimes):
        rows = r == k
        X = sm.add_constant(np.column_stack([x1["X"][rows], x2["X"][rows]]))
        fit = sm.OLS(out["X"].to_numpy()[rows], X).fit()
        truth = np.array([reg["constant"] * DT, *reg["weights"]])
        assert np.all(np.abs(fit.params - truth) < Z * fit.bse)
        sigma_hat = np.sqrt(fit.mse_resid / DT)
        assert abs(sigma_hat - reg["sigma"]) < Z * reg["sigma"] / np.sqrt(2 * rows.sum())
