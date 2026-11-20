"""Synthetic recovery: known elasticities, confounded prices.

The fast version runs in CI. The full study (more seeds, more draws) is in
`elasticity.pipeline` and is marked slow.
"""

import numpy as np
import pytest

from elasticity import bayes, classical
from elasticity.features import aggregate
from elasticity.simulate import SimConfig, simulate


def _truth(sim, index):
    return np.array([sim.beta[p] for p in index])


def test_naive_is_biased_towards_zero():
    cfg = SimConfig(n_products=4, n_stores=6, n_weeks=160, price_on_xi=1.0, seed=21)
    sim = simulate(cfg)
    a = aggregate(sim.panel)
    naive = classical.naive(a).set_index("product").elasticity
    truth = _truth(sim, naive.index)
    # bias towards zero: estimates are less negative than the truth on average
    assert (naive.to_numpy() - truth).mean() > 0.4


def test_iv_recovers_truth_within_interval():
    cfg = SimConfig(n_products=4, n_stores=6, n_weeks=160, price_on_xi=1.0, seed=21)
    sim = simulate(cfg)
    a = aggregate(sim.panel)
    r = classical.iv(a).set_index("product")
    truth = _truth(sim, r.index)
    inside = (r.lo90.to_numpy() <= truth) & (truth <= r.hi90.to_numpy())
    assert inside.mean() >= 0.5
    assert np.abs(r.elasticity.to_numpy() - truth).mean() < 0.5


def test_bayesian_iv_beats_naive_on_bias():
    cfg = SimConfig(n_products=3, n_stores=4, n_weeks=120, price_on_xi=1.0, seed=8)
    sim = simulate(cfg)
    a = aggregate(sim.panel)
    naive = classical.naive(a).set_index("product").elasticity
    idata, _, _ = bayes.fit(
        a, iv=True, draws=200, tune=200, chains=2, cores=1, seed=0, sampler="pymc"
    )
    est = bayes.per_product(idata).set_index("product")["mean"]
    truth = _truth(sim, est.index)
    assert (
        np.abs(est.to_numpy() - truth).mean()
        < np.abs(naive.loc[est.index].to_numpy() - truth).mean()
    )


@pytest.mark.slow
def test_full_recovery_study():
    """Full study: 5 seeds, 4 chains, 500 draws. Run with `pytest -m slow`."""
    covered, errs = [], []
    for seed in range(5):
        sim = simulate(SimConfig(seed=seed))
        a = aggregate(sim.panel)
        idata, _, _ = bayes.fit(a, iv=True, draws=500, tune=500, chains=4, seed=seed)
        e = bayes.per_product(idata).set_index("product")
        truth = _truth(sim, e.index)
        covered += list((e.lo90.to_numpy() <= truth) & (truth <= e.hi90.to_numpy()))
        errs += list(e["mean"].to_numpy() - truth)
    assert np.mean(covered) >= 0.7
    assert abs(np.mean(errs)) < 0.2
