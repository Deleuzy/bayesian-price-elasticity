import numpy as np

from elasticity import classical
from elasticity.features import aggregate
from elasticity.simulate import SimConfig, simulate


def test_naive_is_biased_toward_zero_under_endogeneity():
    sim = simulate(SimConfig(n_products=3, n_stores=6, n_weeks=120, seed=11, price_on_xi=1.0))
    a = aggregate(sim.panel)
    est = classical.naive(a).set_index("product").elasticity
    truth = np.array([sim.beta[p] for p in est.index])
    assert np.all(est.to_numpy() > truth)


def test_iv_recovers_truth_on_large_sample():
    sim = simulate(SimConfig(n_products=3, n_stores=6, n_weeks=200, seed=11, price_on_xi=1.0))
    r = classical.iv(sim.panel, instrument="log_cost").set_index("product")
    truth = np.array([sim.beta[p] for p in r.index])
    assert np.all(np.abs(r.elasticity.to_numpy() - truth) < 0.6)
    assert (r.first_stage_f > 10).all()


def test_iv_reports_first_stage_f(small_sim):
    r = classical.iv(small_sim.panel)
    assert {"first_stage_f", "partial_r2"} <= set(r.columns)
    assert np.isfinite(r.first_stage_f).all()
