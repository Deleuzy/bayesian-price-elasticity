import numpy as np

from elasticity.simulate import SimConfig, simulate


def test_shapes_and_columns():
    cfg = SimConfig(n_products=2, n_stores=3, n_weeks=20)
    sim = simulate(cfg)
    assert len(sim.panel) == 2 * 3 * 20
    assert set(sim.beta) == {"P1", "P2"}
    for col in ["log_p", "log_q", "log_cost", "hausman", "deal", "quarter"]:
        assert col in sim.panel


def test_seed_is_reproducible():
    a = simulate(SimConfig(n_products=2, n_stores=2, n_weeks=10, seed=7)).panel
    b = simulate(SimConfig(n_products=2, n_stores=2, n_weeks=10, seed=7)).panel
    np.testing.assert_allclose(a.log_q.to_numpy(), b.log_q.to_numpy())


def test_price_is_endogenous_to_demand_shock():
    """Price moves with the demand shock xi, which the analyst never sees."""
    d = simulate(SimConfig(seed=1)).panel
    g = d.groupby(["product", "store"], observed=True)
    corr = np.corrcoef(d.log_p - g.log_p.transform("mean"), d.xi - g.xi.transform("mean"))[0, 1]
    assert corr > 0.3


def test_cost_is_a_valid_instrument():
    """Cost shifts price but has no direct effect on demand."""
    d = simulate(SimConfig(seed=2)).panel
    p_resid = d.log_p - d.groupby("product", observed=True).log_p.transform("mean")
    c_resid = d.log_cost - d.groupby("product", observed=True).log_cost.transform("mean")
    assert np.corrcoef(p_resid, c_resid)[0, 1] > 0.5
