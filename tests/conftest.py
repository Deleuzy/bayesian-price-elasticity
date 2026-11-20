import pytest

from elasticity.features import aggregate
from elasticity.simulate import SimConfig, simulate


@pytest.fixture(scope="session")
def small_sim():
    """Small synthetic panel with known elasticities, used by fast tests."""
    cfg = SimConfig(n_products=3, n_stores=4, n_weeks=80, seed=3)
    return simulate(cfg)


@pytest.fixture(scope="session")
def small_weekly(small_sim):
    return aggregate(small_sim.panel)
