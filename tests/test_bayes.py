"""Fast Bayesian tests: tiny models, few draws, so they run in CI."""

import numpy as np
import pytest

from elasticity import bayes
from elasticity.features import aggregate, time_split
from elasticity.simulate import SimConfig, simulate


@pytest.fixture(scope="module")
def fitted_small():
    sim = simulate(SimConfig(n_products=2, n_stores=3, n_weeks=60, seed=5))
    a = aggregate(sim.panel)
    train, test = time_split(a, holdout_weeks=12)
    idata, meta, m = bayes.fit(
        train, iv=True, draws=150, tune=150, chains=2, cores=1, seed=0, sampler="pymc"
    )
    return sim, train, test, idata, meta, m


def test_posterior_has_expected_variables(fitted_small):
    _, _, _, idata, _, _ = fitted_small
    for v in ["beta", "delta", "level", "lambda", "sigma", "mu_beta"]:
        assert v in idata.posterior


def test_elasticities_are_negative(fitted_small):
    _, _, _, idata, _, _ = fitted_small
    e = bayes.per_product(idata)
    assert (e["mean"] < 0).all()
    assert (e["lo90"] <= e["mean"]).all() and (e["mean"] <= e["hi90"]).all()


def test_diagnostics_structure(fitted_small):
    _, _, _, idata, _, _ = fitted_small
    d = bayes.diagnostics(idata)
    assert d["chains"] == 2 and d["draws"] == 150
    assert d["max_rhat"] > 0 and d["min_ess_bulk"] > 0


def test_predict_shape_and_finite(fitted_small):
    _, _, test, idata, meta, _ = fitted_small
    draws = bayes.predict(idata, meta, test, n_draws=50, seed=1)
    assert draws.shape == (50, len(test))
    assert np.isfinite(draws).all()


def test_non_centred_parameters_present(fitted_small):
    _, _, _, _, _, m = fitted_small
    names = {v.name for v in m.free_RVs}
    assert {"z_beta", "sigma_beta", "mu_beta"} <= names
