"""Anderson-Rubin sets, the unpooled ablation and the training-only store filter."""

import numpy as np
import pandas as pd

from elasticity import bayes, classical
from elasticity.features import aggregate
from elasticity.simulate import SimConfig, simulate


def _weekly(seed=11, weeks=200):
    sim = simulate(SimConfig(n_products=3, n_stores=6, n_weeks=weeks, seed=seed, price_on_xi=1.0))
    return sim, aggregate(sim.panel)


def test_ar_set_contains_2sls_and_is_close_to_wald_when_strong():
    _, a = _weekly()
    ar = classical.anderson_rubin(a).set_index("product")
    r = classical.iv(a).set_index("product")
    assert ar["ar_bounded"].all()
    assert ((ar.ar_lo90 <= r.elasticity) & (r.elasticity <= ar.ar_hi90)).all()
    ratio = (ar.ar_hi90 - ar.ar_lo90) / (r.hi90 - r.lo90)
    assert ((ratio > 0.5) & (ratio < 2.0)).all()


def test_ar_set_is_unbounded_when_instrument_is_noise():
    _, a = _weekly(seed=3, weeks=120)
    a = a.assign(log_cost=np.random.default_rng(0).standard_normal(len(a)))
    ar = classical.anderson_rubin(a)
    assert (~ar["ar_bounded"]).sum() >= 2


def test_unpooled_model_drops_the_hyperprior():
    _, a = _weekly(weeks=60)
    m, _ = bayes.build_model(a, iv=True, pooled=False)
    names = {v.name for v in m.free_RVs}
    assert "z_beta" in names
    assert not {"mu_beta", "sigma_beta"} & names


def test_store_filter_uses_only_weeks_before_the_cut():
    rows = []
    for week in range(1, 9):
        n = 10 if week <= 2 else 7  # stores report less often after week 2
        for store in range(n):
            rows.append(
                {
                    "product": "A",
                    "store": store,
                    "week": week,
                    "units": 5.0,
                    "log_p": 0.0,
                    "log_cost": -0.5,
                    "deal": 0.0,
                    "date": pd.Timestamp("1989-09-14") + pd.Timedelta(days=7 * (week - 1)),
                }
            )
    d = pd.DataFrame(rows)
    assert len(aggregate(d)) == 8  # full-sample median 7: every week kept
    assert sorted(aggregate(d, before_week=3)["week"]) == [1, 2]  # training median 10
