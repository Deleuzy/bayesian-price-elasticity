"""Synthetic scanner panel with known elasticities and price endogeneity.

The retailer sees a product-week demand shock `xi` that the analyst does not see,
and raises the price when demand is high. Price therefore correlates with the
error term of the demand equation, and OLS of log units on log price is biased
towards zero. Unit cost moves the price but does not enter demand, so it is a
valid instrument. Because `xi` is common to all stores in a week (as DFF's
chainwide promotions are), the price of the same product in other stores
(a Hausman instrument) carries `xi` too and is NOT valid here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import add_time_features


@dataclass
class SimConfig:
    n_products: int = 6
    n_stores: int = 6
    n_weeks: int = 100
    mu_beta: float = -2.0
    sd_beta: float = 0.4
    deal_effect: float = 0.3
    deal_rate: float = 0.15
    deal_depth: float = 0.2
    xi_sd: float = 0.2
    price_on_xi: float = 0.5
    cost_rho: float = 0.9
    cost_sd: float = 0.05
    store_price_sd: float = 0.03
    store_sd: float = 0.3
    eps_sd: float = 0.3
    season_amp: float = 0.1
    seed: int = 0


@dataclass
class Simulated:
    panel: pd.DataFrame
    beta: dict[str, float]
    config: SimConfig = field(default_factory=SimConfig)


def simulate(cfg: SimConfig | None = None) -> Simulated:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    J, S, T = cfg.n_products, cfg.n_stores, cfg.n_weeks
    names = [f"P{j + 1}" for j in range(J)]
    beta = cfg.mu_beta + cfg.sd_beta * rng.standard_normal(J)
    alpha = 3.0 + 0.5 * rng.standard_normal(J)
    store = cfg.store_sd * rng.standard_normal(S)
    base_cost = np.log(rng.uniform(1.0, 3.0, J))
    markup = np.log(rng.uniform(1.2, 1.6, J))

    # product x week: AR(1) cost, demand shock, chainwide deal
    cost = np.zeros((J, T))
    innov = cfg.cost_sd * rng.standard_normal((J, T))
    cost[:, 0] = innov[:, 0] / np.sqrt(1 - cfg.cost_rho**2)
    for t in range(1, T):
        cost[:, t] = cfg.cost_rho * cost[:, t - 1] + innov[:, t]
    log_cost = base_cost[:, None] + cost
    xi = cfg.xi_sd * rng.standard_normal((J, T))
    deal = (rng.uniform(size=(J, T)) < cfg.deal_rate).astype(float)

    j, s, t = np.meshgrid(np.arange(J), np.arange(S), np.arange(T), indexing="ij")
    j, s, t = j.ravel(), s.ravel(), t.ravel()
    week = t + 1
    date = pd.Timestamp("1989-09-14") + pd.to_timedelta(t * 7, unit="D")
    doy = date.day_of_year.to_numpy()
    season = cfg.season_amp * np.sin(2 * np.pi * doy / 365.25)

    log_p = (
        markup[j]
        + log_cost[j, t]
        + cfg.price_on_xi * xi[j, t]
        - cfg.deal_depth * deal[j, t]
        + cfg.store_price_sd * rng.standard_normal(j.size)
    )
    log_q = (
        alpha[j]
        + store[s]
        + beta[j] * (log_p - markup[j] - base_cost[j])
        + cfg.deal_effect * deal[j, t]
        + season
        + xi[j, t]
        + cfg.eps_sd * rng.standard_normal(j.size)
    )
    d = pd.DataFrame(
        {
            "product": pd.Categorical([names[k] for k in j], categories=names),
            "store": s + 1,
            "week": week,
            "date": date,
            "log_p": log_p,
            "log_q": log_q,
            "log_cost": log_cost[j, t],
            "deal": deal[j, t],
            "xi": xi[j, t],  # demand shock: diagnostic only, never given to a model
        }
    )
    d["price"] = np.exp(d.log_p)
    d["units"] = np.exp(d.log_q)
    d["cost"] = np.exp(d.log_cost)
    g = d.groupby(["product", "week"], observed=True).log_p
    d["hausman"] = (g.transform("sum") - d.log_p) / (g.transform("count") - 1)
    d = add_time_features(d)
    return Simulated(panel=d, beta=dict(zip(names, beta, strict=True)), config=cfg)
