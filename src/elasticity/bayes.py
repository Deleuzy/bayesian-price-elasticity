"""Bayesian hierarchical demand model in PyMC (Model B), optionally with a
jointly estimated first stage that makes it a Bayesian instrumental variable
model (control function form).

Data are product x week rows (chain level, see `features.aggregate`). For
product j in week t:

    log_q = a_j + beta_j * (log_p - pbar_j) + delta_j * deal + level_j[block]
            + season + lambda_j * v + e,           e ~ Student-t(nu, 0, sigma_j)

level_j is a Gaussian random walk over 13-week blocks (slow drift in the
product's base demand; a linear trend extrapolated badly out of sample).

beta_j = mu_beta + sigma_beta * z_beta_j: product elasticities partially pooled
toward a category mean (non-centred). delta_j (promotion effect at a given
price) is pooled the same way.

With `iv=True` the first stage is part of the likelihood:

    log_p - pbar_j = pi0_j + pi_j * (log_cost - cbar_j) + controls + v,   v ~ Normal(0, sigma_v_j)

and the demand equation gets lambda_j * v. The joint density of price and
quantity factorises as p(price) * p(quantity | price), and with a linear first
stage the endogenous part of the demand error is lambda_j * v. So beta_j is
identified only by cost driven price movements, and the uncertainty of the
first stage flows into beta_j because both are sampled together. Without `iv`
price is treated as exogenous (lambda_j = 0), the usual implicit assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

SEASON = ["s1", "c1", "s2", "c2"]


@dataclass
class Meta:
    products: list[str]
    pbar: np.ndarray
    cbar: np.ndarray
    q0: int
    n_blocks: int
    iv: bool


def make_meta(d: pd.DataFrame, iv: bool) -> Meta:
    present = set(d["product"].astype(str))
    if hasattr(d["product"], "cat"):
        products = [str(p) for p in d["product"].cat.categories if str(p) in present]
    else:
        products = sorted(present)
    g = d.groupby(d["product"].astype(str))
    return Meta(
        products=products,
        pbar=g["log_p"].mean().reindex(products).to_numpy(),
        cbar=g["log_cost"].mean().reindex(products).to_numpy(),
        q0=int(d["quarter"].min()),
        n_blocks=int(d["quarter"].max() - d["quarter"].min() + 1),
        iv=iv,
    )


def _design(d: pd.DataFrame, meta: Meta) -> dict[str, np.ndarray]:
    j = pd.Categorical(d["product"].astype(str), categories=meta.products).codes
    if (j < 0).any():
        raise ValueError("product not seen in training")
    j = j.astype(int)
    return {
        "j": j,
        "x": d["log_p"].to_numpy() - meta.pbar[j],
        "z": d["log_cost"].to_numpy() - meta.cbar[j],
        "deal": d["deal"].to_numpy(dtype=float),
        "q": d["quarter"].to_numpy().astype(int) - meta.q0,
        "season": d[SEASON].to_numpy(dtype=float),
        "y": d["log_q"].to_numpy(),
    }


def build_model(d: pd.DataFrame, iv: bool = False) -> tuple[pm.Model, Meta]:
    meta = make_meta(d, iv)
    X = _design(d, meta)
    j = X["j"]
    ybar = d.groupby(d["product"].astype(str))["log_q"].mean().reindex(meta.products).to_numpy()
    coords = {
        "product": meta.products,
        "season": SEASON,
        "block": np.arange(meta.n_blocks),
        "step": np.arange(meta.n_blocks - 1),
    }
    q = X["q"]
    with pm.Model(coords=coords) as m:
        # elasticities: partial pooling toward a category mean, non-centred
        mu_beta = pm.Normal("mu_beta", -1.5, 1.0)
        sigma_beta = pm.HalfNormal("sigma_beta", 0.5)
        z_beta = pm.Normal("z_beta", 0, 1, dims="product")
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * z_beta, dims="product")
        # promotion (deal) effect at a given price, pooled the same way
        mu_delta = pm.Normal("mu_delta", 0, 0.5)
        sigma_delta = pm.HalfNormal("sigma_delta", 0.3)
        z_delta = pm.Normal("z_delta", 0, 1, dims="product")
        delta = pm.Deterministic("delta", mu_delta + sigma_delta * z_delta, dims="product")
        a = pm.Normal("a", ybar, 1.0, dims="product")
        sigma_rw = pm.HalfNormal("sigma_rw", 0.1, dims="product")
        z_rw = pm.Normal("z_rw", 0, 1, dims=("product", "step"))
        steps = sigma_rw[:, None] * z_rw
        level = pm.Deterministic(
            "level",
            pm.math.concatenate(
                [pm.math.zeros((len(meta.products), 1)), steps.cumsum(axis=1)], axis=1
            ),
            dims=("product", "block"),
        )
        gamma = pm.Normal("gamma", 0, 0.3, dims="season")
        sigma = pm.HalfNormal("sigma", 0.5, dims="product")
        nu = pm.Gamma("nu", 2.0, 0.1)

        mu = (
            a[j]
            + beta[j] * X["x"]
            + delta[j] * X["deal"]
            + level[j, q]
            + pm.math.dot(X["season"], gamma)
        )
        if iv:
            pi0 = pm.Normal("pi0", 0, 0.5, dims="product")
            mu_pi = pm.Normal("mu_pi", 0.5, 0.5)
            sigma_pi = pm.HalfNormal("sigma_pi", 0.5)
            z_pi = pm.Normal("z_pi", 0, 1, dims="product")
            pi = pm.Deterministic("pi", mu_pi + sigma_pi * z_pi, dims="product")
            pi_deal = pm.Normal("pi_deal", 0, 0.5, dims="product")
            sigma_prw = pm.HalfNormal("sigma_prw", 0.05, dims="product")
            z_prw = pm.Normal("z_prw", 0, 1, dims=("product", "step"))
            psteps = sigma_prw[:, None] * z_prw
            plevel = pm.Deterministic(
                "plevel",
                pm.math.concatenate(
                    [pm.math.zeros((len(meta.products), 1)), psteps.cumsum(axis=1)], axis=1
                ),
                dims=("product", "block"),
            )
            pi_season = pm.Normal("pi_season", 0, 0.3, dims="season")
            sigma_v = pm.HalfNormal("sigma_v", 0.3, dims="product")
            mu_p = (
                pi0[j]
                + pi[j] * X["z"]
                + pi_deal[j] * X["deal"]
                + plevel[j, q]
                + pm.math.dot(X["season"], pi_season)
            )
            pm.Normal("price", mu_p, sigma_v[j], observed=X["x"])
            lam = pm.Normal("lambda", 0, 2.0, dims="product")
            mu = mu + lam[j] * (X["x"] - mu_p)
        pm.StudentT("log_q", nu=nu, mu=mu, sigma=sigma[j], observed=X["y"])
    return m, meta


def sample(
    m: pm.Model,
    draws: int = 500,
    tune: int = 500,
    chains: int = 4,
    seed: int = 0,
    target_accept: float = 0.9,
    sampler: str = "nutpie",
    cores: int = 1,
):
    """NUTS via nutpie (default, faster on Apple silicon) or PyMC's own sampler."""
    kw = (
        {"cores": cores}
        if sampler == "nutpie"
        else {"compute_convergence_checks": False, "cores": cores}
    )
    with m:
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=seed,
            target_accept=target_accept,
            nuts_sampler=sampler,
            progressbar=False,
            **kw,
        )


def fit(d: pd.DataFrame, iv: bool = False, **kw) -> tuple[az.InferenceData, Meta, pm.Model]:
    m, meta = build_model(d, iv=iv)
    return sample(m, **kw), meta, m


def prior_predictive(m: pm.Model, draws: int = 500, seed: int = 0) -> az.InferenceData:
    with m:
        return pm.sample_prior_predictive(draws=draws, random_seed=seed)


def posterior_predictive(m: pm.Model, idata: az.InferenceData, seed: int = 0) -> np.ndarray:
    """Replicated log units for the training rows, shape (draws, rows)."""
    with m:
        pp = pm.sample_posterior_predictive(
            idata, var_names=["log_q"], random_seed=seed, progressbar=False
        )
    return pp.posterior_predictive["log_q"].stack(sample=("chain", "draw")).to_numpy().T


def diagnostics(idata: az.InferenceData) -> dict:
    """R-hat and ESS over all sampled (free) parameters, plus divergences."""
    derived = ("beta", "delta", "pi", "level", "plevel")
    free = [v for v in idata.posterior.data_vars if v not in derived]
    summ = az.summary(idata, var_names=free, kind="diagnostics")
    return {
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "min_ess_tail": float(summ["ess_tail"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum()),
        "chains": int(idata.posterior.sizes["chain"]),
        "draws": int(idata.posterior.sizes["draw"]),
        "parameters": int(len(summ)),
    }


def per_product(idata: az.InferenceData, var: str = "beta", mass: float = 0.9) -> pd.DataFrame:
    flat = idata.posterior[var].stack(sample=("chain", "draw"))
    lo, hi = (1 - mass) / 2, 1 - (1 - mass) / 2
    return pd.DataFrame(
        {
            "product": [str(p) for p in flat["product"].values],
            "mean": flat.mean("sample").values,
            "sd": flat.std("sample").values,
            "lo90": flat.quantile(lo, "sample").values,
            "hi90": flat.quantile(hi, "sample").values,
            "p_below_minus1": (flat < -1).mean("sample").values,
        }
    )


def _extend(level: np.ndarray, sigma: np.ndarray, n_blocks: int, rng) -> np.ndarray:
    """Continue the random walk (draws, product, block) into future blocks."""
    D, J, B = level.shape
    if n_blocks <= B:
        return level
    steps = sigma[:, :, None] * rng.standard_normal((D, J, n_blocks - B))
    future = level[:, :, -1:] + steps.cumsum(axis=2)
    return np.concatenate([level, future], axis=2)


def predict(
    idata: az.InferenceData,
    meta: Meta,
    d: pd.DataFrame,
    n_draws: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Posterior predictive draws of log units for new rows, shape (draws, rows).

    Future 13-week blocks continue each product's random walk, so the forecast
    interval widens with the horizon. Price and cost of the new rows are taken as
    known: a conditional forecast, which is the pricing use case (what happens at
    the planned price). In the IV
    model the first-stage residual of the new row enters through lambda_j, so a
    price that is high because demand is high still forecasts well.
    """
    rng = np.random.default_rng(seed)
    X = _design(d, meta)
    post = idata.posterior.stack(sample=("chain", "draw"))
    n = post.sizes["sample"]
    idx = np.sort(rng.choice(n, size=min(n_draws, n), replace=False))

    def g(name):
        return np.moveaxis(post[name].values[..., idx], -1, 0)  # (draws, ...)

    j, q = X["j"], X["q"]
    level = _extend(g("level"), g("sigma_rw"), q.max() + 1, rng)
    mu = (
        g("a")[:, j]
        + g("beta")[:, j] * X["x"]
        + g("delta")[:, j] * X["deal"]
        + level[:, j, q]
        + g("gamma") @ X["season"].T
    )
    if meta.iv:
        mu_p = (
            g("pi0")[:, j]
            + g("pi")[:, j] * X["z"]
            + g("pi_deal")[:, j] * X["deal"]
            + _extend(g("plevel"), g("sigma_prw"), q.max() + 1, rng)[:, j, q]
            + g("pi_season") @ X["season"].T
        )
        mu = mu + g("lambda")[:, j] * (X["x"] - mu_p)
    sigma = g("sigma")[:, j]
    nu = g("nu")[:, None]
    return mu + sigma * rng.standard_t(nu, size=mu.shape)
