"""End-to-end run: real Dominick's cheese data plus the synthetic recovery study.

    OMP_NUM_THREADS=1 uv run python -m elasticity.pipeline [--quick]

Writes results/*.json, results/*.csv and figures/*.png. The README numbers are
rendered from these files by `elasticity.report`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import bayes, classical, validate  # noqa: E402
from .data import load_panel  # noqa: E402
from .features import aggregate, time_split  # noqa: E402
from .simulate import SimConfig, simulate  # noqa: E402

RESULTS, FIGURES = Path("results"), Path("figures")
# reference categorical palette (dataviz skill), fixed order per method
COL = {
    "naive": "#2a78d6",
    "controls": "#eb6834",
    "iv_2sls": "#1baf7a",
    "hier": "#eda100",
    "hier_iv": "#6250d6",
}
NAMES = {
    "naive": "A. Naive OLS",
    "controls": "OLS with controls",
    "iv_2sls": "2SLS (cost IV)",
    "hier": "B. Hierarchical",
    "hier_iv": "B. Hierarchical + IV",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax):
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def _dump(obj, name):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def _r(x, k=3):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


# --------------------------------------------------------------------------- real data
def real(quick: bool) -> None:
    FIGURES.mkdir(exist_ok=True)
    kw = dict(draws=200, tune=200, chains=2) if quick else dict(draws=1000, tune=1000, chains=4)
    store_panel = load_panel(12)
    weekly = aggregate(store_panel)
    train, test = time_split(weekly, holdout_weeks=52)

    # ---- classical, per product, on chain-level training weeks
    naive = classical.naive(train).set_index("product")
    ctrl = classical.with_controls(train).set_index("product")
    iv = classical.iv(train).set_index("product")

    # ---- Hausman robustness on store-level rows (training weeks only)
    st_train = store_panel[store_panel["week"] < test["week"].min()]
    haus = classical.iv(st_train, instrument="hausman").set_index("product")
    ctrl_store = classical.with_controls(st_train).set_index("product")

    # ---- Bayesian, with prior and posterior predictive checks
    out = {}
    for name, flag in (("hier", False), ("hier_iv", True)):
        t0 = time.time()
        m, meta = bayes.build_model(train, iv=flag)
        idata = bayes.sample(m, seed=1, **kw)
        secs = time.time() - t0
        prior = bayes.prior_predictive(m, draws=500, seed=2)
        rep = bayes.posterior_predictive(m, idata, seed=3)
        hold = bayes.predict(idata, meta, test, n_draws=1000, seed=4)
        out[name] = dict(idata=idata, meta=meta, prior=prior, rep=rep, hold=hold, secs=secs)

    # ---- elasticity table
    hb = bayes.per_product(out["hier"]["idata"]).set_index("product")
    hi = bayes.per_product(out["hier_iv"]["idata"]).set_index("product")
    products = list(hb.index)
    margin = store_panel.assign(m=1 - store_panel["cost"] / store_panel["price"])
    margin = margin.groupby(margin["product"].astype(str))["m"].median()
    tab = pd.DataFrame(index=products)
    tab["naive"], tab["naive_lo"], tab["naive_hi"] = naive.elasticity, naive.lo90, naive.hi90
    tab["controls"], tab["controls_lo"], tab["controls_hi"] = ctrl.elasticity, ctrl.lo90, ctrl.hi90
    tab["iv_2sls"], tab["iv_lo"], tab["iv_hi"] = iv.elasticity, iv.lo90, iv.hi90
    tab["first_stage_f"] = iv.first_stage_f
    tab["hier"], tab["hier_lo"], tab["hier_hi"] = hb["mean"], hb.lo90, hb.hi90
    tab["hier_iv"], tab["hier_iv_lo"], tab["hier_iv_hi"] = hi["mean"], hi.lo90, hi.hi90
    tab["hier_iv_p_elastic"] = hi.p_below_minus1
    tab["hausman_store_2sls"] = haus.elasticity
    tab["hausman_f"] = haus.first_stage_f
    tab["controls_store_ols"] = ctrl_store.elasticity
    tab["dff_median_margin"] = margin.reindex(products)
    tab["lerner_margin_hier_iv"] = -1 / tab["hier_iv"]
    tab.index.name = "product"
    tab.round(4).to_csv(RESULTS / "elasticities.csv")

    # ---- promotion read (hierarchical IV)
    post = out["hier_iv"]["idata"].posterior.stack(sample=("chain", "draw"))
    delta = post["delta"].values  # (product, samples)
    beta = post["beta"].values
    promo = []
    for k, p in enumerate(products):
        g = train[train["product"].astype(str) == p]
        on, off = g[g["deal"] >= 0.5], g[g["deal"] < 0.1]
        depth = 1 - np.exp(on["log_p"].median() - off["log_p"].median()) if len(on) else np.nan
        flag_lift = np.exp(delta[k]) - 1
        total = np.exp(beta[k] * np.log(1 - depth) + delta[k]) - 1
        promo.append(
            {
                "product": p,
                "deal_weeks_share": float((g["deal"] >= 0.5).mean()),
                "median_discount": _r(depth),
                "flag_lift_mean": _r(flag_lift.mean()),
                "flag_lift_lo90": _r(np.quantile(flag_lift, 0.05)),
                "flag_lift_hi90": _r(np.quantile(flag_lift, 0.95)),
                "total_lift_mean": _r(np.nanmean(total)),
                "total_lift_lo90": _r(np.nanquantile(total, 0.05)),
                "total_lift_hi90": _r(np.nanquantile(total, 0.95)),
            }
        )
    pd.DataFrame(promo).to_csv(RESULTS / "promo.csv", index=False)

    # ---- validation on the 52-week holdout
    val = {}
    val["baseline"] = validate.metrics(test, *validate.baseline(train, test))
    val["naive"] = validate.metrics(test, *validate.ols_forecast(train, test, controls=False))
    val["controls"] = validate.metrics(test, *validate.ols_forecast(train, test, controls=True))
    for name in ("hier", "hier_iv"):
        val[name] = validate.metrics(test, *validate.from_draws(out[name]["hold"]))

    # ---- diagnostics, prior and posterior predictive summaries
    y = train["log_q"].to_numpy()
    checks = {}
    for name in ("hier", "hier_iv"):
        o = out[name]
        rep = o["rep"]
        pr = o["prior"].prior_predictive["log_q"].stack(sample=("chain", "draw")).values.T
        pb = o["prior"].prior["beta"].stack(sample=("chain", "draw")).values
        lo, hi = np.quantile(rep, 0.05, axis=0), np.quantile(rep, 0.95, axis=0)
        checks[name] = {
            **bayes.diagnostics(o["idata"]),
            "seconds": round(o["secs"], 1),
            "prior_beta_90": [_r(np.quantile(pb, 0.05)), _r(np.quantile(pb, 0.95))],
            "prior_pred_log_q_90": [_r(np.quantile(pr, 0.05)), _r(np.quantile(pr, 0.95))],
            "observed_log_q_range": [_r(y.min()), _r(y.max())],
            "ppc_p_sd": _r((rep.std(axis=1) > y.std()).mean()),
            "ppc_p_mean": _r((rep.mean(axis=1) > y.mean()).mean()),
            "ppc_p_min": _r((rep.min(axis=1) > y.min()).mean()),
            "ppc_p_max": _r((rep.max(axis=1) > y.max()).mean()),
            "ppc_in_sample_coverage90": _r(((y >= lo) & (y <= hi)).mean()),
            "mu_beta_mean": _r(float(o["idata"].posterior["mu_beta"].mean())),
            "mu_beta_90": [
                _r(float(o["idata"].posterior["mu_beta"].quantile(0.05))),
                _r(float(o["idata"].posterior["mu_beta"].quantile(0.95))),
            ],
            "sigma_beta_mean": _r(float(o["idata"].posterior["sigma_beta"].mean())),
        }
    lam = out["hier_iv"]["idata"].posterior["lambda"]
    checks["hier_iv"]["lambda_90_excludes_0"] = int(
        ((lam.quantile(0.05, ("chain", "draw")) > 0) | (lam.quantile(0.95, ("chain", "draw")) < 0))
        .sum()
        .item()
    )

    strong = tab["first_stage_f"] >= 10
    summary = {
        "data": {
            "category": "cheese (Dominick's 'che')",
            "products": len(products),
            "stores": int(store_panel["store"].nunique()),
            "store_week_rows": int(len(store_panel)),
            "product_week_rows": int(len(weekly)),
            "train_weeks": int(train["week"].nunique()),
            "test_weeks": int(test["week"].nunique()),
            "first_date": str(weekly["date"].min().date()),
            "split_date": str(test["date"].min().date()),
            "last_date": str(weekly["date"].max().date()),
            "corr_log_p_log_cost": _r(np.corrcoef(train["log_p"], train["log_cost"])[0, 1]),
        },
        "means": {
            k: _r(tab[k].mean())
            for k in ("naive", "controls", "iv_2sls", "hier", "hier_iv", "hausman_store_2sls")
        },
        "means_strong_iv": {
            "n": int(strong.sum()),
            **{
                k: _r(tab.loc[strong, k].mean()) for k in ("controls", "iv_2sls", "hier", "hier_iv")
            },
        },
        "weak_iv_products": int((tab["first_stage_f"] < 10).sum()),
        "median_first_stage_f": _r(tab["first_stage_f"].median(), 1),
        "median_hausman_f": _r(tab["hausman_f"].median(), 0),
        "elastic_products_p95": int((tab["hier_iv_p_elastic"] > 0.95).sum()),
        "validation": val,
        "checks": checks,
        "settings": kw,
    }
    _dump(summary, "real.json")
    _figures_real(tab, out, train, y)


def _figures_real(tab, out, train, y):
    # forest plot: elasticity per product by method
    methods = ["naive", "controls", "iv_2sls", "hier", "hier_iv"]
    lohi = {
        "naive": ("naive_lo", "naive_hi"),
        "controls": ("controls_lo", "controls_hi"),
        "iv_2sls": ("iv_lo", "iv_hi"),
        "hier": ("hier_lo", "hier_hi"),
        "hier_iv": ("hier_iv_lo", "hier_iv_hi"),
    }
    n = len(tab)
    fig, ax = plt.subplots(figsize=(8, 0.62 * n + 1.2))
    for k, mth in enumerate(methods):
        yy = np.arange(n) + (k - 2) * 0.15
        lo, hi = tab[lohi[mth][0]].clip(-8, 2), tab[lohi[mth][1]].clip(-8, 2)
        ax.hlines(yy, lo, hi, color=COL[mth], lw=2)
        ax.plot(tab[mth].clip(-8, 2), yy, "o", color=COL[mth], ms=4.5, label=NAMES[mth])
    ax.axvline(-1, color=MUTED, lw=1, ls="--")
    ax.set_yticks(
        np.arange(n),
        [f"{p}  (F={f:.1f})" for p, f in zip(tab.index, tab.first_stage_f, strict=True)],
    )
    ax.invert_yaxis()
    ax.set_xlim(-8, 2)
    ax.set_xlabel(
        "own-price elasticity, 90% interval (clipped to [-8, 2])", color=MUTED, fontsize=8
    )
    ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0))
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "elasticities.png", dpi=150)
    plt.close(fig)

    # shrinkage: no pooling (OLS with controls) vs partial pooling (hierarchical)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    mu = out["hier"]["idata"].posterior["mu_beta"].mean().item()
    for p in tab.index:
        ax.annotate(
            "",
            xy=(1, tab.loc[p, "hier"]),
            xytext=(0, tab.loc[p, "controls"]),
            arrowprops=dict(arrowstyle="->", color=COL["hier"], lw=1.2),
        )
    ax.plot(np.zeros(len(tab)), tab["controls"], "o", color=COL["controls"], ms=6)
    ax.plot(np.ones(len(tab)), tab["hier"], "o", color=COL["hier"], ms=6)
    ax.axhline(mu, color=MUTED, lw=1, ls="--")
    ax.text(1.05, mu, f"category mean {mu:.2f}", color=MUTED, fontsize=8, va="center")
    ax.set_xticks([0, 1], ["no pooling\n(OLS per product)", "partial pooling\n(hierarchical B)"])
    ax.set_xlim(-0.3, 1.6)
    ax.set_ylabel("elasticity", color=MUTED, fontsize=8)
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "shrinkage.png", dpi=150)
    plt.close(fig)

    # prior and posterior predictive checks (hierarchical IV)
    o = out["hier_iv"]
    pr = o["prior"].prior_predictive["log_q"].stack(sample=("chain", "draw")).values.T
    rep = o["rep"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    bins = np.linspace(-2, 10, 80)
    for i in range(40):
        axes[0].hist(
            np.clip(pr[i], -2, 10),
            bins=bins,
            histtype="step",
            color=COL["hier_iv"],
            alpha=0.25,
            lw=0.8,
        )
    axes[0].hist(y, bins=bins, histtype="step", color=INK, lw=1.6)
    axes[0].set_title("prior predictive vs observed log units", fontsize=9, color=INK)
    bins = np.linspace(y.min() - 0.5, y.max() + 0.5, 60)
    for i in range(40):
        axes[1].hist(rep[i], bins=bins, histtype="step", color=COL["hier_iv"], alpha=0.25, lw=0.8)
    axes[1].hist(y, bins=bins, histtype="step", color=INK, lw=1.6)
    axes[1].set_title("posterior predictive vs observed log units", fontsize=9, color=INK)
    for ax in axes:
        _style(ax)
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGURES / "predictive_checks.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- synthetic
def synthetic(quick: bool, seeds: int) -> None:
    """Recovery of known elasticities: one detailed run plus a seed study."""
    kw = dict(draws=300, tune=300, chains=2) if quick else dict(draws=500, tune=500, chains=4)
    rows = []
    for seed in range(seeds):
        sim = simulate(SimConfig(seed=seed))
        a = aggregate(sim.panel)
        truth = pd.Series(sim.beta)
        est = {
            "naive": classical.naive(a).set_index("product"),
            "controls": classical.with_controls(a).set_index("product"),
            "iv_2sls": classical.iv(a).set_index("product"),
        }
        for k, e in est.items():
            for p in e.index:
                rows.append(
                    _simrow(
                        seed,
                        k,
                        p,
                        truth[p],
                        e.loc[p, "elasticity"],
                        e.loc[p, "lo90"],
                        e.loc[p, "hi90"],
                    )
                )
        hstore = sim.panel
        h = classical.iv(hstore, instrument="hausman").set_index("product")
        for p in h.index:
            rows.append(
                _simrow(
                    seed,
                    "hausman_store_2sls",
                    p,
                    truth[p],
                    h.loc[p, "elasticity"],
                    h.loc[p, "lo90"],
                    h.loc[p, "hi90"],
                )
            )
        for name, flag in (("hier", False), ("hier_iv", True)):
            idata, _, _ = bayes.fit(a, iv=flag, seed=seed, **kw)
            e = bayes.per_product(idata).set_index("product")
            for p in e.index:
                rows.append(
                    _simrow(
                        seed,
                        name,
                        p,
                        truth[p],
                        e.loc[p, "mean"],
                        e.loc[p, "lo90"],
                        e.loc[p, "hi90"],
                    )
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "synthetic_runs.csv", index=False)
    s = df.groupby("method").agg(
        bias=("error", "mean"),
        mae=("abs_error", "mean"),
        coverage90=("covered", "mean"),
        width90=("width", "mean"),
        n=("error", "size"),
    )
    cfg = SimConfig()
    _dump(
        {
            "seeds": seeds,
            "products_per_seed": cfg.n_products,
            "stores": cfg.n_stores,
            "weeks": cfg.n_weeks,
            "true_mean_elasticity": cfg.mu_beta,
            "price_on_demand_shock": cfg.price_on_xi,
            "settings": kw,
            "methods": {m: {k: _r(v) for k, v in r.items()} for m, r in s.iterrows()},
        },
        "synthetic.json",
    )


def _simrow(seed, method, product, true, est, lo, hi):
    return {
        "seed": seed,
        "method": method,
        "product": product,
        "true": float(true),
        "estimate": float(est),
        "error": float(est - true),
        "abs_error": float(abs(est - true)),
        "covered": bool(lo <= true <= hi),
        "width": float(hi - lo),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few draws, for a smoke run")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--only", choices=["real", "synthetic"])
    a = ap.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    if a.only != "synthetic":
        real(a.quick)
    if a.only != "real":
        synthetic(a.quick, a.seeds)


if __name__ == "__main__":
    main()
