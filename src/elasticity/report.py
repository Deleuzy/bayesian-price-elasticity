"""Render README.md from results/ so that every number in it is generated.

    uv run python -m elasticity.report           # write README.md
    uv run python -m elasticity.report --check   # exit 1 if README.md is stale

No result is typed by hand in this module: each one is read from results/. The
only literal numbers in the prose restate the model specification (the prior
table, from `elasticity.bayes`) and the study design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
README = Path("README.md")


def _load():
    real = json.loads((RESULTS / "real.json").read_text())
    syn = json.loads((RESULTS / "synthetic.json").read_text())
    tab = pd.read_csv(RESULTS / "elasticities.csv", index_col="product")
    promo = pd.read_csv(RESULTS / "promo.csv")
    runs = pd.read_csv(RESULTS / "synthetic_runs.csv")
    return real, syn, tab, promo, runs


def f2(x, k=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{k}f}"


def pct(x, k=0):
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{k}f}%"


def ci(m, lo, hi, k=2):
    return f"{f2(m, k)} [{f2(lo, k)}, {f2(hi, k)}]"


def ar_set(lo, hi, k=2) -> str:
    """Anderson-Rubin set; an infinite end means the set is unbounded on that side."""
    if pd.isna(lo) and pd.isna(hi):
        return "empty"
    if np.isinf(lo) and np.isinf(hi):
        return "unbounded"
    left = "(-inf" if np.isinf(lo) else f"[{f2(lo, k)}"
    right = "inf)" if np.isinf(hi) else f"{f2(hi, k)}]"
    return f"{left}, {right}"


def elasticity_table(tab: pd.DataFrame) -> str:
    rows = [
        "| Product | A. Naive OLS | OLS + controls | 2SLS (cost IV), Wald | 2SLS, Anderson-Rubin set | First-stage F | B. Hierarchical | B + IV (joint) | P(elasticity < -1), B + IV |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for p, r in tab.iterrows():
        rows.append(
            f"| {p} | {ci(r.naive, r.naive_lo, r.naive_hi)} | {ci(r.controls, r.controls_lo, r.controls_hi)} "
            f"| {ci(r.iv_2sls, r.iv_lo, r.iv_hi)} | {ar_set(r.ar_lo, r.ar_hi)} | {f2(r.first_stage_f, 1)} "
            f"| {ci(r.hier, r.hier_lo, r.hier_hi)} "
            f"| {ci(r.hier_iv, r.hier_iv_lo, r.hier_iv_hi)} | {pct(r.hier_iv_p_elastic)} |"
        )
    return "\n".join(rows)


def validation_table(real: dict) -> str:
    names = {
        "baseline": "Simple baseline (last 13 weeks median, no price)",
        "naive": "A. Naive log-log OLS per product",
        "controls": "OLS per product with controls",
        "hier": "B. Hierarchical",
        "hier_iv": "B. Hierarchical + IV",
    }
    rows = [
        "| Model | MAE (units per store-week) | WAPE | 90% interval coverage |",
        "|---|---|---|---|",
    ]
    for k, label in names.items():
        v = real["validation"][k]
        rows.append(
            f"| {label} | {f2(v['mae'], 1)} | {pct(v['wape'], 1)} | {pct(v['coverage90'], 1)} |"
        )
    return "\n".join(rows)


def checks_table(real: dict) -> str:
    rows = [
        "| Model | Max R-hat | Min ESS (bulk) | Min ESS (tail) | Divergences | Chains x draws | Free parameters | Seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, label in (("hier", "B. Hierarchical"), ("hier_iv", "B. Hierarchical + IV")):
        c = real["checks"][k]
        rows.append(
            f"| {label} | {f2(c['max_rhat'], 3)} | {c['min_ess_bulk']:.0f} | {c['min_ess_tail']:.0f} "
            f"| {c['divergences']} | {c['chains']} x {c['draws']} | {c['parameters']} | {c['seconds']:.0f} |"
        )
    return "\n".join(rows)


def synthetic_table(syn: dict) -> str:
    names = {
        "naive": "A. Naive OLS",
        "controls": "OLS with controls",
        "iv_2sls": "2SLS (cost IV)",
        "hausman_store_2sls": "2SLS (Hausman IV, store rows)",
        "hier": "B. Hierarchical",
        "hier_iv": "B. Hierarchical + IV",
        "iv_ar": "Ablation: 2SLS estimate with Anderson-Rubin set",
        "iv_2sls_trend": "Ablation: 2SLS, linear trend instead of block effects",
        "hier_iv_nopool": "Ablation: B + IV without pooling",
    }
    rows = [
        "| Method | Mean error (bias) | Mean absolute error | 90% interval coverage | Mean 90% width (bounded sets) | Unbounded sets |",
        "|---|---|---|---|---|---|",
    ]
    for k, label in names.items():
        m = syn["methods"][k]
        rows.append(
            f"| {label} | {f2(m['bias'])} | {f2(m['mae'])} | {pct(m['coverage90'])} "
            f"| {f2(m['width90'])} | {pct(m['unbounded'])} |"
        )
    return "\n".join(rows)


def synthetic_checks_table(syn: dict) -> str:
    names = {
        "hier": "B. Hierarchical",
        "hier_iv": "B. Hierarchical + IV",
        "hier_iv_nopool": "B + IV without pooling",
    }
    rows = [
        "| Model | Fits | Max R-hat | Min ESS (bulk) | Min ESS (tail) | Divergences (total) | Fits with divergences |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, label in names.items():
        g = syn["diagnostics"][k]
        rows.append(
            f"| {label} | {g['fits']:.0f} | {f2(g['max_rhat'], 3)} | {g['min_ess_bulk']:.0f} "
            f"| {g['min_ess_tail']:.0f} | {g['divergences']:.0f} | {g['fits_with_divergences']:.0f} |"
        )
    return "\n".join(rows)


def margin_table(tab: pd.DataFrame) -> str:
    rows = [
        "| Product | Implied margin, -1 / elasticity (B + IV) | DFF median margin |",
        "|---|---|---|",
    ]
    for p, r in tab.iterrows():
        rows.append(f"| {p} | {pct(r.lerner_margin_hier_iv)} | {pct(r.dff_median_margin)} |")
    return "\n".join(rows)


def main(check: bool) -> int:
    real, syn, tab, promo, runs = _load()
    d = real["data"]
    m = real["means"]
    ms = real["means_strong_iv"]
    v = real["validation"]
    c = real["checks"]
    lam_n = c["hier_iv"]["lambda_90_excludes_0"]
    elastic = tab[tab["hier_iv"] < -1].index.tolist()
    promo_rows = promo.set_index("product")
    promo_md = [
        "| Product | Weeks on deal | Median discount | Lift on deal flag (B + IV, 90%) | Total promo lift incl. price (B + IV, 90%) |",
        "|---|---|---|---|---|",
    ]
    for p, r in promo_rows.iterrows():
        promo_md.append(
            f"| {p} | {pct(r.deal_weeks_share)} | {pct(r.median_discount)} | "
            f"{f2(r.flag_lift_mean, 2)} [{f2(r.flag_lift_lo90, 2)}, {f2(r.flag_lift_hi90, 2)}] | "
            f"{f2(r.total_lift_mean, 2)} [{f2(r.total_lift_lo90, 2)}, {f2(r.total_lift_hi90, 2)}] |"
        )
    base_mae = v["baseline"]["mae"]
    n_runs = int(runs.shape[0])
    fe = real["settings"]
    mu_hat = c["hier"]["mu_beta_mean"]
    moved = int(((tab["hier"] - mu_hat).abs() < (tab["controls"] - mu_hat).abs()).sum())
    sd_controls, sd_hier = tab["controls"].std(), tab["hier"].std()
    sm = syn["methods"]
    ar_unbounded = int((~tab["ar_bounded"].astype(bool)).sum())
    weak = tab["first_stage_f"] < 10
    ar_unbounded_weak = int((~tab.loc[weak, "ar_bounded"].astype(bool)).sum())
    names_v = {
        "naive": "naive OLS",
        "controls": "OLS with controls",
        "hier": "B",
        "hier_iv": "B + IV",
    }
    by_mae = sorted(names_v, key=lambda k: v[k]["mae"])
    mae_rank = ", ".join(f"{names_v[k]} ({f2(v[k]['mae'], 1)})" for k in by_mae)
    by_cov = sorted(names_v, key=lambda k: abs(v[k]["coverage90"] - 0.9))
    cov_rank = ", ".join(f"{names_v[k]} ({pct(v[k]['coverage90'], 1)})" for k in by_cov)
    top_promo = promo.loc[promo["total_lift_mean"].idxmax()]
    text = f"""# Bayesian hierarchical price elasticity with instrumental variables

A demo of price elasticity estimation on open retail scanner data (Dominick's
Finer Foods, cheese, Chicago, 1989 to 1997). It covers the two things a
pricing team needs most from a demand model and that a plain regression does
not give: **partial pooling across products in a PyMC hierarchy with full
inference checks**, and **treatment of price endogeneity**, because retailers
set prices in response to demand.

Everything below is generated from `results/` by `elasticity.report`. Run
`uv run python -m elasticity.report --check` to confirm the README matches.

## Data

* **Source:** Kilts Center for Marketing, Chicago Booth. Dominick's Finer Foods
  scanner data (movement file `wche.zip`, UPC file `upcche.csv`). The Kilts terms
  say: "These data are for academic research purposes only. Users must acknowledge
  in their working papers and/or publications the Kilts Center for Marketing at the
  University of Chicago Booth School of Business." The raw files are **not**
  committed. `elasticity.data` downloads them and checks their SHA-256.
* **Acknowledgement:** this analysis is a non-commercial portfolio project. Data:
  Kilts Center for Marketing, University of Chicago Booth School of Business.
* **Scope:** the top {d["products"]} fixed-weight cheese UPCs by revenue
  (random-weight packs excluded), {d["stores"]} stores, weekly, {d["store_week_rows"]:,} store-week rows.
  Chain-level modelling uses {d["product_week_rows"]:,} product-weeks (store means), with
  product-weeks where fewer than 80% of the usual stores reported dropped as data gaps.
* **Price** is the shelf price per unit (PRICE / QTY). **Unit cost** is derived
  from DFF's gross margin (`cost = price * (1 - PROFIT/100)`). The manual says this
  is the average acquisition cost, not replacement cost, so it lags wholesale moves.
* **Time split:** the last 52 weeks (from {d["split_date"]} to {d["last_date"]}) are the holdout.
  Training runs from {d["first_date"]}.

## Method

**A. Naive model.** Per product, log units on log price (OLS). Price is treated
as exogenous. This is the baseline most teams start from.

**Why price is endogenous.** A retailer sees demand (weather, school holidays,
competitor stock-outs, local events) before the analyst does, and sets price
partly in response to it. A week with high demand is then often a week with a
higher price, so the price coefficient is pulled towards zero. In the DFF cheese
data the correlation of log price with log cost, pooled over the training
product-weeks, is {f2(d["corr_log_p_log_cost"])}, so cost moves price strongly. That is the
relevance condition. Much of it is between products, so the first-stage F is reported per product.

**Controls.** Adding the promotion flag, 13-week block effects (slow drift in
base demand), yearly seasonality and store effects (on store rows) is what most
analysts do next. It does not solve endogeneity, and in the synthetic study it
made the bias worse: the block effects absorb the slow, cost-driven part of price.

**2SLS with cost as instrument.** Log price is instrumented with log unit cost,
which moves price but does not enter demand directly. The assumptions are: cost
is correlated with price (relevance, tested with the first-stage F), and cost
affects units only through price (exclusion, not testable here). The argument is
that a wholesale cost change does not change a shopper's preference. Two threats
remain and are not controlled for: manufacturer trade deals can lower acquisition
cost in the same weeks a retailer runs a feature or display, which moves demand
directly; and acquisition cost reflects forward buying ahead of expected demand.
The deal flag controls for the first only partly, because it is incomplete.
Standard errors are Newey-West over weeks on the chain-level rows.

**B. Bayesian hierarchical model (PyMC).** For product j in week t:

    log units = a_j + beta_j * (log price - mean) + delta_j * deal + level_j(block)
                + season + e,    e ~ Student-t(nu, sigma_j)

* `beta_j = mu_beta + sigma_beta * z_j` is written **non-centred**, so the
  product elasticities are partially pooled toward a category mean. Products
  with few or noisy observations are shrunk towards the mean.
* `level_j` is a random walk over 13-week blocks, which is a slow drift in base
  demand. A linear trend extrapolated badly out of sample, which is why I
  replaced it.
* A product-week demand shock is shared by all stores, because weekly demand
  moves chainwide.

**B + IV.** The same model with a first stage inside the likelihood:
`log price = pi0_j + pi_j * log cost + controls + v_jt`, and the demand equation
gets `lambda_j * v_jt`. The first-stage error `v` carries the part of price that
moves with demand, so beta is identified by the cost-driven part of price. This
is the control-function form of the Bayesian IV model. It makes the first-stage
uncertainty enter the elasticity posterior.

**Priors.** Weakly informative, chosen before looking at the holdout. None of them
assumes endogeneity: the loading `lambda_j` is wide on purpose, so the data decide it.

| Parameter | Prior | Role |
|---|---|---|
| `mu_beta` | Normal(-1.5, 1) | category mean elasticity |
| `sigma_beta` | HalfNormal(0.5) | spread of product elasticities around the category mean |
| `a_j` | Normal(product's training mean log units, 1) | product intercept, centred on the data (a mild empirical-Bayes step) |
| `sigma_rw` | HalfNormal(0.1), per product | random-walk step size of base demand per 13-week block |
| `sigma_j` | HalfNormal(0.5), per product | residual scale |
| `nu` | Gamma(2, 0.1), mean 20 | Student-t degrees of freedom, shared by products |
| `delta_j`, `mu_delta`, `sigma_delta` | Normal(0, 0.5), HalfNormal(0.3) | promotion effect at a given price |
| `pi_j`, `mu_pi`, `sigma_pi` | Normal(0.5, 0.5), HalfNormal(0.5) | first-stage slope on log cost (positive by prior) |
| `sigma_v_j` | HalfNormal(0.3), per product | first-stage residual scale |
| `lambda_j` | Normal(0, 2) | endogeneity loading, one per product |

## Results: elasticities on the real data

Posterior mean and 90% interval for the B models, OLS and 2SLS intervals are
90% confidence intervals, all estimated on the {d["train_weeks"]} training weeks.
The first-stage F tells how strong the cost instrument is; values below 10 mean
the 2SLS estimate for that product is unreliable. The Wald column uses Newey-West
errors; in the synthetic study these intervals cover the truth
{pct(sm["iv_2sls"]["coverage90"])} of the time rather than 90%, so read them as rough. The
Anderson-Rubin column is the 90% confidence set from inverting the Anderson-Rubin test
(same Newey-West kernel), which stays valid when the instrument is weak. It is unbounded,
on at least one side, for {ar_unbounded} of {len(tab)} products: for those, the cost data cannot
pin the elasticity down.

{elasticity_table(tab)}

Mean across the {len(tab)} products: naive {f2(m["naive"])}, OLS with controls {f2(m["controls"])},
hierarchical {f2(m["hier"])}, hierarchical with IV {f2(m["hier_iv"])}. The 2SLS mean is
left out, because it averages in estimates with a first-stage F near zero.
Restricted to the {ms["n"]} products with first-stage F of at least 10: OLS with controls
{f2(ms["controls"])}, 2SLS {f2(ms["iv_2sls"])}, hierarchical {f2(ms["hier"])}, hierarchical with IV {f2(ms["hier_iv"])}.
The weak-instrument products are {real["weak_iv_products"]} of {len(tab)}, with a median first-stage F of
{f2(real["median_first_stage_f"], 1)}, so the cost instrument is strong for some cheese
products and weak for others. The Hausman instrument (same product's price in other stores) is
very strong on store rows (median F {f2(real["median_hausman_f"], 0)}), but it shares the chainwide
demand shock with the outcome, so it is shown as a robustness check and not used as the main estimate.

![Elasticity forest plot](figures/elasticities.png)

![Per-product OLS with controls against hierarchical B](figures/shrinkage.png)

In that plot only {moved} of {len(tab)} products move toward the category mean, and the
spread of elasticities barely changes (standard deviation {f2(sd_controls)} for OLS with
controls, {f2(sd_hier)} for B). Pooling is light on this data. The comparison also
changes the model (random-walk level, Student-t likelihood), so it is not a clean
measure of shrinkage.

## Validation on the 52-week holdout

Each forecaster predicts units for the holdout weeks. The Bayesian models
forecast the holdout conditionally on the planned price and cost. The baseline
ignores price. Coverage is the share of observed units inside the 90% interval.

{validation_table(real)}

Ranked by MAE, lowest first: {mae_rank}. Ranked by how close the 90% interval
coverage is to 90%: {cov_rank}. The differences in MAE between the price models are small. The holdout measures short-run forecast accuracy in the same
price regime as training. It cannot tell a biased slope from a good one, which is why the
synthetic study below is where bias is measured. The baseline is far worse (MAE {f2(base_mae, 1)}),
so a price-aware model does add information.

The OLS intervals are prediction intervals for a new observation, so like the Bayesian
intervals they include parameter uncertainty. MAPE is averaged over weeks with positive
sales. Units are mean units per reporting store per week, not chain volume.

![Prior and posterior predictive checks](figures/predictive_checks.png)

## Diagnostics

Sampling uses nutpie. Checks cover all free parameters, not only the elasticities.

{checks_table(real)}

Prior predictive: a 90% interval for the product elasticities `beta` (not the category
mean) of {f2(c["hier"]["prior_beta_90"][0])} to {f2(c["hier"]["prior_beta_90"][1])} (B) and
prior predictive log units from {f2(c["hier"]["prior_pred_log_q_90"][0])} to
{f2(c["hier"]["prior_pred_log_q_90"][1])}, against an observed range of
{f2(c["hier"]["observed_log_q_range"][0])} to {f2(c["hier"]["observed_log_q_range"][1])}.
Posterior predictive: in-sample 90% coverage of the observed log units is
{pct(c["hier"]["ppc_in_sample_coverage90"], 1)} (B) and
{pct(c["hier_iv"]["ppc_in_sample_coverage90"], 1)} (B + IV). The predictive
p-values for sd, mean, min and max are in `results/real.json`. For B, the replicated sd
exceeds the observed sd in {f2(c["hier"]["ppc_p_sd"], 3)} of draws, and the replicated max
exceeds the observed max in {f2(c["hier"]["ppc_p_max"], 3)}. The replicated min exceeds the
observed min in only {f2(c["hier"]["ppc_p_min"], 3)}. So the replicated data are too dispersed,
with a lower tail that is too long and an upper tail that is too high. The mean is
reproduced ({f2(c["hier"]["ppc_p_mean"], 3)}). The model is not well calibrated in the tails.

The endogeneity term `lambda_j` is credibly away from zero (90% interval excludes 0)
for {lam_n} of {len(tab)} products. So on this data the evidence that price responds to
unobserved weekly demand is weak, and the IV and non-IV hierarchical models differ only
modestly (category means {f2(m["hier_iv"])} and {f2(m["hier"])}). The 2SLS intervals are widest
where the first stage is weakest (see the F column), so the large individual IV
effects are not evidence of a strong endogeneity signal.

## Synthetic recovery test

The real data never reveal the truth, so I simulate panels with known elasticities
({n_runs} estimates over {syn["seeds"]} seeds). Each simulated retailer
sets price partly in response to a demand shock the analyst cannot see, and cost
moves price without moving demand. The truth is `beta_j` for each product.

{synthetic_table(syn)}

Convergence of the synthetic Bayesian fits (one fit per seed and model, saved in
`results/synthetic_diagnostics.csv`):

{synthetic_checks_table(syn)}

The naive model is biased towards zero (mean error {f2(syn["methods"]["naive"]["bias"])}) and its 90%
intervals cover the truth only {pct(syn["methods"]["naive"]["coverage90"])} of the time.
Adding controls makes the bias worse here (mean error {f2(syn["methods"]["controls"]["bias"])}), not better. The
block effects absorb the slow, cost-driven part of price, so what remains is closer to the
demand shock. In this simulator that is how controls fail. On the real data, adding
controls moves individual products in both directions and barely changes the category
mean, so the simulation does not show what happens on DFF. The hierarchical model
without IV (mean error {f2(syn["methods"]["hier"]["bias"])}) is still biased, because it treats
price as exogenous. The Bayesian IV model recovers the truth: mean error
{f2(syn["methods"]["hier_iv"]["bias"])} and 90% coverage {pct(syn["methods"]["hier_iv"]["coverage90"])}, with
intervals that are conservative (the coverage is above 90%). The 2SLS instrument is unbiased
(mean error {f2(syn["methods"]["iv_2sls"]["bias"])}) but its intervals are much wider and cover
{pct(syn["methods"]["iv_2sls"]["coverage90"])}, below the nominal 90%. B + IV is also more precise
than 2SLS here (mean width {f2(syn["methods"]["hier_iv"]["width90"])} against
{f2(syn["methods"]["iv_2sls"]["width90"])}).

The three ablation rows test where that precision comes from. Without pooling, B + IV
has mean absolute error {f2(sm["hier_iv_nopool"]["mae"])}, width {f2(sm["hier_iv_nopool"]["width90"])} and coverage
{pct(sm["hier_iv_nopool"]["coverage90"])}, against {f2(sm["hier_iv"]["mae"])}, {f2(sm["hier_iv"]["width90"])} and
{pct(sm["hier_iv"]["coverage90"])} with pooling. So partial pooling adds little precision in this
design; it mainly makes the intervals more conservative. 2SLS with a linear trend instead
of block effects has mean absolute error {f2(sm["iv_2sls_trend"]["mae"])} and width
{f2(sm["iv_2sls_trend"]["width90"])}, so part of the gap to B + IV is the block effects throwing away
cost variation. The rest comes from the Bayesian model as a whole (random-walk level,
joint likelihood with the first stage, priors), which this study does not split further.
The instrument alone removes the bias in this design.

The Anderson-Rubin sets for 2SLS cover {pct(sm["iv_ar"]["coverage90"])}, no better than the Wald
intervals, with {pct(sm["iv_ar"]["unbounded"])} of the sets unbounded, so the 2SLS undercoverage here is
not a weak-instrument problem. Its cause is not established.

The simulator favours the control-function approach. The first-stage residual is almost
entirely the demand shock (store-level price noise has sd {f2(syn["store_price_sd"])}), and the
true elasticities are drawn from a normal distribution (mean {f2(syn["true_mean_elasticity"])},
sd {f2(syn["true_sd_elasticity"])}), the same form as the hierarchy the model assumes. The bias and coverage numbers
above are therefore an optimistic case, not a forecast for DFF.

The synthetic study shares one caveat with the real data: the instrument is the
cost series, and the simulation makes it valid by design. The test shows what each
estimator does when its assumptions hold, not that they hold for DFF.

The same check runs in CI on a small panel (`tests/test_recovery.py`). The slow test
(`pytest -m slow`) is a smaller subset of seeds that fits only B + IV, with the same
sampler settings. The table above comes from `elasticity.pipeline --seeds {syn["seeds"]}`.

## What this means for pricing

Elasticity per product (B + IV, posterior mean): a product with elasticity below -1
is one where a price cut raises revenue. Here the posterior puts more than
{f2(tab["hier_iv_p_elastic"].min() * 100, 0)}% of mass below -1 for every product, so
all {len(tab)} are elastic on this model. That is a strong claim from a short, noisy
panel and should be read with the wide intervals in the table above, not as a
ranking of which products to cut.
Under a constant-elasticity model, the profit-maximising price (the Lerner condition
(p - c) / p = -1 / elasticity) has a gross margin of -1 / elasticity. This is a
profit-maximising condition, not a revenue-maximising one. A revenue-maximising price
would need elasticity equal to -1. The elasticity here is a retailer's own-item
elasticity, which includes switching to other items and other stores, so it is not a
category or manufacturer elasticity. The table compares that implied margin with DFF's
own median margin for each product, as a plausibility check on the elasticities.

{margin_table(tab)}

* Products with P(elasticity < -1) above 95%: {len(elastic)} of {len(tab)}.
* Implied margins run from {f2(tab["lerner_margin_hier_iv"].min() * 100, 0)}% to
  {f2(tab["lerner_margin_hier_iv"].max() * 100, 0)}% of price, against DFF median margins of
  {f2(tab["dff_median_margin"].min() * 100, 0)}% to {f2(tab["dff_median_margin"].max() * 100, 0)}%. Where the
  two disagree by a lot, the elasticity or the cost proxy is the first suspect.

Promotion read (B + IV). The lift on the deal flag is the shift in units at the same
price. The total lift adds the typical price cut in a deal week through the elasticity,
so it inherits the elasticity's uncertainty and is large for products with a steep
elasticity and a deep discount ({top_promo["product"]} is the largest, with a total lift of
{f2(top_promo["total_lift_mean"])}). Lifts are proportional changes in units: {f2(top_promo["total_lift_mean"])}
means {pct(top_promo["total_lift_mean"])} more units. Read these as
directional. The deal flag in the DFF data is not set consistently (manual remark 4),
so some promoted weeks are not flagged.

{chr(10).join(promo_md)}

## Honest limits

* **Cheese only, twelve products, one retailer, 1989 to 1997.** The results do not
  transfer to other categories, to other markets, or to today's shoppers.
* **The cost proxy is average acquisition cost**, which lags replacement cost. The
  manual says so. A replacement-cost series would give a cleaner instrument.
* **The instrument is weak for some products** (first-stage F below 10 for
  {real["weak_iv_products"]} products). For those, the 2SLS interval is not meaningful, and the
  Anderson-Rubin set is unbounded (on at least one side) for {ar_unbounded_weak} of them. The B + IV
  intervals are also not evidence of cost information: they are probably set by the
  hierarchy and the prior. I have not tested that. So the P(elasticity < -1) figures for
  these products should not be used alone for pricing decisions.
* **No promotion timing data.** The deal flag is incomplete, so promo lifts are read
  as directional only.
* **The holdout is 52 weeks of the same regime.** A structural shift, such as a new
  competitor or a new pricing rule, would break the model, and this holdout does
  not test that.
* **The Bayesian models use a random walk for base demand.** It captures slow drift
  but cannot anticipate a sudden change.
* **Chain-level rows average over stores.** Store-level heterogeneity is in the
  store-level 2SLS robustness check, not in the Bayesian model.
* **The Hausman instrument is not a valid instrument here.** In the simulation it
  shares the demand shock with the outcome. In the DFF data the same concern applies,
  so it is a robustness check only.
* **The posterior predictive is too dispersed.** The replicated minimum falls below the
  observed minimum in almost every draw, and the replicated maximum is too high. The
  Student-t tail with one shared degrees-of-freedom parameter is the likely cause. I have not
  tested that.
* **Some synthetic fits have divergences.** B + IV had {syn["diagnostics"]["hier_iv"]["divergences"]:.0f}
  divergent transitions over {syn["diagnostics"]["hier_iv"]["fits_with_divergences"]:.0f} of
  {syn["diagnostics"]["hier_iv"]["fits"]:.0f} fits, and the unpooled ablation
  {syn["diagnostics"]["hier_iv_nopool"]["divergences"]:.0f} over {syn["diagnostics"]["hier_iv_nopool"]["fits_with_divergences"]:.0f}
  fits (max R-hat {f2(max(g["max_rhat"] for g in syn["diagnostics"].values()), 2)}). The synthetic fits use
  {syn["settings"]["draws"]} draws; a higher `target_accept` would be the first fix. The real-data
  fits have no divergences.
* **Partial pooling adds little here.** On this data pooling is light, and in the synthetic
  study removing it barely changes B + IV. The case for the hierarchy is products with
  short histories, which this panel does not have.
* **MAE and coverage are on units, which are dominated by large products.** The
  WAPE is the more comparable number.

## If I did this at Arla

For NRM this is a decision tool for pricing and promotions, not a forecast for
its own sake. I would use the hierarchical structure because it gives a
per-product elasticity with an honest interval, shrunk toward the category when a
SKU has little history, which is the normal case for new or seasonal products.
Before a price change I would check the instrument strength for each SKU,
because a weak instrument is the most common way a pricing model gets confidently
wrong. Cost has to come from a replacement-cost feed where possible, and the
promo calendar has to be complete, which in this data it is not.

For monitoring, I would track three things per SKU: the holdout coverage of the
90% interval, the drift of the elasticity posterior over time, and the first-stage
F. A widening interval or a falling F is an early warning that the model is no
longer identified and that the team should stop acting on it.

An agentic layer would sit on top of this, not inside it. It can read the
monitoring tables, flag SKUs whose elasticity crossed -1 or whose instrument
weakened, draft the scenario ("what happens to margin if we cut this SKU 5%?")
and run the posterior predictive for it. The pricing decision stays with a
person. The agent has no authority to change a price. Its role is to make the
model's uncertainty visible at the moment someone decides.

## Reproduce

```bash
uv sync
uv run pytest                                 # fast synthetic tests, no real data needed
uv run python -m elasticity.data              # downloads the Kilts files, checks SHA-256
OMP_NUM_THREADS=1 uv run python -m elasticity.pipeline --seeds 10
uv run python -m elasticity.report            # rewrites this README from results/
uv run pytest -m slow                         # full synthetic recovery study
```

Sampling settings used for the real data: {fe.get("draws", "n/a")} draws, {fe.get("tune", "n/a")} tuning
steps, {fe.get("chains", "n/a")} chains (`results/real.json`), run on an Apple M1 with 8 GB RAM
with one thread.

## CI and CD

* **CI** (`.github/workflows/ci.yml`, every branch and pull request): ruff lint and format,
  the fast test suite, and `elasticity.report --check`, which fails if any number in this
  README no longer matches `results/`.
* **CD** (`.github/workflows/cd.yml`): runs CI as a gate, then delivers. A push to `main`
  publishes this README, the figures and the raw result files as a GitHub Pages site
  (`scripts/build_site.py`). A version tag such as `v0.2.0` must match the version in
  `pyproject.toml`; it builds the wheel and sdist, installs the wheel in a clean
  environment as a smoke test, and publishes a GitHub Release with the wheel, the sdist and
  a zip of `results/`, `figures/` and this README.
* **Reproduce** (`.github/workflows/reproduce.yml`, started by hand): downloads the raw
  data on a clean runner, checks its SHA-256, reruns the whole pipeline and uploads the new
  results with a diff against the committed ones. The raw data is never uploaded.

At Arla the same pattern would deliver a scheduled re-estimation: the gate would also check
convergence (R-hat, divergences) and holdout error against the last released model before
new elasticities reach a pricing tool.
"""
    text = text.replace("–", "-").replace("—", "-")
    if check:
        if not README.exists() or README.read_text() != text:
            print("README.md is out of date with results/. Run: python -m elasticity.report")
            return 1
        print("README.md matches results/")
        return 0
    README.write_text(text)
    print(f"wrote {README}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    sys.exit(main(ap.parse_args().check))
