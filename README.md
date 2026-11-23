# Bayesian hierarchical price elasticity with instrumental variables

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
* **Scope:** the top 12 fixed-weight cheese UPCs by revenue
  (random-weight packs excluded), 93 stores, weekly, 368,839 store-week rows.
  Chain-level modelling uses 4,478 product-weeks (store means), with
  product-weeks where fewer than 80% of the usual stores reported dropped as data gaps.
* **Price** is the shelf price per unit (PRICE / QTY). **Unit cost** is derived
  from DFF's gross margin (`cost = price * (1 - PROFIT/100)`). The manual says this
  is the average acquisition cost, not replacement cost, so it lags wholesale moves.
* **Time split:** the last 52 weeks (from 1996-05-09 to 1997-05-01) are the holdout.
  Training runs from 1989-09-14.

## Method

**A. Naive model.** Per product, log units on log price (OLS). Price is treated
as exogenous. This is the baseline most teams start from.

**Why price is endogenous.** A retailer sees demand (weather, school holidays,
competitor stock-outs, local events) before the analyst does, and sets price
partly in response to it. A week with high demand is then often a week with a
higher price, so the price coefficient is pulled towards zero. In the DFF cheese
data the correlation of log price with log cost, pooled over the training
product-weeks, is 0.95, so cost moves price strongly. That is the
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
90% confidence intervals, all estimated on the 337 training weeks.
The first-stage F tells how strong the cost instrument is; values below 10 mean
the 2SLS estimate for that product is unreliable. The Wald column uses Newey-West
errors; in the synthetic study these intervals cover the truth
77% of the time rather than 90%, so read them as rough. The
Anderson-Rubin column is the 90% confidence set from inverting the Anderson-Rubin test
(same Newey-West kernel), which stays valid when the instrument is weak. It is unbounded,
on at least one side, for 5 of 12 products: for those, the cost data cannot
pin the elasticity down.

| Product | A. Naive OLS | OLS + controls | 2SLS (cost IV), Wald | 2SLS, Anderson-Rubin set | First-stage F | B. Hierarchical | B + IV (joint) | P(elasticity < -1), B + IV |
|---|---|---|---|---|---|---|---|---|
| Kraft American Singles 12 oz | -3.19 [-3.48, -2.89] | -3.42 [-3.65, -3.20] | -3.84 [-4.31, -3.37] | [-4.35, -3.39] | 48.2 | -3.63 [-3.82, -3.44] | -3.53 [-3.92, -3.15] | 100% |
| Kraft Phila Cream Chees 8 oz | -4.19 [-4.47, -3.92] | -3.34 [-3.73, -2.94] | -3.70 [-4.60, -2.81] | [-4.83, -2.89] | 21.3 | -3.90 [-4.19, -3.59] | -3.86 [-4.26, -3.39] | 100% |
| Kraft American Singles 16 oz | -2.63 [-3.02, -2.23] | -3.10 [-3.61, -2.60] | -3.46 [-5.13, -1.79] | (-inf, -1.94] | 2.4 | -3.04 [-3.36, -2.71] | -2.77 [-3.32, -2.16] | 100% |
| Kraft Grtd Parmesan 8 oz | -3.19 [-3.65, -2.73] | -4.52 [-5.15, -3.88] | -6.85 [-8.39, -5.31] | [-8.71, -5.43] | 25.4 | -3.66 [-4.08, -3.27] | -3.03 [-3.74, -2.32] | 100% |
| DFF own Amer Sngl Chs Fo 12 oz | -3.80 [-4.16, -3.44] | -3.68 [-4.10, -3.26] | -1.50 [-14.49, 11.48] | unbounded | 0.7 | -3.32 [-3.64, -2.98] | -3.13 [-3.64, -2.56] | 100% |
| Kraft Velveeta Loaf 32 oz | -3.56 [-4.21, -2.91] | -3.12 [-3.61, -2.63] | -2.88 [-3.98, -1.79] | [-4.06, -1.83] | 60.5 | -2.85 [-3.24, -2.44] | -2.49 [-3.03, -1.92] | 100% |
| DFF own Cream Cheese 8 oz | -2.23 [-2.80, -1.66] | -2.76 [-3.72, -1.81] | -1.45 [-3.16, 0.27] | [-3.70, 0.34] | 10.0 | -2.64 [-3.12, -2.15] | -2.61 [-3.22, -1.96] | 100% |
| Kraft Philly Light Crm. 8 oz | -3.87 [-4.10, -3.63] | -3.64 [-3.93, -3.36] | -3.44 [-3.99, -2.88] | [-4.03, -2.89] | 30.3 | -3.79 [-4.03, -3.57] | -3.59 [-3.93, -3.23] | 100% |
| DFF own Amer Sngl Chs Fo 16 oz | -3.39 [-3.75, -3.03] | -2.75 [-3.41, -2.09] | -5.29 [-12.28, 1.70] | unbounded | 1.0 | -2.30 [-2.67, -1.94] | -2.38 [-2.89, -1.85] | 100% |
| Kraft Free Singles 12 oz | -2.51 [-3.11, -1.92] | -2.32 [-3.00, -1.63] | -2.32 [-3.59, -1.06] | [-3.68, -1.03] | 28.7 | -2.07 [-2.38, -1.77] | -1.79 [-2.28, -1.30] | 99% |
| DFF own Grtd Parmsn Chse 8 oz | -1.88 [-2.29, -1.46] | -2.30 [-3.18, -1.42] | -14.21 [-212.56, 184.15] | unbounded | 0.0 | -2.44 [-2.91, -1.96] | -2.20 [-2.95, -1.40] | 99% |
| DFF own Shredded Mozzare 12 oz | -1.57 [-1.91, -1.22] | -1.81 [-2.11, -1.51] | -0.43 [-3.44, 2.59] | unbounded | 1.3 | -1.85 [-2.09, -1.61] | -2.23 [-2.82, -1.63] | 100% |

Mean across the 12 products: naive -3.00, OLS with controls -3.06,
hierarchical -2.96, hierarchical with IV -2.80. The 2SLS mean is
left out, because it averages in estimates with a first-stage F near zero.
Restricted to the 7 products with first-stage F of at least 10: OLS with controls
-3.30, 2SLS -3.50, hierarchical -3.22, hierarchical with IV -2.98.
The weak-instrument products are 5 of 12, with a median first-stage F of
15.7, so the cost instrument is strong for some cheese
products and weak for others. The Hausman instrument (same product's price in other stores) is
very strong on store rows (median F 90524), but it shares the chainwide
demand shock with the outcome, so it is shown as a robustness check and not used as the main estimate.

![Elasticity forest plot](figures/elasticities.png)

![Per-product OLS with controls against hierarchical B](figures/shrinkage.png)

In that plot only 6 of 12 products move toward the category mean, and the
spread of elasticities barely changes (standard deviation 0.73 for OLS with
controls, 0.71 for B). Pooling is light on this data. The comparison also
changes the model (random-walk level, Student-t likelihood), so it is not a clean
measure of shrinkage.

## Validation on the 52-week holdout

Each forecaster predicts units for the holdout weeks. The Bayesian models
forecast the holdout conditionally on the planned price and cost. The baseline
ignores price. Coverage is the share of observed units inside the 90% interval.

| Model | MAE (units per store-week) | WAPE | 90% interval coverage |
|---|---|---|---|
| Simple baseline (last 13 weeks median, no price) | 43.8 | 54.7% | 76.9% |
| A. Naive log-log OLS per product | 30.4 | 38.0% | 89.3% |
| OLS per product with controls | 29.8 | 37.3% | 81.4% |
| B. Hierarchical | 31.0 | 38.8% | 84.5% |
| B. Hierarchical + IV | 30.9 | 38.6% | 85.1% |

Ranked by MAE, lowest first: OLS with controls (29.8), naive OLS (30.4), B + IV (30.9), B (31.0). Ranked by how close the 90% interval
coverage is to 90%: naive OLS (89.3%), B + IV (85.1%), B (84.5%), OLS with controls (81.4%). The differences in MAE between the price models are small. The holdout measures short-run forecast accuracy in the same
price regime as training. It cannot tell a biased slope from a good one, which is why the
synthetic study below is where bias is measured. The baseline is far worse (MAE 43.8),
so a price-aware model does add information.

The OLS intervals are prediction intervals for a new observation, so like the Bayesian
intervals they include parameter uncertainty. MAPE is averaged over weeks with positive
sales. Units are mean units per reporting store per week, not chain volume.

![Prior and posterior predictive checks](figures/predictive_checks.png)

## Diagnostics

Sampling uses nutpie. Checks cover all free parameters, not only the elasticities.

| Model | Max R-hat | Min ESS (bulk) | Min ESS (tail) | Divergences | Chains x draws | Free parameters | Seconds |
|---|---|---|---|---|---|---|---|
| B. Hierarchical | 1.010 | 374 | 556 | 0 | 4 x 1000 | 408 | 157 |
| B. Hierarchical + IV | 1.010 | 603 | 1072 | 0 | 4 x 1000 | 823 | 392 |

Prior predictive: a 90% interval for the product elasticities `beta` (not the category
mean) of -3.47 to 0.30 (B) and
prior predictive log units from 1.57 to
6.47, against an observed range of
1.72 to 7.68.
Posterior predictive: in-sample 90% coverage of the observed log units is
90.1% (B) and
89.9% (B + IV). The predictive
p-values for sd, mean, min and max are in `results/real.json`. For B, the replicated sd
exceeds the observed sd in 0.981 of draws, and the replicated max
exceeds the observed max in 0.975. The replicated min exceeds the
observed min in only 0.002. So the replicated data are too dispersed,
with a lower tail that is too long and an upper tail that is too high. The mean is
reproduced (0.136). The model is not well calibrated in the tails.

The endogeneity term `lambda_j` is credibly away from zero (90% interval excludes 0)
for 1 of 12 products. So on this data the evidence that price responds to
unobserved weekly demand is weak, and the IV and non-IV hierarchical models differ only
modestly (category means -2.80 and -2.96). The 2SLS intervals are widest
where the first stage is weakest (see the F column), so the large individual IV
effects are not evidence of a strong endogeneity signal.

## Synthetic recovery test

The real data never reveal the truth, so I simulate panels with known elasticities
(540 estimates over 10 seeds). Each simulated retailer
sets price partly in response to a demand shock the analyst cannot see, and cost
moves price without moving demand. The truth is `beta_j` for each product.

| Method | Mean error (bias) | Mean absolute error | 90% interval coverage | Mean 90% width (bounded sets) | Unbounded sets |
|---|---|---|---|---|---|
| A. Naive OLS | 0.49 | 0.49 | 12% | 0.57 | 0% |
| OLS with controls | 1.46 | 1.46 | 0% | 0.46 | 0% |
| 2SLS (cost IV) | -0.01 | 0.39 | 77% | 1.26 | 0% |
| 2SLS (Hausman IV, store rows) | 1.47 | 1.47 | 0% | 0.45 | 0% |
| B. Hierarchical | 1.27 | 1.27 | 0% | 0.43 | 0% |
| B. Hierarchical + IV | 0.04 | 0.12 | 98% | 0.58 | 0% |
| Ablation: 2SLS estimate with Anderson-Rubin set | -0.01 | 0.39 | 75% | 1.38 | 0% |
| Ablation: 2SLS, linear trend instead of block effects | -0.04 | 0.28 | 80% | 0.99 | 0% |
| Ablation: B + IV without pooling | 0.06 | 0.14 | 92% | 0.60 | 0% |

Convergence of the synthetic Bayesian fits (one fit per seed and model, saved in
`results/synthetic_diagnostics.csv`):

| Model | Fits | Max R-hat | Min ESS (bulk) | Min ESS (tail) | Divergences (total) | Fits with divergences |
|---|---|---|---|---|---|---|
| B. Hierarchical | 10 | 1.020 | 249 | 203 | 1 | 1 |
| B. Hierarchical + IV | 10 | 1.020 | 278 | 367 | 6 | 4 |
| B + IV without pooling | 10 | 1.020 | 261 | 101 | 18 | 6 |

The naive model is biased towards zero (mean error 0.49) and its 90%
intervals cover the truth only 12% of the time.
Adding controls makes the bias worse here (mean error 1.46), not better. The
block effects absorb the slow, cost-driven part of price, so what remains is closer to the
demand shock. In this simulator that is how controls fail. On the real data, adding
controls moves individual products in both directions and barely changes the category
mean, so the simulation does not show what happens on DFF. The hierarchical model
without IV (mean error 1.27) is still biased, because it treats
price as exogenous. The Bayesian IV model recovers the truth: mean error
0.04 and 90% coverage 98%, with
intervals that are conservative (the coverage is above 90%). The 2SLS instrument is unbiased
(mean error -0.01) but its intervals are much wider and cover
77%, below the nominal 90%. B + IV is also more precise
than 2SLS here (mean width 0.58 against
1.26).

The three ablation rows test where that precision comes from. Without pooling, B + IV
has mean absolute error 0.14, width 0.60 and coverage
92%, against 0.12, 0.58 and
98% with pooling. So partial pooling adds little precision in this
design; it mainly makes the intervals more conservative. 2SLS with a linear trend instead
of block effects has mean absolute error 0.28 and width
0.99, so part of the gap to B + IV is the block effects throwing away
cost variation. The rest comes from the Bayesian model as a whole (random-walk level,
joint likelihood with the first stage, priors), which this study does not split further.
The instrument alone removes the bias in this design.

The Anderson-Rubin sets for 2SLS cover 75%, no better than the Wald
intervals, with 0% of the sets unbounded, so the 2SLS undercoverage here is
not a weak-instrument problem. Its cause is not established.

The simulator favours the control-function approach. The first-stage residual is almost
entirely the demand shock (store-level price noise has sd 0.03), and the
true elasticities are drawn from a normal distribution (mean -2.00,
sd 0.40), the same form as the hierarchy the model assumes. The bias and coverage numbers
above are therefore an optimistic case, not a forecast for DFF.

The synthetic study shares one caveat with the real data: the instrument is the
cost series, and the simulation makes it valid by design. The test shows what each
estimator does when its assumptions hold, not that they hold for DFF.

The same check runs in CI on a small panel (`tests/test_recovery.py`). The slow test
(`pytest -m slow`) is a smaller subset of seeds that fits only B + IV, with the same
sampler settings. The table above comes from `elasticity.pipeline --seeds 10`.

## What this means for pricing

Elasticity per product (B + IV, posterior mean): a product with elasticity below -1
is one where a price cut raises revenue. Here the posterior puts more than
99% of mass below -1 for every product, so
all 12 are elastic on this model. That is a strong claim from a short, noisy
panel and should be read with the wide intervals in the table above, not as a
ranking of which products to cut.
Under a constant-elasticity model, the profit-maximising price (the Lerner condition
(p - c) / p = -1 / elasticity) has a gross margin of -1 / elasticity. This is a
profit-maximising condition, not a revenue-maximising one. A revenue-maximising price
would need elasticity equal to -1. The elasticity here is a retailer's own-item
elasticity, which includes switching to other items and other stores, so it is not a
category or manufacturer elasticity. The table compares that implied margin with DFF's
own median margin for each product, as a plausibility check on the elasticities.

| Product | Implied margin, -1 / elasticity (B + IV) | DFF median margin |
|---|---|---|
| Kraft American Singles 12 oz | 28% | 28% |
| Kraft Phila Cream Chees 8 oz | 26% | 29% |
| Kraft American Singles 16 oz | 36% | 27% |
| Kraft Grtd Parmesan 8 oz | 33% | 27% |
| DFF own Amer Sngl Chs Fo 12 oz | 32% | 45% |
| Kraft Velveeta Loaf 32 oz | 40% | 15% |
| DFF own Cream Cheese 8 oz | 38% | 33% |
| Kraft Philly Light Crm. 8 oz | 28% | 29% |
| DFF own Amer Sngl Chs Fo 16 oz | 42% | 50% |
| Kraft Free Singles 12 oz | 56% | 24% |
| DFF own Grtd Parmsn Chse 8 oz | 45% | 29% |
| DFF own Shredded Mozzare 12 oz | 45% | 40% |

* Products with P(elasticity < -1) above 95%: 12 of 12.
* Implied margins run from 26% to
  56% of price, against DFF median margins of
  15% to 50%. Where the
  two disagree by a lot, the elasticity or the cost proxy is the first suspect.

Promotion read (B + IV). The lift on the deal flag is the shift in units at the same
price. The total lift adds the typical price cut in a deal week through the elasticity,
so it inherits the elasticity's uncertainty and is large for products with a steep
elasticity and a deep discount (Kraft Phila Cream Chees 8 oz is the largest, with a total lift of
1.46). Lifts are proportional changes in units: 1.46
means 146% more units. Read these as
directional. The deal flag in the DFF data is not set consistently (manual remark 4),
so some promoted weeks are not flagged.

| Product | Weeks on deal | Median discount | Lift on deal flag (B + IV, 90%) | Total promo lift incl. price (B + IV, 90%) |
|---|---|---|---|---|
| Kraft American Singles 12 oz | 38% | 6% | 0.04 [-0.04, 0.10] | 0.32 [0.24, 0.39] |
| Kraft Phila Cream Chees 8 oz | 37% | 19% | 0.11 [0.03, 0.23] | 1.46 [1.30, 1.63] |
| Kraft American Singles 16 oz | 35% | 7% | 0.08 [0.02, 0.15] | 0.32 [0.27, 0.38] |
| Kraft Grtd Parmesan 8 oz | 39% | 7% | 0.07 [0.00, 0.13] | 0.34 [0.28, 0.39] |
| DFF own Amer Sngl Chs Fo 12 oz | 31% | 22% | 0.11 [0.03, 0.24] | 1.44 [1.21, 1.68] |
| Kraft Velveeta Loaf 32 oz | 29% | 5% | 0.09 [0.03, 0.17] | 0.24 [0.18, 0.30] |
| DFF own Cream Cheese 8 oz | 41% | 13% | 0.10 [0.03, 0.21] | 0.58 [0.47, 0.70] |
| Kraft Philly Light Crm. 8 oz | 37% | 18% | 0.09 [0.02, 0.17] | 1.26 [1.12, 1.41] |
| DFF own Amer Sngl Chs Fo 16 oz | 16% | 14% | 0.08 [0.00, 0.17] | 0.53 [0.44, 0.62] |
| Kraft Free Singles 12 oz | 34% | 13% | 0.07 [0.01, 0.14] | 0.36 [0.30, 0.43] |
| DFF own Grtd Parmsn Chse 8 oz | 36% | 6% | 0.08 [0.02, 0.15] | 0.23 [0.19, 0.27] |
| DFF own Shredded Mozzare 12 oz | 18% | 16% | 0.09 [0.02, 0.17] | 0.61 [0.50, 0.73] |

## Honest limits

* **Cheese only, twelve products, one retailer, 1989 to 1997.** The results do not
  transfer to other categories, to other markets, or to today's shoppers.
* **The cost proxy is average acquisition cost**, which lags replacement cost. The
  manual says so. A replacement-cost series would give a cleaner instrument.
* **The instrument is weak for some products** (first-stage F below 10 for
  5 products). For those, the 2SLS interval is not meaningful, and the
  Anderson-Rubin set is unbounded (on at least one side) for 5 of them. The B + IV
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
* **Some synthetic fits have divergences.** B + IV had 6
  divergent transitions over 4 of
  10 fits, and the unpooled ablation
  18 over 6
  fits (max R-hat 1.02). The synthetic fits use
  500 draws; a higher `target_accept` would be the first fix. The real-data
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

Sampling settings used for the real data: 1000 draws, 1000 tuning
steps, 4 chains (`results/real.json`), run on an Apple M1 with 8 GB RAM
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
