"""Frequentist estimators, one regression per product.

* naive: log units on log price only (Model A).
* controls: adds deal flag, 13-week block fixed effects (slow level changes),
  yearly seasonality and, on store rows, store fixed effects
  (the no-pooling counterpart of the hierarchical model).
* iv: two-stage least squares with the same controls, instrumenting log price
  with log unit cost (or with the Hausman price of the same product elsewhere).

Standard errors: on store x week rows they are clustered by week, because price
and the demand shock both move chainwide within a week. On chain-level weekly
rows they are Newey-West with 4 lags, for autocorrelation over weeks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

CONTROLS_RHS = "deal + C(quarter) + s1 + c1 + s2 + c2"


def _stores_rhs(d: pd.DataFrame) -> str:
    return " + C(store)" if "store" in d and d["store"].nunique() > 1 else ""


def _store_level(d: pd.DataFrame) -> bool:
    return "store" in d and d["store"].nunique() > 1


def _sm_cov(g: pd.DataFrame) -> dict:
    """Cluster by week for store-level rows, Newey-West (4 lags) for weekly rows."""
    if _store_level(g):
        return {"cov_type": "cluster", "cov_kwds": {"groups": g["week"]}}
    return {"cov_type": "HAC", "cov_kwds": {"maxlags": 4}}


def _lm_cov(g: pd.DataFrame) -> dict:
    if _store_level(g):
        return {"cov_type": "clustered", "clusters": g["week"]}
    return {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": 4}


def _row(product, est, se, n, **extra) -> dict:
    return {
        "product": product,
        "elasticity": float(est),
        "se": float(se),
        "lo90": float(est - 1.6449 * se),
        "hi90": float(est + 1.6449 * se),
        "n": int(n),
        **extra,
    }


def naive(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, g in d.groupby("product", observed=True):
        r = smf.ols("log_q ~ log_p", g).fit(**_sm_cov(g))
        rows.append(
            _row(p, r.params["log_p"], r.bse["log_p"], r.nobs, resid_sd=float(np.sqrt(r.scale)))
        )
    return pd.DataFrame(rows)


def with_controls(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, g in d.groupby("product", observed=True):
        f = f"log_q ~ log_p + {CONTROLS_RHS}{_stores_rhs(g)}"
        r = smf.ols(f, g).fit(**_sm_cov(g))
        rows.append(_row(p, r.params["log_p"], r.bse["log_p"], r.nobs))
    return pd.DataFrame(rows)


def iv(d: pd.DataFrame, instrument: str = "log_cost") -> pd.DataFrame:
    """2SLS per product; reports the first-stage partial F of the instrument."""
    rows = []
    for p, g in d.groupby("product", observed=True):
        g = g.reset_index(drop=True)
        f = f"log_q ~ 1 + {CONTROLS_RHS}{_stores_rhs(g)} + [log_p ~ {instrument}]"
        r = IV2SLS.from_formula(f, g).fit(**_lm_cov(g))
        diag = r.first_stage.diagnostics.loc["log_p"]
        rows.append(
            _row(
                p,
                r.params["log_p"],
                r.std_errors["log_p"],
                r.nobs,
                first_stage_f=float(diag["f.stat"]),
                partial_r2=float(diag["partial.rsquared"]),
            )
        )
    return pd.DataFrame(rows)
