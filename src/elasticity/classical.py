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
TREND_RHS = "deal + t + s1 + c1 + s2 + c2"
CHI2_1_90 = 2.705543  # 90% critical value of chi-square with 1 df
NW_LAGS = 4


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


def iv(d: pd.DataFrame, instrument: str = "log_cost", controls: str = CONTROLS_RHS) -> pd.DataFrame:
    """2SLS per product; reports the first-stage partial F of the instrument.

    `controls=TREND_RHS` swaps the 13-week block effects for a linear trend, an
    ablation used in the synthetic study."""
    rows = []
    for p, g in d.groupby("product", observed=True):
        g = g.reset_index(drop=True)
        f = f"log_q ~ 1 + {controls}{_stores_rhs(g)} + [log_p ~ {instrument}]"
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


def _nw_var(s: np.ndarray, lags: int = NW_LAGS) -> np.ndarray:
    """Newey-West (Bartlett) long-run variance of the columns of s (rows are weeks)."""
    v = (s * s).sum(axis=0)
    for lag in range(1, lags + 1):
        v = v + 2 * (1 - lag / (lags + 1)) * (s[lag:] * s[:-lag]).sum(axis=0)
    return v


def anderson_rubin(
    d: pd.DataFrame,
    instrument: str = "log_cost",
    controls: str = CONTROLS_RHS,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """90% Anderson-Rubin confidence set per product, by grid inversion.

    For each candidate elasticity b, regress log_q - b * log_p on the controls and
    the instrument, and keep b if the instrument's coefficient is not significant
    (Newey-West Wald test, same kernel as the 2SLS errors). The set stays valid when
    the instrument is weak; it can be unbounded, which is reported. Chain-level
    (one row per week) data only.
    """
    import patsy

    if _store_level(d):
        raise ValueError("anderson_rubin expects chain-level weekly rows")
    grid = np.arange(-30.0, 10.0 + 1e-9, 0.01) if grid is None else grid
    rows = []
    for p, g in d.groupby("product", observed=True):
        g = g.sort_values("week").reset_index(drop=True)
        X = np.asarray(patsy.dmatrix(f"1 + {controls} + {instrument}", g))
        k = X.shape[1] - 1  # the instrument is the last column
        xtx_inv = np.linalg.pinv(X.T @ X)
        w = (xtx_inv @ X.T)[k]  # gamma = w @ y
        hat = X @ (xtx_inv @ X.T)
        q, lp = g["log_q"].to_numpy(), g["log_p"].to_numpy()
        uq, up = q - hat @ q, lp - hat @ lp
        gamma = w @ q - grid * (w @ lp)
        resid = uq[:, None] - grid[None, :] * up[:, None]
        var = _nw_var(w[:, None] * resid)
        keep = gamma**2 <= CHI2_1_90 * var
        if keep.any():
            lo, hi = grid[keep].min(), grid[keep].max()
            bounded = not (keep[0] or keep[-1])
            gaps = bool((np.diff(np.flatnonzero(keep)) > 1).any())
        else:
            lo = hi = np.nan
            bounded, gaps = True, False
        rows.append(
            {
                "product": p,
                "ar_lo90": float(lo) if bounded or not keep[0] else -np.inf,
                "ar_hi90": float(hi) if bounded or not keep[-1] else np.inf,
                "ar_bounded": bool(bounded),
                "ar_disjoint": gaps,
            }
        )
    return pd.DataFrame(rows)
