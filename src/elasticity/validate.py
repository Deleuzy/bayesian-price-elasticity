"""Out-of-sample forecasts and metrics on a time-based holdout.

Every forecaster returns, per holdout row, a point forecast of units (median)
and a central 90% interval, so the same metrics apply to all of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .classical import CONTROLS_RHS


def metrics(test: pd.DataFrame, point: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    y = test["units"].to_numpy()
    err = np.abs(point - y)
    pos = y > 0  # MAPE is undefined for zero sales; average over positive rows only
    return {
        "mae": float(err.mean()),
        "wape": float(err.sum() / y.sum()),
        "mape": float((err[pos] / y[pos]).mean()) if pos.any() else float("nan"),
        "coverage90": float(((y >= lo) & (y <= hi)).mean()),
        "rows": int(len(y)),
    }


def baseline(train: pd.DataFrame, test: pd.DataFrame, recent: int = 13, band: int = 52):
    """Per product (and store, if present): median units of the last `recent` training weeks,
    interval from the 5% and 95% quantiles of the last `band` weeks. Ignores price."""
    last = train["week"].max()
    key = ["product", "store"] if "store" in train else ["product"]
    r = train[train["week"] > last - recent].groupby(key, observed=True)["units"].median()
    b = train[train["week"] > last - band].groupby(key, observed=True)["units"]
    q = pd.concat(
        [r.rename("point"), b.quantile(0.05).rename("lo"), b.quantile(0.95).rename("hi")], axis=1
    )
    m = test[key].merge(q.reset_index(), on=key, how="left")
    fallback = train["units"].median()
    return (
        m["point"].fillna(fallback).to_numpy(),
        m["lo"].fillna(train["units"].quantile(0.05)).to_numpy(),
        m["hi"].fillna(train["units"].quantile(0.95)).to_numpy(),
    )


def ols_forecast(train: pd.DataFrame, test: pd.DataFrame, controls: bool):
    """Per product log-log OLS. The 90% interval is the OLS prediction interval for a new
    observation on the log scale, so it includes parameter uncertainty as well as the
    residual scale (the same information the Bayesian predictive intervals carry)."""
    point = np.empty(len(test))
    lo, hi = np.empty(len(test)), np.empty(len(test))
    test = test.reset_index(drop=True)
    for p, g in train.groupby("product", observed=True):
        rhs = "log_p"
        if controls:
            rhs += f" + {CONTROLS_RHS}" + (
                " + C(store)" if "store" in g and g["store"].nunique() > 1 else ""
            )
        r = smf.ols(f"log_q ~ {rhs}", g).fit()
        idx = np.flatnonzero((test["product"] == p).to_numpy())
        new = test.iloc[idx].copy()
        # future 13-week blocks are unknown: carry the last training block forward
        new["quarter"] = new["quarter"].clip(upper=g["quarter"].max())
        pred = r.get_prediction(new).summary_frame(alpha=0.10)
        point[idx], lo[idx], hi[idx] = (
            np.exp(pred["mean"].to_numpy()),
            np.exp(pred["obs_ci_lower"].to_numpy()),
            np.exp(pred["obs_ci_upper"].to_numpy()),
        )
    return point, lo, hi


def from_draws(log_draws: np.ndarray):
    units = np.exp(log_draws)
    return (
        np.median(units, axis=0),
        np.quantile(units, 0.05, axis=0),
        np.quantile(units, 0.95, axis=0),
    )
