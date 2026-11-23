"""Time features shared by the real and the simulated panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

ORIGIN = pd.Timestamp("1989-09-14")
CONTROLS = ["deal", "quarter", "s1", "c1", "s2", "c2"]


def add_time_features(d: pd.DataFrame) -> pd.DataFrame:
    """Years since week 1, 13-week block index, and two yearly Fourier harmonics."""
    d = d.copy()
    days = (d["date"] - ORIGIN).dt.days.to_numpy()
    d["t"] = days / 365.25
    d["quarter"] = np.floor(d["t"] * 4).astype(int)  # 13-week blocks since week 1
    angle = 2 * np.pi * d["date"].dt.day_of_year.to_numpy() / 365.25
    d["s1"], d["c1"] = np.sin(angle), np.cos(angle)
    d["s2"], d["c2"] = np.sin(2 * angle), np.cos(2 * angle)
    return d


def holdout_cut(d: pd.DataFrame, holdout_weeks: int = 52) -> int:
    """First week of the test set: the last `holdout_weeks` distinct weeks are held out."""
    weeks = np.sort(d["week"].unique())
    if len(weeks) <= holdout_weeks:
        raise ValueError(f"{len(weeks)} weeks is too few for a {holdout_weeks}-week holdout")
    return int(weeks[-holdout_weeks])


def time_split(d: pd.DataFrame, holdout_weeks: int = 52) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last `holdout_weeks` distinct weeks are the test set."""
    cut = holdout_cut(d, holdout_weeks)
    return d[d["week"] < cut].copy(), d[d["week"] >= cut].copy()


def aggregate(d: pd.DataFrame, before_week: int | None = None) -> pd.DataFrame:
    """Store x week rows to chain-level product x week rows.

    units: mean units per reporting store (so store openings do not look like
    demand); log_p and log_cost: mean over stores; deal: share of stores with a
    deal flag. Product-weeks where fewer than 80% of the usual number of stores
    reported are dropped as data gaps. The confounding demand shock and most price
    moves are chainwide, so this is the grain at which price endogeneity lives (and the grain a
    manufacturer sees in retailer sell-out data).

    With `before_week`, the usual number of stores is the median over weeks before
    it only (the training weeks), so the holdout does not shape the filter.
    """
    g = d.groupby(["product", "week"], observed=True)
    a = g.agg(
        units=("units", "mean"),
        log_p=("log_p", "mean"),
        log_cost=("log_cost", "mean"),
        deal=("deal", "mean"),
        date=("date", "first"),
        n_stores=("units", "size"),
    ).reset_index()
    # weeks where only a fraction of stores reported are data gaps, not demand
    ref = a if before_week is None else a[a["week"] < before_week]
    usual = ref.groupby("product", observed=True)["n_stores"].median()
    full = a["product"].map(usual).astype(float)
    a = a[a["n_stores"] >= 0.8 * full].reset_index(drop=True)
    a["log_q"] = np.log(a["units"])
    a["price"] = np.exp(a["log_p"])
    a["cost"] = np.exp(a["log_cost"])
    return add_time_features(a)
