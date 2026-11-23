"""Holdout metrics on a hand-computed example."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from elasticity.features import holdout_cut, time_split
from elasticity.validate import metrics, ols_forecast


def _test_frame():
    return pd.DataFrame({"product": ["A", "A"], "week": [1, 2], "units": [10.0, 20.0]})


def test_mae_wape_and_mape_by_hand():
    m = metrics(_test_frame(), np.array([12.0, 15.0]), np.array([8, 14]), np.array([11, 30]))
    # errors 2 and 5
    assert np.isclose(m["mae"], 3.5)
    assert np.isclose(m["wape"], 7 / 30)
    assert np.isclose(m["mape"], (0.2 + 0.25) / 2)


def test_mape_skips_zero_sales_weeks():
    t = pd.DataFrame({"product": ["A", "A"], "week": [1, 2], "units": [10.0, 0.0]})
    m = metrics(t, np.array([12.0, 3.0]), np.array([0, 0]), np.array([20, 20]))
    assert np.isfinite(m["mape"])
    assert np.isclose(m["mape"], 0.2)
    assert np.isclose(m["mae"], 2.5)  # zero-sales weeks still count for MAE


def test_ols_interval_includes_parameter_uncertainty():
    rng = np.random.default_rng(0)
    n = 30
    log_p = rng.normal(0, 0.1, n + 5)
    log_q = 3 - 2 * log_p + rng.normal(0, 0.2, n + 5)
    d = pd.DataFrame({"product": "A", "log_p": log_p, "log_q": log_q, "quarter": 0})
    train, test = d.iloc[:n], d.iloc[n:].copy()
    test.loc[test.index[-1], "log_p"] = 1.0  # far outside the training prices
    point, lo, hi = ols_forecast(train, test, controls=False)
    r = smf.ols("log_q ~ log_p", train).fit()
    residual_only = 2 * 1.6449 * np.sqrt(r.scale)
    width = np.log(hi) - np.log(lo)
    assert np.all(width > residual_only)
    # extrapolating far from the training prices widens the interval
    assert width[-1] > width[0] * 1.5
    assert np.allclose(np.log(point), r.predict(test))


def test_holdout_cut_is_shared_by_split():
    d = pd.DataFrame({"week": np.repeat(np.arange(1, 11), 2)})
    assert holdout_cut(d, 3) == 8
    train, test = time_split(d, 3)
    assert train["week"].max() == 7 and test["week"].min() == 8
    with pytest.raises(ValueError):
        holdout_cut(d, 10)


def test_coverage_counts_units_inside_the_interval():
    inside = metrics(_test_frame(), np.array([12.0, 15.0]), np.array([8, 14]), np.array([11, 30]))
    assert inside["coverage90"] == 1.0
    half = metrics(_test_frame(), np.array([12.0, 15.0]), np.array([8, 21]), np.array([11, 30]))
    assert half["coverage90"] == 0.5
