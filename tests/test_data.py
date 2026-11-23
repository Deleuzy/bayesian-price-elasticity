"""Tidy-panel rules on a hand-made movement table, no Kilts download needed."""

import numpy as np
import pandas as pd

from elasticity.data import tidy


def _movement():
    # one product, three stores, one week: prices 2, 4, 6 at 50% margin
    return pd.DataFrame(
        {
            "STORE": [1, 2, 3],
            "UPC": [100, 100, 100],
            "WEEK": [1, 1, 1],
            "MOVE": [10, 20, 30],
            "QTY": [1, 1, 1],
            "PRICE": [2.0, 4.0, 6.0],
            "SALE": [np.nan, "B", np.nan],
            "PROFIT": [50.0, 50.0, 50.0],
        }
    )


def _products():
    return pd.DataFrame({"UPC": [100], "label": ["Test cheese 8 oz"]})


def test_cost_is_price_times_one_minus_margin():
    out = tidy(_movement(), _products()).sort_values("store")
    np.testing.assert_allclose(out["cost"].to_numpy(), [1.0, 2.0, 3.0])


def test_deal_flag_is_set_only_where_sale_is_present():
    out = tidy(_movement(), _products()).sort_values("store")
    assert out["deal"].tolist() == [0.0, 1.0, 0.0]


def test_hausman_is_leave_one_out_mean_of_other_stores():
    out = tidy(_movement(), _products()).sort_values("store").reset_index(drop=True)
    # store 1 sees the mean log price of stores 2 and 3
    assert np.isclose(out.loc[0, "hausman"], (np.log(4.0) + np.log(6.0)) / 2)
    assert np.isclose(out.loc[2, "hausman"], (np.log(2.0) + np.log(4.0)) / 2)


def test_week_one_starts_on_the_manual_date():
    out = tidy(_movement(), _products())
    assert set(out["date"]) == {pd.Timestamp("1989-09-14")}


def test_rows_without_a_second_store_are_dropped():
    mv = _movement().iloc[[0]]
    out = tidy(mv, _products())
    assert len(out) == 0
