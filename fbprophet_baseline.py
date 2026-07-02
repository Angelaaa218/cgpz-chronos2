#!/usr/bin/env python
"""Standalone reproduction of the book's Prophet baseline on the CGPZ dataset.

Run on an x86-64 machine (Linux cluster or Intel Mac) where the original
fbprophet works. Per SKU, a Prophet model (single yearly-seasonality term, US
public holidays) is fit on the first 68 weeks of data_processed.csv, following
notebook 7. It prints out-of-sample R^2 and WMAPE on the 30-week test horizon
for two forecast constructions:

  book   -- the book's ``make_future_dataframe(periods, freq="W")`` horizon,
            whose weekly dates are Sunday-anchored and one day off the Monday
            data. This is what reproduces the book's numbers (~0.265).
  true   -- forecasting at the true observed test dates (~0.215).

The gap between the two is entirely the forecast-date construction: it appears
with fbprophet and with the maintained prophet package alike, so the book's
0.265 is a date artifact, not a package difference. The script prefers fbprophet
(the book's package); if it is not installed it falls back to prophet and says
so, so you can confirm the two give the same figures.

Environment (conda, x86-64 -- works natively on a Linux cluster):
    conda create -n fbp -c conda-forge python=3.8 fbprophet=0.7.1 \
        "holidays=0.11" pandas scikit-learn numpy -y
    conda activate fbp

Usage:
    python fbprophet_baseline.py --data data/data_processed.csv
"""
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
for _n in ("fbprophet", "prophet", "cmdstanpy", "pystan"):
    logging.getLogger(_n).setLevel(logging.ERROR)

try:
    from fbprophet import Prophet
    PACKAGE = "fbprophet"
except ImportError:
    from prophet import Prophet
    PACKAGE = "prophet"

TRAIN_W = 68  # the book's split on data_processed.csv (98 weeks per SKU)


def _scores(y, p):
    pc = np.clip(p, 0, None)  # the paper clips point forecasts at 0
    return r2_score(y, pc), float(np.abs(y - pc).sum() / y.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data_processed.csv",
                    help="path to data_processed.csv")
    args = ap.parse_args()

    sales = pd.read_csv(args.data)
    sales["week"] = pd.to_datetime(sales["week"])
    sales = sales.sort_values(["sku", "week"])

    y_test, book_pred, true_pred = [], [], []
    for _, g in sales.groupby("sku", sort=True):
        g = g.reset_index(drop=True)
        size = len(g) - TRAIN_W
        m = Prophet(yearly_seasonality=1)
        m.add_country_holidays(country_name="US")
        m.fit(pd.DataFrame({"ds": g["week"][:TRAIN_W], "y": g["weekly_sales"][:TRAIN_W]}))

        # book recipe: Sunday-anchored make_future_dataframe dates
        book = m.predict(m.make_future_dataframe(periods=size, freq="W"))["yhat"].to_numpy()
        # true dates: forecast at the actual (Monday) test weeks
        true = m.predict(pd.DataFrame({"ds": g["week"]}))["yhat"].to_numpy()

        y_test.append(g["weekly_sales"].to_numpy()[TRAIN_W:])
        book_pred.append(book[-size:])
        true_pred.append(true[TRAIN_W:])

    y = np.concatenate(y_test)
    book_r2, book_w = _scores(y, np.concatenate(book_pred))
    true_r2, true_w = _scores(y, np.concatenate(true_pred))

    print(f"package = {PACKAGE}")
    print(f"SKUs    = {sales['sku'].nunique()},  test weeks/SKU = {len(y_test[0])}")
    print(f"book recipe (make_future_dataframe): R2 = {book_r2:.4f}   WMAPE = {book_w:.4f}")
    print(f"true dates                         : R2 = {true_r2:.4f}   WMAPE = {true_w:.4f}")


if __name__ == "__main__":
    main()
