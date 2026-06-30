#!/usr/bin/env python
"""Standalone reproduction of the book's Prophet baseline on the CGPZ dataset.

Run on an x86-64 machine (Linux cluster or Intel Mac) where the original
fbprophet works. Prints out-of-sample R^2 and WMAPE on the 30-week test horizon,
matching the book's notebook 7 recipe: per-SKU Prophet with a single
yearly-seasonality term and US public holidays, fit on the first 68 weeks of
data_processed.csv.

The script prefers fbprophet (the book's package); if it is not installed it
falls back to the maintained prophet package and says so, so you can compare.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data_processed.csv",
                    help="path to data_processed.csv")
    args = ap.parse_args()

    sales = pd.read_csv(args.data)
    sales["week"] = pd.to_datetime(sales["week"])
    sales = sales.sort_values(["sku", "week"])

    y_test, y_pred = [], []
    for _, g in sales.groupby("sku", sort=True):
        g = g.reset_index(drop=True)
        train = pd.DataFrame({"ds": g["week"][:TRAIN_W], "y": g["weekly_sales"][:TRAIN_W]})
        m = Prophet(yearly_seasonality=1)
        m.add_country_holidays(country_name="US")
        m.fit(train)
        yhat = m.predict(pd.DataFrame({"ds": g["week"]}))["yhat"].to_numpy()
        y_test.append(g["weekly_sales"].to_numpy()[TRAIN_W:])
        y_pred.append(yhat[TRAIN_W:])

    y = np.concatenate(y_test)
    p = np.concatenate(y_pred)
    pc = np.clip(p, 0, None)  # the paper clips point forecasts at 0

    print(f"package        = {PACKAGE}")
    print(f"SKUs           = {sales['sku'].nunique()},  test weeks/SKU = {len(y_test[0])}")
    print(f"R2    (raw)    = {r2_score(y, p):.4f}     WMAPE (raw)    = {np.abs(y - p).sum() / y.sum():.4f}")
    print(f"R2    (clip0)  = {r2_score(y, pc):.4f}     WMAPE (clip0)  = {np.abs(y - pc).sum() / y.sum():.4f}")


if __name__ == "__main__":
    main()
