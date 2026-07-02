"""Prophet models used in the paper.

Two distinct uses:

* ``add_teacher_covariate`` fits a per-SKU Prophet with the listed price and
  its two lags as regressors (no holidays) and exposes Prophet's point
  forecast ``yhat`` as a future-known covariate for Chronos-2. This is the
  distilled covariate behind Table 10.
* ``book_prophet_baselines`` reproduces the book's own Prophet baseline
  (per-SKU Prophet with US public holidays), both standalone and as an extra
  feature fed to the decentralized Elastic Net, as reported in Table 8.

Prophet's own logging is silenced so the reproduction output stays readable.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TRAIN_W

logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
os.environ.setdefault("STAN_NUM_THREADS", "1")

CACHE = Path(__file__).resolve().parent.parent / "outputs" / "prophet_teacher.csv"


def _r2_wape(y, yhat):
    from sklearn.metrics import r2_score
    yhat = np.clip(yhat, 0, None)
    return float(r2_score(y, yhat)), float(np.abs(y - yhat).sum() / y.sum())


def _fit_sku(weeks, sales, reg, train_w, us_holidays, skip=0):
    """Fit Prophet on weeks ``[skip, train_w)`` and return yhat over all weeks.

    The first ``skip`` weeks (whose price lags are undefined) are excluded from
    the fit and returned as NaN, so Chronos-2 reads them as missing rather than
    imputed. ``skip=0`` recovers the plain all-weeks fit.
    """
    from prophet import Prophet
    train = pd.DataFrame({"ds": weeks[skip:train_w], "y": sales[skip:train_w]})
    m = Prophet(yearly_seasonality=1)
    if us_holidays:
        m.add_country_holidays(country_name="US")
    if reg is not None:
        for c in reg.columns:
            m.add_regressor(c)
            train[c] = reg[c].to_numpy()[skip:train_w]
    m.fit(train)
    future = pd.DataFrame({"ds": weeks[skip:]})
    if reg is not None:
        for c in reg.columns:
            future[c] = reg[c].to_numpy()[skip:]
    yhat = m.predict(future)["yhat"].to_numpy()
    return np.concatenate([np.full(skip, np.nan), yhat])


def add_teacher_covariate(df, regressors=("price", "price-1", "price-2"),
                          train_w=TRAIN_W, use_cache=True):
    """Append a ``prophet_yhat`` column (per-SKU Prophet forecast) to ``df``.

    The two price-lag regressors are undefined for the first two weeks, so those
    weeks are dropped from the Prophet fit and their ``prophet_yhat`` is left
    missing (NaN); Chronos-2 reads them as missing covariates.
    """
    if "prophet_yhat" in df.columns:
        return df
    if use_cache and CACHE.exists():
        cache = pd.read_csv(CACHE, parse_dates=["week"])
        return df.merge(cache, on=["sku", "week"], how="left", validate="one_to_one")

    skip = max((int(c.rsplit("-", 1)[1]) for c in regressors if c[-1].isdigit()), default=0)
    out = []
    for sku, g in df.groupby("sku", sort=True):
        g = g.sort_values("week").reset_index(drop=True)
        yhat = _fit_sku(g["week"].to_numpy(), g["weekly_sales"].to_numpy(),
                        g[list(regressors)], train_w, us_holidays=False, skip=skip)
        out.append(pd.DataFrame({"sku": sku, "week": g["week"].values, "prophet_yhat": yhat}))
    cache = pd.concat(out, ignore_index=True)
    CACHE.parent.mkdir(exist_ok=True)
    cache.to_csv(CACHE, index=False)
    return df.merge(cache, on=["sku", "week"], how="left", validate="one_to_one")


def teacher_standalone(df, train_w=TRAIN_W):
    """Out-of-sample R^2 / WMAPE of the teacher Prophet forecast on the test horizon."""
    d = df if "prophet_yhat" in df.columns else add_teacher_covariate(df, train_w=train_w)
    y, yhat = [], []
    for _, g in d.groupby("sku", sort=True):
        g = g.sort_values("week")
        y.append(g["weekly_sales"].to_numpy()[train_w:])
        yhat.append(g["prophet_yhat"].to_numpy()[train_w:])
    return _r2_wape(np.concatenate(y), np.concatenate(yhat))


def book_prophet_baselines(train_w=68):
    """The book's Prophet baseline, transcribed from notebook 7.

    Per SKU, a Prophet model (``yearly_seasonality=1``, US public holidays) is
    fit on the first 68 weeks of ``data_processed.csv``; the forecast horizon is
    built with the book's ``make_future_dataframe(periods, freq="W")`` and read
    as ``yhat[-size:]``. It is scored standalone, and as one extra feature fed to
    the decentralized Elastic Net (``l1_ratio=0.7``) after dropping the trend and
    month dummies, exactly as in the notebook. R^2 is on raw predictions, the
    book's metric.

    The ``make_future_dataframe`` path is what yields the book's numbers (0.265
    standalone, 0.568 with Elastic Net): it generates Sunday-anchored forecast
    dates one day off the Monday data, shifting Prophet's seasonal/holiday
    effects. This reproduces with any Prophet version, so the figures come from
    the date construction, not the package. (Predicting at the true test dates
    instead gives 0.215 / 0.565.)
    """
    from prophet import Prophet
    from sklearn.linear_model import ElasticNet
    from sklearn.metrics import r2_score

    sales = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "data_processed.csv",
                        parse_dates=["week"]).sort_values(["sku", "week"])
    drop = ["trend"] + [f"month_{m}" for m in range(2, 13)]
    feat = [c for c in sales.columns if c not in ["week", "weekly_sales", "sku"] + drop]

    def _r2_w(y, p):
        return float(r2_score(y, p)), float(np.abs(y - p).sum() / y.sum())

    y_test, yhat_std, en_pred = [], [], []
    for _, g in sales.groupby("sku", sort=True):
        g = g.reset_index(drop=True)
        size = len(g) - train_w
        m = Prophet(yearly_seasonality=1)
        m.add_country_holidays(country_name="US")
        m.fit(pd.DataFrame({"ds": g["week"][:train_w], "y": g["weekly_sales"][:train_w]}))
        future = m.make_future_dataframe(periods=size, freq="W")
        yhat_all = m.predict(future)["yhat"].to_numpy()      # 68 history + size future
        y = g["weekly_sales"].to_numpy()
        y_test.append(y[train_w:])
        yhat_std.append(yhat_all[-size:])
        X = np.column_stack([g[feat].to_numpy(), yhat_all])  # book drops trend/months, adds Prophet
        en = ElasticNet(alpha=0.05, l1_ratio=0.7).fit(X[:train_w], y[:train_w])
        en_pred.append(en.predict(X[train_w:]))
    y = np.concatenate(y_test)
    standalone = _r2_w(y, np.concatenate(yhat_std))
    downstream = _r2_w(y, np.concatenate(en_pred))
    return pd.DataFrame([
        {"Method": "Prophet (book recipe), standalone", "R2": standalone[0], "WMAPE": standalone[1]},
        {"Method": "Prophet + Decentralized Elastic Net", "R2": downstream[0], "WMAPE": downstream[1]},
    ])
