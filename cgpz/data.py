"""CGPZ dataset loading, feature engineering, and Chronos-2 windowing.

The CGPZ dataset (Cohen, Gras, Pentecoste, Zhang; the demand-prediction
textbook of Cohen 2022) holds 100 weeks of weekly sales for 44 tech-gadget
SKUs. ``data/data_raw.csv`` has the raw columns; ``load_cgpz`` reconstructs
the book's 45 engineered features from the raw data so that all 100 weeks are
available (the book's processed file drops the first weeks used for lags).

Evaluation protocol (matches the paper): first 70 weeks train, last 30 test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "data_raw.csv"

TOTAL_W, TEST_W = 100, 30
TRAIN_W = TOTAL_W - TEST_W                 # 70
VAL_CTX_W = TRAIN_W - TEST_W               # 40  (validation context length)

# Point-forecast quantile grid used to select q* and to score the tables
# (0.05..0.95). This is narrower than the quantile levels Chronos-2 optimizes
# internally during fine-tuning; see cgpz/chronos.py.
QUANTILES = [round(0.05 * k, 2) for k in range(1, 20)]


def load_cgpz() -> tuple[pd.DataFrame, list[str]]:
    """Return ``(df, all_cov_cols)``.

    ``df`` is sorted by (sku, week) with the book's engineered features added:
    two price lags, month one-hots, a year-trend flag, and static one-hots for
    color, vendor, and functionality (book naming and reference levels).
    ``all_cov_cols`` is the full 45-feature stack: every predictive column, i.e.
    all columns except the identifiers ``week``, ``sku`` and the target
    ``weekly_sales``. The paper's headline Chronos-2 run uses ``price`` alone.
    """
    df = pd.read_csv(DATA_RAW)
    df["week"] = pd.to_datetime(df["week"])
    df = df.sort_values(["sku", "week"]).reset_index(drop=True)
    df["feat_main_page"] = df["feat_main_page"].astype(int)
    df["price"] = df["price"].astype(float)

    df["price-1"] = df.groupby("sku")["price"].shift(1).fillna(df["price"])
    df["price-2"] = df.groupby("sku")["price"].shift(2).fillna(df["price"])

    df["month"] = df["week"].dt.month
    for m in range(2, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)

    df["year"] = df["week"].dt.year
    first_year = df.groupby("sku")["year"].transform("min")
    df["trend"] = (df["year"] > first_year).astype(int)

    color = pd.get_dummies(df["color"], prefix="color").astype(int)
    color = color.drop(columns=["color_black"], errors="ignore")  # reference level
    vendor = pd.get_dummies(df["vendor"].astype(int), prefix="vendor", drop_first=True).astype(int)
    func = pd.get_dummies(df["functionality"], prefix="functionality", drop_first=True).astype(int)

    df = pd.concat([df, color, vendor, func], axis=1)
    static_cov = list(color.columns) + list(vendor.columns) + list(func.columns)
    dynamic_cov = ["price", "price-1", "price-2", "feat_main_page", "trend"] + \
        [f"month_{m}" for m in range(2, 13)]
    return df, dynamic_cov + static_cov


def build_window(df, cov_cols, start_w, ctx_len, horizon):
    """Slice every SKU into a Chronos-2 context/future/test triple.

    Returns ``(context_df, future_df, test_df)``. ``context_df`` carries
    ``[id, timestamp, target, <cov>]`` for ``ctx_len`` weeks from ``start_w``;
    ``future_df`` carries ``[id, timestamp, <cov>]`` for the next ``horizon``
    weeks (the covariates are future-known); ``test_df`` carries the matching
    ground-truth ``y_true``. ``future_df`` is ``None`` when ``cov_cols`` is empty.
    """
    ctx, fut, test = [], [], []
    for sku, g in df.groupby("sku", sort=True):
        g = g.sort_values("week").reset_index(drop=True)
        c = g.iloc[start_w:start_w + ctx_len]
        f = g.iloc[start_w + ctx_len:start_w + ctx_len + horizon]
        sid = str(sku)
        crow = {"id": sid, "timestamp": c["week"].values,
                "target": c["weekly_sales"].astype(float).values}
        frow = {"id": sid, "timestamp": f["week"].values}
        for cc in cov_cols:
            crow[cc] = c[cc].astype(float).values
            frow[cc] = f[cc].astype(float).values
        ctx.append(pd.DataFrame(crow))
        fut.append(pd.DataFrame(frow))
        test.append(pd.DataFrame({"id": sid, "sku": sku,
                                  "timestamp": f["week"].values,
                                  "y_true": f["weekly_sales"].astype(float).values}))
    context_df = pd.concat(ctx, ignore_index=True)
    future_df = pd.concat(fut, ignore_index=True) if cov_cols else None
    return context_df, future_df, pd.concat(test, ignore_index=True)


def fit_inputs(df, cov_cols, start_w, n_weeks):
    """List-of-dicts for ``Chronos2Pipeline.fit`` over weeks [start_w, start_w+n_weeks)."""
    out = []
    for _, g in df.groupby("sku", sort=True):
        g = g.sort_values("week").reset_index(drop=True)
        tr = g.iloc[start_w:start_w + n_weeks]
        task = {"target": tr["weekly_sales"].to_numpy(dtype=np.float32).reshape(1, -1)}
        if cov_cols:
            task["past_covariates"] = {c: tr[c].to_numpy(dtype=np.float32) for c in cov_cols}
            task["future_covariates"] = {c: np.zeros(0, dtype=np.float32) for c in cov_cols}
        out.append(task)
    return out
