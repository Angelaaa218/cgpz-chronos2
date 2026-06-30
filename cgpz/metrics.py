"""Scoring and the two-stage, train-only quantile-selection protocol.

Metrics match the paper: out-of-sample ``R^2`` is the pooled
``sklearn.r2_score`` (global-mean baseline); ``WMAPE`` is
``sum|y - yhat| / sum(y)``. Point forecasts are clipped at 0.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score

from .data import QUANTILES


def score_at_q(pred_df, test_df, q):
    """R^2 and WMAPE for one quantile column ``q`` (preds clipped at 0)."""
    col = str(q)
    m = test_df.merge(pred_df[["id", "timestamp", col]].rename(columns={col: "yhat"}),
                      on=["id", "timestamp"], how="left", validate="one_to_one")
    y = m["y_true"].to_numpy()
    yhat = np.clip(m["yhat"].to_numpy(), 0, None)
    r2 = r2_score(y, yhat)
    wape = float(np.abs(y - yhat).sum() / y.sum())
    return r2, wape


def per_quantile_table(pred_df, test_df):
    """DataFrame ``[q, R2, WMAPE]`` over every native quantile."""
    import pandas as pd
    rows = [{"q": q, "R2": score_at_q(pred_df, test_df, q)[0],
             "WMAPE": score_at_q(pred_df, test_df, q)[1]} for q in QUANTILES]
    return pd.DataFrame(rows)


def best_q(pred_df, test_df, by="R2"):
    """Quantile that maximizes R^2 (``by="R2"``) or minimizes WMAPE (``by="WMAPE"``)."""
    best = None
    for q in QUANTILES:
        r2, wape = score_at_q(pred_df, test_df, q)
        val = r2 if by == "R2" else -wape
        if best is None or val > best[0]:
            best = (val, q)
    return best[1]
