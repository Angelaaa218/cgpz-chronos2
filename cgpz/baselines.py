"""CGPZ baselines, transcribed from the companion notebooks of Cohen et al.'s
demand-prediction textbook (the public ``public-main`` repository: notebook 3
"Common Demand Prediction Methods", notebook 4 "Tree-Based Methods", and
notebook 5 "Clustering Techniques").

The data file (``data_processed.csv``), the per-SKU feature set, the 68/30
split (``np.split([68])``), the model choices, and every hyperparameter are
taken directly from those notebooks; only the surrounding loop is reorganized so
the baselines can be called as functions. Reproduces the per-family numbers in
the book's Figure 6.1 (Table 8 of the paper). Metric: out-of-sample R^2 (the
book's metric), with WMAPE added.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parent.parent / "data" / "data_processed.csv"


def _sku_data():
    """Build per-SKU X/y dictionaries split at week 68 (notebooks 3-4)."""
    sales = pd.read_csv(DATA)
    colnames = [c for c in sales.columns if c not in ["week", "weekly_sales", "sku"]]
    skuSet = list(sales.sku.unique())
    X_dict, y_dict = {}, {}
    for i in skuSet:
        df_i = sales[sales.sku == i]
        X_train_i, X_test_i = np.split(df_i[colnames].values, [68])
        y_train_i, y_test_i = np.split(df_i.weekly_sales.values, [68])
        X_dict[i] = {"train": X_train_i, "test": X_test_i}
        y_dict[i] = {"train": y_train_i, "test": y_test_i}
    return skuSet, X_dict, y_dict, colnames


def _score(y, yhat):
    return {"R2": float(r2_score(y, yhat)),
            "WMAPE": float(np.abs(y - yhat).sum() / y.sum())}


def _decentralized(model_factory):
    """One model fit per SKU (notebook 3/4 decentralized loop)."""
    skuSet, X_dict, y_dict, _ = _sku_data()
    y_test, y_pred = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in skuSet:
            model_i = model_factory().fit(X_dict[i]["train"], y_dict[i]["train"])
            y_pred += list(model_i.predict(X_dict[i]["test"]))
            y_test += list(y_dict[i]["test"])
    return np.array(y_test), np.array(y_pred)


def _centralized(model_factory):
    """One model pooled across all SKUs (notebook 3/4 centralized loop)."""
    skuSet, X_dict, y_dict, _ = _sku_data()
    X_cen_train = np.concatenate([X_dict[i]["train"] for i in skuSet], axis=0)
    y_train = np.concatenate([y_dict[i]["train"] for i in skuSet], axis=0)
    X_cen_test = np.concatenate([X_dict[i]["test"] for i in skuSet], axis=0)
    y_test = np.concatenate([y_dict[i]["test"] for i in skuSet], axis=0)
    model_cen = model_factory().fit(X_cen_train, y_train)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")  # pooled OLS is ill-conditioned by design
        return y_test, model_cen.predict(X_cen_test)


def _kmeans():
    """Notebook 5 K-means: cluster SKUs on the standardized [mean, std] of
    (price, weekly sales) over the train window, fit one LinearRegression per
    cluster. The cluster count is chosen by validation on a 48/20 sub-split of
    the 68 training weeks, then refit on the full 68 and scored on the 30-week
    test (the book selects 5)."""
    skuSet, X_dict, y_dict, colnames = _sku_data()
    p = colnames.index("price")

    def _features(getX, gety):
        rows = []
        for s in skuSet:
            pr, sa = getX(s)[:, p], gety(s)
            rows.append([pr.mean(), sa.mean(), pr.std(), sa.std()])
        return StandardScaler().fit_transform(np.array(rows))

    def _cluster_fit(z, getX, gety, getXte, getyte):
        labels = KMeans(n_clusters=z, n_init=10, random_state=0).fit_predict(_features(getX, gety))
        yte, yp = [], []
        for c in range(z):
            members = [skuSet[k] for k in range(len(skuSet)) if labels[k] == c]
            if not members:
                continue
            Xtr = np.concatenate([getX(s) for s in members], axis=0)
            ytr = np.concatenate([gety(s) for s in members], axis=0)
            m = LinearRegression().fit(Xtr, ytr)
            for s in members:
                yp += list(m.predict(getXte(s)))
                yte += list(getyte(s))
        return r2_score(yte, yp), (np.array(yte), np.array(yp))

    # validation sub-split inside the 68 train weeks: subtrain 48, validation 20
    sub = (lambda s: X_dict[s]["train"][:48], lambda s: y_dict[s]["train"][:48],
           lambda s: X_dict[s]["train"][48:], lambda s: y_dict[s]["train"][48:])
    best_z = max(range(2, 15), key=lambda z: _cluster_fit(z, *sub)[0])

    full = (lambda s: X_dict[s]["train"], lambda s: y_dict[s]["train"],
            lambda s: X_dict[s]["test"], lambda s: y_dict[s]["test"])
    return _cluster_fit(best_z, *full)[1]


def table8_baselines() -> pd.DataFrame:
    """Best baseline per family from the book's Figure 6.1, as in Table 8."""
    rows = []
    add = lambda name, yp: rows.append({"Method": name, **_score(*yp)})
    # The pooled OLS is ill-conditioned by design (the book reports R^2 = 0.114),
    # so its predictions overflow harmlessly; silence the numeric warnings.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        # Hyperparameters are the book's reported choices (notebooks 3-4).
        add("Decentralized Elastic Net (traditional)",
            _decentralized(lambda: ElasticNet(alpha=0.05, l1_ratio=0.3)))
        add("Centralized OLS (traditional)", _centralized(LinearRegression))
        add("Decentralized Random Forest (tree-based)",
            _decentralized(lambda: RandomForestRegressor(max_features=44, max_depth=8, random_state=0)))
        add("Centralized Random Forest (tree-based)",
            _centralized(lambda: RandomForestRegressor(max_features=31, max_depth=4, random_state=0)))
        add("K-means clustering", _kmeans())
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(table8_baselines().round(3).to_string(index=False))
