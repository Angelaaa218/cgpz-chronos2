"""Section EC.4: same-covariate baseline comparisons in Table EC.5.

Refit the five book baseline families on price only, Prophet forecast only,
and the original 45 features plus the Prophet forecast. Also retain the
45-feature reference. Use the processed data's 68/30 split and score raw,
unclipped predictions.

Chronos reference rows come from the included per-quantile CSVs at the
validation-selected quantile. Regenerate them with reproduce.py chronos.
All inputs and outputs are local to this package; see README for commands.
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

from .data import load_cgpz
from .prophet_teacher import add_teacher_covariate

ROOT = Path(__file__).resolve().parent.parent

TRAIN_W = 68                       # the book's split on data_processed.csv
PROPHET = "prophet_yhat"

def load_panel():
    """Load the processed data and the fixed week-70 Prophet covariate."""
    cache = ROOT / "outputs" / "prophet_teacher.csv"
    if not cache.exists():
        add_teacher_covariate(load_cgpz()[0])
    sales = pd.read_csv(ROOT / "data" / "data_processed.csv", parse_dates=["week"])
    teacher = pd.read_csv(cache, parse_dates=["week"])
    sales = sales.merge(teacher, on=["sku", "week"], how="left", validate="one_to_one")
    sales = sales.sort_values(["sku", "week"]).reset_index(drop=True)
    if sales[PROPHET].isna().any():
        raise ValueError("Prophet covariate has missing values on the processed grid")
    features = [c for c in sales if c not in ("week", "sku", "weekly_sales", PROPHET)]
    if len(features) != 45:
        raise ValueError(f"Expected 45 features, found {len(features)}")
    panel = {sku: group.reset_index(drop=True) for sku, group in sales.groupby("sku", sort=False)}
    return panel, {
        "full45": features,
        "price_only": ["price"],
        "full45+yhat": features + [PROPHET],
        "yhat_only": [PROPHET],
    }


def _split(panel, s, feats):
    g = panel[s]
    X, y = g[feats].to_numpy(float), g["weekly_sales"].to_numpy(float)
    return X[:TRAIN_W], y[:TRAIN_W], X[TRAIN_W:], y[TRAIN_W:]


def decentralized(panel, factory, feats):
    yt, yp = [], []
    for s in panel:
        Xtr, ytr, Xte, yte = _split(panel, s, feats)
        yp.append(factory().fit(Xtr, ytr).predict(Xte))
        yt.append(yte)
    return map(np.concatenate, (yt, yp))


def centralized(panel, factory, feats):
    parts = [_split(panel, s, feats) for s in panel]
    m = factory().fit(np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
    Xte, yte = np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts])
    return yte, m.predict(Xte)


def kmeans(panel, feats):
    """Notebook 5: cluster SKUs on the standardized [mean, std] of (price,
    sales) over the train window, one LinearRegression per cluster, cluster
    count chosen on a 48/20 sub-split of the 68 training weeks."""
    skus = list(panel)

    def profile(lo, hi):
        rows = []
        for s in panel:
            g = panel[s]
            pr, sa = g["price"].to_numpy(float)[lo:hi], g["weekly_sales"].to_numpy(float)[lo:hi]
            rows.append([pr.mean(), sa.mean(), pr.std(), sa.std()])
        return StandardScaler().fit_transform(np.array(rows))

    def fit(z, tr_lo, tr_hi, te_lo, te_hi):
        labels = KMeans(n_clusters=z, n_init=10, random_state=0).fit_predict(profile(tr_lo, tr_hi))
        yt, yp = [], []
        for c in range(z):
            members = [skus[k] for k in range(len(skus)) if labels[k] == c]
            if not members:
                continue
            X = lambda s, a, b: panel[s][feats].to_numpy(float)[a:b]
            Y = lambda s, a, b: panel[s]["weekly_sales"].to_numpy(float)[a:b]
            m = LinearRegression().fit(np.concatenate([X(s, tr_lo, tr_hi) for s in members]),
                                       np.concatenate([Y(s, tr_lo, tr_hi) for s in members]))
            for s in members:
                yp.append(m.predict(X(s, te_lo, te_hi)))
                yt.append(Y(s, te_lo, te_hi))
        yt, yp = map(np.concatenate, (yt, yp))
        return float(r2_score(yt, yp)), (yt, yp)

    best_z = max(range(2, 15), key=lambda z: fit(z, 0, 48, 48, TRAIN_W)[0])
    return fit(best_z, 0, TRAIN_W, TRAIN_W, len(panel[skus[0]]))[1], best_z


def score(y, p):
    out = {"R2": float(r2_score(y, p)), "WMAPE": float(np.abs(y - p).sum() / y.sum())}
    return out


def main():
    panel, feature_sets = load_panel()
    families = [
        ("Decentralized Elastic Net (traditional)",
         lambda f: decentralized(panel, lambda: ElasticNet(alpha=0.05, l1_ratio=0.3), f)),
        ("Centralized OLS (traditional)", lambda f: centralized(panel, LinearRegression, f)),
        ("Decentralized Random Forest (tree-based)",
         lambda f: decentralized(panel, lambda: RandomForestRegressor(max_features=min(44, len(f)),
                                                               max_depth=8, random_state=0), f)),
        ("Centralized Random Forest (tree-based)",
         lambda f: centralized(panel, lambda: RandomForestRegressor(max_features=min(31, len(f)),
                                                             max_depth=4, random_state=0), f)),
        ("K-means clustering", lambda f: kmeans(panel, f)[0]),
    ]

    rows = []
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")            # pooled OLS is ill-conditioned by design
        for fs, feats in feature_sets.items():
            for name, run in families:
                y, p = run(feats)
                rows.append({"feature_set": fs, "n_feat": len(feats), "Method": name, **score(y, p)})
                r = rows[-1]
                print(f"{fs:12} {len(feats):>2} {name:42} R2={r['R2']:7.3f} W={r['WMAPE']:6.3f}")

    # --- reference rows: the teacher, and Chronos-2-small on the same covariate ---
    yt = np.concatenate([panel[s]["weekly_sales"].to_numpy(float)[TRAIN_W:] for s in panel])
    yp = np.concatenate([panel[s][PROPHET].to_numpy(float)[TRAIN_W:] for s in panel])
    rows.append({"feature_set": "reference", "n_feat": 1, "Method": "Prophet yhat standalone", **score(yt, yp)})
    print(f"\nreference    1 Prophet yhat standalone{'':19} R2={rows[-1]['R2']:7.3f} W={rows[-1]['WMAPE']:6.3f}")

    # Add the saved, validation-selected Chronos reference rows.
    for filename, feature_set, run_name in [
        ("chronos_price.csv", "price_only", "zero-shot"),
        ("chronos.csv", "yhat_only", "zero-shot"),
        ("chronos.csv", "yhat_only", "FT 5 ep"),
    ]:
        path = ROOT / "outputs" / filename
        if not path.exists():
            raise FileNotFoundError(f"{path}: regenerate with reproduce.py chronos; see README")
        ch = pd.read_csv(path)
        selected = ch[(ch["stage"] == "test") & (ch["run"] == run_name)
                      & np.isclose(ch["q"], ch["q_star"])]
        assert len(selected) == 1, f"Expected one selected row: {filename}, {run_name}"
        r = selected.iloc[0]
        rows.append({"feature_set": feature_set, "n_feat": 1,
                     "Method": "Chronos-2 " + run_name, "R2": r.R2,
                     "WMAPE": r.WMAPE})

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "ec4_prophet_featuresets.csv", index=False)
    print(out.to_string(index=False))
    print("Saved outputs/ec4_prophet_featuresets.csv")


if __name__ == "__main__":
    main()
