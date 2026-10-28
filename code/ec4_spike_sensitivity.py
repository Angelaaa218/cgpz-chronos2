"""Section EC.4: compare the Prophet-only random forest and Chronos predictions.

Reports the full test set and exclusion of the SKU 25 spike observation.
Exclusions are post-hoc sensitivity checks, not the primary evaluation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent


def main():
    sales = pd.read_csv(ROOT / 'data/data_processed.csv', parse_dates=['week'])
    teacher = pd.read_csv(ROOT / 'outputs/prophet_teacher.csv', parse_dates=['week'])
    sales = sales.merge(teacher, on=['sku', 'week'], validate='one_to_one')
    rows = []
    for sku, group in sales.groupby('sku', sort=True):
        group = group.sort_values('week').reset_index(drop=True)
        model = RandomForestRegressor(max_features=1, max_depth=8, random_state=0)
        model.fit(group[['prophet_yhat']].iloc[:68], group.weekly_sales.iloc[:68])
        test = group.iloc[68:][['sku', 'week', 'weekly_sales']].copy()
        test['random_forest'] = model.predict(group[['prophet_yhat']].iloc[68:])
        rows.append(test)
    results = pd.read_csv(ROOT / 'outputs/chronos.csv')
    selected = results[(results.run == 'FT 5 ep') & (results.stage == 'test')
                       & np.isclose(results.q, results.q_star)]
    assert len(selected) == 1
    q = str(float(selected.iloc[0].q))
    pred = pd.read_csv(ROOT / 'outputs/preds_prophet_yhat_small_ft.csv',
                       parse_dates=['timestamp'])
    pred = pred.rename(columns={'id': 'sku', 'timestamp': 'week', q: 'chronos'})
    test = pd.concat(rows).merge(pred[['sku', 'week', 'chronos']],
                                 on=['sku', 'week'], validate='one_to_one')
    assert len(test) == 44 * 30
    spike = (test.sku == 25) & (test.week == pd.Timestamp('2018-07-09'))
    assert spike.sum() == 1
    print('Spike observation:')
    print(test.loc[spike].to_string(index=False))
    out = []
    for label, mask in [('Full test set', np.ones(len(test), dtype=bool)),
                        ('Exclude SKU 25 on 2018-07-09', ~spike)]:
        part = test.loc[mask]
        y = part.weekly_sales.to_numpy()
        for model in ['random_forest', 'chronos']:
            p = part[model].to_numpy()
            out.append({'sample': label, 'model': model, 'n': len(y),
                        'R2': r2_score(y, p), 'WMAPE': np.abs(y-p).sum()/y.sum()})
    summary = pd.DataFrame(out)
    print(summary.to_string(index=False))
    summary.to_csv(ROOT / 'outputs/ec4_spike_sensitivity.csv', index=False)


if __name__ == "__main__":
    main()
