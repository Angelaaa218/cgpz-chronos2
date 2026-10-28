"""Chronos-2 forecasting on CGPZ: zero-shot and fine-tuned.

A single forecast follows the two-stage quantile-selection protocol: a validation
stage (weeks 1-40 forecast weeks 41-70) selects the point-forecast quantile
``q*``; the test stage (weeks 1-70 forecast weeks 71-100) is scored at ``q*``.
Fine-tuned runs refit the model in each stage with the recipe in ``FT``.
Prophet inputs retain the week-70 fit during validation, as documented in README.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import QUANTILES, TEST_W, TRAIN_W, VAL_CTX_W, build_window, fit_inputs, load_cgpz
from .prophet_teacher import add_teacher_covariate
from .metrics import best_q, per_quantile_table

MODELS = {"small": "autogluon/chronos-2-small", "base": "autogluon/chronos-2"}

# Fine-tuning recipe. Matches the Alibaba study (Section 4 of the paper) except
# for the batch size (16 vs 256) and the warmup, which is dropped because the
# 15-step CGPZ budget is too short for a 100-step warmup.
FT = dict(
    finetune_mode="full", learning_rate=1e-5, batch_size=16, num_steps=15, min_past=10,
    weight_decay=0.01, max_grad_norm=1.0, lr_scheduler_type="cosine", warmup_steps=0,
)


def _pipeline(model, device):
    from chronos import Chronos2Pipeline
    return Chronos2Pipeline.from_pretrained(MODELS[model], device_map=device)


def _predict(pipe, ctx, fut, cross_learning):
    kw = dict(prediction_length=TEST_W, quantile_levels=QUANTILES, id_column="id",
              timestamp_column="timestamp", target="target",
              cross_learning=cross_learning, batch_size=64)
    return pipe.predict_df(ctx, future_df=fut, **kw) if fut is not None \
        else pipe.predict_df(ctx, **kw)


def _finetune(model, df, cov, start_w, n_weeks, device):
    import tempfile
    pipe = _pipeline(model, device)
    extra = {k: FT[k] for k in ("weight_decay", "max_grad_norm", "lr_scheduler_type", "warmup_steps")}
    with tempfile.TemporaryDirectory() as ckpt:
        return pipe.fit(
            fit_inputs(df, cov, start_w, n_weeks), prediction_length=TEST_W,
            finetune_mode=FT["finetune_mode"], learning_rate=FT["learning_rate"],
            num_steps=FT["num_steps"], batch_size=FT["batch_size"], min_past=FT["min_past"],
            output_dir=ckpt, remove_printer_callback=True, report_to=[], **extra,
        )


def forecast(df, cov, model="small", device="mps", finetune=False,
             cross_learning=False, future_known=True):
    """Run the two-stage protocol and return ``(per_quantile_df, q_star)``.

    ``cov`` is the covariate column list (empty for univariate). Set
    ``future_known=False`` to provide the covariate over history only (it stays
    in the context but is withheld from the forecast horizon).
    """
    val_ctx, val_fut, val_test = build_window(df, cov, 0, VAL_CTX_W, TEST_W)
    test_ctx, test_fut, test_df = build_window(df, cov, 0, TRAIN_W, TEST_W)
    if not future_known:
        val_fut = test_fut = None

    if finetune:
        pv = _finetune(model, df, cov, 0, VAL_CTX_W, device)
        q = best_q(_predict(pv, val_ctx, val_fut, cross_learning), val_test)
        pt = _finetune(model, df, cov, 0, TRAIN_W, device)
        pred = _predict(pt, test_ctx, test_fut, cross_learning)
    else:
        pipe = _pipeline(model, device)
        q = best_q(_predict(pipe, val_ctx, val_fut, cross_learning), val_test)
        pred = _predict(pipe, test_ctx, test_fut, cross_learning)

    table = per_quantile_table(pred, test_df)
    return table, q


def score_at_qstar(table, q):
    """Pull the (R2, WMAPE) row at the selected quantile from a per-quantile table."""
    row = table.loc[table["q"].round(2) == round(q, 2)].iloc[0]
    return float(row["R2"]), float(row["WMAPE"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cov', choices=['price', 'prophet_yhat'], default='prophet_yhat')
    ap.add_argument('--runs', choices=['both', 'zs', 'ft'], default='both')
    ap.add_argument('--model', choices=['small', 'base'], default='small')
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    df = load_cgpz()[0]
    if args.cov == 'prophet_yhat':
        df = add_teacher_covariate(df)
    # Keep the week-70 Prophet fit fixed for validation and final testing.
    cov = [args.cov]
    vc, vf, vy = build_window(df, cov, 0, VAL_CTX_W, TEST_W)
    tc, tf, ty = build_window(df, cov, 0, TRAIN_W, TEST_W)
    runs = {'both': [('zero-shot', False), ('FT 5 ep', True)],
            'zs': [('zero-shot', False)], 'ft': [('FT 5 ep', True)]}[args.runs]
    output = Path(__file__).resolve().parent.parent / 'outputs'
    output.mkdir(exist_ok=True)
    rows = []
    for label, ft in runs:
        torch.manual_seed(0)
        np.random.seed(0)
        if ft:
            pv = _finetune(args.model, df, cov, 0, VAL_CTX_W, args.device)
            val = _predict(pv, vc, vf, False)
            pt = _finetune(args.model, df, cov, 0, TRAIN_W, args.device)
            test = _predict(pt, tc, tf, False)
        else:
            pipe = _pipeline(args.model, args.device)
            val = _predict(pipe, vc, vf, False)
            test = _predict(pipe, tc, tf, False)
        q = best_q(val, vy)
        tag = 'ft' if ft else 'zs'
        test.to_csv(output / f'preds_{args.cov}_{args.model}_{tag}.csv', index=False)
        for stage, pred, truth in [('val', val, vy), ('test', test, ty)]:
            scores = per_quantile_table(pred, truth)
            scores.insert(0, 'q_star', q)
            scores.insert(0, 'stage', stage)
            scores.insert(0, 'run', label)
            rows.append(scores)
        print(label, 'q* =', q)
        print(rows[-1].loc[np.isclose(rows[-1].q, q)].to_string(index=False))
    suffix = '' if args.cov == 'prophet_yhat' else '_price'
    model_suffix = '' if args.model == 'small' else '_base'
    pd.concat(rows, ignore_index=True).to_csv(
        output / f'chronos{suffix}{model_suffix}.csv', index=False)


if __name__ == '__main__':
    main()
