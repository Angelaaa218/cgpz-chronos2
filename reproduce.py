"""Out-of-sample R^2 and WMAPE for each model on the CGPZ dataset.

    python reproduce.py --list                                  # show model names
    python reproduce.py --model "Chronos-2 (Small, price, zero-shot)"
    python reproduce.py                                         # every model
    python reproduce.py --table 8                               # a full paper table (8, 9, 10)

Each model reports a single (R^2, WMAPE) on the 30-week test horizon. Chronos-2
models are scored at the validation-selected quantile. ``--table`` reproduces a
whole Section 6 table: Table 8 (baselines), Table 9 (covariate/size/CL grid),
or Table 10 (Prophet-yhat quantile sweep, zero-shot vs fine-tuned). The device
auto-detects CUDA, then Apple MPS, then CPU; override with --device.
"""
import argparse
import functools

from cgpz import baselines, chronos, prophet_teacher
from cgpz.data import load_cgpz

_DEVICE = "cpu"


def _device(arg):
    if arg != "auto":
        return arg
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@functools.lru_cache(maxsize=None)
def _baseline_scores():
    return {r["Method"]: (r["R2"], r["WMAPE"])
            for _, r in baselines.table8_baselines().iterrows()}


@functools.lru_cache(maxsize=None)
def _prophet_scores():
    return {r["Method"]: (r["R2"], r["WMAPE"])
            for _, r in prophet_teacher.book_prophet_baselines().iterrows()}


@functools.lru_cache(maxsize=None)
def _df_plain():
    return load_cgpz()[0]


@functools.lru_cache(maxsize=None)
def _df_yhat():
    return prophet_teacher.add_teacher_covariate(load_cgpz()[0])


def _chronos(cov, finetune, model="small"):
    df = _df_yhat() if cov == ["prophet_yhat"] else _df_plain()
    table, q = chronos.forecast(df, cov, model=model, device=_DEVICE, finetune=finetune)
    return chronos.score_at_qstar(table, q)


def registry():
    """Ordered map: model name -> callable returning (R^2, WMAPE)."""
    reg = {}
    for name in ("Decentralized Elastic Net (traditional)",
                 "Centralized OLS (traditional)",
                 "Decentralized Random Forest (tree-based)",
                 "Centralized Random Forest (tree-based)",
                 "K-means clustering"):
        reg[name] = (lambda n=name: _baseline_scores()[n])
    for name in ("Prophet (book recipe), standalone",
                 "Prophet + Decentralized Elastic Net"):
        reg[name] = (lambda n=name: _prophet_scores()[n])
    reg["Chronos-2 (Small, price, zero-shot)"] = lambda: _chronos(["price"], False, "small")
    reg["Chronos-2 (Base, price, zero-shot)"] = lambda: _chronos(["price"], False, "base")
    reg["Chronos-2 (Small, Prophet yhat, zero-shot)"] = lambda: _chronos(["prophet_yhat"], False, "small")
    reg["Chronos-2 (Small, Prophet yhat, fine-tuned)"] = lambda: _chronos(["prophet_yhat"], True, "small")
    return reg


def _fmt(r2, w):
    return f"R2={r2:6.3f}  WMAPE={w:6.3f}"


def table8():
    """Table 8: headline zero-shot Chronos-2 (price) vs the book's baselines."""
    print("Table 8  --  zero-shot Chronos-2 (price) vs book baselines")
    r2, w = _chronos(["price"], False, "small")
    print(f"  {'Chronos-2 (Small, price, zero-shot)':44s} {_fmt(r2, w)}")
    for name, (r2, w) in {**_baseline_scores(), **_prophet_scores()}.items():
        print(f"  {name:44s} {_fmt(r2, w)}")


def table9():
    """Table 9: zero-shot R2/WMAPE by covariate set, model size, and CL flag."""
    all_cov = load_cgpz()[1]
    cov_sets = [("None (univariate)", [], True),
                ("price (future-known)", ["price"], True),
                ("price (history only)", ["price"], False),
                ("All 45 features", all_cov, True)]
    print("Table 9  --  zero-shot R2/WMAPE by covariate set (q* chosen per column)")
    header = f"  {'Covariate set':22s}"
    for model in ("Small", "Base"):
        for cl in ("CLoff", "CLon"):
            header += f"| {model}/{cl:5s}R2 WMAPE "
    print(header)
    for label, cov, fk in cov_sets:
        line = f"  {label:22s}"
        for model in ("small", "base"):
            for cl in (False, True):
                table, q = chronos.forecast(_df_plain(), cov, model=model, device=_DEVICE,
                                            finetune=False, cross_learning=cl, future_known=fk)
                r2, w = chronos.score_at_qstar(table, q)
                line += f"| {r2:6.3f} {w:5.3f} "
        print(line)


def table10():
    """Table 10: Prophet-yhat covariate, full quantile sweep, zero-shot vs fine-tuned."""
    print("Table 10  --  Prophet-yhat covariate: R2/WMAPE over the quantile grid")
    df = _df_yhat()
    zs, zq = chronos.forecast(df, ["prophet_yhat"], model="small", device=_DEVICE, finetune=False)
    ft, fq = chronos.forecast(df, ["prophet_yhat"], model="small", device=_DEVICE, finetune=True)
    print(f"  {'q':>5} | {'ZS R2':>7} {'ZS WMAPE':>8} | {'FT R2':>7} {'FT WMAPE':>8}")
    for _, r in zs.iterrows():
        q = r["q"]
        fr = ft.loc[ft["q"] == q].iloc[0]
        star = " *" if round(q, 2) in (round(zq, 2), round(fq, 2)) else ""
        print(f"  {q:5.2f} | {r['R2']:7.3f} {r['WMAPE']:8.3f} | {fr['R2']:7.3f} {fr['WMAPE']:8.3f}{star}")
    r2, w = prophet_teacher.teacher_standalone(df)
    print(f"  Prophet standalone: {_fmt(r2, w)}    (* = validation-selected q*)")


_TABLES = {"8": table8, "9": table9, "10": table10}


def main():
    global _DEVICE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="model name (default: run all)")
    ap.add_argument("--table", default=None, choices=["8", "9", "10"],
                    help="reproduce a full paper table (8, 9, or 10)")
    ap.add_argument("--list", action="store_true", help="list model names and exit")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    args = ap.parse_args()

    reg = registry()
    if args.list:
        for name in reg:
            print(name)
        return

    _DEVICE = _device(args.device)
    if args.table:
        _TABLES[args.table]()
        return
    names = [args.model] if args.model else list(reg)
    for name in names:
        if name not in reg:
            raise SystemExit(f"unknown model: {name!r}\nrun with --list to see the names")
        r2, wmape = reg[name]()
        print(f"{name}: R2={r2:.3f}  WMAPE={wmape:.3f}")


if __name__ == "__main__":
    main()
