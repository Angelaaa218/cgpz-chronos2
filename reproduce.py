"""Out-of-sample R^2 and WMAPE for each model on the CGPZ dataset.

    python reproduce.py --list                                  # show model names
    python reproduce.py --model "Chronos-2 (Small, price, zero-shot)"
    python reproduce.py                                         # every model

Each model reports a single (R^2, WMAPE) on the 30-week test horizon. Chronos-2
models are scored at the validation-selected quantile. The device auto-detects
CUDA, then Apple MPS, then CPU; override with --device.
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


def main():
    global _DEVICE
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="model name (default: run all)")
    ap.add_argument("--list", action="store_true", help="list model names and exit")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    args = ap.parse_args()

    reg = registry()
    if args.list:
        for name in reg:
            print(name)
        return

    _DEVICE = _device(args.device)
    names = [args.model] if args.model else list(reg)
    for name in names:
        if name not in reg:
            raise SystemExit(f"unknown model: {name!r}\nrun with --list to see the names")
        r2, wmape = reg[name]()
        print(f"{name}: R2={r2:.3f}  WMAPE={wmape:.3f}")


if __name__ == "__main__":
    main()
