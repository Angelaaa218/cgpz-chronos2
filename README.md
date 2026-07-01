# Chronos-2 on the CGPZ retail dataset

Code to reproduce the public-dataset (CGPZ) results in *A Bitter Lesson for
Retail Demand Forecasting*. It evaluates Chronos-2 against the demand-prediction
baselines of Cohen et al.'s textbook, and shows that a single future-known
covariate, optionally distilled from a per-SKU Prophet model, matches or beats
those baselines, with fine-tuning improving the result further.

## Dataset

The CGPZ dataset is 100 weeks of weekly sales for 44 tech-gadget SKUs, released
with the textbook's companion repository. Two files are shipped here:

- `data/data_raw.csv` — raw weekly sales, listed price, the main-page promotion
  flag, and static product attributes.
- `data/data_processed.csv` — the book's 45-feature engineered matrix, used by
  the baselines.

Evaluation follows the book: first 70 weeks train, last 30 weeks test,
out-of-sample R² and WMAPE (predictions clipped at 0). The baselines use the
book's 68/32 split, matching the numbers in its Figure 6.1.

## Installation

```bash
pip install -r requirements.txt
```

Prophet builds on cmdstanpy; the first import compiles its model once.

## Reporting results

Each model reports a single out-of-sample R² and WMAPE, keyed by model name:

```bash
python reproduce.py --list                                   # list model names
python reproduce.py --model "Chronos-2 (Small, price, zero-shot)"
python reproduce.py                                          # run every model
```

Example output:

```
K-means clustering: R2=0.560  WMAPE=0.763
Chronos-2 (Small, price, zero-shot): R2=0.667  WMAPE=0.520
```

Chronos-2 models are scored at the validation-selected quantile and select a
device automatically (CUDA, then Apple MPS, then CPU); pass
`--device {cuda,mps,cpu}` to override.

## Method

**Quantile selection.** Chronos-2 emits quantile forecasts, so a point forecast
requires choosing a quantile. We use a two-stage, train-only protocol: a
validation stage (weeks 1–40 forecast weeks 41–70) selects the quantile `q*`
that maximizes validation R²; the test stage (weeks 1–70 forecast weeks 71–100)
is scored at `q*`. Fine-tuned runs refit the model in each stage so that `q*` is
never chosen on data the scored model has seen.

**Fine-tuning.** Full-parameter fine-tuning with the quantile loss over
Chronos-2's 21 native quantile levels: learning rate 1e-5, weight decay 0.01,
gradient clipping at norm 1.0, cosine schedule, batch size 16, five epochs
(see `cgpz/chronos.py::FT`).

**Prophet covariate.** For the distillation experiment we fit a per-SKU Prophet
model with the listed price and its two lags as regressors and feed its point
forecast to Chronos-2 as a single future-known covariate. The two price lags are
undefined for the first two weeks, so Prophet is fit on the remaining weeks and
its forecast is marked missing there, which Chronos-2 handles natively.

**Note on the Prophet baseline.** The book's numbers (0.265 standalone, 0.568
with Elastic Net) come from its `make_future_dataframe(periods, freq="W")`
forecast dates, which are Sunday-anchored and one day off the Monday data,
shifting Prophet's seasonal/holiday effects. This reproduces with any Prophet
version (`fbprophet` or `prophet`) -- the figures follow from the date
construction, not the package. Predicting at the true test dates instead gives
0.215 / 0.565. `book_prophet_baselines` follows the book's recipe and reports
the former; `fbprophet_baseline.py` lets you confirm the 0.265 on x86 hardware.

## Layout

```
data/                 CGPZ raw and processed data
cgpz/
  data.py             loading, feature engineering, windowing
  metrics.py          R², WMAPE, two-stage quantile selection
  baselines.py        the book's baselines, transcribed from its notebooks (Table 8)
  prophet_teacher.py  Prophet teacher covariate and the book's Prophet baseline
  chronos.py          zero-shot and fine-tuned Chronos-2
reproduce.py          report R^2 / WMAPE per model (by name)
```

The baseline code in `cgpz/baselines.py` is transcribed directly from Cohen et
al.'s public demand-prediction notebooks (notebooks 3 "Common Demand Prediction
Methods", 4 "Tree-Based Methods", and 5 "Clustering Techniques"): the data file,
the 68/30 split, the model choices, and every hyperparameter are theirs, and the
numbers match their Figure 6.1. The book's Prophet baseline (`prophet_teacher.py`)
follows its notebook 7 recipe.

## Data provenance and citation

The CGPZ dataset and the baseline recipes are from the companion materials of
Cohen, Gras, Pentecoste, and Zhang, *Demand Prediction in Retail: A Practical
Guide to Leverage Data and Predictive Analytics* (Springer). The data files in
`data/` are redistributed here for reproducibility and remain subject to the
terms of that original source; please cite the book when using them.

If you use this code, please cite *A Bitter Lesson for Retail Demand
Forecasting* and the dataset above.

## Requirements

See `requirements.txt`. Chronos-2 runs on CPU, CUDA, or Apple MPS. The baselines
and Prophet teacher take a few minutes (Prophet is fit once per SKU and cached
to `outputs/`); the Chronos-2 models download model weights from the Hugging
Face Hub on first use.
