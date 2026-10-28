# Chronos-2 on the CGPZ retail dataset

Code for the public-dataset experiments in *A Bitter Lesson for Retail Demand Forecasting*. It evaluates Chronos-2 with price or a distilled Prophet forecast as its covariate, in zero-shot and fine-tuned settings, against the baselines from Cohen et al.'s textbook. It also includes the same-covariate comparisons in Section EC.4.

## Dataset

The CGPZ dataset contains 100 weeks of weekly sales for 44 tech-gadget SKUs. Two files are included:

- `data/data_raw.csv`: weekly sales, listed price, the main-page promotion flag, and product attributes.
- `data/data_processed.csv`: the book's 45-feature matrix, with 98 weeks per SKU after dropping the two price-lag warm-up weeks. Features exclude the identifiers `week` and `sku` and the target `weekly_sales`.

Evaluation uses the first 70 weeks for training and the last 30 weeks for testing. Baselines use the corresponding 68/30 split of the processed data. We report out-of-sample R² and WMAPE using predictions without clipping at zero.

## Installation

```bash
pip install -r requirements.txt
```

Chronos-2 supports CPU, CUDA, and Apple MPS. Model weights download from Hugging Face on first use. Prophet uses CmdStanPy and caches the fitted covariate in `outputs/prophet_teacher.csv`.

## Running the experiments

Run commands from this folder. To list models, evaluate one model, or evaluate all models:

```bash
python reproduce.py compare --list
python reproduce.py compare --model "Chronos-2 (Small, price, zero-shot)"
python reproduce.py compare
```

Each model reports one test R² and WMAPE at its validation-selected quantile. The `compare` command selects CUDA, then MPS, then CPU according to availability. Use `--device cpu`, `--device cuda`, or `--device mps` to choose explicitly.

To run a complete comparison:

```bash
python reproduce.py compare --table 8    # headline model comparison
python reproduce.py compare --table 9    # covariates, model size, and cross-learning
python reproduce.py compare --table 10   # Prophet-covariate quantile comparison
```

The numbers 8, 9, and 10 are retained command identifiers from the original code, not table numbers in the revised manuscript.

To generate saved Chronos forecasts and then run Section EC.4:

```bash
python reproduce.py chronos --cov price --runs zs --device cpu
python reproduce.py chronos --cov prophet_yhat --runs both --device cpu
python reproduce.py ec4
python reproduce.py sensitivity
```

These commands write to `outputs/` and replace existing result files. To use the included Chronos results, skip the first two commands. The `ec4` command refits the baselines with the same covariates as Chronos-2. The `sensitivity` command compares results with and without the SKU 25 demand spike on 9 July 2018, using saved Chronos predictions.

## Method

**Quantile selection.** Forecast weeks 41–70 from weeks 1–40 and select the quantile that maximizes validation R². Forecast test weeks 71–100 from weeks 1–70 using this selected quantile. The search grid in `code/data.py` contains 21 levels: 0.01, 0.05–0.95 in steps of 0.05, and 0.99. The Prophet covariate uses the fixed fit described below.

**Fine-tuning.** Full-parameter fine-tuning uses quantile loss, learning rate 1e-5, weight decay 0.01, gradient clipping at norm 1.0, cosine learning-rate decay, batch size 16, and no warmup. We fine-tune for 15 optimization steps, corresponding to five nominal epochs under our training-budget convention. One nominal epoch is `ceil(44 / 16) = 3` steps. Because windows are sampled with replacement, these are not exhaustive passes over a fixed set of training instances. Validation and test models are fine-tuned separately from the public checkpoint. Settings are defined in `code/chronos.py` under `FT`.

**Prophet covariate.** We fit a separate Prophet model for each SKU using yearly seasonality, current price, and two lagged prices. Its point forecast becomes the single Chronos-2 covariate. Prophet fits weeks 3–70, with missing covariate values for weeks 1–2 because the price lags are unavailable. Fitting only through week 40 was unstable in our diagnostic run and produced negative total predicted demand over validation weeks 41–70. We therefore retain the week-70 Prophet fit for both validation and testing. No test sales enter fitting or quantile selection, but validation sales enter the Prophet fit, so validation is not fully held out for the combined pipeline.

**Book's Prophet baseline.** The book uses Sunday forecast dates, one day before the Monday observations. `book_prophet_baselines` in `code/prophet_teacher.py` preserves that recipe. The distilled Prophet covariate instead uses the actual observation dates.

**Saved results.** The included paper results used the earlier 19-level quantile grid. They have not been regenerated with the expanded grid. The CPU verification run selected 0.60 for fine-tuning (R² 0.692, WMAPE 0.545), while the saved paper run selected 0.55 (R² 0.713, WMAPE 0.505). Exact reproduction of that fine-tuned result remains unresolved.

## Code layout

```text
reproduce.py                 Command-line entry point
code/
  data.py                    Data loading, windows, and quantile grid
  metrics.py                 R², WMAPE, and quantile selection
  chronos.py                 Chronos inference, fine-tuning, and saved forecasts
  prophet_teacher.py         Prophet covariate and book baseline
  baselines.py               Classical forecasting baselines
  comparisons.py             Model and covariate comparisons
  ec4_prophet_featuresets.py  Section EC.4 baseline comparisons
  ec4_spike_sensitivity.py    Single-observation sensitivity analysis
data/                        Raw and processed datasets
outputs/                     Saved scores and predictions
```

## Data provenance and citation

The data and baseline recipes come from the companion materials of Cohen, Gras, Pentecoste, and Zhang, *Demand Prediction in Retail: A Practical Guide to Leverage Data and Predictive Analytics* (Springer). The baseline implementations follow notebooks 3–5, and the book's Prophet baseline follows notebook 7.

The included data remain subject to the original source's terms. If you use this code, please cite the textbook and *A Bitter Lesson for Retail Demand Forecasting*.
