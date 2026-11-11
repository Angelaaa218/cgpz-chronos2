# Chronos-2 on the CGPZ retail dataset

Code, data, and saved outputs for the public-dataset experiments in *A Bitter Lesson for Retail Demand Forecasting*. The experiments compare Chronos-2 with the book's baselines, evaluate price and Prophet-derived covariates, and fine-tune Chronos-2 Small using the per-SKU Prophet forecast. Section EC.4 includes the same-covariate baseline comparisons and the single-observation sensitivity analysis.

## Installation

The Chronos experiments use an Apple M4 Pro with the MPS backend, macOS 15.7.4, and Python 3.13.5. `requirements.txt` pins the package versions used in this environment.

```bash
pip install -r requirements.txt
```

Model weights download from Hugging Face on first use. Run the commands below from this folder.

## Run the experiments

List available models or evaluate one model:

```bash
python reproduce.py compare --list
python reproduce.py compare --model "Chronos-2 (Small, price, zero-shot)" --device mps
```

Run the paper's comparisons:

```bash
python reproduce.py compare --table headline --device mps
python reproduce.py compare --table covariates --device mps
python reproduce.py compare --table prophet --device mps
```

Generate Chronos scores and predictions:

```bash
python reproduce.py chronos --cov price --runs zs --device mps
python reproduce.py chronos --cov prophet_yhat --runs both --device mps
```

Run the Section EC.4 comparisons and sensitivity analysis:

```bash
python reproduce.py ec4
python reproduce.py sensitivity
```

The required saved Chronos outputs and Prophet covariate are included, so the last two commands can run without regenerating the forecasts. Commands that save results write to `outputs/` and replace the corresponding files. The code also accepts `--device cuda` or `--device cpu`; omitting the device for `compare` selects an available backend automatically.

## Data and method

**Data.** CGPZ contains 100 weeks of sales for 44 tech-gadget SKUs. `data/data_raw.csv` contains the original observations; `data/data_processed.csv` contains the book's 45 engineered features and 98 weeks per SKU after the two price-lag warm-up weeks. Chronos uses the first 70 weeks for training and the final 30 weeks for testing. The baselines use the corresponding 68/30 split of the processed data. Predictions are scored using out-of-sample R² and WMAPE without clipping at zero.

**Quantile selection.** Forecast weeks 41–70 from weeks 1–40 and select the quantile that maximizes validation R². Forecast weeks 71–100 from weeks 1–70 using that quantile. The evaluation grid contains 21 levels: 0.01, 0.05–0.95 in steps of 0.05, and 0.99, matching Chronos-2 Base's native grid. Chronos-2 Small has 13 native levels; its pipeline interpolates the additional requested levels. The included score files contain the 19 levels from 0.05 to 0.95.

**Fine-tuning.** Full-parameter fine-tuning uses quantile loss, learning rate 1e-5, weight decay 0.01, gradient clipping at norm 1.0, cosine learning-rate decay, batch size 16, no warmup, and 15 optimization steps. With one target and one covariate, each step samples eight SKU windows with replacement: 15 steps are approximately 2.73 nominal epochs, counting one sampled window per SKU as one epoch. Validation and test models are fine-tuned separately from the public checkpoint. The `chronos` command initializes PyTorch and NumPy with seed 0 for each run; the Trainer uses seed 42. Settings are defined in `code/chronos.py` under `FT`.

**Prophet covariate.** A separate Prophet model is fitted for each SKU using yearly seasonality, current price, and two lagged prices. Its full forecast is the single Chronos covariate. Prophet fits weeks 3–70; the first two covariate values are missing because price lags are unavailable. The same fit is retained for validation and testing because the shorter week-40 fit yielded unstable forecasts. Validation is therefore not fully held out for the combined pipeline; test sales are excluded from fitting and quantile selection. The saved covariate is `outputs/prophet_teacher.csv`.

## Files

```text
reproduce.py                 Command-line entry point
requirements.txt             Tested package versions
code/
  data.py                    Data loading, windows, and quantile grid
  metrics.py                 R², WMAPE, and quantile selection
  chronos.py                 Chronos inference and fine-tuning
  prophet_teacher.py         Prophet covariate and book baseline
  baselines.py               Classical forecasting baselines
  comparisons.py             Model and covariate comparisons
  ec4_prophet_featuresets.py  Section EC.4 baseline comparisons
  ec4_spike_sensitivity.py    Single-observation sensitivity analysis
data/                        Raw and processed datasets
outputs/                     Paper scores, predictions, and Prophet covariate
```

## Data provenance and citation

The data and baseline recipes come from the [companion materials](https://github.com/demand-prediction-guide/public) for Cohen, Gras, Pentecoste, and Zhang, *Demand Prediction in Retail: A Practical Guide to Leverage Data and Predictive Analytics* (Springer). The baseline implementations follow notebooks 3–5; the book's Prophet baseline follows notebook 7.

The included data remain subject to the original source's terms. If you use this code, please cite the textbook and *A Bitter Lesson for Retail Demand Forecasting*.
