# Parking Violation Prediction with Negative Binomial Regression

This project adapts a classroom exercise on count regression. It compares Poisson and negative binomial models on simulated parking violation counts and generates batch predictions from a CSV.

## Run locally

From the repository folder on Linux or WSL, run these commands with Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python model.py train
python model.py predict --data data/prediction_example.csv
```

Training reads `data/training.csv` by default and writes `artifacts/model.joblib` and `artifacts/metrics.json`. Batch prediction writes `artifacts/predictions.csv` with the original columns and an `expected_violations` column. The predicted mean can be fractional. Use `--data path/to/file.csv` to supply your own data and `--help` after either command to see its options.

Only load Joblib files you trust. Loading a Joblib file can execute code.

## Data and evaluation

The demonstration uses simulated data for 180 fictional countries, each observed in two periods. Counts were drawn from a negative binomial distribution to illustrate overdispersion. The scenario was inspired by a study of diplomatic parking violations. [Fisman and Miguel’s study](https://doi.org/10.1086/527495)

| Column | Meaning |
| --- | --- |
| `staff` | Staff size as a nonnegative integer |
| `post` | Period indicator, either `yes` or `no` |
| `corruption` | Finite numeric indicator |
| `code` | Country code, required only for training |
| `violations` | Observed count as a nonnegative integer, required only for training |

The split keeps each country's records together. Mean absolute error (MAE) evaluates the predicted mean, and mean negative log-likelihood evaluates the probability assigned to the observed count. Applying the model to real data would require validation with observed counts.

## Demonstration results

The test set contains 45 held-out fictional countries.

| Model | MAE (violations) | Mean negative log-likelihood |
| --- | ---: | ---: |
| Poisson | 5.12 | 4.62 |
| Negative binomial | 5.10 | 2.45 |

The negative binomial model assigned more probability to the held-out counts, while the two models had similar MAE.

## Notebook

Open [analysis.ipynb](analysis.ipynb) in VS Code with the Jupyter extension, select the virtual environment as the kernel, and run all cells. The notebook compares predicted means against the simulated counts.
