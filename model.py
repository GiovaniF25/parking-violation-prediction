"""Estimate expected parking violations in batches and compare count models."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import statsmodels.api as sm
from scipy.stats import nbinom, poisson
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

PROJECT = "parking-violation-prediction"
FEATURES = ["staff", "post", "corruption"]
TARGET = "violations"


def validate(data, training):
    numbers = data[["staff", "corruption"] + ([TARGET] if training else [])].to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError("Numeric variables must be finite.")
    for column in ["staff"] + ([TARGET] if training else []):
        values = data[column].to_numpy(dtype=float)
        if ((values < 0) | (values != np.floor(values))).any():
            raise ValueError(f"{column} must contain nonnegative integers.")
    if not data["post"].isin(["yes", "no"]).all():
        raise ValueError("post must be yes or no.")
    if training and ("code" not in data or data["code"].isna().any()):
        raise ValueError("Training requires a country code in every row.")


def design_matrix(data):
    return pd.DataFrame({
        "const": 1.0,
        "staff": data["staff"].astype(float),
        "post": data["post"].eq("yes").astype(float),
        "corruption": data["corruption"].astype(float),
    }, index=data.index)


def fit_models(data):
    if data["code"].nunique() < 2:
        raise ValueError("Training requires at least two distinct country codes.")
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    i, j = next(split.split(data, groups=data["code"]))
    training, test = data.iloc[i], data.iloc[j]
    X_train, X_test = design_matrix(training), design_matrix(test)
    nb = sm.NegativeBinomial(training[TARGET], X_train).fit(method="bfgs", maxiter=300, disp=False)
    pois = sm.Poisson(training[TARGET], X_train).fit(disp=False)
    if not (nb.mle_retvals["converged"] and pois.mle_retvals["converged"]):
        raise ValueError("Model fitting did not converge on the supplied data.")
    nb_mean, poisson_mean = nb.predict(X_test), pois.predict(X_test)
    size = 1 / nb.params["alpha"]
    y = test[TARGET]
    return nb.params.drop("alpha"), {
        "training_countries": training["code"].nunique(),
        "test_countries": test["code"].nunique(),
        "alpha": float(nb.params["alpha"]),
        "negative_binomial": {
            "mae": mean_absolute_error(y, nb_mean),
            "mean_negative_log_likelihood": float(-nbinom.logpmf(y, size, size / (size + nb_mean)).mean()),
        },
        "poisson": {
            "mae": mean_absolute_error(y, poisson_mean),
            "mean_negative_log_likelihood": float(-poisson.logpmf(y, poisson_mean).mean()),
        },
    }


def predict(coefficients, data):
    return {"expected_violations": np.exp(design_matrix(data) @ coefficients)}


def read_csv(path, training=False):
    data = pd.read_csv(path)
    required = FEATURES + ([TARGET] if training else [])
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if data.empty or data[required].isna().any().any():
        raise ValueError("The CSV must contain rows and values in all required columns.")
    validate(data, training)
    return data


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train, evaluate, and save the model")
    train.add_argument("--data", type=Path, default=root / "data" / "training.csv")
    train.add_argument("--output", type=Path, default=root / "artifacts")
    prediction = commands.add_parser("predict", help="Generate predictions from a CSV")
    prediction.add_argument("--data", type=Path, required=True)
    prediction.add_argument("--model", type=Path, default=root / "artifacts" / "model.joblib")
    prediction.add_argument("--output", type=Path, default=root / "artifacts" / "predictions.csv")
    args = parser.parse_args()

    try:
        if args.command == "train":
            model, metrics = fit_models(read_csv(args.data, training=True))
            args.output.mkdir(parents=True, exist_ok=True)
            joblib.dump({"project": PROJECT, "model": model}, args.output / "model.joblib")
            report = json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False)
            (args.output / "metrics.json").write_text(report + "\n", encoding="utf-8")
            print(report)
            print(f"Saved model and metrics to {args.output}")
        else:
            data = read_csv(args.data)
            bundle = joblib.load(args.model)
            if bundle["project"] != PROJECT:
                raise ValueError("The model belongs to another project.")
            result = data.assign(**predict(bundle["model"], data))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
            print(f"{len(result)} predictions saved to {args.output}")
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
