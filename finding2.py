import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import QuantileTransformer
from data_loader import load_dataset, DATASETS

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Noise levels follow the paper: 0 is baseline, then increasing irrelevant features
NOISE_LEVELS = [0, 5, 10, 20, 50]

# Multiple seeds to average over noise content and model initialization
# Data split stays fixed (seed=42 in data_loader) for comparability with benchmark
SEEDS = [42, 7, 13, 21, 99]

def make_models(task, seed):
    if task == "classification":
        return {
            "GBT": GradientBoostingClassifier(n_estimators=200, random_state=seed),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(256, 256),
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=seed
            ),
        }
    else:
        return {
            "GBT": GradientBoostingRegressor(n_estimators=200, random_state=seed),
            "MLP": MLPRegressor(
                hidden_layer_sizes=(256, 256),
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=seed
            ),
        }

def run_finding2(dataset_name):
    # GBT is used as the sole tree representative here.
    # It achieves the most consistent benchmark performance across all 3 datasets.
    # Using all tree models would clutter the visualization without adding
    # new information — the goal is to contrast tree behavior vs MLP as a class.
    ds = load_dataset(dataset_name)
    task = ds["task"]
    metric_fn = accuracy_score if task == "classification" else r2_score
    metric_name = "Accuracy" if task == "classification" else "R²"

    # scores[model][noise_level] = list of scores across seeds
    scores = {"GBT": {n: [] for n in NOISE_LEVELS},
              "MLP": {n: [] for n in NOISE_LEVELS}}

    for seed in SEEDS:
        rng = np.random.RandomState(seed)

        for n_noise in NOISE_LEVELS:
            # Generate noise features — fresh per seed so we average over content
            noise_train = rng.randn(ds["X_train"].shape[0], n_noise)
            noise_test  = rng.randn(ds["X_test"].shape[0],  n_noise)

            noise_cols = [f"noise_{i}" for i in range(n_noise)]

            X_train_noisy = pd.concat([
                ds["X_train"].reset_index(drop=True),
                pd.DataFrame(noise_train, columns=noise_cols)
            ], axis=1)
            X_test_noisy = pd.concat([
                ds["X_test"].reset_index(drop=True),
                pd.DataFrame(noise_test, columns=noise_cols)
            ], axis=1)

            # Gaussianize for MLP — fit on train only to avoid leakage
            qt = QuantileTransformer(output_distribution="normal", random_state=seed)
            X_train_nn = pd.DataFrame(
                qt.fit_transform(X_train_noisy), columns=X_train_noisy.columns
            )
            X_test_nn = pd.DataFrame(
                qt.transform(X_test_noisy), columns=X_test_noisy.columns
            )

            models = make_models(task, seed)

            models["GBT"].fit(X_train_noisy, ds["y_train"])
            models["MLP"].fit(X_train_nn,    ds["y_train"])

            scores["GBT"][n_noise].append(
                metric_fn(ds["y_test"], models["GBT"].predict(X_test_noisy))
            )
            scores["MLP"][n_noise].append(
                metric_fn(ds["y_test"], models["MLP"].predict(X_test_nn))
            )

    # Aggregate results
    print(f"\n--- Finding 2: {dataset_name} ({metric_name}) ---")
    print(f"  {'noise':>6} | {'GBT mean':>10} {'±std':>7} | {'MLP mean':>10} {'±std':>7}")
    print(f"  {'-'*52}")

    rows = []
    for n_noise in NOISE_LEVELS:
        gbt_mean = np.mean(scores["GBT"][n_noise])
        gbt_std  = np.std(scores["GBT"][n_noise])
        mlp_mean = np.mean(scores["MLP"][n_noise])
        mlp_std  = np.std(scores["MLP"][n_noise])

        print(f"  {n_noise:>6} | {gbt_mean:>10.4f} {gbt_std:>7.4f} | "
              f"{mlp_mean:>10.4f} {mlp_std:>7.4f}")

        rows.append({
            "dataset":   dataset_name,
            "task":      task,
            "n_noise":   n_noise,
            "GBT_mean":  round(gbt_mean, 4),
            "GBT_std":   round(gbt_std,  4),
            "MLP_mean":  round(mlp_mean, 4),
            "MLP_std":   round(mlp_std,  4),
        })

    return pd.DataFrame(rows)

all_results = []
for name in DATASETS:
    df = run_finding2(name)
    all_results.append(df)

final = pd.concat(all_results, ignore_index=True)
final.to_csv("finding2_results.csv", index=False)
print("\nSaved to finding2_results.csv")
