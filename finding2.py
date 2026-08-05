# finding2.py

import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import QuantileTransformer

from config import DATASETS, NOISE_LEVELS, NOISE_SEEDS, FINDING_N_ESTIMATORS
from data_loader import load_dataset
from models import get_models
from experiment_utils import (
    IncrementalCSVWriter, ExperimentTracker, log_stage,
    log_finished, log_failed, STAGE_COMPUTATION, STAGE_WRITE,
)


def run_finding2(dataset_name):
    # GBT is used as the sole tree representative here.
    # It achieves the most consistent benchmark performance across all 3 datasets.
    # Using all tree models would clutter the visualization without adding
    # new information — the goal is to contrast tree behavior vs MLP as a class.
    log_stage(dataset_name, "Loading...")
    ds = load_dataset(dataset_name)
    task = ds["task"]
    metric_fn = accuracy_score if task == "classification" else r2_score
    metric_name = "Accuracy" if task == "classification" else "R²"

    log_stage(dataset_name, "Running...")
    scores = {"GBT": {n: [] for n in NOISE_LEVELS},
              "MLP": {n: [] for n in NOISE_LEVELS}}

    for seed in NOISE_SEEDS:
        rng = np.random.RandomState(seed)

        for n_noise in NOISE_LEVELS:
            noise_train = rng.randn(ds["X_train"].shape[0], n_noise)
            noise_test = rng.randn(ds["X_test"].shape[0], n_noise)
            noise_cols = [f"noise_{i}" for i in range(n_noise)]

            X_train_noisy = pd.concat([
                ds["X_train"].reset_index(drop=True),
                pd.DataFrame(noise_train, columns=noise_cols)
            ], axis=1)
            X_test_noisy = pd.concat([
                ds["X_test"].reset_index(drop=True),
                pd.DataFrame(noise_test, columns=noise_cols)
            ], axis=1)

            # NOTE: random_state=seed here is intentional and must stay
            # tied to the per-iteration noise seed, not MASTER_SEED.
            qt = QuantileTransformer(output_distribution="normal", random_state=seed)
            X_train_nn = pd.DataFrame(
                qt.fit_transform(X_train_noisy), columns=X_train_noisy.columns
            )
            X_test_nn = pd.DataFrame(
                qt.transform(X_test_noisy), columns=X_test_noisy.columns
            )

            models = get_models(task, seed=seed, n_estimators=FINDING_N_ESTIMATORS)
            models["GBT"].fit(X_train_noisy, ds["y_train"])
            models["MLP"].fit(X_train_nn, ds["y_train"])

            scores["GBT"][n_noise].append(
                metric_fn(ds["y_test"], models["GBT"].predict(X_test_noisy))
            )
            scores["MLP"][n_noise].append(
                metric_fn(ds["y_test"], models["MLP"].predict(X_test_nn))
            )

    print(f"\n--- Finding 2: {dataset_name} ({metric_name}) ---")
    print(f"  {'noise':>6} | {'GBT mean':>10} {'±std':>7} | {'MLP mean':>10} {'±std':>7}")
    print(f"  {'-'*52}")

    rows = []
    for n_noise in NOISE_LEVELS:
        gbt_mean = np.mean(scores["GBT"][n_noise])
        gbt_std = np.std(scores["GBT"][n_noise])
        mlp_mean = np.mean(scores["MLP"][n_noise])
        mlp_std = np.std(scores["MLP"][n_noise])

        print(f"  {n_noise:>6} | {gbt_mean:>10.4f} {gbt_std:>7.4f} | "
              f"{mlp_mean:>10.4f} {mlp_std:>7.4f}")

        rows.append({
            "dataset": dataset_name,
            "task": task,
            "n_noise": n_noise,
            "GBT_mean": round(gbt_mean, 4),
            "GBT_std": round(gbt_std, 4),
            "MLP_mean": round(mlp_mean, 4),
            "MLP_std": round(mlp_std, 4),
        })

    return rows


run_start = time.perf_counter()
writer = IncrementalCSVWriter("finding2_results.csv")
tracker = ExperimentTracker()

for name in DATASETS:
    start = time.perf_counter()
    try:
        rows = run_finding2(name)
    except Exception as exc:
        tracker.record_failure(name, exc, stage=STAGE_COMPUTATION)
        log_failed(name, STAGE_COMPUTATION, start, exc)
        continue

    try:
        writer.add_rows(rows)
    except Exception as exc:
        tracker.record_failure(name, exc, stage=STAGE_WRITE)
        log_failed(name, STAGE_WRITE, start, exc)
        continue

    tracker.record_success(name)
    log_finished(name, start)

total_runtime = time.perf_counter() - run_start
tracker.print_summary(total_runtime)
print(f"\nSaved to {writer.path}")