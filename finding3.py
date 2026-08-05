# finding3.py

import time
import numpy as np
from scipy.stats import special_ortho_group
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import QuantileTransformer

from config import DATASETS, N_ROTATIONS, ROTATION_MODEL_SEEDS, MASTER_SEED, FINDING_N_ESTIMATORS
from data_loader import load_dataset
from models import get_models
from experiment_utils import (
    IncrementalCSVWriter, ExperimentTracker, log_stage,
    log_finished, log_failed, STAGE_COMPUTATION, STAGE_WRITE,
)


def run_finding3(dataset_name):
    # GBT is used as the sole tree representative here.
    # It achieves the most consistent benchmark performance across all 3 datasets.
    # Using all tree models would clutter the slope charts without adding
    # new information — the goal is to isolate the rotation effect, not compare
    # tree architectures against each other.
    log_stage(dataset_name, "Loading...")
    ds = load_dataset(dataset_name)
    task = ds["task"]
    n_features = ds["X_train"].shape[1]
    metric_fn = accuracy_score if task == "classification" else r2_score
    metric_name = "Accuracy" if task == "classification" else "R²"

    log_stage(dataset_name, "Running...")
    # NOTE: uses MASTER_SEED, not a per-iteration seed -- runs once, before
    # either seed loop below, unlike finding2.py's per-noise QuantileTransformer.
    qt = QuantileTransformer(output_distribution="normal", random_state=MASTER_SEED)
    X_train_g = qt.fit_transform(ds["X_train"])
    X_test_g = qt.transform(ds["X_test"])

    scores = {"original": {"GBT": [], "MLP": []},
              "rotated": {"GBT": [], "MLP": []}}

    for seed in ROTATION_MODEL_SEEDS:
        models = get_models(task, seed=seed, n_estimators=FINDING_N_ESTIMATORS)
        for mname in ["GBT", "MLP"]:
            model = models[mname]
            model.fit(X_train_g, ds["y_train"])
            scores["original"][mname].append(
                metric_fn(ds["y_test"], model.predict(X_test_g))
            )

    for rot_seed in range(N_ROTATIONS):
        R = special_ortho_group.rvs(n_features, random_state=rot_seed)
        X_train_r = X_train_g @ R
        X_test_r = X_test_g @ R

        for seed in ROTATION_MODEL_SEEDS:
            models = get_models(task, seed=seed, n_estimators=FINDING_N_ESTIMATORS)
            for mname in ["GBT", "MLP"]:
                model = models[mname]
                model.fit(X_train_r, ds["y_train"])
                scores["rotated"][mname].append(
                    metric_fn(ds["y_test"], model.predict(X_test_r))
                )

    print(f"\n--- Finding 3: {dataset_name} ({metric_name}) ---")
    print(f"  {'setting':>10} | {'GBT mean':>10} {'±std':>7} | "
          f"{'MLP mean':>10} {'±std':>7}")
    print(f"  {'-'*54}")

    rows = []
    for setting in ["original", "rotated"]:
        gbt_mean = np.mean(scores[setting]["GBT"])
        gbt_std = np.std(scores[setting]["GBT"])
        mlp_mean = np.mean(scores[setting]["MLP"])
        mlp_std = np.std(scores[setting]["MLP"])

        print(f"  {setting:>10} | {gbt_mean:>10.4f} {gbt_std:>7.4f} | "
              f"{mlp_mean:>10.4f} {mlp_std:>7.4f}")

        rows.append({
            "dataset": dataset_name,
            "task": task,
            "setting": setting,
            "GBT_mean": round(gbt_mean, 4),
            "GBT_std": round(gbt_std, 4),
            "MLP_mean": round(mlp_mean, 4),
            "MLP_std": round(mlp_std, 4),
        })

    return rows


run_start = time.perf_counter()
writer = IncrementalCSVWriter("finding3_results.csv")
tracker = ExperimentTracker()

for name in DATASETS:
    start = time.perf_counter()
    try:
        rows = run_finding3(name)
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