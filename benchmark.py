# benchmark.py

import time
from sklearn.metrics import accuracy_score, r2_score

from config import DATASETS
from data_loader import load_dataset
from models import get_models
from experiment_utils import (
    IncrementalCSVWriter, ExperimentTracker, log_stage, format_exception,
    STAGE_COMPUTATION, STAGE_WRITE,
)


def evaluate(model, X_train, X_test, y_train, y_test, task):
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = round(time.time() - start, 2)

    preds = model.predict(X_test)
    if task == "classification":
        score = round(accuracy_score(y_test, preds), 4)
        metric = "Accuracy"
    else:
        score = round(r2_score(y_test, preds), 4)
        metric = "R²"

    return score, metric, elapsed


def run_benchmark_for_dataset(name):
    """
    Full benchmark computation for one dataset -- identical logic to what
    used to live directly in the top-level loop, moved into a function
    only so the loop can wrap it in a try/except at the dataset boundary.
    No change to what is computed or how.
    """
    log_stage(name, "Loading...")
    ds = load_dataset(name)
    task = ds["task"]
    models = get_models(task)

    log_stage(name, "Running...")
    rows = []
    print(f"\n--- {name} ({task}) ---")
    for model_name, model in models.items():
        # MLP gets gaussianized features, trees get raw features
        # this distinction follows the paper's preprocessing methodology
        if model_name == "MLP":
            X_tr, X_te = ds["X_train_nn"], ds["X_test_nn"]
        else:
            X_tr, X_te = ds["X_train"], ds["X_test"]

        score, metric, elapsed = evaluate(
            model, X_tr, X_te, ds["y_train"], ds["y_test"], task
        )
        print(f"  {model_name:15s} {metric}: {score} ({elapsed}s)")

        rows.append({
            "dataset": name,
            "task": task,
            "model": model_name,
            "metric": metric,
            "score": score,
            "time_s": elapsed,
        })
    return rows


writer = IncrementalCSVWriter("benchmark_results.csv")
tracker = ExperimentTracker()

for name in DATASETS:
    try:
        rows = run_benchmark_for_dataset(name)
    except Exception as exc:
        tracker.record_failure(name, exc, stage=STAGE_COMPUTATION)
        log_stage(name, f"Failed (computation): {format_exception(exc)}")
        continue

    try:
        writer.add_rows(rows)
    except Exception as exc:
        tracker.record_failure(name, exc, stage=STAGE_WRITE)
        log_stage(name, f"Failed (write): {format_exception(exc)}")
        continue

    tracker.record_success(name)
    log_stage(name, "Finished.")

tracker.print_summary()
print(f"\nResults saved to {writer.path}")