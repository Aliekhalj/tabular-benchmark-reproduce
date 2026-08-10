import time
from sklearn.metrics import accuracy_score, r2_score

from config import DATASETS
from data_loader import load_dataset, validate_dataset_registry
from models import get_models
from experiment_utils import (
    IncrementalCSVWriter, ExperimentTracker, log_stage,
    log_finished, log_failed, STAGE_COMPUTATION, STAGE_WRITE,
)


def evaluate(model, X_train, X_test, y_train, y_test, task):
    # NOTE: uses time.time(), not time.perf_counter() -- this is
    # pre-existing, produces the "time_s" CSV column, and is
    # deliberately left untouched. See design notes from the previous
    # commit (structured logging).
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = round(time.time() - start, 2)

    preds = model.predict(X_test)
    if task == "classification":
        score = round(accuracy_score(y_test, preds), 4)
        metric = "Accuracy"
    else:
        score = round(r2_score(y_test, preds), 4)
        metric = "R\u00b2"

    return score, metric, elapsed


def run_benchmark_for_dataset(name):
    """
    Full benchmark computation for one dataset. Unchanged -- this commit
    only adds the registry validation call below, before the loop.
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


if __name__ == "__main__":
    validate_dataset_registry()

    run_start = time.perf_counter()
    writer = IncrementalCSVWriter("benchmark_results.csv")
    tracker = ExperimentTracker()

    for name in DATASETS:
        start = time.perf_counter()
        try:
            rows = run_benchmark_for_dataset(name)
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
    print(f"\nResults saved to {writer.path}")