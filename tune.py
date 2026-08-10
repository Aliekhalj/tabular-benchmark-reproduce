"""
Phase 2 Commit 2: RandomizedSearchCV-based hyperparameter tuning against
an isolated validation split.

Design summary (see Phase 2 Commit 2 architecture discussion for the full
reasoning):

- data_loader.py is untouched. The validation split is carved out of
  ds["X_train"]/ds["y_train"] entirely within this module -- one split
  per dataset, computed once, reused identically across all 4 models.
- ds["X_test"]/ds["X_test_nn"]/ds["y_test"] are never passed into any
  tuning function in this file. Only the final evaluation step (called
  once, after tuning is fully complete) references them -- this is
  structural, not a matter of care: search/refit functions' signatures
  simply don't accept test data as an argument.
- MLP-specific leakage note: ds["X_train_nn"] was produced by a
  QuantileTransformer fit on the FULL X_train inside load_dataset() --
  verified directly against the current data_loader.py before writing
  this module. Carving a validation subset out of ds["X_train_nn"]
  directly would mean that transform already "saw" the validation rows
  during its own fit. This module fits a second, tuning-scoped
  QuantileTransformer on the train_train rows only, and applies it to
  both train_train and val -- see _build_search_inputs().
- refit=False: RandomizedSearchCV's automatic refit would train the
  final model on the train_train-only-Gaussianized array (for MLP),
  not on ds["X_train_nn"] (fit on the full X_train) -- a train/test
  feature-space mismatch. Instead, refit_and_evaluate() constructs a
  fresh model with search.best_params_ applied and fits it on
  ds["X_train"]/ds["X_train_nn"] directly, matching what ds["X_test"]/
  ds["X_test_nn"] were derived from.

RESOLVED: this module originally needed a local duplicate of
benchmark.py's evaluate(), because benchmark.py had no __main__ guard --
importing anything from it would have executed its full top-level
13-dataset run as a side effect, including rewriting benchmark_results.csv.
benchmark.py now has that guard (approved as an explicit, minimal,
behavior-preserving exception to this commit's "don't touch benchmark.py"
scope). evaluate() is imported directly below; no duplicate logic exists
in this file.
"""

import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, PredefinedSplit, RandomizedSearchCV
from sklearn.preprocessing import QuantileTransformer

from config import (
    DATASETS, MASTER_SEED,
    TUNING_N_ITER, TUNING_VAL_FRACTION, TUNING_VAL_SPLIT_SEED, TUNING_SEARCH_SEED,
)
from data_loader import load_dataset, validate_dataset_registry
from models import get_models
from hyperparameter_spaces import get_search_space
from benchmark import evaluate
from experiment_utils import (
    IncrementalCSVWriter, ExperimentTracker, log_stage,
    log_finished, log_failed, STAGE_COMPUTATION, STAGE_WRITE,
)

MODEL_NAMES = ["RandomForest", "GBT", "XGBoost", "MLP"]
# RF/GBT/XGBoost's own n_jobs is set to 1 for the base estimator passed
# into the search -- RandomizedSearchCV owns parallelism instead, avoiding
# the nested-parallelism-oversubscription problem flagged when models.py's
# n_jobs parameter was added. GBT/MLP have no n_jobs of their own.
BASE_ESTIMATOR_N_JOBS = {"RandomForest": 1, "XGBoost": 1, "GBT": -1, "MLP": -1}


def _make_predefined_split(n_train_train, n_val):
    """
    -1 for rows used as the search's training pool, 0 for the single
    validation fold (sklearn.model_selection.PredefinedSplit convention).
    Built once per dataset (see _split_for_tuning) and reused identically
    across all 4 models, so every model is tuned against literally the
    same validation rows, not just a same-sized independent draw.
    """
    test_fold = np.concatenate([
        np.full(n_train_train, -1),
        np.full(n_val, 0),
    ])
    return PredefinedSplit(test_fold)


def _split_for_tuning(ds):
    """
    Carves a validation split out of ds["X_train"]/ds["y_train"] only.
    Returns row *indices*, not data -- ds["X_test"]/ds["y_test"] are
    never referenced here.
    """
    n = len(ds["X_train"])
    idx = np.arange(n)
    stratify = ds["y_train"] if ds["task"] == "classification" else None

    idx_train_train, idx_val = train_test_split(
        idx, test_size=TUNING_VAL_FRACTION,
        random_state=TUNING_VAL_SPLIT_SEED, stratify=stratify,
    )
    return idx_train_train, idx_val


def _build_search_inputs(ds, idx_train_train, idx_val, model_name):
    """
    Returns (X, y, predefined_split) ready for RandomizedSearchCV.

    Trees: ds["X_train"] rows directly, reordered to [train_train rows,
    val rows] to match the PredefinedSplit's fold assignment.

    MLP: a QuantileTransformer fit on the train_train rows ONLY, applied
    to train_train and val -- not ds["X_train_nn"], which was fit on the
    full X_train (see module docstring).
    """
    order = np.concatenate([idx_train_train, idx_val])
    n_train_train = len(idx_train_train)

    if model_name == "MLP":
        X_raw = ds["X_train"].iloc[order].reset_index(drop=True)
        qt = QuantileTransformer(output_distribution="normal", random_state=MASTER_SEED)
        X_tt = qt.fit_transform(X_raw.iloc[:n_train_train])
        X_v = qt.transform(X_raw.iloc[n_train_train:])
        X = pd.DataFrame(np.vstack([X_tt, X_v]), columns=X_raw.columns)
    else:
        X = ds["X_train"].iloc[order].reset_index(drop=True)

    y = ds["y_train"].iloc[order].reset_index(drop=True)
    split = _make_predefined_split(n_train_train, len(idx_val))
    return X, y, split


def tune_one(model_name, ds):
    """
    Tunes a single (dataset, model) pair. Does not accept or reference
    test data -- the test set genuinely isn't in scope of this function,
    not merely unused by convention.
    """
    task = ds["task"]
    idx_train_train, idx_val = _split_for_tuning(ds)
    X, y, split = _build_search_inputs(ds, idx_train_train, idx_val, model_name)

    scoring = "accuracy" if task == "classification" else "r2"
    base_estimator = get_models(
        task, n_jobs=BASE_ESTIMATOR_N_JOBS[model_name]
    )[model_name]

    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=get_search_space(model_name, task),
        n_iter=TUNING_N_ITER,
        scoring=scoring,
        cv=split,
        refit=False,
        random_state=TUNING_SEARCH_SEED,
        n_jobs=-1,
    )
    start = time.perf_counter()
    search.fit(X, y)
    search_time = time.perf_counter() - start

    return search.best_params_, search.best_score_, search_time


def refit_and_evaluate(model_name, ds, best_params):
    """
    Fresh model, best_params applied, fit on ds["X_train"]/ds["X_train_nn"]
    -- data_loader.py's own versions, matching what ds["X_test"]/
    ds["X_test_nn"] were derived from -- then scored once on the test set.
    This is the only function in this module that touches test data, and
    it's called exactly once per (dataset, model), after tuning completes.
    """
    task = ds["task"]
    model = get_models(task)[model_name]
    model.set_params(**best_params)

    X_tr = ds["X_train_nn"] if model_name == "MLP" else ds["X_train"]
    X_te = ds["X_test_nn"] if model_name == "MLP" else ds["X_test"]

    return evaluate(model, X_tr, X_te, ds["y_train"], ds["y_test"], task)


def run_tuning_for_dataset(name):
    log_stage(name, "Loading...")
    ds = load_dataset(name)

    log_stage(name, "Running...")
    rows = []
    for model_name in MODEL_NAMES:
        best_params, val_score, search_time = tune_one(model_name, ds)
        test_score, metric, refit_time = refit_and_evaluate(model_name, ds, best_params)

        rows.append({
            "dataset": name,
            "task": ds["task"],
            "model": model_name,
            "metric": metric,
            "val_score": round(val_score, 4),
            "test_score": test_score,
            "best_params": json.dumps(best_params, default=str),
            "n_iter": TUNING_N_ITER,
            "time_s": round(search_time + refit_time, 2),
        })
    return rows


if __name__ == "__main__":
    validate_dataset_registry()

    run_start = time.perf_counter()
    writer = IncrementalCSVWriter("tuning_results.csv")
    tracker = ExperimentTracker()

    for name in DATASETS:
        start = time.perf_counter()
        try:
            rows = run_tuning_for_dataset(name)
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