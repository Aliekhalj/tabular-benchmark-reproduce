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
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, PredefinedSplit, RandomizedSearchCV
from sklearn.preprocessing import QuantileTransformer

from config import (
    DATASETS, NEW_DATASETS, MASTER_SEED,
    TUNING_N_ITER, TUNING_VAL_FRACTION, TUNING_VAL_SPLIT_SEED, TUNING_SEARCH_SEED,
)
from data_loader import load_dataset, validate_dataset_registry
from models import get_models
from hyperparameter_spaces import get_search_space
from benchmark import evaluate
from experiment_utils import (
    IncrementalCSVWriter, ExperimentTracker, log_stage,
    log_finished, log_failed, select_datasets, STAGE_COMPUTATION, STAGE_WRITE,
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


def _load_completed_datasets(path):
    """
    Detect which datasets have been fully tuned in an existing results CSV.
    
    A dataset is considered COMPLETE if it has exactly 4 rows corresponding
    to the 4 expected models: RandomForest, GBT, XGBoost, MLP.
    
    Arguments:
        path: Path to the tuning results CSV (e.g., "tuning_results_new.csv")
    
    Returns:
        Set of dataset names that are complete. Empty set if file does not exist.
    
    Raises:
        ValueError: If the CSV is malformed or has inconsistent data:
        - Missing required columns (dataset, model)
        - A dataset has 4+ rows (indicates duplicate or unexpected model entries)
    
    Note: If a dataset has 1-3 rows (incomplete), it is NOT added to the
    returned set, so it will be rerun on the next invocation.
    """
    if not os.path.exists(path):
        return set()
    
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read existing {path}: {exc}")
    
    # Validate required columns exist
    required_cols = {"dataset", "model"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{path} is missing required columns. "
            f"Found: {set(df.columns)}, Expected: {required_cols}"
        )
    
    completed = set()
    expected_models = {"RandomForest", "GBT", "XGBoost", "MLP"}
    
    for dataset_name in df["dataset"].unique():
        subset = df[df["dataset"] == dataset_name]
        models_in_subset = set(subset["model"].unique())
        n_rows = len(subset)
        
        if n_rows == 4 and models_in_subset == expected_models:
            # Exactly 4 rows, all expected models present, no duplicates
            completed.add(dataset_name)
        elif n_rows < 4:
            # Incomplete: 0-3 rows. Will be rerun. Do not add to completed.
            pass
        else:
            # n_rows > 4 or unexpected model names: data corruption
            duplicates = [m for m in expected_models 
                         if len(subset[subset["model"] == m]) > 1]
            unexpected = models_in_subset - expected_models
            raise ValueError(
                f"Dataset '{dataset_name}' has {n_rows} rows with models {models_in_subset}. "
                f"Expected exactly 4 rows (one per model). "
                f"Duplicates: {duplicates if duplicates else 'none'}, "
                f"Unexpected: {unexpected if unexpected else 'none'}. "
                f"This indicates a data corruption issue. Please inspect {path}."
            )
    
    return completed


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", choices=["all", "new"], default="all")
    args = parser.parse_args()

    dataset_names = select_datasets(DATASETS, NEW_DATASETS, args.datasets)
    suffix = "" if args.datasets == "all" else "_new"
    output_path = f"tuning_results{suffix}.csv"

    validate_dataset_registry()  # always checks the FULL registry, deliberately

    # Detect which datasets are already complete
    try:
        completed_datasets = _load_completed_datasets(output_path)
    except ValueError as exc:
        print(f"Error reading {output_path}: {exc}")
        sys.exit(1)

    # Filter to datasets that still need to be run
    original_count = len(dataset_names)
    dataset_names = [d for d in dataset_names if d not in completed_datasets]
    new_count = len(dataset_names)

    # Log what's being skipped
    if completed_datasets:
        print(f"\n{len(completed_datasets)} completed dataset(s) detected, skipping:")
        for dataset_name in sorted(completed_datasets):
            log_stage(dataset_name, "Already complete, skipping")
        print()

    if not dataset_names:
        print(f"All {original_count} selected dataset(s) are complete. Nothing to do.")
        sys.exit(0)

    if len(completed_datasets) > 0:
        print(f"Running remaining {new_count} of {original_count} dataset(s).\n")

    # Create writer and load existing results (only from complete datasets)
    # Partial rows from incomplete datasets are excluded and will be replaced by
    # fresh complete runs. This prevents accumulation of partial rows (1-3 models
    # per dataset) which would corrupt the CSV on resume.
    writer = IncrementalCSVWriter(output_path)
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            # Keep only rows from datasets we're skipping (complete ones)
            complete_rows_df = existing_df[existing_df['dataset'].isin(completed_datasets)]
            writer.rows = complete_rows_df.to_dict('records')
            print(f"Loaded {len(writer.rows)} complete result rows from {output_path}")
            if len(existing_df) > len(complete_rows_df):
                partial_rows = len(existing_df) - len(complete_rows_df)
                print(f"Removed {partial_rows} incomplete/partial row(s) (will be replaced on rerun)\n")
            else:
                print()
        except Exception as exc:
            print(f"Error loading existing rows from {output_path}: {exc}")
            sys.exit(1)

    run_start = time.perf_counter()
    tracker = ExperimentTracker()

    for name in dataset_names:
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