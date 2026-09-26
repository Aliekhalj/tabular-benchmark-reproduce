"""
Phase 3: 5-fold cross-validation of the already-tuned (frozen)
hyperparameters from tuning_results.csv, within the original 70%
training pool only.

Confirmed design (Phase 3 audit, both open questions resolved by the
user before this was written):

- Fold pool = A1: folds are carved from ds["X_train"]/ds["y_train"]
  only. ds["X_test"]/ds["X_test_nn"]/ds["y_test"] are never referenced
  by any function in this file -- structural, not a matter of care,
  the same discipline tune.py already applies to the real test set.
  CV here is a supplementary robustness/variance estimate alongside the
  existing single-split test_score in tuning_results.csv, not a
  replacement for it.
- Hyperparameters are frozen, read directly from tuning_results.csv --
  no search, no RandomizedSearchCV, no re-selection of any kind
  anywhere in this file. This is NOT nested CV.
- MLP preprocessing follows the exact precedent already verified in
  tune.py's _build_search_inputs(): a fresh QuantileTransformer fit on
  each fold's training rows only, applied to that fold's training and
  validation rows -- never ds["X_train_nn"], which was fit on the full
  X_train and would leak this fold's validation rows into its own fit.
  random_state=MASTER_SEED for the transform itself, matching tune.py's
  established division of labor (MASTER_SEED governs transform/model
  initialization; a dedicated seed governs row partitioning --
  CV_FOLD_SEED plays that role here, the same way TUNING_VAL_SPLIT_SEED
  does in tune.py).
- Fold construction: StratifiedKFold for classification (consistent
  with every other split in this project), plain KFold for regression,
  both shuffle=True, random_state=CV_FOLD_SEED.
- Failure-isolation granularity matches tune.py: per-dataset, not per
  (dataset, model) -- a single model's CV failure loses that dataset's
  CV rows for all 4 models. Matches tune.py's existing tradeoff exactly,
  not a new decision.

Two outputs: cv_fold_results.csv (one row per dataset/model/fold, 260
rows if all succeed) and cv_results.csv (one aggregate row per
dataset/model, 52 rows if all succeed).

KNOWN, ACCEPTED LIMITATION, flagged rather than engineered around:
run_cv_for_dataset() returns rows for two separate IncrementalCSVWriter
instances, written in one try block. If the first write succeeds and
the second fails, the two output files could briefly disagree about
that one dataset until fixed and rerun. Not solved here -- would need a
transactional multi-file writer, which is more machinery than this
narrow, never-yet-observed failure mode (the write step never failed
once across the entire 18.9-hour tuning run) justifies building.
"""

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

from config import DATASETS, MASTER_SEED, CV_N_FOLDS, CV_FOLD_SEED
from data_loader import load_dataset, validate_dataset_registry
from models import get_models
from benchmark import evaluate
from tune import MODEL_NAMES
from experiment_utils import (
    IncrementalCSVWriter, ExperimentTracker, log_stage,
    log_finished, log_failed, STAGE_COMPUTATION, STAGE_WRITE,
)


def _load_tuned_params(path="tuning_results.csv"):
    """
    best_params keyed by (dataset, model), read once from
    tuning_results.csv. Frozen input to this whole module -- no search
    occurs anywhere in this file.
    """
    tuning = pd.read_csv(path)
    return {
        (row["dataset"], row["model"]): json.loads(row["best_params"])
        for _, row in tuning.iterrows()
    }


def _load_completed_datasets(agg_path, fold_path):
    """
    Resume/skip support, mirroring tune.py's _load_completed_datasets()
    but requiring agreement between CV's two output files.

    A dataset is COMPLETE only if BOTH:
      - agg_path has exactly 4 rows for it (one per model in
        {RandomForest, GBT, XGBoost, MLP}, no duplicates/extras).
      - fold_path has exactly 20 rows for it: each of those 4 models
        has exactly CV_N_FOLDS rows, with fold_index values
        {0, ..., CV_N_FOLDS-1} and no duplicate (model, fold_index)
        pairs.

    A dataset missing or under-represented in either file (0 up to
    one row short of complete) is simply left out of the returned set
    -- will be rerun, from scratch, in both files. A dataset with a
    row count/structure that can't happen from a clean run (too many
    rows, duplicate model/fold entries, unexpected model names) is a
    data-corruption signal, not a "just rerun it" situation, so it
    raises ValueError instead of being silently treated as incomplete.

    Returns the set of dataset names complete in both files. A missing
    file contributes no complete datasets (equivalent to a fresh run
    for that file) rather than raising.
    """
    expected_models = {"RandomForest", "GBT", "XGBoost", "MLP"}

    def _agg_complete(path):
        if not os.path.exists(path):
            return set()
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Failed to read existing {path}: {exc}")

        required_cols = {"dataset", "model"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"{path} is missing required columns. "
                f"Found: {set(df.columns)}, Expected: {required_cols}"
            )

        complete = set()
        for name, subset in df.groupby("dataset"):
            n = len(subset)
            models = set(subset["model"])
            if n == 4 and models == expected_models:
                complete.add(name)
            elif n < 4:
                continue  # incomplete -- dataset will be rerun
            else:
                dup = [m for m in expected_models if (subset["model"] == m).sum() > 1]
                unexpected = models - expected_models
                raise ValueError(
                    f"Dataset '{name}' in {path} has {n} rows with models {models}. "
                    f"Expected exactly 4 (one per model). "
                    f"Duplicates: {dup or 'none'}, Unexpected: {unexpected or 'none'}. "
                    f"Inspect {path} before rerunning."
                )
        return complete

    def _fold_complete(path):
        if not os.path.exists(path):
            return set()
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise ValueError(f"Failed to read existing {path}: {exc}")

        required_cols = {"dataset", "model", "fold_index"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"{path} is missing required columns. "
                f"Found: {set(df.columns)}, Expected: {required_cols}"
            )

        expected_folds = set(range(CV_N_FOLDS))
        expected_total = len(expected_models) * CV_N_FOLDS  # 20 at CV_N_FOLDS=5

        complete = set()
        for name, subset in df.groupby("dataset"):
            n = len(subset)
            models = set(subset["model"])
            pairs = list(zip(subset["model"], subset["fold_index"]))
            no_dup_pairs = len(set(pairs)) == len(pairs)
            per_model_folds_ok = no_dup_pairs and all(
                set(subset.loc[subset["model"] == m, "fold_index"]) == expected_folds
                for m in expected_models
            )

            if n == expected_total and models == expected_models and per_model_folds_ok:
                complete.add(name)
            elif n < expected_total:
                continue  # incomplete -- dataset will be rerun
            else:
                dup_pairs = sorted({p for p in pairs if pairs.count(p) > 1})
                unexpected = models - expected_models
                raise ValueError(
                    f"Dataset '{name}' in {path} has {n} rows spanning models {models}. "
                    f"Expected exactly {expected_total} "
                    f"({len(expected_models)} models x {CV_N_FOLDS} folds), no duplicates. "
                    f"Duplicate (model, fold_index) pairs: {dup_pairs or 'none'}, "
                    f"Unexpected models: {unexpected or 'none'}. "
                    f"Inspect {path} before rerunning."
                )
        return complete

    return _agg_complete(agg_path) & _fold_complete(fold_path)


def _make_folds(ds):
    """
    Returns CV_N_FOLDS (train_idx, val_idx) index-array pairs, drawn
    exclusively from ds["X_train"]/ds["y_train"]'s own row range.
    ds["X_test"]/ds["y_test"] are never referenced here.
    """
    n = len(ds["X_train"])
    if ds["task"] == "classification":
        splitter = StratifiedKFold(
            n_splits=CV_N_FOLDS, shuffle=True, random_state=CV_FOLD_SEED
        )
        return list(splitter.split(np.arange(n), ds["y_train"]))
    else:
        splitter = KFold(
            n_splits=CV_N_FOLDS, shuffle=True, random_state=CV_FOLD_SEED
        )
        return list(splitter.split(np.arange(n)))


def _fold_data(ds, train_idx, val_idx, model_name):
    """
    Same conceptual pattern as tune.py's _build_search_inputs(), adapted
    for K rotating folds instead of one fixed validation split.
    """
    if model_name == "MLP":
        X_tr_raw = ds["X_train"].iloc[train_idx]
        X_va_raw = ds["X_train"].iloc[val_idx]
        qt = QuantileTransformer(output_distribution="normal", random_state=MASTER_SEED)
        X_tr = pd.DataFrame(qt.fit_transform(X_tr_raw), columns=X_tr_raw.columns)
        X_va = pd.DataFrame(qt.transform(X_va_raw), columns=X_va_raw.columns)
    else:
        X_tr = ds["X_train"].iloc[train_idx].reset_index(drop=True)
        X_va = ds["X_train"].iloc[val_idx].reset_index(drop=True)

    y_tr = ds["y_train"].iloc[train_idx].reset_index(drop=True)
    y_va = ds["y_train"].iloc[val_idx].reset_index(drop=True)
    return X_tr, X_va, y_tr, y_va


def cv_one(model_name, ds, best_params):
    """
    Runs CV_N_FOLDS folds for one (dataset, model), frozen best_params,
    no search of any kind. Never accepts or references ds["X_test"]/
    ds["X_test_nn"]/ds["y_test"] -- structural, matching tune.py's own
    discipline for the real test set.
    """
    task = ds["task"]
    folds = _make_folds(ds)

    results = []
    for fold_index, (train_idx, val_idx) in enumerate(folds):
        X_tr, X_va, y_tr, y_va = _fold_data(ds, train_idx, val_idx, model_name)

        model = get_models(task)[model_name]
        model.set_params(**best_params)

        score, metric, _ = evaluate(model, X_tr, X_va, y_tr, y_va, task)
        results.append((fold_index, score, metric))
    return results


def run_cv_for_dataset(name, tuned_params):
    log_stage(name, "Loading...")
    ds = load_dataset(name)

    log_stage(name, "Running...")
    fold_rows, agg_rows = [], []
    for model_name in MODEL_NAMES:
        best_params = tuned_params[(name, model_name)]
        fold_results = cv_one(model_name, ds, best_params)

        scores = [s for _, s, _ in fold_results]
        metric = fold_results[0][2]

        for fold_index, score, _ in fold_results:
            fold_rows.append({
                "dataset": name, "task": ds["task"], "model": model_name,
                "fold_index": fold_index, "score": score,
            })

        agg_rows.append({
            "dataset": name, "task": ds["task"], "model": model_name,
            "metric": metric,
            "mean_score": round(float(np.mean(scores)), 4),
            "std_score": round(float(np.std(scores)), 4),
            "n_folds": len(scores),
        })
    return fold_rows, agg_rows


if __name__ == "__main__":

    
    dataset_names = list(DATASETS)
    agg_path = f"cv_results.csv"
    fold_path = f"cv_fold_results.csv"

    validate_dataset_registry()
    tuned_params = _load_tuned_params(path=f"tuning_results.csv")

    # Resume/skip: a dataset is only skipped if BOTH cv_results and
    # cv_fold_results already hold a complete result for it. Otherwise
    # it's rerun from scratch in both files -- see _load_completed_datasets().
    try:
        completed_datasets = _load_completed_datasets(agg_path, fold_path)
    except ValueError as exc:
        print(f"Error reading existing CV results: {exc}")
        sys.exit(1)

    original_count = len(dataset_names)
    dataset_names = [d for d in dataset_names if d not in completed_datasets]

    if completed_datasets:
        print(f"\n{len(completed_datasets)} completed dataset(s) detected, skipping:")
        for name in sorted(completed_datasets):
            log_stage(name, "Already complete, skipping")
        print()

    if not dataset_names:
        print(f"All {original_count} selected dataset(s) are complete. Nothing to do.")
        sys.exit(0)

    if completed_datasets:
        print(f"Running remaining {len(dataset_names)} of {original_count} dataset(s).\n")

    run_start = time.perf_counter()
    fold_writer = IncrementalCSVWriter(fold_path)
    agg_writer = IncrementalCSVWriter(agg_path)

    # Preload only rows belonging to datasets we're skipping (complete in
    # both files). Rows for any other dataset -- including one complete
    # in only one of the two files -- are excluded here, so a rerun of
    # that dataset can't accumulate on top of stale/partial rows. Same
    # fix already applied to tune.py's resume logic.
    if os.path.exists(fold_path):
        try:
            existing_fold = pd.read_csv(fold_path)
            fold_writer.rows = existing_fold[
                existing_fold["dataset"].isin(completed_datasets)
            ].to_dict("records")
        except Exception as exc:
            print(f"Error loading existing rows from {fold_path}: {exc}")
            sys.exit(1)

    if os.path.exists(agg_path):
        try:
            existing_agg = pd.read_csv(agg_path)
            agg_writer.rows = existing_agg[
                existing_agg["dataset"].isin(completed_datasets)
            ].to_dict("records")
        except Exception as exc:
            print(f"Error loading existing rows from {agg_path}: {exc}")
            sys.exit(1)

    tracker = ExperimentTracker()

    for name in dataset_names:
        start = time.perf_counter()
        try:
            fold_rows, agg_rows = run_cv_for_dataset(name, tuned_params)
        except Exception as exc:
            tracker.record_failure(name, exc, stage=STAGE_COMPUTATION)
            log_failed(name, STAGE_COMPUTATION, start, exc)
            continue

        try:
            fold_writer.add_rows(fold_rows)
            agg_writer.add_rows(agg_rows)
        except Exception as exc:
            tracker.record_failure(name, exc, stage=STAGE_WRITE)
            log_failed(name, STAGE_WRITE, start, exc)
            continue

        tracker.record_success(name)
        log_finished(name, start)

    total_runtime = time.perf_counter() - run_start
    tracker.print_summary(total_runtime)
    print(f"\nResults saved to {fold_writer.path} and {agg_writer.path}")