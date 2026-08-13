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
    validate_dataset_registry()
    tuned_params = _load_tuned_params()

    run_start = time.perf_counter()
    fold_writer = IncrementalCSVWriter("cv_fold_results.csv")
    agg_writer = IncrementalCSVWriter("cv_results.csv")
    tracker = ExperimentTracker()

    for name in DATASETS:
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