"""
Shared model factory for the tabular-benchmark-reproduce project.

get_models() is the single place model configurations are defined --
it replaces the two near-duplicate implementations that used to live
in benchmark.py (get_models) and finding2.py/finding3.py (make_models,
identical in both files). That duplication is exactly how the
300-vs-200 n_estimators mismatch happened; see config.py's comment
on BENCHMARK_N_ESTIMATORS / FINDING_N_ESTIMATORS for the full story.

Design note (Phase 0 review): this always builds and returns all four
models (RandomForest, GBT, XGBoost, MLP), even though finding2.py and
finding3.py only ever read "GBT" and "MLP" back out. Constructing an
unused sklearn/XGBoost estimator does no computation -- it just stores
hyperparameters until .fit() is called -- so the two unused objects
built on each finding2/3 call cost nothing measurable. One function,
one behavior, rather than a subsetting parameter to avoid something
that's already free.

Design note (Phase 0 review): this interface is PROVISIONAL. Phase 3
(hyperparameter tuning) needs each model paired with a search space
for RandomizedSearchCV, not a single fixed configuration -- different
enough that get_models() will likely be reworked or replaced then.
Not a place to over-invest further.

Output-preservation note: n_estimators is applied uniformly to
RandomForest, GBT, and XGBoost. benchmark.py's call leaves it at the
default (BENCHMARK_N_ESTIMATORS=300), matching benchmark.py exactly
for all three models. finding2.py/finding3.py call with
n_estimators=FINDING_N_ESTIMATORS (200); this also affects the
RandomForest/XGBoost objects built inside that call, but those two
are never used by finding2.py/finding3.py, so it has no observable
effect on their results -- only "GBT" (now correctly at 200, matching
today) and "MLP" (unaffected by n_estimators) are ever read.
"""

import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor

from config import (
    MASTER_SEED,
    BENCHMARK_N_ESTIMATORS,
    MLP_HIDDEN_LAYER_SIZES,
    MLP_MAX_ITER,
    MLP_VALIDATION_FRACTION,
    MLP_N_ITER_NO_CHANGE,
)

# Moved here from benchmark.py / finding2.py / finding3.py, which each
# set this identically. A module-level statement is the right scope --
# this is global environment setup, not per-caller behavior.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def get_models(task, seed=MASTER_SEED, n_estimators=BENCHMARK_N_ESTIMATORS):
    """
    Build the four-model roster (RandomForest, GBT, XGBoost, MLP) for
    a given task.

    task: "classification" or "regression"
    seed: random_state passed to every model. Defaults to MASTER_SEED,
        matching benchmark.py's fixed seed=42 today. finding2.py/
        finding3.py pass their per-iteration seed explicitly.
    n_estimators: tree count for RandomForest, GBT, and XGBoost.
        Defaults to BENCHMARK_N_ESTIMATORS (300), matching benchmark.py
        today. finding2.py/finding3.py pass FINDING_N_ESTIMATORS (200)
        explicitly -- see module docstring for why this is safe to
        apply uniformly even though only GBT is affected in practice.
    """
    mlp_kwargs = dict(
        hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
        max_iter=MLP_MAX_ITER,
        early_stopping=True,
        validation_fraction=MLP_VALIDATION_FRACTION,
        n_iter_no_change=MLP_N_ITER_NO_CHANGE,
        random_state=seed,
    )

    if task == "classification":
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=n_estimators, random_state=seed, n_jobs=-1
            ),
            "GBT": GradientBoostingClassifier(
                n_estimators=n_estimators, random_state=seed
            ),
            "XGBoost": XGBClassifier(
                n_estimators=n_estimators, random_state=seed,
                eval_metric="logloss", verbosity=0
            ),
            "MLP": MLPClassifier(**mlp_kwargs),
        }
    else:
        return {
            "RandomForest": RandomForestRegressor(
                n_estimators=n_estimators, random_state=seed, n_jobs=-1
            ),
            "GBT": GradientBoostingRegressor(
                n_estimators=n_estimators, random_state=seed
            ),
            "XGBoost": XGBRegressor(
                n_estimators=n_estimators, random_state=seed, verbosity=0
            ),
            "MLP": MLPRegressor(**mlp_kwargs),
        }