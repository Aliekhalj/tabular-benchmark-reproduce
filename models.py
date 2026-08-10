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

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def get_models(task, seed=MASTER_SEED, n_estimators=BENCHMARK_N_ESTIMATORS, n_jobs=-1):
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
                n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs
            ),
            "GBT": GradientBoostingClassifier(
                n_estimators=n_estimators, random_state=seed
            ),
            "XGBoost": XGBClassifier(
                n_estimators=n_estimators, random_state=seed,
                eval_metric="logloss", verbosity=0, n_jobs=n_jobs
            ),
            "MLP": MLPClassifier(**mlp_kwargs),
        }
    else:
        return {
            "RandomForest": RandomForestRegressor(
                n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs
            ),
            "GBT": GradientBoostingRegressor(
                n_estimators=n_estimators, random_state=seed
            ),
            "XGBoost": XGBRegressor(
                n_estimators=n_estimators, random_state=seed,
                verbosity=0, n_jobs=n_jobs
            ),
            "MLP": MLPRegressor(**mlp_kwargs),
        }