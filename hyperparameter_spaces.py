# hyperparameter_spaces.py


"""
Paper-sourced hyperparameter search spaces for RandomForest, GBT, XGBoost,
and MLP.

Source: Grinsztajn, Oyallon, Varoquaux (2022), "Why do tree-based models
still outperform deep learning on tabular data?", arXiv:2207.08815,
Appendix A.3, Tables 5 (XGBoost), 6 (RandomForest), 7 (GradientBoosting),
3 (MLP).

This module defines search spaces only -- no RandomizedSearchCV wiring,
no evaluation logic. Every distribution here is either a plain Python
list (sampled uniformly by RandomizedSearchCV, or -- where the paper
specifies unequal weights -- a list with values repeated in the exact
proportion that reproduces those weights exactly, since every weight the
paper specifies is a multiple of 0.05), a native scipy.stats distribution
(loguniform, uniform, lognorm), or one of the two small custom
distributions below, both exposing .rvs(random_state=None) for direct
RandomizedSearchCV compatibility.
"""

import math
from scipy.stats import loguniform, randint, uniform, lognorm


class LogUniformInt:
    """
    Log-uniform distribution over integers, matching the paper's
    "LogUniformInt[low, high]" notation (Hyperopt-sklearn convention):
    continuous log-uniform sampling over [low, high], then rounded to
    the nearest integer. The paper's non-integer bounds (e.g. [9.5,
    3000.5]) are exactly this convention -- chosen to avoid rounding
    bias at the edges -- and are preserved here rather than rounded to
    whole numbers.
    """
    def __init__(self, low, high):
        self._dist = loguniform(low, high)

    def rvs(self, random_state=None):
        return int(round(self._dist.rvs(random_state=random_state)))


class MLPArchitecture:
    """
    Compound distribution reproducing the paper's MLP architecture
    search (Table 3): "Num layers" (UniformInt[1,8]) and "Layer size"
    (UniformInt[16,1024]) are independent parameters in the paper,
    combined into a network with `num_layers` layers all of `layer_size`
    width. sklearn's MLPClassifier/MLPRegressor has no equivalent two-
    parameter architecture spec -- hidden_layer_sizes is a single fixed
    tuple -- so this samples both independently on every .rvs() call and
    constructs the tuple, preserving the paper's independent-sampling
    intent rather than pre-generating a fixed list of candidate shapes.
    """
    def __init__(self):
        self._num_layers = randint(1, 9)      # UniformInt[1,8] inclusive
        self._layer_size = randint(16, 1025)   # UniformInt[16,1024] inclusive

    def rvs(self, random_state=None):
        n = self._num_layers.rvs(random_state=random_state)
        size = self._layer_size.rvs(random_state=random_state)
        return tuple([int(size)] * int(n))


def _weighted_list(values, weights):
    """
    Reproduces the paper's weighted discrete choices (e.g. "[2, 3]
    ([0.95, 0.05])") as a plain Python list with values repeated in
    exact proportion, sampled uniformly by RandomizedSearchCV. Exact,
    not approximate, because every weight the paper specifies is a
    multiple of 0.05 -- asserted below so this fails loudly rather than
    silently producing a wrong ratio if that assumption is ever violated
    by a value added here later.
    """
    assert all(abs(w * 20 - round(w * 20)) < 1e-9 for w in weights), (
        "weights must be exact multiples of 0.05 for _weighted_list to "
        "reproduce them exactly"
    )
    assert abs(sum(weights) - 1.0) < 1e-9, "weights must sum to 1.0"
    out = []
    for v, w in zip(values, weights):
        out.extend([v] * round(w * 20))
    return out


# ── RandomForest -- Table 6. Clean match: paper's RF is sklearn's
# RandomForestClassifier/Regressor, same as this project already uses. ──

def _rf_space(task):
    space = {
        "max_depth": _weighted_list([None, 2, 3, 4], [0.7, 0.1, 0.1, 0.1]),
        "n_estimators": LogUniformInt(9.5, 3000.5),
        # Paper lists "sqrt" twice (no explicit weight parentheses on this
        # row, unlike the others) -- preserved literally as a 13-item list
        # so "sqrt" is sampled with double the probability of every other
        # option, matching the paper's table exactly as printed.
        "max_features": ["sqrt", "sqrt", "log2", None,
                          0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "min_samples_split": _weighted_list([2, 3], [0.95, 0.05]),
        "min_samples_leaf": LogUniformInt(1.5, 50.5),
        "bootstrap": [True, False],
        "min_impurity_decrease": _weighted_list(
            [0.0, 0.01, 0.02, 0.05], [0.85, 0.05, 0.05, 0.05]
        ),
    }
    if task == "classification":
        space["criterion"] = ["gini", "entropy"]
    else:
        space["criterion"] = ["squared_error", "absolute_error"]
    return space


# ── GradientBoosting -- Table 7. Paper uses plain GradientBoostingTrees
# for numerical-features-only datasets (HistGradientBoostingTrees is
# used only "when using categorical features", Sec 4.1) -- all 13 of
# this project's datasets are drawn from the paper's numerical-only
# tracks, so GradientBoostingClassifier/Regressor (already used in this
# project's models.py) is the correct, paper-consistent choice here, not
# a gap to flag. ──

def _gbt_space(task):
    space = {
        # LogNormal[log(0.01), log(10)]: standard log-normal notation,
        # underlying Normal(mu=log(0.01), sigma=log(10)). scipy.stats.lognorm
        # parameterizes as Normal(mean=log(scale), std=s), so s=log(10),
        # scale=0.01.
        "learning_rate": lognorm(s=math.log(10), scale=0.01),
        "subsample": uniform(loc=0.5, scale=0.5),  # Uniform[0.5, 1]
        "n_estimators": LogUniformInt(10.5, 1000.5),
        # friedman_mse / squared_error only -- current sklearn versions
        # still support both; this is the only appearance of "Friedman"
        # anywhere in the paper (a split-quality criterion, unrelated to
        # any dataset).
        "criterion": ["friedman_mse", "squared_error"],
        "max_depth": _weighted_list(
            [None, 2, 3, 4, 5], [0.1, 0.1, 0.6, 0.1, 0.1]
        ),
        "min_samples_split": _weighted_list([2, 3], [0.95, 0.05]),
        "min_samples_leaf": LogUniformInt(1.5, 50.5),
        "min_impurity_decrease": _weighted_list(
            [0.0, 0.01, 0.02, 0.05], [0.85, 0.05, 0.05, 0.05]
        ),
        "max_leaf_nodes": _weighted_list(
            [None, 5, 10, 15], [0.85, 0.05, 0.05, 0.05]
        ),
    }
    if task == "classification":
        # Paper: [deviance, exponential]. sklearn renamed "deviance" to
        # "log_loss" (GradientBoostingClassifier's `loss` parameter no
        # longer accepts "deviance" in current versions) -- same paper-
        # specified option, current API name.
        space["loss"] = ["log_loss", "exponential"]
    else:
        space["loss"] = ["squared_error", "absolute_error", "huber"]
    return space


# ── XGBoost -- Table 5. Clean match: same parameters, two renamed via
# XGBoost's sklearn-compatible API ("lambda"/"alpha" are Python reserved-
# word-adjacent, renamed reg_lambda/reg_alpha in XGBClassifier/Regressor). ──

def _xgboost_space(task):
    return {
        "max_depth": randint(1, 12),               # UniformInt[1, 11]
        # UniformInt[100, 6000, 200]: Hyperopt-style quniform notation
        # (uniform, quantized to steps of 200) -- represented as the
        # exact, finite enumerated list rather than a custom distribution,
        # since it's small and this is simplest and exact.
        "n_estimators": list(range(100, 6001, 200)),
        "min_child_weight": LogUniformInt(1, 100),    # LogUniformInt[1, 1e2]
        "subsample": uniform(loc=0.5, scale=0.5),      # Uniform[0.5, 1]
        "learning_rate": loguniform(1e-5, 0.7),
        "colsample_bylevel": uniform(loc=0.5, scale=0.5),
        "colsample_bytree": uniform(loc=0.5, scale=0.5),
        "gamma": loguniform(1e-8, 7),
        "reg_lambda": loguniform(1, 4),      # paper: "Lambda"
        "reg_alpha": loguniform(1e-8, 1e2),  # paper: "Alpha"
    }


# ── MLP -- Table 3. IMPORTANT: the paper's MLP (Gorishniy et al. 2021's
# PyTorch implementation) does not correspond to sklearn's MLPClassifier/
# MLPRegressor, which this project uses. Several of the paper's tunable
# parameters have no sklearn equivalent at all and are excluded below
# rather than approximated. Only what both (a) the paper specifies and
# (b) sklearn's MLP actually supports is included. ──

def _mlp_space(task):
    return {
        # Table 3 "Num layers" x "Layer size" -> sklearn's hidden_layer_sizes.
        # No direct sklearn equivalent for this two-parameter architecture
        # spec; see MLPArchitecture docstring.
        "hidden_layer_sizes": MLPArchitecture(),
        "learning_rate_init": loguniform(1e-5, 1e-2),
        "batch_size": [256, 512, 1024],
        # NOT included, and why:
        #   dropout             -- paper-specified (Uniform[0,0.5]), but
        #                          sklearn's MLP has no dropout parameter
        #                          at all. Not approximable with anything
        #                          else in the sklearn API.
        #   learning-rate sched -- paper-specified (True/False, PyTorch
        #                          ReduceLROnPlateau); sklearn's
        #                          `learning_rate` string param only
        #                          applies to solver='sgd', not 'adam'.
        #   category embedding  -- paper-specified but not applicable:
        #                          this project's 13 datasets are all
        #                          numerical-only, no categorical features.
        #   alpha (L2)          -- sklearn supports this, but the paper's
        #                          Table 3 does not specify any range for
        #                          it at all (weight_decay appears only
        #                          in Table 2, Resnet's space, not MLP's).
        #                          Excluded rather than inventing a range.
    }


def get_search_space(model_name, task):
    """
    Returns the paper-sourced RandomizedSearchCV-compatible
    param_distributions dict for the given model and task
    ("classification" or "regression"), mirroring models.get_models(task)'s
    existing task-aware shape.
    """
    builders = {
        "RandomForest": _rf_space,
        "GBT": _gbt_space,
        "XGBoost": _xgboost_space,
        "MLP": _mlp_space,
    }
    if model_name not in builders:
        raise ValueError(f"No search space defined for model '{model_name}'")
    return builders[model_name](task)


if __name__ == "__main__":
    from sklearn.model_selection import ParameterSampler
    from models import get_models

    for model_name in ["RandomForest", "GBT", "XGBoost", "MLP"]:
        for task in ["classification", "regression"]:
            space = get_search_space(model_name, task)

            # Parameter names match the actual estimator's API.
            estimator = get_models(task)[model_name]
            valid_params = set(estimator.get_params().keys())
            unsupported = set(space.keys()) - valid_params
            assert not unsupported, (
                f"{model_name}/{task}: search space has parameters not "
                f"recognized by the estimator: {unsupported}"
            )

            # Compatible with RandomizedSearchCV -- ParameterSampler is
            # the exact internal mechanism RandomizedSearchCV uses to draw
            # combinations from param_distributions; sampling directly
            # from it is a precise test that these objects work as
            # RandomizedSearchCV will actually use them, not an
            # approximation of that behavior.
            sampler = ParameterSampler(space, n_iter=20, random_state=0)
            samples = list(sampler)
            assert len(samples) == 20

            print(f"{model_name:12s} {task:15s} OK "
                  f"({len(space)} params) e.g. {samples[0]}")

    print("\nAll search spaces importable, parameter names valid, "
          "RandomizedSearchCV-compatible.")