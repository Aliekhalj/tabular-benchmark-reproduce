import time
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from data_loader import load_dataset, DATASETS

# Suppress convergence warnings — we handle this via early stopping instead
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def get_models(task):
    """
    Tree models use default-ish settings with a fixed random state.
    MLP uses early stopping instead of a hard iteration cap —
    this is more principled and matches the paper's approach.
    """
    if task == "classification":
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1
            ),
            "GBT": GradientBoostingClassifier(
                n_estimators=300, random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300, random_state=42,
                eval_metric="logloss", verbosity=0
            ),
            # 2 hidden layers matches the paper's MLP setup more closely
            # early_stopping replaces max_iter cutoff — fairer and cleaner
            "MLP": MLPClassifier(
                hidden_layer_sizes=(256, 256),
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            ),
        }
    else:
        return {
            "RandomForest": RandomForestRegressor(
                n_estimators=300, random_state=42, n_jobs=-1
            ),
            "GBT": GradientBoostingRegressor(
                n_estimators=300, random_state=42
            ),
            "XGBoost": XGBRegressor(
                n_estimators=300, random_state=42, verbosity=0
            ),
            "MLP": MLPRegressor(
                hidden_layer_sizes=(256, 256),
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=42
            ),
        }

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

results = []

for name in DATASETS:
    ds = load_dataset(name)
    task = ds["task"]
    models = get_models(task)

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
        print(f"  {model_name:15s} {metric}: {score}  ({elapsed}s)")

        results.append({
            "dataset": name,
            "task":    task,
            "model":   model_name,
            "metric":  metric,
            "score":   score,
            "time_s":  elapsed,
        })

df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)
print("\nResults saved to benchmark_results.csv")