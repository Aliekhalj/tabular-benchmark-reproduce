# benchmark.py

import time
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from config import DATASETS
from data_loader import load_dataset
from models import get_models


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
        print(f"  {model_name:15s} {metric}: {score} ({elapsed}s)")

        results.append({
            "dataset": name,
            "task": task,
            "model": model_name,
            "metric": metric,
            "score": score,
            "time_s": elapsed,
        })

df = pd.DataFrame(results)
df.to_csv("benchmark_results.csv", index=False)
print("\nResults saved to benchmark_results.csv")