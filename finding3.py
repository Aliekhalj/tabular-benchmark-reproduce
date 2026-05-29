import warnings
import numpy as np
import pandas as pd
from scipy.stats import special_ortho_group
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import QuantileTransformer
from data_loader import load_dataset, DATASETS

warnings.filterwarnings("ignore", category=ConvergenceWarning)

N_ROTATIONS = 10  # number of random rotation matrices to average over
MODEL_SEEDS = [42, 7, 13]  # model initialization seeds to average over

def make_models(task, seed):
    """Both models receive identical input in this experiment —
    the goal is to isolate the effect of rotation, not preprocessing."""
    if task == "classification":
        return {
            "GBT": GradientBoostingClassifier(n_estimators=200, random_state=seed),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(256, 256),
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=seed
            ),
        }
    else:
        return {
            "GBT": GradientBoostingRegressor(n_estimators=200, random_state=seed),
            "MLP": MLPRegressor(
                hidden_layer_sizes=(256, 256),
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                random_state=seed
            ),
        }

def run_finding3(dataset_name):
    # GBT is used as the sole tree representative here.
    # It achieves the most consistent benchmark performance across all 3 datasets.
    # Using all tree models would clutter the slope charts without adding
    # new information — the goal is to isolate the rotation effect, not compare
    # tree architectures against each other.
    ds = load_dataset(dataset_name)
    task = ds["task"]
    n_features = ds["X_train"].shape[1]
    metric_fn = accuracy_score if task == "classification" else r2_score
    metric_name = "Accuracy" if task == "classification" else "R²"

    # Gaussianize features before rotation — same as the paper.
    # This removes scale differences between features so the rotation
    # is a pure orientation change, not confounded by feature scales.
    # Both models receive the same preprocessed data in this experiment.
    qt = QuantileTransformer(output_distribution="normal", random_state=42)
    X_train_g = qt.fit_transform(ds["X_train"])
    X_test_g  = qt.transform(ds["X_test"])

    scores = {"original": {"GBT": [], "MLP": []},
              "rotated":  {"GBT": [], "MLP": []}}

    # Original: vary only model seed — no rotation applied
    for seed in MODEL_SEEDS:
        models = make_models(task, seed)
        for mname, model in models.items():
            model.fit(X_train_g, ds["y_train"])
            scores["original"][mname].append(
                metric_fn(ds["y_test"], model.predict(X_test_g))
            )

    # Rotated: vary both rotation matrix and model seed
    for rot_seed in range(N_ROTATIONS):
        R = special_ortho_group.rvs(n_features, random_state=rot_seed)
        X_train_r = X_train_g @ R
        X_test_r  = X_test_g  @ R

        for seed in MODEL_SEEDS:
            models = make_models(task, seed)
            for mname, model in models.items():
                model.fit(X_train_r, ds["y_train"])
                scores["rotated"][mname].append(
                    metric_fn(ds["y_test"], model.predict(X_test_r))
                )

    # Report results
    print(f"\n--- Finding 3: {dataset_name} ({metric_name}) ---")
    print(f"  {'setting':>10} | {'GBT mean':>10} {'±std':>7} | "
          f"{'MLP mean':>10} {'±std':>7}")
    print(f"  {'-'*54}")

    rows = []
    for setting in ["original", "rotated"]:
        gbt_mean = np.mean(scores[setting]["GBT"])
        gbt_std  = np.std(scores[setting]["GBT"])
        mlp_mean = np.mean(scores[setting]["MLP"])
        mlp_std  = np.std(scores[setting]["MLP"])

        print(f"  {setting:>10} | {gbt_mean:>10.4f} {gbt_std:>7.4f} | "
              f"{mlp_mean:>10.4f} {mlp_std:>7.4f}")

        rows.append({
            "dataset":  dataset_name,
            "task":     task,
            "setting":  setting,
            "GBT_mean": round(gbt_mean, 4),
            "GBT_std":  round(gbt_std,  4),
            "MLP_mean": round(mlp_mean, 4),
            "MLP_std":  round(mlp_std,  4),
        })

    return pd.DataFrame(rows)

all_results = []
for name in DATASETS:
    df = run_finding3(name)
    all_results.append(df)

final = pd.concat(all_results, ignore_index=True)
final.to_csv("finding3_results.csv", index=False)
print("\nSaved to finding3_results.csv")
