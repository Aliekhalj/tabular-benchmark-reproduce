# visualize.py

"""
Plots for the benchmark and the two findings (noise sensitivity,
rotation invariance), across all 23 datasets.

Design note: with 23 datasets, one-subplot-per-dataset (the previous
version's grid layout) stops being readable. Every plot here instead
puts all datasets from one task type into a SINGLE panel -- datasets
as rows (benchmark), lines per dataset with an aggregate overlay
(finding 2), or dots per dataset with a distribution summary
(finding 3). The only place multiple panels appear is one-panel-per-
TASK-TYPE (classification vs regression), never one-panel-per-dataset.

Classification and regression are always plotted separately: Accuracy
and R\u00b2 are not on a comparable scale.

Inputs (unchanged CSV schemas, written by benchmark.py / finding2.py /
finding3.py -- this script only reads them):
  benchmark_results.csv : dataset, task, model, metric, score, time_s
  finding2_results.csv  : dataset, task, n_noise,
                           GBT_mean, GBT_std, MLP_mean, MLP_std
  finding3_results.csv  : dataset, task, setting,
                           GBT_mean, GBT_std, MLP_mean, MLP_std

Note finding2/finding3 only ever contain GBT and MLP columns --
finding2.py/finding3.py hardcode GBT as the sole tree representative
(see their own comments), so RandomForest/XGBoost are intentionally
absent from those two plots, not a bug here.

Outputs, written to FIGURES_DIR (created if missing):
  fig1_benchmark_classification.png / fig1_benchmark_regression.png
  fig2_benchmark_mean_rank.png
  fig3_finding2_classification.png / fig3_finding2_regression.png
  fig4_finding3_classification.png / fig4_finding3_regression.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIGURES_DIR = "figures"

MODELS_ALL = ["RandomForest", "GBT", "XGBoost", "MLP"]
MODEL_COLORS = {
    "RandomForest": "#2ca02c",  # green
    "GBT": "#1f77b4",           # blue
    "XGBoost": "#ff7f0e",       # orange
    "MLP": "#d62728",           # red
}
TASKS = ["classification", "regression"]
TASK_METRIC = {"classification": "Accuracy", "regression": "R\u00b2"}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _label(dataset_name):
    """Readable axis label for a dataset name, e.g. 'house_16H' -> 'House 16H'."""
    return dataset_name.replace("_", " ").title()


def _save(fig, filename):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def _require_columns(df, required, source):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source} is missing required column(s): {missing}")


# ── Figure 1: per-dataset benchmark scores, one panel per task ──────────────
def plot_benchmark_scores():
    df = pd.read_csv("benchmark_results.csv")
    _require_columns(df, {"dataset", "task", "model", "score"}, "benchmark_results.csv")

    for task in TASKS:
        sub = df[df["task"] == task]
        if sub.empty:
            print(f"No {task} rows in benchmark_results.csv -- skipping fig1_benchmark_{task}.png")
            continue

        pivot = sub.pivot(index="dataset", columns="model", values="score")
        present_models = [m for m in MODELS_ALL if m in pivot.columns]
        # Case-insensitive alphabetical sort: this project's dataset names mix
        # capitalization (e.g. "Ailerons", "MiamiHousing2016" vs "california"),
        # and a plain sort_index() would sort ASCII-style (all-capitals first),
        # not the a-to-z order a reader expects.
        pivot = pivot[present_models].sort_index(key=lambda idx: idx.str.lower())

        n = len(pivot)
        fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * n + 2)))

        data = pivot.values
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                        vmin=data.min(), vmax=data.max())

        ax.set_xticks(range(len(present_models)))
        ax.set_xticklabels(present_models, fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels([_label(d) for d in pivot.index], fontsize=8)

        # Cell gridlines (thin white lines between cells, standard heatmap style)
        ax.set_xticks(np.arange(-0.5, len(present_models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Value inside each cell. Text color picked per-cell from the
        # cell's own color luminance (via im.norm/im.cmap, so no extra
        # colormap object or import needed) so numbers stay legible at
        # both the dark-green and dark-red ends of the scale.
        for i in range(n):
            for j in range(len(present_models)):
                r, g, b, _ = im.cmap(im.norm(data[i, j]))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                        fontsize=7.5, color=text_color)

        ax.set_title(f"Benchmark \u2014 {task.title()} ({n} datasets)", fontsize=12, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(TASK_METRIC[task], fontsize=9)

        _save(fig, f"fig1_benchmark_{task}.png")


# ── Figure 2: mean rank across datasets, one panel per task ─────────────────
def plot_benchmark_mean_rank():
    df = pd.read_csv("benchmark_results.csv")
    _require_columns(df, {"dataset", "task", "model", "score"}, "benchmark_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, task in zip(axes, TASKS):
        sub = df[df["task"] == task]
        if sub.empty:
            ax.set_visible(False)
            continue

        pivot = sub.pivot(index="dataset", columns="model", values="score")
        present_models = [m for m in MODELS_ALL if m in pivot.columns]
        # Rank 1 = best (highest score) on that dataset; averaged across datasets.
        ranks = pivot[present_models].rank(axis=1, ascending=False)
        mean_rank = ranks.mean(axis=0).sort_values()

        colors = [MODEL_COLORS[m] for m in mean_rank.index]
        ax.barh(mean_rank.index, mean_rank.values, color=colors)
        ax.invert_yaxis()  # best (lowest rank) at top
        for i, v in enumerate(mean_rank.values):
            ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)

        ax.set_xlabel("Mean rank (1 = best)")
        ax.set_xlim(0, len(present_models) + 0.6)
        ax.set_title(f"{task.title()} (n={pivot.shape[0]})", fontsize=11)

    fig.suptitle("Benchmark \u2014 Mean Rank Across Datasets", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig2_benchmark_mean_rank.png")


# ── Figure 3: finding 2 -- noise sensitivity, one panel per task ────────────
def plot_finding2():
    df = pd.read_csv("finding2_results.csv")
    _require_columns(
        df, {"dataset", "task", "n_noise", "GBT_mean", "MLP_mean"}, "finding2_results.csv"
    )

    for task in TASKS:
        sub = df[df["task"] == task]
        if sub.empty:
            print(f"No {task} rows in finding2_results.csv -- skipping fig3_finding2_{task}.png")
            continue

        noise_levels = sorted(sub["n_noise"].unique())
        fig, ax = plt.subplots(figsize=(8, 5))

        # One thin line per dataset per model: score change from that
        # dataset's own noise=0 baseline. Collected into per-model dicts
        # so the aggregate (mean-across-datasets) line can be overlaid
        # afterward without a second pass over the CSV.
        per_model_deltas = {"GBT": [], "MLP": []}
        for dataset, dsub in sub.groupby("dataset"):
            dsub = dsub.sort_values("n_noise")
            base_row = dsub[dsub["n_noise"] == 0]
            if base_row.empty:
                continue  # this dataset has no noise=0 baseline row -- skip it, don't guess one
            for model in ["GBT", "MLP"]:
                base = base_row[f"{model}_mean"].values[0]
                delta = dsub[f"{model}_mean"].values - base
                ax.plot(dsub["n_noise"], delta, color=MODEL_COLORS[model],
                        alpha=0.25, linewidth=1, zorder=1)
                per_model_deltas[model].append(dict(zip(dsub["n_noise"], delta)))

        for model in ["GBT", "MLP"]:
            mean_delta = [
                np.mean([d[n] for d in per_model_deltas[model] if n in d])
                for n in noise_levels
            ]
            ax.plot(noise_levels, mean_delta, color=MODEL_COLORS[model],
                    linewidth=3, marker="o", zorder=3, label=f"{model} (mean)")

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Noise features added")
        ax.set_ylabel(f"Change in {TASK_METRIC[task]} from baseline (noise=0)")
        ax.set_title(f"Finding 2 \u2014 Noise Sensitivity, {task.title()} "
                     f"({sub['dataset'].nunique()} datasets)", fontsize=12, fontweight="bold")
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        ax.text(0.98, 0.02, "thin lines = individual datasets", transform=ax.transAxes,
                fontsize=7, color="gray", ha="right")
        ax.grid(linestyle="--", alpha=0.4)

        _save(fig, f"fig3_finding2_{task}.png")


# ── Figure 4: finding 3 -- rotation sensitivity, one panel per task ─────────
def plot_finding3():
    df = pd.read_csv("finding3_results.csv")
    _require_columns(
        df, {"dataset", "task", "setting", "GBT_mean", "MLP_mean"}, "finding3_results.csv"
    )

    rng = np.random.RandomState(42)  # fixed seed: jitter is for readability only, not a result

    for task in TASKS:
        sub = df[df["task"] == task]
        if sub.empty:
            print(f"No {task} rows in finding3_results.csv -- skipping fig4_finding3_{task}.png")
            continue

        deltas = {"GBT": [], "MLP": []}
        for dataset, dsub in sub.groupby("dataset"):
            orig = dsub[dsub["setting"] == "original"]
            rot = dsub[dsub["setting"] == "rotated"]
            if orig.empty or rot.empty:
                continue  # this dataset is missing one of the two settings -- skip it, don't guess
            for model in ["GBT", "MLP"]:
                deltas[model].append(rot[f"{model}_mean"].values[0] - orig[f"{model}_mean"].values[0])

        model_order = ["GBT","MLP"]
        positions = {m: i + 1 for i, m in enumerate(model_order)}

        fig, ax = plt.subplots(figsize=(8, 4))

        # Distribution summary: boxplot (median + IQR) per model
        bp = ax.boxplot(
            [deltas[m] for m in model_order],
            positions=[positions[m] for m in model_order],
            vert=False, widths=0.5, showfliers=False, patch_artist=True,
        )
        for patch, m in zip(bp["boxes"], model_order):
            patch.set_facecolor(MODEL_COLORS[m])
            patch.set_alpha(0.25)
        for median, m in zip(bp["medians"], model_order):
            median.set_color(MODEL_COLORS[m])
            median.set_linewidth(2)

        # Individual datasets: jittered dots on top of the boxplot
        for m in model_order:
            y_jitter = positions[m] + rng.uniform(-0.15, 0.15, len(deltas[m]))
            ax.scatter(deltas[m], y_jitter, color=MODEL_COLORS[m], alpha=0.7,
                       s=30, zorder=3, edgecolors="white", linewidths=0.4)

        ax.axvline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_yticks([positions[m] for m in model_order])
        ax.set_yticklabels(model_order)
        ax.set_xlabel(f"Change in {TASK_METRIC[task]} (rotated \u2212 original)")
        ax.set_title(f"Finding 3 \u2014 Rotation Sensitivity, {task.title()} "
                     f"({sub['dataset'].nunique()} datasets)", fontsize=12, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)

        _save(fig, f"fig4_finding3_{task}.png")


if __name__ == "__main__":
    plot_benchmark_scores()
    plot_benchmark_mean_rank()
    plot_finding2()
    plot_finding3()