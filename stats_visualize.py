# stats_visualize.py

"""
Visualization of the final statistical analysis (stats.py's PRIMARY
output only -- tuning_results_merged.csv's test_score results). No
new statistics are computed anywhere in this file: every number
plotted is read directly from stats_summary.csv / stats_pairwise.csv.
The one piece of layout logic that isn't a straight CSV read is
clique-finding for the CD diagram (which models' already-computed
ranks fall within the already-computed Nemenyi CD of each other) --
that's a drawing decision, not a statistical calculation.

stats_summary_cv.csv / stats_pairwise_cv.csv (the CV sensitivity
results) are intentionally not visualized here, per scope.

Inputs (unchanged CSV schemas, written by stats.py):
  stats_summary.csv  : task, model, mean_rank, mean_score, n_datasets,
                        friedman_chi2, friedman_p,
                        friedman_significant_0.05
  stats_pairwise.csv : task, model_a, model_b, mean_score_a,
                        mean_score_b, wilcoxon_stat, wilcoxon_p_raw,
                        cohens_d_z, mean_rank_diff, nemenyi_cd,
                        nemenyi_significant, wilcoxon_p_holm,
                        wilcoxon_significant_holm_0.05

Outputs, written to FIGURES_DIR (created if missing):
  fig5_stats_mean_rank.png
  fig6_critical_difference_classification.png
  fig6_critical_difference_regression.png
  fig7_pairwise_effects_classification.png
  fig7_pairwise_effects_regression.png
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

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


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


def _present_models(model_series):
    """Fixed MODELS_ALL order, restricted to whatever models actually
    appear in this task's rows (so a partial run doesn't crash)."""
    return [m for m in MODELS_ALL if m in set(model_series)]


# ── Figure 5: mean rank, one panel per task, Friedman result annotated ──────
def plot_mean_rank():
    summary = pd.read_csv("stats_summary.csv")
    _require_columns(
        summary,
        {"task", "model", "mean_rank", "n_datasets", "friedman_chi2",
         "friedman_p", "friedman_significant_0.05"},
        "stats_summary.csv",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, task in zip(axes, TASKS):
        sub = summary[summary["task"] == task].sort_values("mean_rank")
        if sub.empty:
            ax.set_visible(False)
            continue

        colors = [MODEL_COLORS[m] for m in sub["model"]]
        ax.barh(sub["model"], sub["mean_rank"], color=colors)
        ax.invert_yaxis()  # best (lowest rank) at top
        for i, v in enumerate(sub["mean_rank"]):
            ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=8)

        n = int(sub["n_datasets"].iloc[0])
        chi2 = sub["friedman_chi2"].iloc[0]
        p = sub["friedman_p"].iloc[0]
        sig = "significant" if sub["friedman_significant_0.05"].iloc[0] else "not significant"

        ax.set_xlabel("Mean rank (1 = best)")
        ax.set_xlim(0, len(MODELS_ALL) + 0.6)
        ax.set_title(
            f"{task.title()} (n={n})\n"
            f"Friedman \u03c7\u00b2={chi2:.2f}, p={p:.4g} ({sig})",
            fontsize=10,
        )

    fig.suptitle("Final Statistical Analysis \u2014 Mean Rank (test_score)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig5_stats_mean_rank.png")


# ── Figure 6: Critical Difference diagram, one file per task ────────────────
def _find_cliques(sorted_ranks, cd):
    """
    sorted_ranks: list of (model, rank), ascending by rank.
    Returns the maximal groups of models whose rank spread (max-min)
    stays within the Nemenyi CD -- i.e. groups not shown as
    significantly different. Both rank and cd are pre-computed values
    read from stats_pairwise.csv; this only decides which consecutive
    runs qualify and keeps the maximal ones (a run fully contained
    inside a larger qualifying run is dropped, standard CD-diagram
    convention).
    """
    n = len(sorted_ranks)
    intervals = []
    for i in range(n):
        j = i
        while j + 1 < n and sorted_ranks[j + 1][1] - sorted_ranks[i][1] <= cd:
            j += 1
        if j > i:
            intervals.append((i, j))
    maximal = [
        (a, b) for a, b in intervals
        if not any(a2 <= a and b <= b2 and (a2, b2) != (a, b) for a2, b2 in intervals)
    ]
    maximal = sorted(set(maximal))
    return [[sorted_ranks[k][0] for k in range(a, b + 1)] for a, b in maximal]


def plot_critical_difference():
    summary = pd.read_csv("stats_summary.csv")
    pairwise = pd.read_csv("stats_pairwise.csv")
    _require_columns(summary, {"task", "model", "mean_rank", "n_datasets"}, "stats_summary.csv")
    _require_columns(pairwise, {"task", "nemenyi_cd"}, "stats_pairwise.csv")

    k = len(MODELS_ALL)
    for task in TASKS:
        sub = summary[summary["task"] == task]
        pw = pairwise[pairwise["task"] == task]
        if sub.empty or pw.empty:
            print(f"No {task} rows -- skipping fig6_critical_difference_{task}.png")
            continue

        cd = pw["nemenyi_cd"].iloc[0]
        n = int(sub["n_datasets"].iloc[0])
        sorted_ranks = sorted(zip(sub["model"], sub["mean_rank"]), key=lambda t: t[1])

        fig, ax = plt.subplots(figsize=(7, 3.2))

        # Main rank axis (1 = best .. k = worst)
        ax.plot([1, k], [1, 1], color="black", linewidth=1)
        for x in range(1, k + 1):
            ax.plot([x, x], [0.95, 1.05], color="black", linewidth=1)
            ax.text(x, 1.18, str(x), ha="center", fontsize=9)

        # Each model: a stem up or down to its label, alternating sides
        # so labels don't overlap when ranks are close together.
        for idx, (model, rank) in enumerate(sorted_ranks):
            if idx % 2 == 0:
                ax.plot([rank, rank], [1, 1.45], color=MODEL_COLORS[model], linewidth=1.5)
                ax.text(rank, 1.5, model, ha="center", va="bottom", fontsize=9,
                        color=MODEL_COLORS[model], fontweight="bold")
            else:
                ax.plot([rank, rank], [1, 0.55], color=MODEL_COLORS[model], linewidth=1.5)
                ax.text(rank, 0.5, model, ha="center", va="top", fontsize=9,
                        color=MODEL_COLORS[model], fontweight="bold")

        # Clique bars: thick horizontal segments joining models that are
        # NOT significantly different under Nemenyi (rank spread <= CD).
        cliques = _find_cliques(sorted_ranks, cd)
        rank_of = dict(sorted_ranks)
        clique_y = 0.88
        for clique in cliques:
            ranks_c = [rank_of[m] for m in clique]
            ax.plot([min(ranks_c), max(ranks_c)], [clique_y, clique_y],
                    color="black", linewidth=4, solid_capstyle="butt")
            clique_y -= 0.09

        # CD reference scale, drawn once in a corner
        cd_x1 = k
        cd_x0 = k - cd
        ax.plot([cd_x0, cd_x1], [1.95, 1.95], color="gray", linewidth=2)
        ax.plot([cd_x0, cd_x0], [1.9, 2.0], color="gray", linewidth=2)
        ax.plot([cd_x1, cd_x1], [1.9, 2.0], color="gray", linewidth=2)
        ax.text((cd_x0 + cd_x1) / 2, 2.05, f"CD = {cd:.3f}", ha="center",
                fontsize=8, color="gray")

        ax.set_xlim(0.5, k + 0.5)
        ax.set_ylim(0.25, 2.2)
        ax.axis("off")
        ax.set_title(
            f"Critical Difference Diagram \u2014 {task.title()} (n={n} datasets)\n"
            f"Models joined by a bar are not significantly different (Nemenyi, \u03b1=0.05)",
            fontsize=11, fontweight="bold",
        )

        _save(fig, f"fig6_critical_difference_{task}.png")


# ── Figure 7: pairwise effect size + significance heatmap, one per task ─────
def plot_pairwise_heatmap():
    pairwise = pd.read_csv("stats_pairwise.csv")
    _require_columns(
        pairwise,
        {"task", "model_a", "model_b", "cohens_d_z", "wilcoxon_p_holm",
         "wilcoxon_significant_holm_0.05"},
        "stats_pairwise.csv",
    )

    for task in TASKS:
        sub = pairwise[pairwise["task"] == task]
        if sub.empty:
            print(f"No {task} rows -- skipping fig7_pairwise_effects_{task}.png")
            continue

        models = _present_models(pd.concat([sub["model_a"], sub["model_b"]]))
        idx = {m: i for i, m in enumerate(models)}
        k = len(models)

        d_matrix = np.full((k, k), np.nan)
        p_matrix = np.full((k, k), np.nan)
        sig_matrix = np.zeros((k, k), dtype=bool)
        for _, row in sub.iterrows():
            i, j = idx[row["model_a"]], idx[row["model_b"]]
            # d_matrix[i, j] > 0 means model_a (row i) outperforms model_b
            # (row diff was defined as pivot[a] - pivot[b] in stats.py).
            d_matrix[i, j] = row["cohens_d_z"]
            d_matrix[j, i] = -row["cohens_d_z"]
            p_matrix[i, j] = p_matrix[j, i] = row["wilcoxon_p_holm"]
            sig_matrix[i, j] = sig_matrix[j, i] = row["wilcoxon_significant_holm_0.05"]

        fig, ax = plt.subplots(figsize=(6, 5))
        vmax = np.nanmax(np.abs(d_matrix)) if k > 1 else 1.0
        im = ax.imshow(d_matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(k))
        ax.set_xticklabels(models, fontsize=9)
        ax.set_yticks(range(k))
        ax.set_yticklabels(models, fontsize=9)

        for i in range(k):
            for j in range(k):
                if i == j:
                    ax.text(j, i, "\u2014", ha="center", va="center", fontsize=11, color="gray")
                    continue
                marker = "*" if sig_matrix[i, j] else ""
                r, g, b, _ = im.cmap(im.norm(d_matrix[i, j]))
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "white" if luminance < 0.5 else "black"
                ax.text(j, i, f"d={d_matrix[i, j]:+.2f}\np={p_matrix[i, j]:.3f}{marker}",
                        ha="center", va="center", fontsize=7.5, color=text_color)

        ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cohen's d (row \u2212 column)", fontsize=9)

        ax.set_title(
            f"Pairwise Effect Size & Significance \u2014 {task.title()}\n"
            f"(* = significant, Holm-corrected Wilcoxon, \u03b1=0.05)",
            fontsize=11, fontweight="bold",
        )

        _save(fig, f"fig7_pairwise_effects_{task}.png")


if __name__ == "__main__":
    plot_mean_rank()
    plot_critical_difference()
    plot_pairwise_heatmap()