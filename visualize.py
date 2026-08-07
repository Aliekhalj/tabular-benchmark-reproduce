# visualize.py

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# Two-model color scheme used in findings 2 and 3
C_GBT = "#1f77b4"  # blue
C_MLP = "#d62728"  # red

# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
})

DATASET_LABELS = {
    "bank_marketing": "Bank Marketing",
    "california": "California",
    "magic_telescope": "Magic Telescope",
}

# Per-panel width, derived from the original fixed 3-panel figsize (14, ...).
# Held constant regardless of grid shape, so panel readability doesn't
# degrade as dataset count grows -- only total figure width/height change.
PANEL_WIDTH = 14 / 3
PANEL_HEIGHT_BENCHMARK = 4.5
PANEL_HEIGHT_FINDING2 = 4.5
PANEL_HEIGHT_FINDING3 = 5

# Datasets up to this count stay in a single row. Chosen to match exactly
# the "3-4" case the layout is required to still handle well, and to
# reproduce today's single-row output for the current dataset count (3)
# without alteration.
SINGLE_ROW_MAX = 4


def _label(name):
    """Dataset display name for plot titles. Falls back to a readable
    auto-generated label for any dataset not yet in DATASET_LABELS."""
    return DATASET_LABELS.get(name, name.replace("_", " ").title())


def _grid_shape(n_datasets):
    """
    Determine (nrows, ncols) for the dataset panel grid.

    n_datasets <= SINGLE_ROW_MAX: single row (nrows=1, ncols=n_datasets).
    Reproduces the layout used before this commit for the current dataset
    count, and keeps small counts as a simple left-to-right comparison --
    the most readable option when everything already fits on one row.

    Beyond that: a near-square grid (ncols = ceil(sqrt(n))), which keeps
    panel width roughly constant as dataset count grows, rather than
    continuing to squeeze an ever-wider single row into a fixed figure
    width, and keeps empty trailing cells small relative to grid size
    across the 10-15 dataset target range (e.g. 12 datasets -> 3x4, 0
    empty cells; 15 -> 4x4, 1 empty cell).
    """
    if n_datasets <= SINGLE_ROW_MAX:
        return 1, n_datasets
    ncols = math.ceil(math.sqrt(n_datasets))
    nrows = math.ceil(n_datasets / ncols)
    return nrows, ncols


# ── Figure 1: Benchmark — dot plot, broken y-axis ────────────────────────────
def plot_benchmark():
    df = pd.read_csv("benchmark_results.csv")
    datasets = df["dataset"].unique()
    n_datasets = len(datasets)
    models = ["RandomForest", "GBT", "XGBoost", "MLP"]

    # Tree models share the blue family; MLP is red
    dot_colors = ["#1f77b4", "#2ca02c", "#aec7e8", "#d62728"]

    nrows, ncols = _grid_shape(n_datasets)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(PANEL_WIDTH * ncols, PANEL_HEIGHT_BENCHMARK * nrows),
        constrained_layout=True
    )
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(
        "Figure 1 — Benchmark: Tree Models vs MLP on Tabular Data",
        fontsize=13, fontweight="bold"
    )

    for ax, name in zip(axes, datasets):
        sub = df[df["dataset"] == name]
        metric = sub["metric"].iloc[0]
        scores = [sub[sub["model"] == m]["score"].values[0] for m in models]

        # Dot plot — one dot per model, y position = score
        for i, (score, color) in enumerate(zip(scores, dot_colors)):
            ax.scatter(i, score, color=color, s=120, zorder=4,
                       edgecolors="white", linewidths=0.8)
            ax.annotate(f"{score:.3f}",
                       xy=(i, score), xytext=(0, 8),
                       textcoords="offset points",
                       ha="center", fontsize=8.5)

        # Draw a faint horizontal line connecting all dots for readability
        ax.plot(range(len(models)), scores,
               color="gray", linewidth=0.8, linestyle="--",
               zorder=2, alpha=0.5)

        # Non-zero y-axis: show meaningful range only
        # Pad by 15% of the score range above and below
        lo, hi = min(scores), max(scores)
        pad = max((hi - lo) * 1.5, 0.03)  # at least 0.03 padding
        ax.set_ylim(lo - pad, hi + pad * 2)

        ax.set_title(_label(name), fontsize=11)
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)

        # Explicit axis-break note — scientific honesty
        ax.annotate("* y-axis does not start at 0",
                   xy=(0.01, 0.01), xycoords="axes fraction",
                   fontsize=7, color="gray")

    # Hide any unused grid cells (grid can exceed n_datasets, e.g. 13
    # datasets in a 4x4 grid leaves 3 empty)
    for ax in axes[n_datasets:]:
        ax.set_visible(False)

    # Shared legend
    handles = [
        plt.scatter([], [], color=c, s=80, label=m)
        for m, c in zip(models, dot_colors)
    ]
    fig.legend(handles=handles, labels=models,
              loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, -0.08), frameon=False)

    plt.savefig("fig1_benchmark.png", bbox_inches="tight")
    print("Saved fig1_benchmark.png")
    plt.show()


# ── Figure 2: Finding 2 — line plot (keep structure, fix colors) ──────────────
def plot_finding2():
    df = pd.read_csv("finding2_results.csv")
    datasets = df["dataset"].unique()
    n_datasets = len(datasets)

    nrows, ncols = _grid_shape(n_datasets)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(PANEL_WIDTH * ncols, PANEL_HEIGHT_FINDING2 * nrows),
        constrained_layout=True
    )
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(
        "Figure 2 — Finding 2: MLP Degrades More Under Uninformative Features",
        fontsize=13, fontweight="bold"
    )

    for ax, name in zip(axes, datasets):
        sub = df[df["dataset"] == name]
        task = sub["task"].iloc[0]
        metric = "Accuracy" if task == "classification" else "R²"

        for mname, col_m, col_s, color in [
            ("GBT", "GBT_mean", "GBT_std", C_GBT),
            ("MLP", "MLP_mean", "MLP_std", C_MLP),
        ]:
            ax.plot(sub["n_noise"], sub[col_m],
                   marker="o", color=color, label=mname,
                   linewidth=2, zorder=3)
            ax.fill_between(
                sub["n_noise"],
                sub[col_m] - sub[col_s],
                sub[col_m] + sub[col_s],
                alpha=0.15, color=color
            )

        # Annotate drop magnitude for GBT and MLP at noise=50
        for col_m, color, va in [("GBT_mean", C_GBT, "top"),
                                  ("MLP_mean", C_MLP, "bottom")]:
            start = sub[sub["n_noise"] == 0][col_m].values[0]
            end = sub[sub["n_noise"] == 50][col_m].values[0]
            drop = start - end
            ax.annotate(f"Δ={drop:.3f}",
                       xy=(50, end),
                       xytext=(-28, 8 if va == "bottom" else -14),
                       textcoords="offset points",
                       fontsize=8, color=color)

        ax.set_title(_label(name), fontsize=11)
        ax.set_xlabel("Noise features added")
        ax.set_ylabel(metric)
        ax.legend(frameon=False, fontsize=9)

    for ax in axes[n_datasets:]:
        ax.set_visible(False)

    plt.savefig("fig2_finding2.png", bbox_inches="tight")
    print("Saved fig2_finding2.png")
    plt.show()


# ── Figure 3: Finding 3 — slope chart ────────────────────────────────────────
# Slope charts are the correct encoding for before/after comparisons.
# They show direction and magnitude of change directly, without requiring
# the reader to mentally subtract two bars.
def plot_finding3():
    df = pd.read_csv("finding3_results.csv")
    datasets = df["dataset"].unique()
    n_datasets = len(datasets)

    nrows, ncols = _grid_shape(n_datasets)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(PANEL_WIDTH * ncols, PANEL_HEIGHT_FINDING3 * nrows),
        constrained_layout=True
    )
    axes = np.atleast_1d(axes).flatten()
    fig.suptitle(
        "Figure 3 — Finding 3: Trees Are Sensitive to Rotation, MLP Is Not",
        fontsize=13, fontweight="bold"
    )

    for ax, name in zip(axes, datasets):
        sub = df[df["dataset"] == name]
        task = sub["task"].iloc[0]
        metric = "Accuracy" if task == "classification" else "R²"

        orig = sub[sub["setting"] == "original"].iloc[0]
        rot = sub[sub["setting"] == "rotated"].iloc[0]

        x = [0, 1]  # original=0, rotated=1

        for mname, col_m, col_s, color in [
            ("GBT", "GBT_mean", "GBT_std", C_GBT),
            ("MLP", "MLP_mean", "MLP_std", C_MLP),
        ]:
            y_orig = orig[col_m]
            y_rot = rot[col_m]
            e_orig = orig[col_s]
            e_rot = rot[col_s]

            # Connecting line — slope tells the story
            ax.plot(x, [y_orig, y_rot],
                   color=color, linewidth=2.5, zorder=3,
                   label=mname,
                   marker="o", markersize=8,
                   markerfacecolor="white", markeredgewidth=2)

            # Error bars at each endpoint
            ax.errorbar([0], [y_orig], yerr=[e_orig],
                       fmt="none", color=color,
                       capsize=4, linewidth=1.5, zorder=4)
            ax.errorbar([1], [y_rot], yerr=[e_rot],
                       fmt="none", color=color,
                       capsize=4, linewidth=1.5, zorder=4)

            # Annotate values
            ax.text(-0.08, y_orig, f"{y_orig:.3f}",
                   ha="right", va="center", fontsize=8, color=color)
            ax.text(1.08, y_rot, f"{y_rot:.3f}",
                   ha="left", va="center", fontsize=8, color=color)

            # Annotate the change
            delta = y_rot - y_orig
            sign = "+" if delta >= 0 else ""
            mid_y = (y_orig + y_rot) / 2
            ax.text(0.5, mid_y, f"{sign}{delta:.3f}",
                   ha="center", va="bottom", fontsize=8,
                   color=color, style="italic")

        # Non-zero y-axis
        all_vals = [orig["GBT_mean"], orig["MLP_mean"],
                   rot["GBT_mean"], rot["MLP_mean"]]
        lo, hi = min(all_vals), max(all_vals)
        pad = max((hi - lo) * 1.2, 0.03)
        ax.set_ylim(lo - pad, hi + pad)

        ax.set_title(_label(name), fontsize=11)
        ax.set_ylabel(metric)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Original", "Rotated"], fontsize=10)
        ax.set_xlim(-0.4, 1.4)
        ax.legend(frameon=False, loc="upper right", fontsize=9)

        ax.annotate("* y-axis does not start at 0",
                   xy=(0.01, 0.01), xycoords="axes fraction",
                   fontsize=7, color="gray")

    for ax in axes[n_datasets:]:
        ax.set_visible(False)

    plt.savefig("fig3_finding3.png", bbox_inches="tight")
    print("Saved fig3_finding3.png")
    plt.show()


if __name__ == "__main__":
    plot_benchmark()
    plot_finding2()
    plot_finding3()