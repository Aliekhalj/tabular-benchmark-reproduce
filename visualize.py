import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── consistent colors throughout all figures ──────────────────────────────────
COLORS = {
    "RandomForest": "#2196F3",
    "GBT":          "#4CAF50",
    "XGBoost":      "#FF9800",
    "MLP":          "#F44336",
}

plt.rcParams.update({
    "figure.dpi":      150,
    "font.size":       11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Figure 1: Benchmark ───────────────────────────────────────────────────────
def plot_benchmark():
    df = pd.read_csv("benchmark_results.csv")
    datasets = df["dataset"].unique()
    models   = ["RandomForest", "GBT", "XGBoost", "MLP"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Figure 1 — Benchmark: Tree Models vs MLP on Tabular Data",
        fontsize=13, fontweight="bold", y=1.02
    )

    bar_width = 0.18
    x = np.arange(len(models))

    for ax, name in zip(axes, datasets):
        sub = df[df["dataset"] == name]
        metric = sub["metric"].iloc[0]

        for i, mname in enumerate(models):
            row = sub[sub["model"] == mname]
            if row.empty:
                continue
            score = row["score"].values[0]
            ax.bar(
                i, score,
                width=bar_width * 3,
                color=COLORS[mname],
                label=mname,
                zorder=3
            )
            ax.text(i, score + 0.005, f"{score:.3f}",
                    ha="center", va="bottom", fontsize=8.5)

        ax.set_title(name.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

        # Highlight MLP bar with a red edge to draw attention
        ax.patches[3].set_edgecolor("black")
        ax.patches[3].set_linewidth(1.2)

    handles = [plt.Rectangle((0,0),1,1, color=COLORS[m]) for m in models]
    fig.legend(handles, models, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.08), frameon=False)

    plt.tight_layout()
    plt.savefig("fig1_benchmark.png", bbox_inches="tight")
    print("Saved fig1_benchmark.png")
    plt.show()


# ── Figure 2: Finding 2 — Uninformative Features ─────────────────────────────
def plot_finding2():
    df = pd.read_csv("finding2_results.csv")
    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Figure 2 — Finding 2: MLP Degrades More Under Uninformative Features",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ax, name in zip(axes, datasets):
        sub  = df[df["dataset"] == name]
        task = sub["task"].iloc[0]
        metric = "Accuracy" if task == "classification" else "R²"

        for mname, col_mean, col_std, color in [
            ("GBT", "GBT_mean", "GBT_std", COLORS["GBT"]),
            ("MLP", "MLP_mean", "MLP_std", COLORS["MLP"]),
        ]:
            ax.plot(sub["n_noise"], sub[col_mean],
                    marker="o", color=color, label=mname, linewidth=2)
            ax.fill_between(
                sub["n_noise"],
                sub[col_mean] - sub[col_std],
                sub[col_mean] + sub[col_std],
                alpha=0.15, color=color
            )

        ax.set_title(name.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Noise features added")
        ax.set_ylabel(metric)
        ax.legend(frameon=False)
        ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("fig2_finding2.png", bbox_inches="tight")
    print("Saved fig2_finding2.png")
    plt.show()


# ── Figure 3: Finding 3 — Rotation Invariance ────────────────────────────────
def plot_finding3():
    df = pd.read_csv("finding3_results.csv")
    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Figure 3 — Finding 3: Trees Are Sensitive to Rotation, MLP Is Not",
        fontsize=13, fontweight="bold", y=1.02
    )

    bar_width = 0.3
    x = np.array([0, 1])  # original, rotated

    for ax, name in zip(axes, datasets):
        sub  = df[df["dataset"] == name]
        task = sub["task"].iloc[0]
        metric = "Accuracy" if task == "classification" else "R²"

        for i, (mname, color) in enumerate([("GBT", COLORS["GBT"]),
                                             ("MLP", COLORS["MLP"])]):
            rows = sub[sub["setting"].isin(["original", "rotated"])]
            means = rows[f"{mname}_mean"].values
            stds  = rows[f"{mname}_std"].values

            offset = (i - 0.5) * bar_width
            bars = ax.bar(
                x + offset, means,
                width=bar_width,
                color=color,
                label=mname,
                yerr=stds,
                capsize=4,
                zorder=3
            )
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(stds) + 0.005,
                        f"{mean:.3f}",
                        ha="center", va="bottom", fontsize=8)

        ax.set_title(name.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(["Original", "Rotated"], fontsize=10)
        ax.legend(frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    plt.tight_layout()
    plt.savefig("fig3_finding3.png", bbox_inches="tight")
    print("Saved fig3_finding3.png")
    plt.show()


if __name__ == "__main__":
    plot_benchmark()
    plot_finding2()
    plot_finding3()