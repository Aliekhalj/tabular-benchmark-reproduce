# stats.py

"""
Phase 4: Final statistical analysis comparing RandomForest, GBT,
XGBoost, and MLP across the 23-dataset benchmark.

PRIMARY analysis: tuning_results.csv's test_score -- one
held-out 30%-test-set evaluation per dataset, frozen best
hyperparameters. Matches the paper's own reporting convention.

SECONDARY / sensitivity analysis: cv_results.csv's mean_score
-- the already-aggregated 5-fold CV mean per dataset (frozen
hyperparameters, training-pool only). Included to check whether
conclusions are sensitive to the specific 30% test split, NOT a
replacement for the primary analysis -- cv.py's own docstring is
explicit that CV is a supplementary robustness estimate alongside the
single test_score, never a substitute for it.

STATISTICAL UNIT throughout: one score per (dataset, model). The 5 CV
folds are never treated as independent observations -- this module
reads cv_results.csv's already-aggregated mean_score, never
cv_fold_results.csv.

Classification and regression are analyzed completely separately
throughout (Accuracy and R\u00b2 are not on a comparable scale).

For each (task, score_source) block, this computes:
  - Friedman test (omnibus, across all 4 models)
  - Mean rank per model (rank 1 = best on that dataset)
  - Nemenyi post-hoc critical difference + pairwise significance,
    for all 6 pairs
  - All 6 pairwise Wilcoxon signed-rank tests, Holm-corrected
  - Paired Cohen's d (d_z) for each of the 6 pairs

No paired t-tests: dataset-level score differences across a
heterogeneous 11-23 dataset benchmark have no basis for a normality
assumption, and the project's own statistical-readiness notes call for
Wilcoxon (distribution-free) as the pairwise test of record. Wilcoxon
is kept as the single pairwise test rather than running both and
picking whichever agrees.

OUTPUTS:
  stats_summary.csv     -- primary (test_score): task, model, mean
                            rank, mean score, Friedman result
  stats_pairwise.csv    -- primary (test_score): all 6 pairs per task,
                            Wilcoxon (raw + Holm), Cohen's d, Nemenyi
  stats_summary_cv.csv  -- secondary (CV mean_score), same schema
  stats_pairwise_cv.csv -- secondary (CV mean_score), same schema

If the input files are missing required columns, contain an
incomplete dataset x model block (Friedman/Nemenyi require every
dataset to have a score for every model), contain duplicate
(dataset, model) rows, or contain NaN scores, this script stops and
reports the problem rather than silently dropping rows or filling
values.
"""

import sys
import itertools
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, studentized_range

MODELS = ["RandomForest", "GBT", "XGBoost", "MLP"]
TASKS = ["classification", "regression"]
ALPHA = 0.05


def _holm_correction(pvalues):
    """
    Holm-Bonferroni step-down correction. Returns adjusted p-values in
    the SAME order as the input list. Implemented directly (this is
    the entire algorithm -- sort ascending, adjust, enforce
    monotonicity by running max, restore original order) rather than
    adding statsmodels as a new project dependency: the project's
    existing dependencies (README.md) are numpy, pandas, matplotlib,
    scipy, scikit-learn, xgboost, and statsmodels is not among them.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]

    adjusted_sorted = np.empty(m)
    running_max = 0.0
    for i in range(m):
        val = (m - i) * sorted_p[i]
        running_max = max(running_max, val)
        adjusted_sorted[i] = min(running_max, 1.0)

    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


def _nemenyi_cd(k, n, alpha=ALPHA):
    """
    Nemenyi critical difference (Demsar 2006, "Statistical Comparisons
    of Classifiers over Multiple Data Sets"): CD = q_alpha *
    sqrt(k(k+1)/(6n)).

    q_alpha is derived from scipy's studentized_range distribution:
    q_alpha = studentized_range.ppf(1-alpha, k, inf) / sqrt(2). Checked
    against Demsar's published table before use here -- reproduces
    2.569 (k=4) and 2.343 (k=3) exactly -- so no separate Nemenyi-table
    package (e.g. scikit-posthocs, not installed in this environment)
    is needed.
    """
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q_alpha * np.sqrt(k * (k + 1) / (6 * n))


def _cohens_d_z(diff):
    """
    Paired Cohen's d (d_z): mean of the paired differences / sample
    std of the paired differences (ddof=1). Computed on dataset-level
    differences only -- never fold-level, per this module's statistical
    unit.
    """
    diff = np.asarray(diff, dtype=float)
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0 if diff.mean() == 0 else float(np.inf) * np.sign(diff.mean())
    return diff.mean() / sd


def _validate_and_pivot(df, score_col, source_label):
    """
    Validates the input long-format dataframe and pivots it to one
    (dataset x model) score matrix per task. Raises rather than
    silently working around any structural problem -- an incomplete
    block breaks Friedman/Nemenyi, which require every dataset to have
    a score for every model.
    """
    required_cols = {"dataset", "task", "model", score_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{source_label}: missing required column(s) {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    unexpected_models = set(df["model"].unique()) - set(MODELS)
    if unexpected_models:
        raise ValueError(
            f"{source_label}: unexpected model name(s) {unexpected_models}. "
            f"Expected exactly {MODELS}."
        )

    unexpected_tasks = set(df["task"].unique()) - set(TASKS)
    if unexpected_tasks:
        raise ValueError(
            f"{source_label}: unexpected task value(s) {unexpected_tasks}. "
            f"Expected exactly {TASKS}."
        )

    dup = df[df.duplicated(subset=["dataset", "model"], keep=False)]
    if len(dup) > 0:
        raise ValueError(
            f"{source_label}: duplicate (dataset, model) rows found:\n"
            f"{dup[['dataset', 'model']]}"
        )

    if df[score_col].isna().any():
        bad = df[df[score_col].isna()][["dataset", "model"]]
        raise ValueError(f"{source_label}: NaN {score_col} values found:\n{bad}")

    pivots = {}
    for task in TASKS:
        sub = df[df["task"] == task]
        pivot = sub.pivot(index="dataset", columns="model", values=score_col)

        if pivot.isna().any().any():
            missing = pivot[pivot.isna().any(axis=1)]
            raise ValueError(
                f"{source_label}/{task}: incomplete block -- the following "
                f"dataset(s) are missing a score for at least one model "
                f"(Friedman/Nemenyi require a complete design):\n{missing}"
            )

        pivots[task] = pivot[MODELS]  # fixed column order

    return pivots


def _rank_matrix(pivot, higher_is_better=True):
    """
    Per-dataset ranks across the 4 models, rank 1 = best. Both
    Accuracy and R\u00b2 are higher-is-better in this project, so
    higher_is_better=False is never exercised in practice, but the
    parameter is kept explicit rather than hardcoding the rank
    direction inside the function. Ties are averaged (pandas' default
    rank method), the standard convention for Friedman/Nemenyi.
    """
    ascending = not higher_is_better
    return pivot.rank(axis=1, ascending=ascending)


def analyze(pivot, task, source_label):
    """
    Runs the full battery (Friedman, mean ranks, Nemenyi, 6x
    Wilcoxon+Holm, 6x Cohen's d) for one (task, score_source) block.
    Returns (summary_rows, pairwise_rows).
    """
    n = len(pivot)  # number of datasets in this task
    ranks = _rank_matrix(pivot, higher_is_better=True)
    mean_ranks = ranks.mean(axis=0)

    columns = [pivot[m].values for m in MODELS]
    friedman_stat, friedman_p = friedmanchisquare(*columns)

    summary_rows = []
    for model in MODELS:
        summary_rows.append({
            "task": task,
            "model": model,
            "mean_rank": round(float(mean_ranks[model]), 4),
            "mean_score": round(float(pivot[model].mean()), 4),
            "n_datasets": n,
            "friedman_chi2": round(float(friedman_stat), 4),
            "friedman_p": friedman_p,
            "friedman_significant_0.05": bool(friedman_p < ALPHA),
        })

    cd = _nemenyi_cd(k=len(MODELS), n=n, alpha=ALPHA)

    pairs = list(itertools.combinations(MODELS, 2))
    raw_pvalues = []
    pair_rows = []
    for m1, m2 in pairs:
        diff = pivot[m1].values - pivot[m2].values
        try:
            wilc_stat, wilc_p = wilcoxon(diff)
        except ValueError:
            # wilcoxon raises if every paired difference is exactly
            # zero; treat as no evidence of a difference rather than
            # letting the whole run crash on one degenerate pair.
            wilc_stat, wilc_p = np.nan, 1.0
        raw_pvalues.append(wilc_p)

        d_z = _cohens_d_z(diff)
        rank_diff = abs(float(mean_ranks[m1] - mean_ranks[m2]))

        pair_rows.append({
            "task": task,
            "model_a": m1,
            "model_b": m2,
            "mean_score_a": round(float(pivot[m1].mean()), 4),
            "mean_score_b": round(float(pivot[m2].mean()), 4),
            "wilcoxon_stat": wilc_stat,
            "wilcoxon_p_raw": wilc_p,
            "cohens_d_z": round(float(d_z), 4),
            "mean_rank_diff": round(rank_diff, 4),
            "nemenyi_cd": round(float(cd), 4),
            "nemenyi_significant": bool(rank_diff > cd),
        })

    holm_p = _holm_correction(raw_pvalues)
    for row, p_adj in zip(pair_rows, holm_p):
        row["wilcoxon_p_raw"] = round(float(row["wilcoxon_p_raw"]), 6)
        row["wilcoxon_p_holm"] = round(float(p_adj), 6)
        row["wilcoxon_significant_holm_0.05"] = bool(p_adj < ALPHA)

    return summary_rows, pair_rows


def run_source(path, score_col, source_label):
    print(f"\n{'=' * 70}\n{source_label}  ({path}, column='{score_col}')\n{'=' * 70}")
    df = pd.read_csv(path)
    pivots = _validate_and_pivot(df, score_col, source_label)

    all_summary, all_pairwise = [], []
    for task in TASKS:
        pivot = pivots[task]
        n = len(pivot)
        print(f"\n--- {task} (n={n} datasets) ---")

        summary_rows, pairwise_rows = analyze(pivot, task, source_label)
        all_summary.extend(summary_rows)
        all_pairwise.extend(pairwise_rows)

        fr = summary_rows[0]
        sig = "SIGNIFICANT" if fr["friedman_significant_0.05"] else "not significant"
        print(f"Friedman: chi2={fr['friedman_chi2']}, p={fr['friedman_p']:.4g} ({sig} at alpha=0.05)")
        print("Mean ranks (1=best):")
        for row in sorted(summary_rows, key=lambda r: r["mean_rank"]):
            print(f"  {row['model']:<13} mean_rank={row['mean_rank']:.3f}  mean_score={row['mean_score']:.4f}")

        n_sig_wilcoxon = sum(r["wilcoxon_significant_holm_0.05"] for r in pairwise_rows)
        n_sig_nemenyi = sum(r["nemenyi_significant"] for r in pairwise_rows)
        print(f"Pairwise: {len(pairwise_rows)}/6 pairs tested -- "
              f"{n_sig_wilcoxon}/6 significant (Wilcoxon, Holm-corrected), "
              f"{n_sig_nemenyi}/6 significant (Nemenyi, CD={pairwise_rows[0]['nemenyi_cd']})")

    return all_summary, all_pairwise


def main():
    try:
        primary_summary, primary_pairwise = run_source(
            "tuning_results.csv", "test_score", "PRIMARY: held-out test_score"
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"\nSTATS ANALYSIS STOPPED (primary source): {exc}")
        sys.exit(1)

    try:
        secondary_summary, secondary_pairwise = run_source(
            "cv_results.csv", "mean_score", "SECONDARY: CV mean_score (sensitivity check)"
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"\nSTATS ANALYSIS STOPPED (secondary/CV source): {exc}")
        sys.exit(1)

    pd.DataFrame(primary_summary).to_csv("stats_summary.csv", index=False)
    pd.DataFrame(primary_pairwise).to_csv("stats_pairwise.csv", index=False)
    pd.DataFrame(secondary_summary).to_csv("stats_summary_cv.csv", index=False)
    pd.DataFrame(secondary_pairwise).to_csv("stats_pairwise_cv.csv", index=False)

    print(f"\n{'=' * 70}")
    print("Saved:")
    print(f"  stats_summary.csv     ({len(primary_summary)} rows)  -- primary")
    print(f"  stats_pairwise.csv    ({len(primary_pairwise)} rows)  -- primary")
    print(f"  stats_summary_cv.csv  ({len(secondary_summary)} rows)  -- CV sensitivity")
    print(f"  stats_pairwise_cv.csv ({len(secondary_pairwise)} rows)  -- CV sensitivity")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()