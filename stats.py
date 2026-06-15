"""
stats.py

Statistical comparison utilities for the reproducibility study.

Provides:
  - wilcoxon_test(a, b)  : Wilcoxon signed-rank test (non-parametric, paired)
  - cohen_d(a, b)        : Cohen's d effect size
  - compare(name_a, scores_a, name_b, scores_b) : full comparison report

Why Wilcoxon and not a t-test?
    The Wilcoxon signed-rank test makes no assumption about the distribution of
    differences. With 5 seeds (benchmark) or 30 rotation scores (finding3) we
    cannot reliably verify normality, so Wilcoxon is the safer choice. This
    matches standard practice in ML benchmarking papers.

Why Cohen's d?
    Statistical significance alone does not tell you whether a difference is
    meaningful. Cohen's d quantifies effect size independently of sample count.
    Conventional thresholds: small=0.2, medium=0.5, large=0.8.
"""

import numpy as np
from scipy.stats import wilcoxon


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Cohen's d for two paired samples.
    Uses the pooled standard deviation of the two groups.
    Returns a signed value: positive means a > b.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mean_diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return mean_diff / pooled_std


def effect_size_label(d: float) -> str:
    """Return a human-readable label for a Cohen's d value."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def wilcoxon_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Run Wilcoxon signed-rank test on paired samples a and b.

    Returns:
        (statistic, p_value)

    Falls back to (nan, nan) if all differences are zero (e.g. identical scores),
    which can happen when both models hit ceiling performance.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    if np.all(diff == 0):
        return float("nan"), float("nan")
    stat, p = wilcoxon(a, b, alternative="two-sided")
    return float(stat), float(p)


def compare(
    name_a: str,
    scores_a: list[float],
    name_b: str,
    scores_b: list[float],
    metric: str = "score",
    alpha: float = 0.05,
) -> dict:
    """
    Full statistical comparison between two sets of paired scores.

    Args:
        name_a, name_b : model/condition names for display
        scores_a, scores_b : paired lists of scores (same length, same conditions)
        metric : name of the metric being compared (e.g. "R²", "Accuracy")
        alpha  : significance threshold (default 0.05)

    Returns:
        dict with keys: name_a, name_b, mean_a, mean_b, std_a, std_b,
                        statistic, p_value, significant, cohen_d, effect_label
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    mean_a, std_a = np.mean(a), np.std(a, ddof=1)
    mean_b, std_b = np.mean(b), np.std(b, ddof=1)
    stat, p = wilcoxon_test(a, b)
    d = cohen_d(a, b)
    label = effect_size_label(d)
    significant = (p < alpha) if not np.isnan(p) else False

    return {
        "name_a": name_a,
        "name_b": name_b,
        "metric": metric,
        "mean_a": round(mean_a, 4),
        "std_a": round(std_a, 4),
        "mean_b": round(mean_b, 4),
        "std_b": round(std_b, 4),
        "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else None,
        "p_value": round(p, 4) if not np.isnan(p) else None,
        "significant": significant,
        "cohen_d": round(d, 4),
        "effect_label": label,
    }


def print_comparison(result: dict, indent: str = "  ") -> None:
    """Pretty-print a single comparison result."""
    sig_str = "YES *" if result["significant"] else "no"
    p_str = f"{result['p_value']:.4f}" if result["p_value"] is not None else "n/a"
    stat_str = f"{result['wilcoxon_stat']:.1f}" if result["wilcoxon_stat"] is not None else "n/a"

    print(
        f"{indent}{result['name_a']} vs {result['name_b']} ({result['metric']})\n"
        f"{indent}  {result['name_a']}: {result['mean_a']:.4f} ± {result['std_a']:.4f}\n"
        f"{indent}  {result['name_b']}: {result['mean_b']:.4f} ± {result['std_b']:.4f}\n"
        f"{indent}  Wilcoxon stat={stat_str}, p={p_str} → significant: {sig_str}\n"
        f"{indent}  Cohen's d={result['cohen_d']:.4f} ({result['effect_label']} effect)"
    )