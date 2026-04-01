"""Statistical significance tests: McNemar, bootstrap CI, paired bootstrap."""
import numpy as np
from loguru import logger


def mcnemar_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> dict:
    """
    McNemar's test: are two classifiers significantly different?
    Only looks at samples where they disagree.

    Returns: {statistic, p_value}
    """
    from statsmodels.stats.contingency_tables import mcnemar
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    table = [
        [int(np.sum(correct_a & correct_b)), int(np.sum(correct_a & ~correct_b))],
        [int(np.sum(~correct_a & correct_b)), int(np.sum(~correct_a & ~correct_b))],
    ]
    result = mcnemar(table, exact=True)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant": result.pvalue < 0.05,
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap 95% CI for any metric.

    Args:
        metric_fn: Callable(y_true, y_pred) -> float

    Returns:
        {mean, ci_lower, ci_upper, std}
    """
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))

    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100
    return {
        "mean": float(np.mean(scores)),
        "ci_lower": float(np.percentile(scores, lower_pct)),
        "ci_upper": float(np.percentile(scores, upper_pct)),
        "std": float(np.std(scores)),
    }


def paired_bootstrap_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap test: is method A significantly better than B?
    Returns p-value for H0: metric(A) <= metric(B)

    Returns:
        {delta, p_value, significant}
    """
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true)
    pred_a = np.array(pred_a)
    pred_b = np.array(pred_b)
    n = len(y_true)

    delta_observed = metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)
    count_not_greater = 0

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        delta = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])
        if delta <= 0:
            count_not_greater += 1

    p_value = count_not_greater / n_bootstrap
    return {
        "delta": float(delta_observed),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


def wilcoxon_test(scores_a: list[float], scores_b: list[float]) -> dict:
    """
    Wilcoxon signed-rank test for human evaluation scores.
    Tests if two methods produce significantly different ratings.
    """
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(scores_a, scores_b)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": p < 0.05,
    }
