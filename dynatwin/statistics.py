# -*- coding: utf-8 -*-
"""
dynatwin/statistics.py
========================
Statistical significance testing restored from the original v4 design.

  bootstrap_ci       95% CI for any scalar metric via resampling
  wilcoxon_test      pairwise Wilcoxon signed-rank between models
  cohens_d           effect size
  compare_models     full cross-model report (Wilcoxon + Bootstrap CI)
  summarise_results  mean ± std table from per-case result dicts
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel


def bootstrap_ci(values: list, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> tuple:
    """
    Non-parametric bootstrap 95% CI for the mean.

    Returns (mean, lower_ci, upper_ci).
    """
    rng  = np.random.default_rng(seed)
    vals = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    if len(vals) == 0:
        return float('nan'), float('nan'), float('nan')
    means = [rng.choice(vals, size=len(vals), replace=True).mean()
             for _ in range(n_boot)]
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(np.mean(vals)), float(lo), float(hi)


def cohens_d(a: list, b: list) -> float:
    """Pooled-SD Cohen's d effect size between two paired samples."""
    a = np.array([v for v in a if not np.isnan(v)])
    b = np.array([v for v in b if not np.isnan(v)])
    n = min(len(a), len(b))
    if n < 2:
        return float('nan')
    diff = a[:n] - b[:n]
    pooled_sd = np.sqrt(
        (np.std(a[:n], ddof=1)**2 + np.std(b[:n], ddof=1)**2) / 2)
    return float(np.mean(diff) / (pooled_sd + 1e-8))


def wilcoxon_test(scores_a: list, scores_b: list) -> dict:
    """
    Wilcoxon signed-rank test between paired per-case scores.
    Returns dict with statistic, p_value, and effect_size (Cohen's d).
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 10:
        return {'statistic': float('nan'), 'p_value': float('nan'),
                'effect_size': float('nan'), 'n': len(a)}
    try:
        stat, p = wilcoxon(a, b, alternative='two-sided')
    except Exception:
        stat, p = float('nan'), float('nan')
    return {'statistic': float(stat), 'p_value': float(p),
            'effect_size': cohens_d(list(a), list(b)), 'n': int(len(a))}


def summarise_results(results: list, metrics=None) -> pd.DataFrame:
    """
    Convert list of per-case result dicts to a summary DataFrame
    with mean ± std and 95% CI for each metric.
    """
    if metrics is None:
        metrics = ['dice_whole_tumour', 'dice_tumour_core', 'dice_enhancing',
                   'dice_necrotic',     'dice_edema',
                   'hd95_whole_tumour', 'hd95_tumour_core', 'hd95_enhancing']
    rows = []
    for m in metrics:
        vals = [r[m] for r in results if m in r]
        mn, lo, hi = bootstrap_ci(vals)
        rows.append({
            'metric':  m,
            'mean':    round(mn, 4),
            'std':     round(float(np.nanstd(vals)), 4),
            'ci_lo':   round(lo, 4),
            'ci_hi':   round(hi, 4),
            'n':       sum(~np.isnan(v) for v in vals),
        })
    return pd.DataFrame(rows)


def compare_models(results_dict: dict, metric: str = 'dice_whole_tumour') -> pd.DataFrame:
    """
    Pairwise Wilcoxon + Bootstrap CI comparison across all models.

    Parameters
    ----------
    results_dict : {'M1': [per_case_dicts], 'M2': [...], 'M3': [...]}
    metric       : metric key to compare

    Returns
    -------
    DataFrame with one row per model-pair.
    """
    model_names = list(results_dict.keys())
    rows = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            na, nb = model_names[i], model_names[j]
            sa = [r.get(metric, float('nan')) for r in results_dict[na]]
            sb = [r.get(metric, float('nan')) for r in results_dict[nb]]
            wx = wilcoxon_test(sa, sb)
            ma, la, ha = bootstrap_ci(sa)
            mb, lb, hb = bootstrap_ci(sb)
            rows.append({
                'model_A':   na,
                'model_B':   nb,
                'metric':    metric,
                f'{na}_mean': round(ma, 4),
                f'{na}_CI':  f"[{la:.4f}, {ha:.4f}]",
                f'{nb}_mean': round(mb, 4),
                f'{nb}_CI':  f"[{lb:.4f}, {hb:.4f}]",
                'wilcoxon_p': round(wx['p_value'], 4),
                'effect_d':   round(wx['effect_size'], 3),
                'significant': wx['p_value'] < 0.05,
            })
    return pd.DataFrame(rows)

print("[statistics] Wilcoxon + Bootstrap CI module ready.")
