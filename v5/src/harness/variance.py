"""
Variance and power analysis for v5.

Provides:
  - Bootstrap confidence intervals (also in mcnemar.py for internal use)
  - Post-hoc power calculation
  - Minimum detectable effect size for given n
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def bootstrap_proportion_ci(
    correct: list[bool],
    iterations: int = 10000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap CI on a single proportion (fraction correct)."""
    arr = np.array(correct, dtype=float)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    props = [rng.choice(arr, size=n, replace=True).mean() for _ in range(iterations)]
    lo = (1.0 - confidence) / 2.0
    return float(np.quantile(props, lo)), float(np.quantile(props, 1.0 - lo))


def mcnemar_power(
    b: int,
    c: int,
    alpha: float = 0.05,
    bonferroni_divisor: int = 5,
) -> float:
    """
    Post-hoc power for McNemar test given observed b, c discordant counts.
    Uses normal approximation to the McNemar statistic.
    """
    n_disc = b + c
    if n_disc == 0:
        return 0.0
    alpha_corr = alpha / bonferroni_divisor
    z_alpha = stats.norm.ppf(1.0 - alpha_corr / 2.0)
    z_beta = (abs(b - c) / math.sqrt(n_disc) - z_alpha)
    power = float(stats.norm.cdf(z_beta))
    return max(0.0, min(1.0, power))


def minimum_detectable_effect(
    n_chains: int,
    alpha: float = 0.05,
    power_target: float = 0.80,
    bonferroni_divisor: int = 5,
) -> float:
    """
    Minimum detectable |p_c - p_b| (proportion difference) given n_chains,
    assuming roughly equal b and c counts (symmetric null).
    Returns MDE as a fraction.
    """
    alpha_corr = alpha / bonferroni_divisor
    z_alpha = stats.norm.ppf(1.0 - alpha_corr / 2.0)
    z_beta = stats.norm.ppf(power_target)
    # Approximate: MDE ≈ (z_alpha + z_beta) / sqrt(n)
    mde = (z_alpha + z_beta) / math.sqrt(max(n_chains, 1))
    return float(mde)


def variance_summary(
    correct_list: list[bool],
    cell: str,
    iterations: int = 10000,
    seed: int = 42,
) -> dict:
    """Full variance characterization for one cell's correct vector."""
    n = len(correct_list)
    proportion = sum(correct_list) / n if n > 0 else 0.0
    ci_lo, ci_hi = bootstrap_proportion_ci(correct_list, iterations=iterations,
                                           seed=seed, confidence=0.95)
    return {
        "cell": cell,
        "n": n,
        "proportion_correct": proportion,
        "ci_95_lower": ci_lo,
        "ci_95_upper": ci_hi,
        "std_dev": float(np.std(correct_list)) if n > 0 else 0.0,
        "std_error": float(np.std(correct_list) / math.sqrt(n)) if n > 0 else 0.0,
    }
