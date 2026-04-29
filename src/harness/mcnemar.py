"""
McNemar's test implementation for v5.

Used to compare the detection rate of two conditions (e.g., with-constraint
vs without-constraint) on paired binary outcomes (correct / incorrect per chain).

This is a direct port of the v4 McNemar pipeline adapted for five parallel cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class McnemarResult:
    """Results of a McNemar test on one cell's paired outcomes."""
    cell: str
    n_chains: int
    n_concordant_both_correct: int      # both conditions correct (a)
    n_discordant_base_only: int         # baseline correct, intervention wrong (b)
    n_discordant_intervention_only: int # baseline wrong, intervention correct (c)
    n_concordant_both_wrong: int        # both conditions wrong (d)
    statistic: float
    p_value: float
    p_value_corrected: float            # after Bonferroni
    bonferroni_divisor: int
    alpha: float
    alpha_corrected: float
    significant: bool
    effect_size_h: float                # Cohen's h
    ci_lower: float                     # bootstrap CI on proportion difference
    ci_upper: float
    notes: str = ""

    @property
    def b(self) -> int:
        return self.n_discordant_base_only

    @property
    def c(self) -> int:
        return self.n_discordant_intervention_only

    @property
    def n_discordant(self) -> int:
        return self.b + self.c

    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"[{self.cell}] McNemar n={self.n_chains} "
            f"b={self.b} c={self.c} χ²={self.statistic:.4f} "
            f"p={self.p_value:.4f} p_corr={self.p_value_corrected:.4f} "
            f"({sig}) h={self.effect_size_h:.4f} "
            f"95%CI=[{self.ci_lower:.4f},{self.ci_upper:.4f}]"
        )


def run_mcnemar(
    baseline_correct: list[bool],
    intervention_correct: list[bool],
    cell: str,
    alpha: float = 0.05,
    bonferroni_divisor: int = 5,
    continuity_correction: bool = True,
    bootstrap_iterations: int = 10000,
    bootstrap_seed: int = 42,
    min_discordant_pairs: int = 10,
) -> McnemarResult:
    """
    Run McNemar's test comparing baseline vs intervention on paired binary outcomes.

    Parameters
    ----------
    baseline_correct     : list of bool, one per chain
    intervention_correct : list of bool, one per chain (same length, same order)
    cell                 : domain cell identifier
    alpha                : nominal significance level (before Bonferroni)
    bonferroni_divisor   : denominator for Bonferroni correction [REQUIRES SIGN-OFF]
    continuity_correction: apply continuity correction (Yates) to chi-squared
    """
    if len(baseline_correct) != len(intervention_correct):
        raise ValueError("baseline and intervention lists must be same length")

    pairs = list(zip(baseline_correct, intervention_correct))
    a = sum(1 for b, i in pairs if b and i)
    b = sum(1 for b, i in pairs if b and not i)
    c = sum(1 for b, i in pairs if not b and i)
    d = sum(1 for b, i in pairs if not b and not i)
    n = len(pairs)

    notes = ""
    if b + c < min_discordant_pairs:
        notes = f"WARNING: only {b+c} discordant pairs (threshold={min_discordant_pairs}); test underpowered"

    # McNemar statistic (with optional continuity correction).
    # When b + c == 0 (all chains concordant), the test is undefined — return
    # chi²=0 and p=1 instead of masking with a fake denominator that yields
    # chi²=1.0 (which would otherwise be a misleading non-zero statistic in
    # the result object).
    if b + c == 0:
        chi2_stat = 0.0
        p_value = 1.0
        if notes:
            notes += "; "
        notes += "no discordant pairs; McNemar undefined (chi²=0, p=1)"
    else:
        if continuity_correction:
            numerator = (abs(b - c) - 1) ** 2
        else:
            numerator = (b - c) ** 2
        chi2_stat = numerator / (b + c)
        p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=1)

    # Bonferroni correction
    alpha_corrected = alpha / bonferroni_divisor
    p_value = float(p_value)          # scipy returns np.float64; ensure plain float
    p_corrected = float(min(p_value * bonferroni_divisor, 1.0))
    significant = bool(p_corrected < alpha)  # prevent np.bool_ leaking into JSON

    # Cohen's h effect size (arcsin transformation of proportion difference)
    p1 = c / n if n > 0 else 0.0  # P(intervention correct, baseline wrong)
    p2 = b / n if n > 0 else 0.0  # P(baseline correct, intervention wrong)
    effect_size_h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))

    # Bootstrap CI on proportion difference (c-b)/n
    ci_lower, ci_upper = _bootstrap_ci(
        np.array(baseline_correct, dtype=int),
        np.array(intervention_correct, dtype=int),
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
        confidence=0.95,
    )

    return McnemarResult(
        cell=cell,
        n_chains=n,
        n_concordant_both_correct=a,
        n_discordant_base_only=b,
        n_discordant_intervention_only=c,
        n_concordant_both_wrong=d,
        statistic=chi2_stat,
        p_value=p_value,
        p_value_corrected=p_corrected,
        bonferroni_divisor=bonferroni_divisor,
        alpha=alpha,
        alpha_corrected=alpha_corrected,
        significant=significant,
        effect_size_h=effect_size_h,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        notes=notes,
    )


def _bootstrap_ci(
    baseline: np.ndarray,
    intervention: np.ndarray,
    iterations: int,
    seed: int,
    confidence: float,
) -> tuple[float, float]:
    """Bootstrap CI on the proportion difference (P(intervention) - P(baseline))."""
    rng = np.random.default_rng(seed)
    n = len(baseline)
    diffs = []
    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        b_sample = baseline[idx]
        i_sample = intervention[idx]
        diff = i_sample.mean() - b_sample.mean()
        diffs.append(diff)
    diffs = np.array(diffs)
    lo = (1.0 - confidence) / 2.0
    hi = 1.0 - lo
    return float(np.quantile(diffs, lo)), float(np.quantile(diffs, hi))


def aggregate_results(cell_results: list[McnemarResult]) -> dict:
    """
    Produce aggregate summary across all cells.
    Uses pooled discordant pairs for a combined McNemar statistic.
    """
    total_b = sum(r.b for r in cell_results)
    total_c = sum(r.c for r in cell_results)
    total_n = sum(r.n_chains for r in cell_results)

    if total_b + total_c == 0:
        pooled_chi2 = 0.0
        pooled_p = 1.0
    else:
        numerator = (abs(total_b - total_c) - 1) ** 2
        pooled_chi2 = numerator / (total_b + total_c)
        pooled_p = 1.0 - stats.chi2.cdf(pooled_chi2, df=1)

    n_significant = sum(1 for r in cell_results if r.significant)

    return {
        "n_cells": len(cell_results),
        "n_cells_significant": n_significant,
        "total_chains": total_n,
        "total_discordant": total_b + total_c,
        "pooled_chi2": pooled_chi2,
        "pooled_p": pooled_p,
        "cell_summaries": [r.summary() for r in cell_results],
    }
