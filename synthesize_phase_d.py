"""
Phase D final synthesis.

Combines:
  RESULTS/phase_d_partial_pubg_nba.json    (PUBG, NBA — already retrieved)
  RESULTS/phase_d_poker_rl_csgo.json       (Poker, RL, CS:GO — from resume run)

Outputs per-cell:
  - n, baseline / intervention detection + FP rates
  - McNemar χ² (with continuity correction) on adversarial detection
  - Exact mid-p binomial p-value on (b,c) for small discordances
  - Bonferroni-corrected p-value (× 5 cells)
  - Bootstrap 95% CI on the detection-rate lift (Δ = Det@Int − Det@Base)
  - Bootstrap 95% CI on the FP-rate change (intervention − baseline)

Writes RESULTS/phase_d_final.json and prints a summary table.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
    _env = Path(__file__).parent / ".env"
    if _env.exists():
        for _line in _env.read_text().splitlines():
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _, _v = _line.partition("=")
            _k = _k.strip()
            _v = _v.strip().strip('"').strip("'")
            if _v and not os.environ.get(_k, "").strip():
                os.environ[_k] = _v
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from retrieve_phase_d_partial import fetch_batch, parse_yn  # noqa: E402

N_CELLS = 5
N_BOOTSTRAP = 5000
RNG_SEED = 20260430


# ---------- Statistical helpers --------------------------------------------

def mcnemar_chi2_continuity(b: int, c: int) -> float:
    """McNemar χ² with continuity correction. Returns χ² (df=1)."""
    if b + c == 0:
        return 0.0
    return (abs(b - c) - 1) ** 2 / (b + c)


def chi2_p_df1(chi2: float) -> float:
    """Survival function of χ²(1) — equals 2 * (1 - Φ(√χ²))."""
    if chi2 <= 0:
        return 1.0
    # P(X² ≥ x) for df=1 = 2 * P(Z ≥ √x) = erfc(√(x/2))
    return math.erfc(math.sqrt(chi2 / 2))


def exact_mcnemar_two_sided_p(b: int, c: int) -> float:
    """Exact two-sided binomial test on (b,c) under H0: p=0.5.
    Use this when b+c is small (recommended <25)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # P(X ≤ k) under Binomial(n, 0.5)
    cum = 0.0
    for i in range(k + 1):
        cum += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2 * cum)


def mcnemar_pvalue(b: int, c: int) -> tuple[float, str]:
    """Pick the appropriate McNemar p-value test based on n_discordant."""
    n = b + c
    if n == 0:
        return 1.0, "no_discordance"
    if n < 25:
        return exact_mcnemar_two_sided_p(b, c), "exact_binomial"
    chi2 = mcnemar_chi2_continuity(b, c)
    return chi2_p_df1(chi2), "chi2_with_cc"


def bootstrap_diff_ci(
    paired_outcomes: list[tuple[int, int]],
    n_iters: int = N_BOOTSTRAP,
    rng_seed: int = RNG_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Cluster-bootstrap on paired outcomes [(baseline, intervention), ...]
    where each value is 0 or 1. Returns (Δ, ci_low, ci_high) for
    Δ = mean(intervention) - mean(baseline).
    """
    rng = random.Random(rng_seed)
    n = len(paired_outcomes)
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = []
    for _ in range(n_iters):
        sample = [paired_outcomes[rng.randrange(n)] for _ in range(n)]
        m_b = sum(p[0] for p in sample) / n
        m_i = sum(p[1] for p in sample) / n
        diffs.append(m_i - m_b)
    diffs.sort()
    ci_low = diffs[int(alpha / 2 * n_iters)]
    ci_high = diffs[int((1 - alpha / 2) * n_iters)]
    point = (
        sum(p[1] for p in paired_outcomes) / n
        - sum(p[0] for p in paired_outcomes) / n
    )
    return point, ci_low, ci_high


# ---------- Per-cell scoring -----------------------------------------------

def pair_results(results: dict[str, str]) -> dict[str, dict[str, str]]:
    """custom_id -> raw text  →  chain_id -> {"baseline": yn, "intervention": yn}.
    Strips the 6-digit positional prefix the new evaluator emits, falls back
    to the old format for the PUBG/NBA batches that ran pre-fix.
    """
    paired: dict[str, dict[str, str]] = defaultdict(dict)
    for cid, text in results.items():
        parts = cid.split("__")
        if len(parts) == 3 and parts[0].isdigit() and len(parts[0]) == 6:
            # New format: 000123__chain_id__variant
            _idx, chain_id, variant = parts
            # Use indexed key so duplicates within the batch don't collapse
            key = f"{_idx}__{chain_id}"
        elif len(parts) >= 2:
            # Old format: chain_id__variant (or chain_id with __ in it)
            chain_id, _, variant = cid.rpartition("__")
            key = chain_id
        else:
            continue
        paired[key][variant] = parse_yn(text)
    return paired


def score_cell_full(
    clean_results: dict[str, str],
    adv_results: dict[str, str],
    cell: str,
) -> dict:
    clean_pairs = pair_results(clean_results)
    adv_pairs = pair_results(adv_results)

    # Adversarial detection paired outcomes (1 = "yes" = correct detection)
    adv_paired_outcomes: list[tuple[int, int]] = []
    b_only = c_only = both_yes = both_no = 0
    for v in adv_pairs.values():
        bv = 1 if v.get("baseline") == "yes" else 0
        iv = 1 if v.get("intervention") == "yes" else 0
        adv_paired_outcomes.append((bv, iv))
        if bv == 1 and iv == 1:
            both_yes += 1
        elif bv == 1 and iv == 0:
            b_only += 1
        elif bv == 0 and iv == 1:
            c_only += 1
        else:
            both_no += 1

    # Clean: 1 = "yes" = false positive
    clean_paired_outcomes: list[tuple[int, int]] = []
    clean_fp_b = clean_fp_i = 0
    for v in clean_pairs.values():
        bv = 1 if v.get("baseline") == "yes" else 0
        iv = 1 if v.get("intervention") == "yes" else 0
        clean_paired_outcomes.append((bv, iv))
        clean_fp_b += bv
        clean_fp_i += iv

    n_clean = len(clean_pairs)
    n_adv = len(adv_pairs)

    # McNemar on adversarial detection
    p_raw, p_method = mcnemar_pvalue(b_only, c_only)
    chi2 = mcnemar_chi2_continuity(b_only, c_only) if (b_only + c_only) > 0 else 0.0
    p_bonf = min(1.0, p_raw * N_CELLS)

    # Bootstrap CIs
    det_delta, det_ci_lo, det_ci_hi = bootstrap_diff_ci(adv_paired_outcomes)
    fp_delta, fp_ci_lo, fp_ci_hi = bootstrap_diff_ci(clean_paired_outcomes)

    return {
        "cell": cell,
        "n_clean": n_clean,
        "n_adversarial": n_adv,
        "det_baseline": sum(p[0] for p in adv_paired_outcomes) / n_adv if n_adv else 0.0,
        "det_intervention": sum(p[1] for p in adv_paired_outcomes) / n_adv if n_adv else 0.0,
        "det_delta": det_delta,
        "det_delta_ci95": [det_ci_lo, det_ci_hi],
        "fp_baseline": clean_fp_b / n_clean if n_clean else 0.0,
        "fp_intervention": clean_fp_i / n_clean if n_clean else 0.0,
        "fp_delta": fp_delta,
        "fp_delta_ci95": [fp_ci_lo, fp_ci_hi],
        "mcnemar": {
            "b_baseline_only": b_only,
            "c_intervention_only": c_only,
            "both_yes": both_yes,
            "both_no": both_no,
            "chi2_with_cc": chi2,
            "p_raw": p_raw,
            "p_bonferroni": p_bonf,
            "p_method": p_method,
        },
    }


# ---------- Main ------------------------------------------------------------

# Phase D batch IDs: cell -> (clean, adversarial)
PHASE_D_BATCHES = {
    "pubg":          ("msgbatch_01VcHLdLj8zwBPjtgrQFw3Lp", "msgbatch_01QxaNgJFjDjJdargiuZbUpF"),
    "nba":           ("msgbatch_01NFDJnX98DCEgzd6hzzo6Fp", "msgbatch_01WmT9EJ3fzbcot4yDMx4FG6"),
    "poker":         ("msgbatch_019CdSVP7bL3vjjiAxoogzoX", "msgbatch_01NSGy2V2ZayjVPGQd2qgLuC"),
    "rocket_league": ("msgbatch_01BPB6U1sYABgv41arrnVPLb", "msgbatch_0186kapuWTHCPLkNcZHfmSgF"),
    # CS:GO batch IDs filled in once they end. The clean batch is queued in
    # the resume run; once it ends, the script auto-launches the adversarial.
    # We hard-code the clean batch ID here from the resume log; the adversarial
    # ID is read from the log dynamically below.
}


def discover_csgo_batch_ids(log_path: Path) -> tuple[str | None, str | None]:
    """Scrape the resume log for csgo clean+adversarial batch IDs."""
    if not log_path.exists():
        return None, None
    text = log_path.read_text()
    import re
    csgo_subs = re.findall(
        r"\[csgo\] Submitted batch (msgbatch_\w+)", text
    )
    clean = csgo_subs[0] if len(csgo_subs) >= 1 else None
    adv = csgo_subs[1] if len(csgo_subs) >= 2 else None
    return clean, adv


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required", file=sys.stderr)
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic()

    # Discover CS:GO batch IDs from log
    log_path = Path("RESULTS/phase_d_resume.log")
    csgo_clean, csgo_adv = discover_csgo_batch_ids(log_path)
    if csgo_clean and csgo_adv:
        PHASE_D_BATCHES["csgo"] = (csgo_clean, csgo_adv)
        print(f"[csgo] using batches {csgo_clean}, {csgo_adv}")
    else:
        print(f"[csgo] not yet ready (clean={csgo_clean}, adv={csgo_adv}); "
              f"synthesizing 4 cells")

    cells_out: dict = {}
    for cell, (clean_id, adv_id) in PHASE_D_BATCHES.items():
        print(f"[{cell}] retrieving + scoring...")
        clean = fetch_batch(client, clean_id)
        adv = fetch_batch(client, adv_id)
        cells_out[cell] = score_cell_full(clean, adv, cell)

    out_path = Path("RESULTS/phase_d_final.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_cells": N_CELLS,
        "bonferroni_divisor": N_CELLS,
        "n_bootstrap": N_BOOTSTRAP,
        "cells": cells_out,
    }, indent=2))

    # Print summary
    print()
    print("=" * 110)
    print("PHASE D FINAL — n=1,200 per cell, Haiku 4.5, violation-detection diagnostic")
    print("=" * 110)
    header = (f"{'Cell':<14} "
              f"{'Det@Base':>9} {'Det@Int':>9} {'Δ':>8} {'95% CI':>16}  "
              f"{'FP@Base':>8} {'FP@Int':>8}  "
              f"{'b':>4} {'c':>5}  {'χ²':>9}  "
              f"{'p_Bonf':>10}")
    print(header)
    print("-" * 110)
    for cell, r in cells_out.items():
        det_ci_lo, det_ci_hi = r["det_delta_ci95"]
        chi2 = r["mcnemar"]["chi2_with_cc"]
        p_bonf = r["mcnemar"]["p_bonferroni"]
        if p_bonf < 1e-300:
            p_str = "<1e-300"
        elif p_bonf < 1e-4:
            p_str = f"{p_bonf:.1e}"
        else:
            p_str = f"{p_bonf:.4f}"
        print(f"{cell:<14} "
              f"{r['det_baseline']:>9.1%} {r['det_intervention']:>9.1%} "
              f"{r['det_delta']:>+8.1%} "
              f"[{det_ci_lo:+.2%}, {det_ci_hi:+.2%}]  "
              f"{r['fp_baseline']:>8.1%} {r['fp_intervention']:>8.1%}  "
              f"{r['mcnemar']['b_baseline_only']:>4} "
              f"{r['mcnemar']['c_intervention_only']:>5}  "
              f"{chi2:>9.1f}  "
              f"{p_str:>10}")
    print("=" * 110)
    print(f"Bonferroni divisor: {N_CELLS}; bootstrap iters: {N_BOOTSTRAP}; "
          f"95% CI = percentile bootstrap on cluster-resampled adversarial pairs")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
