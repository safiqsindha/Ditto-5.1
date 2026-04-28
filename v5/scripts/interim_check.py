"""
Interim analysis script for v5 McNemar experiment.

Pre-registered usage (RISK_MITIGATIONS M1): run after collecting the first
600 chains/cell. If observed |c-b|/n < 0.04 in any cell, pause that cell
before full acquisition.

Usage
-----
    python -m v5.scripts.interim_check --results <results.json>
    python -m v5.scripts.interim_check --results <results.json> --interim-fraction 0.5

Input format (results.json)
---------------------------
    {
        "cells": {
            "nba": {
                "baseline_correct": [true, false, ...],
                "intervention_correct": [true, false, ...]
            }
        }
    }

Output
------
    Per-cell interim McNemar + projected power for remaining chains.
    Emits a PAUSE recommendation if |c-b|/n < 0.04 for any cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

from v5.src.harness.mcnemar import run_mcnemar

PAUSE_THRESHOLD = 0.04
BONFERRONI_DIVISOR = 5
FULL_N_PER_CELL = 1200
ALPHA = 0.05
POWER_TARGET = 0.80


class CellInterimResult(NamedTuple):
    cell: str
    n_interim: int
    n_chains_full_target: int
    effect_delta: float
    p_discordant: float
    interim_significant: bool
    projected_power: float
    recommend_pause: bool


def _compute_projected_power(
    n_full: int,
    delta: float,
    p_discordant: float,
    alpha_corrected: float,
) -> float:
    """
    Approximate power for McNemar at n_full chains given the observed delta
    and discordant rate from the interim.

    Uses normal approximation: chi2 ≈ (n*delta - 1)² / (n*p_d).
    """
    if p_discordant <= 0 or delta <= 0:
        return 0.0
    from scipy import stats
    expected_disc = n_full * p_discordant
    expected_abs_diff = n_full * abs(delta)
    if expected_disc <= 0:
        return 0.0
    chi2_expected = (max(expected_abs_diff - 1, 0)) ** 2 / expected_disc
    chi2_crit = stats.chi2.ppf(1.0 - alpha_corrected, df=1)
    # Non-centrality parameter approximation
    ncp = chi2_expected
    power = 1.0 - stats.ncx2.cdf(chi2_crit, df=1, nc=ncp)
    return max(0.0, min(1.0, float(power)))


def check_cell(
    cell: str,
    baseline_correct: list[bool],
    intervention_correct: list[bool],
    interim_fraction: float = 0.5,
    full_n: int = FULL_N_PER_CELL,
) -> CellInterimResult:
    n = len(baseline_correct)
    cut = max(1, int(n * interim_fraction))
    bl_interim = baseline_correct[:cut]
    iv_interim = intervention_correct[:cut]

    result = run_mcnemar(
        bl_interim, iv_interim, cell=cell,
        bonferroni_divisor=BONFERRONI_DIVISOR,
        bootstrap_iterations=500,
    )

    n_interim = cut
    b, c = result.b, result.c
    delta = (c - b) / n_interim if n_interim > 0 else 0.0
    p_disc = (b + c) / n_interim if n_interim > 0 else 0.0
    alpha_corr = ALPHA / BONFERRONI_DIVISOR

    projected = _compute_projected_power(full_n, delta, p_disc, alpha_corr)
    pause = abs(delta) < PAUSE_THRESHOLD

    return CellInterimResult(
        cell=cell,
        n_interim=n_interim,
        n_chains_full_target=full_n,
        effect_delta=delta,
        p_discordant=p_disc,
        interim_significant=result.significant,
        projected_power=projected,
        recommend_pause=pause,
    )


def print_report(results: list[CellInterimResult]) -> None:
    print("=" * 70)
    print("v5 INTERIM ANALYSIS REPORT")
    print(f"Pause threshold: |delta| < {PAUSE_THRESHOLD}")
    print(f"Full-run target: n={FULL_N_PER_CELL}/cell")
    print("=" * 70)
    any_pause = False
    for r in results:
        status = "🔴 PAUSE RECOMMENDED" if r.recommend_pause else "🟢 continue"
        print(f"\n[{r.cell}] n_interim={r.n_interim}")
        print(f"  delta=(c-b)/n = {r.effect_delta:+.4f}  |  p_disc = {r.p_discordant:.4f}")
        print(f"  interim significant: {r.interim_significant}")
        print(f"  projected power at n={r.n_chains_full_target}: {r.projected_power:.1%}")
        print(f"  recommendation: {status}")
        if r.recommend_pause:
            any_pause = True

    print("\n" + "=" * 70)
    if any_pause:
        print("ACTION REQUIRED: One or more cells below effect-size threshold.")
        print("Review sample-size assumptions before continuing (see RISK_MITIGATIONS M1).")
    else:
        print("All cells above threshold. Proceed with full acquisition.")
    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v5 interim McNemar analysis")
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument(
        "--interim-fraction", type=float, default=0.5,
        help="Fraction of chains to treat as interim (default: 0.5)",
    )
    parser.add_argument(
        "--full-n", type=int, default=FULL_N_PER_CELL,
        help=f"Full-run target per cell (default: {FULL_N_PER_CELL})",
    )
    args = parser.parse_args(argv)

    path = Path(args.results)
    if not path.exists():
        print(f"ERROR: results file not found: {path}", file=sys.stderr)
        return 1

    with path.open() as f:
        data = json.load(f)

    cells_data = data.get("cells", {})
    if not cells_data:
        print("ERROR: no 'cells' key in results JSON", file=sys.stderr)
        return 1

    cell_results = []
    for cell, cell_data in cells_data.items():
        bl = cell_data.get("baseline_correct", [])
        iv = cell_data.get("intervention_correct", [])
        if len(bl) < 10:
            print(f"WARNING: [{cell}] only {len(bl)} chains — skipping", file=sys.stderr)
            continue
        r = check_cell(cell, bl, iv, args.interim_fraction, args.full_n)
        cell_results.append(r)

    if not cell_results:
        print("ERROR: no cells had enough data for analysis", file=sys.stderr)
        return 1

    print_report(cell_results)
    any_pause = any(r.recommend_pause for r in cell_results)
    return 1 if any_pause else 0


if __name__ == "__main__":
    sys.exit(main())
