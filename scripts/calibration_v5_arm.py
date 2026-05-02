"""
v5-arm calibration: rerun a small sample of chain pairs through the v5 prompt
template (no strict/nonstrict suffix) so we can compare v5 phase-D detection
rates to v5.1.

Why this exists
---------------
v5.1 expands v5's 2-condition design (baseline / intervention) into 8
conditions by adding `marker × strict` axes via prompt_variants.py. The
v5.1 "nonstrict" suffix appends:
    "Reply YES if you believe a violation has occurred, NO otherwise."
That suffix did NOT exist in v5.

Phase 3 analysis on v5.1 batch results showed claude-haiku detecting at
~4% on the v5.1 condition closest to v5 (baseline_nomarker_nonstrict),
vs 60% in v5 phase D. The gap is much larger than the strict-format
effect alone would predict — pointing at the v5.1 nonstrict suffix as
a possible suppressor.

This script runs the v5 prompt EXACTLY (PromptBuilder.build → no suffix)
on a small sample, so we can isolate whether the suffix or the
chain-construction differences are the cause.

Output
------
results/v5_1/calibration_v5arm_<TIMESTAMP>/
    raw/<model>_results.jsonl    one JSON object per (chain_pair, variant)
    summary.json                 per-cell detection + FP rates
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibration_v5arm")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Reuse existing infrastructure — same .env loader as the main runner
from run_v5_1_replication import (  # noqa: E402
    build_chain_pairs_for_cell,
    load_env,
    _ENV_PATH,
    _raise_fd_limit,
)


def evaluate_one_model(
    model_id: str,
    pairs_by_cell: dict[str, list],
    out_dir: Path,
) -> dict:
    """
    Run one model against the v5 prompt set across all cells.
    Uses AnthropicRunner directly (sync, since the volume is small).
    Returns per-cell stats.
    """
    from src.harness.runner_native import AnthropicRunner, V51Result, _parse_response

    runner = AnthropicRunner(model_id=model_id, use_batch=False, dry_run=False)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe = model_id.replace("/", "_").replace(":", "_")
    raw_path = raw_dir / f"{safe}_results.jsonl"

    per_cell: dict[str, dict] = {}
    all_results: list[V51Result] = []
    with open(raw_path, "w") as raw_f:
        for cell, pairs in pairs_by_cell.items():
            logger.info(f"[{model_id}] {cell}: {len(pairs)} chain pairs")
            results = runner.evaluate(pairs, condition="v5_arm",
                                      usage_log_path=out_dir / "usage_log.jsonl")
            for r in results:
                raw_f.write(json.dumps(dataclasses.asdict(r)) + "\n")
            all_results.extend(results)
            # Per-cell stats: detection on adversarial = intervention_parsed=='yes'
            #                 FP on clean             = baseline_parsed=='yes'
            det_n = sum(1 for r in results if r.intervention_parsed in ("yes", "no"))
            det_y = sum(1 for r in results if r.intervention_parsed == "yes")
            fp_n  = sum(1 for r in results if r.baseline_parsed in ("yes", "no"))
            fp_y  = sum(1 for r in results if r.baseline_parsed == "yes")
            per_cell[cell] = {
                "n_pairs": len(pairs),
                "det_n": det_n, "det_y": det_y,
                "fp_n":  fp_n,  "fp_y":  fp_y,
                "det_rate": det_y / max(det_n, 1),
                "fp_rate":  fp_y  / max(fp_n,  1),
            }
            logger.info(f"  → det={100*det_y/max(det_n,1):.1f}% "
                        f"fp={100*fp_y/max(fp_n,1):.1f}%")

    return {
        "model_id": model_id,
        "n_total":  len(all_results),
        "per_cell": per_cell,
        "raw_path": str(raw_path),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-chains", type=int, default=50,
                   help="Chain pairs per cell (default 50)")
    p.add_argument("--model",    action="append", dest="models",
                   help="Anthropic model id (repeat for multiple)")
    p.add_argument("--cell",     action="append", dest="cells",
                   help="Cells to include (default: all 5)")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    _raise_fd_limit()
    load_env(_ENV_PATH)

    cells = args.cells or ["poker", "pubg", "nba", "csgo", "rocket_league"]
    models = args.models or ["claude-haiku-4-5-20251001"]

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "results" / "v5_1" / f"calibration_v5arm_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {out_dir}")
    logger.info(f"Models: {models}")
    logger.info(f"Cells:  {cells}  ({args.n_chains} chains each)")

    # Build chain pairs (reuses existing v5/v5.1 pipeline) and convert each
    # chain pair into a v5 PromptPair via PromptBuilder.build() — this is the
    # crucial step: no strict/nonstrict suffix, no marker stripping.
    from src.harness.prompts import PER_CELL_PROMPT_BUILDERS

    pairs_by_cell: dict[str, list] = {}
    for cell in cells:
        builder_cls = PER_CELL_PROMPT_BUILDERS.get(cell)
        if builder_cls is None:
            logger.warning(f"No PromptBuilder for cell '{cell}', skipping")
            continue
        builder = builder_cls()

        chain_pairs = build_chain_pairs_for_cell(cell=cell, n_chains=args.n_chains)
        v5_pairs = []
        for clean_chain, adv_chain in chain_pairs:
            # v5 prompt: build baseline (no constraint) and intervention (with
            # constraint) on the SAME chain.  In the V51Result schema,
            # baseline_parsed = answer to clean chain, intervention_parsed =
            # answer to adversarial chain.  So we build the v5 baseline prompt
            # on the clean chain (no constraint) and the v5 intervention
            # prompt on the adversarial chain (no constraint either, since v5
            # didn't have a warning prompt — the violation is detected purely
            # from chain structure).
            #
            # Wait: v5's "intervention" arm DID add a Constraint Context block.
            # That block was the v5 equivalent of v5.1's "intervention prompt".
            # For calibration we want to compare both arms:
            #   v5_arm_baseline:    clean chain + no constraint  ↔ adv chain + no constraint
            #   v5_arm_intervention: clean chain + constraint    ↔ adv chain + constraint
            #
            # We want to replicate v5's reported numbers exactly, so we do
            # both. Use PromptBuilder.build() on each chain — it produces a
            # PromptPair where .baseline_prompt = no-constraint and
            # .intervention_prompt = with-constraint on that single chain.
            from src.harness.prompts import PromptPair
            clean_pair = builder.build(clean_chain)
            adv_pair   = builder.build(adv_chain)
            # Pack into V5.1's PromptPair shape: baseline_prompt = clean,
            # intervention_prompt = adversarial. Test the V5 BASELINE arm
            # (no constraint) for the headline calibration vs v5 phase D.
            v5_pairs.append(PromptPair(
                chain_id=clean_chain.chain_id,
                cell=cell,
                baseline_prompt=clean_pair.baseline_prompt,    # clean, no constraint
                intervention_prompt=adv_pair.baseline_prompt,  # adv,   no constraint
                metadata={"v5_arm": "baseline_no_constraint",
                          "n_events_clean": len(clean_chain.events),
                          "n_events_adv":   len(adv_chain.events)},
            ))
        pairs_by_cell[cell] = v5_pairs
        logger.info(f"[{cell}] built {len(v5_pairs)} v5-arm prompt pairs "
                    f"(no v5.1 suffix, no constraint context)")

    # Evaluate each model
    summary = {"models": {}, "config": {
        "n_chains": args.n_chains, "cells": cells, "models": models,
        "v5_arm": "baseline_no_constraint",
        "note": "Uses v5's PromptBuilder.build() output without any v5.1 strict/nonstrict suffix and without v5's intervention constraint context. Closest analog to v5 phase D's baseline arm.",
    }}
    for mid in models:
        result = evaluate_one_model(mid, pairs_by_cell, out_dir)
        summary["models"][mid] = result

    # Aggregate + print
    print()
    print("=" * 80)
    print("V5-ARM CALIBRATION SUMMARY (vs v5 phase D haiku baseline ~60% det, ~2% fp)")
    print("=" * 80)
    for mid, mres in summary["models"].items():
        print(f"\n{mid}  (n={mres['n_total']} total pairs)")
        print(f"  {'cell':<14s} {'det%':>5s} {'fp%':>5s}")
        det_total_n = det_total_y = fp_total_n = fp_total_y = 0
        for cell, c in mres["per_cell"].items():
            print(f"  {cell:<14s} {100*c['det_rate']:4.1f}% {100*c['fp_rate']:4.1f}%")
            det_total_n += c['det_n']; det_total_y += c['det_y']
            fp_total_n  += c['fp_n'];  fp_total_y  += c['fp_y']
        if det_total_n:
            print(f"  {'AVG':<14s} {100*det_total_y/det_total_n:4.1f}% "
                  f"{100*fp_total_y/fp_total_n:4.1f}%")

    out_dir.joinpath("summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Summary written → {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
