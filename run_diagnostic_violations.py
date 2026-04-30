"""
Tier-1 violation-detection diagnostic.

Per the 4-way reviewer synthesis (Gemini, ChatGPT, Opus #1, Opus #2)
documented in DECISION_LOG D-42, the pre-Phase-D pilot's 0% YES rate on
4 of 5 cells is a *floor effect* — the model defaults NO under the
"is this consistent with the rules" framing because 8-event chains don't
look like complete real games. Adding chain enrichment (A4-A7) didn't
help and may have made things worse by requiring more cross-event
coherence verification.

This diagnostic flips the dependent variable from "consistency rating"
to "violation detection." We take a real chain, plant ONE explicit
constraint violation, and ask:

    "Does the sequence of events above contain any event that violates
     the rules of {domain}? Reply with exactly one word: YES or NO."

For adversarial chains (with planted violation), GT="yes". For clean
chains (no planted violation), GT="no". Detection accuracy under
baseline vs intervention is the new measurement.

If intervention boosts detection accuracy on the failing cells, the
constraint context IS being used by the model — just on a different
task than the original validity-certification framing afforded. That
unblocks Phase D with a redesigned scoring approach.

Usage:
    python run_diagnostic_violations.py --batch \\
        --cells nba pubg poker --n-per-cell 20 \\
        --output RESULTS/diagnostic_violations.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_diagnostic_violations")

DIAGNOSTIC_QUESTION = (
    "Does the sequence of events above contain any event that violates "
    "the rules of {domain}?\n\n"
    "Most sequences are valid. Only reply YES if you can identify both "
    "(a) which specific listed rule is broken, and (b) which specific "
    "event in the sequence breaks it. Do not infer violations from unusual, "
    "rare, or incomplete patterns. Ignore artifacts in formatting, timing, "
    "or score reporting.\n\n"
    "Reply with exactly one word: YES (a violation is present) or NO."
)


def main():
    parser = argparse.ArgumentParser(description="v5 violation-detection diagnostic")
    parser.add_argument("--cells", nargs="+",
                        default=["nba", "pubg", "poker"],
                        choices=["pubg", "nba", "csgo", "rocket_league", "poker"])
    parser.add_argument("--n-per-cell", type=int, default=20,
                        help="Number of clean+adversarial chain pairs per cell")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch", action="store_true",
                        help="Use Anthropic Batches API (50%% discount)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use mock API responses (no spend)")
    parser.add_argument("--ignore-timestamps", action="store_true",
                        help="Add 'ignore timestamps when evaluating rule "
                             "violations' to the question. Instrument "
                             "correction for CS:GO synthetic-timestamp "
                             "confound (D-43).")
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set; use --dry-run for zero-cost test")
        sys.exit(1)

    from src.cells.csgo.pipeline import CSGOPipeline
    from src.cells.nba.pipeline import NBAPipeline
    from src.cells.poker.pipeline import PokerPipeline
    from src.cells.pubg.pipeline import PUBGPipeline
    from src.cells.rocket_league.pipeline import RocketLeaguePipeline
    from src.common.config import load_cell_configs, load_harness_config
    from src.harness.actionables import compute_retention_rate
    from src.harness.model_evaluator import HAIKU_MODEL, ModelEvaluator
    from src.harness.prompts import PER_CELL_PROMPT_BUILDERS, PromptPair
    from src.harness.violation_injector import inject
    from src.interfaces.chain_builder import FixedPerCellChainBuilder
    from src.interfaces.translation import DOMAIN_T_STUBS

    cell_configs = load_cell_configs()
    harness_config = load_harness_config()

    pipeline_map = {
        "pubg": PUBGPipeline, "nba": NBAPipeline, "csgo": CSGOPipeline,
        "rocket_league": RocketLeaguePipeline, "poker": PokerPipeline,
    }
    chain_lengths = {"pubg": 8, "nba": 5, "csgo": 10, "rocket_league": 12, "poker": 8}
    chain_builder = FixedPerCellChainBuilder(per_cell_chain_length=chain_lengths)
    evaluator = ModelEvaluator(
        model=HAIKU_MODEL, dry_run=args.dry_run,
        allowed_predictions=["yes", "no"],
        use_batch=args.batch and not args.dry_run,
    )

    domain_names = {
        "pubg": "PUBG", "nba": "NBA basketball", "csgo": "Counter-Strike",
        "rocket_league": "Rocket League", "poker": "poker",
    }

    # For each cell: build clean chains, generate adversarial twins,
    # render with diagnostic question, evaluate.
    results: dict = {}
    for cell in args.cells:
        cfg = cell_configs.get(cell)
        if cfg is None:
            logger.error(f"No config for {cell}")
            continue

        logger.info(f"[{cell}] Fetching streams...")
        pipeline = pipeline_map[cell](config=cfg)
        streams = pipeline.run(force_mock=False)
        logger.info(f"[{cell}] {len(streams)} streams, "
                    f"{sum(len(s) for s in streams)} events")

        t_fn = DOMAIN_T_STUBS.get(cell)
        if t_fn is None:
            logger.error(f"[{cell}] No T")
            continue

        candidates = []
        for stream in streams:
            candidates.extend(t_fn.translate(stream))
        chains = chain_builder.build_from_candidates(candidates, cell=cell)
        compute_retention_rate(chains, floor=harness_config.gate2_retention_floor)
        chains_passed = [c for c in chains if c.is_actionable]
        chains_passed = chains_passed[: args.n_per_cell]
        if not chains_passed:
            logger.warning(f"[{cell}] No chains passed Gate 2; skipping")
            continue

        logger.info(f"[{cell}] {len(chains_passed)} clean chains")

        # Generate adversarial twins
        adversarials = []
        for c in chains_passed:
            inj = inject(cell, c)
            if inj is not None:
                adversarials.append(inj)
        logger.info(f"[{cell}] {len(adversarials)} adversarial chains generated")

        # Render prompts via existing PromptBuilder but override the question
        builder = PER_CELL_PROMPT_BUILDERS[cell]()
        domain = domain_names[cell]
        diag_q = DIAGNOSTIC_QUESTION.format(domain=domain)
        if args.ignore_timestamps:
            diag_q = (
                "Note: ignore timestamps when evaluating rule violations. "
                "Focus only on whether any event in the sequence breaks "
                "the stated rules.\n\n" + diag_q
            )

        clean_pairs = []
        for c in chains_passed:
            chain_block = builder.format_chain(c)
            constraint_block = builder.format_constraint_context(c)
            clean_pairs.append(PromptPair(
                chain_id=c.chain_id, cell=cell,
                baseline_prompt=builder._compose(
                    chain_block=chain_block, constraint_block=None,
                    question=diag_q,
                ),
                intervention_prompt=builder._compose(
                    chain_block=chain_block, constraint_block=constraint_block,
                    question=diag_q,
                ),
                metadata={"adversarial": False, "n_events": len(c.events)},
            ))

        adv_pairs = []
        adv_meta = []
        for inj in adversarials:
            chain_block = builder.format_chain(inj.chain)
            constraint_block = builder.format_constraint_context(inj.chain)
            adv_pairs.append(PromptPair(
                chain_id=inj.chain.chain_id, cell=cell,
                baseline_prompt=builder._compose(
                    chain_block=chain_block, constraint_block=None,
                    question=diag_q,
                ),
                intervention_prompt=builder._compose(
                    chain_block=chain_block, constraint_block=constraint_block,
                    question=diag_q,
                ),
                metadata={"adversarial": True, "n_events": len(inj.chain.events)},
            ))
            adv_meta.append(inj)

        # Evaluate clean and adversarial separately
        logger.info(f"[{cell}] Evaluating {len(clean_pairs)} clean pairs...")
        _, clean_b, clean_i = evaluator.evaluate_pairs(clean_pairs)

        logger.info(f"[{cell}] Evaluating {len(adv_pairs)} adversarial pairs...")
        _, adv_b, adv_i = evaluator.evaluate_pairs(adv_pairs)

        # Score:
        #   Clean chains: GT="no" (no violation)
        #   Adversarial chains: GT="yes" (violation present)
        clean_b_correct = sum(1 for r in clean_b if r == "no")
        clean_i_correct = sum(1 for r in clean_i if r == "no")
        adv_b_correct = sum(1 for r in adv_b if r == "yes")
        adv_i_correct = sum(1 for r in adv_i if r == "yes")

        # Detection-rate McNemar on adversarial chains: did intervention catch
        # more violations than baseline?
        b_pairs = [r == "yes" for r in adv_b]
        i_pairs = [r == "yes" for r in adv_i]
        b_only = sum(1 for b, i in zip(b_pairs, i_pairs) if b and not i)
        c_only = sum(1 for b, i in zip(b_pairs, i_pairs) if not b and i)

        # False-positive rate on clean chains under intervention
        clean_fp_baseline = sum(1 for r in clean_b if r == "yes")
        clean_fp_intervention = sum(1 for r in clean_i if r == "yes")

        results[cell] = {
            "n_clean": len(clean_pairs),
            "n_adversarial": len(adv_pairs),
            "clean_baseline_correct_no": clean_b_correct,
            "clean_intervention_correct_no": clean_i_correct,
            "adv_baseline_correct_yes": adv_b_correct,
            "adv_intervention_correct_yes": adv_i_correct,
            "violation_detection_rate_baseline": (
                adv_b_correct / len(adv_pairs) if adv_pairs else 0.0
            ),
            "violation_detection_rate_intervention": (
                adv_i_correct / len(adv_pairs) if adv_pairs else 0.0
            ),
            "false_positive_rate_baseline": (
                clean_fp_baseline / len(clean_pairs) if clean_pairs else 0.0
            ),
            "false_positive_rate_intervention": (
                clean_fp_intervention / len(clean_pairs) if clean_pairs else 0.0
            ),
            "mcnemar_b_baseline_only": b_only,
            "mcnemar_c_intervention_only": c_only,
            "violation_clauses_used": list({m.violation_clause for m in adv_meta}),
        }

    # Print summary
    print("\n" + "=" * 78)
    print("VIOLATION-DETECTION DIAGNOSTIC")
    print("=" * 78)
    print(f"{'Cell':<14} {'Det@Base':>10} {'Det@Int':>10} {'Δ':>8} "
          f"{'FP@Base':>9} {'FP@Int':>9} {'b':>4} {'c':>4}")
    print("-" * 78)
    for cell, r in results.items():
        det_b = r["violation_detection_rate_baseline"]
        det_i = r["violation_detection_rate_intervention"]
        delta = det_i - det_b
        fp_b = r["false_positive_rate_baseline"]
        fp_i = r["false_positive_rate_intervention"]
        print(f"{cell:<14} {det_b:>10.1%} {det_i:>10.1%} {delta:>+8.1%} "
              f"{fp_b:>9.1%} {fp_i:>9.1%} "
              f"{r['mcnemar_b_baseline_only']:>4} "
              f"{r['mcnemar_c_intervention_only']:>4}")
    print("=" * 78)
    print("Det@Base = baseline violation-detection rate on adversarial chains")
    print("Det@Int  = intervention violation-detection rate on adversarial chains")
    print("Δ        = positive ⇒ constraint context helps detect violations")
    print("FP       = false positives on CLEAN chains (lower is better)")
    print("b / c    = baseline-only-correct / intervention-only-correct (McNemar)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "cells": results,
                "diagnostic_question": DIAGNOSTIC_QUESTION,
                "timestamp": time.time(),
            }, f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
