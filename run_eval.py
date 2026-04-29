"""
v5 Phase D evaluation entry point.

Runs the full evaluation pipeline across all five cells
(fortnite, nba, csgo, rocket_league, poker):
  real data (or mock) → T → Gate 2 → PromptBuilder → Haiku API → McNemar → report

CF-3=A: 1× shuffled-event control per real chain evaluated in parallel.
Primary McNemar: real chains (GT="yes"). Shuffle-control McNemar: shuffled
chains (GT="no"). Both reported together. Use --no-shuffle to skip controls.

Requires ANTHROPIC_API_KEY in environment for real evaluation.
Use --dry-run for a zero-cost integration check with mock API responses.

Usage
-----
# Dry run (no API spend, deterministic responses):
  python run_eval.py --dry-run --output RESULTS/eval_dry_run.json

# Real evaluation (~$15 for 1,200 chains/cell × 5 cells × 2 calls + controls):
  python run_eval.py --output RESULTS/eval_phase_d.json

# Single cell only:
  python run_eval.py --cells nba --dry-run

# Skip CF-3 shuffled controls:
  python run_eval.py --no-shuffle --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env from project root (or any parent directory)
except ImportError:
    pass  # python-dotenv not installed; set env vars manually or via CI secrets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_eval")

ALL_CELLS = ["fortnite", "nba", "csgo", "rocket_league", "poker"]
_SHUFFLE_SEED = 42  # deterministic CF-3=A controls


def run_eval(
    cells: list[str],
    output_path: Path | None = None,
    dry_run: bool = False,
    force_mock: bool = False,
    include_shuffle: bool = True,
) -> bool:
    from src.cells.csgo.pipeline import CSGOPipeline
    from src.cells.fortnite.pipeline import FortnitePipeline
    from src.cells.nba.pipeline import NBAPipeline
    from src.cells.poker.pipeline import PokerPipeline
    from src.cells.rocket_league.pipeline import RocketLeaguePipeline
    from src.common.config import load_cell_configs, load_harness_config
    from src.harness.actionables import compute_retention_rate
    from src.harness.cell_runner import CellRunner
    from src.harness.mcnemar import McnemarResult, run_mcnemar
    from src.harness.model_evaluator import HAIKU_MODEL, ModelEvaluator
    from src.harness.prompts import PER_CELL_PROMPT_BUILDERS
    from src.harness.scoring import extract_binary_vectors, score_batch
    from src.interfaces.chain_builder import FixedPerCellChainBuilder
    from src.interfaces.translation import DOMAIN_T_STUBS

    cell_configs = load_cell_configs()
    harness_config = load_harness_config()

    pipeline_map = {
        "fortnite": FortnitePipeline,
        "nba": NBAPipeline,
        "csgo": CSGOPipeline,
        "rocket_league": RocketLeaguePipeline,
        "poker": PokerPipeline,
    }
    chain_lengths = {
        "fortnite": 8, "nba": 5, "csgo": 10,
        "rocket_league": 12, "poker": 8,
    }

    chain_builder = FixedPerCellChainBuilder(per_cell_chain_length=chain_lengths)
    runner = CellRunner(config=harness_config, chain_builder=chain_builder)
    evaluator = ModelEvaluator(
        model=HAIKU_MODEL,
        dry_run=dry_run,
        allowed_predictions=["yes", "no"],
    )

    # Primary eval accumulators (real chains, GT="yes")
    streams_by_cell: dict = {}
    baseline_responses: dict = {}
    intervention_responses: dict = {}
    ground_truths: dict = {}

    # CF-3=A shuffle-control accumulators (shuffled chains, GT="no")
    shuffle_results: dict[str, McnemarResult] = {}

    for cell in cells:
        if cell not in pipeline_map:
            logger.warning(f"Unknown cell '{cell}', skipping")
            continue
        config = cell_configs.get(cell)
        if config is None:
            logger.error(f"No config for cell '{cell}'")
            continue

        logger.info(f"[{cell}] Fetching event streams...")
        pipeline = pipeline_map[cell](config=config)
        streams = pipeline.run(force_mock=force_mock or dry_run)
        streams_by_cell[cell] = streams
        logger.info(
            f"[{cell}] {len(streams)} streams, "
            f"{sum(len(s) for s in streams)} events"
        )

        t_fn = DOMAIN_T_STUBS.get(cell)
        if t_fn is None:
            logger.error(f"No T for cell '{cell}'")
            continue
        runner.register_cell(cell, t_fn)

        # Translate → chain_builder → Gate 2
        logger.info(f"[{cell}] Translating streams to chains...")
        candidates = []
        for stream in streams:
            candidates.extend(t_fn.translate(stream))
        chains = chain_builder.build_from_candidates(candidates, cell=cell)
        retention = compute_retention_rate(chains, floor=harness_config.gate2_retention_floor)
        chains_passed = [c for c in chains if c.is_actionable]
        logger.info(
            f"[{cell}] {len(chains)} chains pre-Gate2 → "
            f"{len(chains_passed)} post-Gate2 "
            f"(retention {retention['retention_rate']:.1%})"
        )
        if not chains_passed:
            logger.warning(f"[{cell}] No chains passed Gate 2, skipping")
            continue

        builder_cls = PER_CELL_PROMPT_BUILDERS.get(cell)
        if builder_cls is None:
            logger.error(f"No PromptBuilder for cell '{cell}'")
            continue
        prompt_builder = builder_cls()

        # --- Primary evaluation (real chains, GT="yes") ----------------------
        pairs_real = [prompt_builder.build(c) for c in chains_passed]
        logger.info(
            f"[{cell}] Evaluating {len(pairs_real)} real pairs "
            f"({'dry-run' if dry_run else HAIKU_MODEL})..."
        )
        _, b_real, i_real = evaluator.evaluate_pairs(pairs_real)
        baseline_responses[cell] = b_real
        intervention_responses[cell] = i_real
        ground_truths[cell] = ["yes"] * len(chains_passed)

        # --- CF-3=A: shuffled-control evaluation (GT="no") ------------------
        if include_shuffle:
            shuffled = chain_builder.shuffle_chains(
                chains_passed, seed=_SHUFFLE_SEED, n_shuffles=1
            )
            if shuffled:
                pairs_shuf = [prompt_builder.build(c) for c in shuffled]
                logger.info(
                    f"[{cell}] CF-3=A: evaluating {len(pairs_shuf)} shuffled "
                    f"control pairs..."
                )
                _, b_shuf, i_shuf = evaluator.evaluate_pairs(pairs_shuf)
                gt_shuf = ["no"] * len(shuffled)

                b_scores = score_batch(shuffled, gt_shuf, b_shuf)
                i_scores = score_batch(shuffled, gt_shuf, i_shuf)
                b_vec, i_vec = extract_binary_vectors(b_scores, i_scores)

                if len(b_vec) >= harness_config.min_discordant_pairs:
                    shuf_mcnemar = run_mcnemar(
                        baseline_correct=b_vec,
                        intervention_correct=i_vec,
                        cell=cell,
                        alpha=harness_config.alpha,
                        bonferroni_divisor=harness_config.bonferroni_divisor,
                        continuity_correction=harness_config.continuity_correction,
                        bootstrap_iterations=harness_config.bootstrap_iterations,
                        bootstrap_seed=harness_config.bootstrap_seed,
                        min_discordant_pairs=harness_config.min_discordant_pairs,
                    )
                    shuffle_results[cell] = shuf_mcnemar
                    logger.info(
                        f"[{cell}] CF-3=A shuffle McNemar: {shuf_mcnemar.summary()}"
                    )
                else:
                    logger.warning(
                        f"[{cell}] CF-3=A: too few non-abstain pairs "
                        f"({len(b_vec)}) for shuffle McNemar"
                    )

    # Primary McNemar across all cells
    logger.info("Running primary McNemar analysis across all cells...")
    report = runner.run(
        streams_by_cell,
        baseline_responses=baseline_responses,
        intervention_responses=intervention_responses,
        ground_truths=ground_truths,
    )

    _print_eval_summary(report, shuffle_results)

    if output_path:
        _save_report(report, shuffle_results, output_path)
        logger.info(f"Report saved to {output_path}")

    return all(r.mcnemar is not None for r in report.cells)


_CF3_LEAKAGE_RATIO_THRESHOLD = 0.50  # |cf3_h|/|primary_h| above this → leakage suspected


def _leakage_diagnosis(cf3_h: float, primary_h: float) -> tuple[float | None, bool]:
    """
    Return (ratio, leakage_suspected).
    ratio  = |cf3_h| / |primary_h|; None when primary effect is near zero.
    leakage_suspected when BOTH effects are same direction (same sign) AND
    ratio > threshold.  Opposite-sign effects indicate the model is correctly
    using constraint context for one condition only — not format leakage.
    """
    if abs(primary_h) < 0.01:
        return None, False
    ratio = abs(cf3_h) / abs(primary_h)
    same_direction = (cf3_h * primary_h) > 0  # both positive or both negative
    suspected = same_direction and (ratio > _CF3_LEAKAGE_RATIO_THRESHOLD)
    return ratio, suspected


def _print_eval_summary(report, shuffle_results: dict) -> None:
    print(f"\n{'='*70}")
    print(f"v5 PHASE D EVALUATION — {report.run_id}")
    print(f"{'='*70}")
    print("  PRIMARY McNemar (real chains, GT=yes):")
    for r in report.cells:
        if r.mcnemar:
            m = r.mcnemar
            sig = "**SIG**" if m.significant else "n.s."
            power_str = f"  pwr={r.power:.2f}" if r.power is not None else ""
            mde_str = f"  MDE={r.mde:.3f}" if r.mde is not None else ""
            print(
                f"    [{r.cell:<14}] n={r.n_chains_post_gate2:>5} "
                f"b={m.b:>4} c={m.c:>4} "
                f"p_corr={m.p_value_corrected:.4f} h={m.effect_size_h:+.4f} {sig}"
                f"{power_str}{mde_str}"
            )
        else:
            errs = "; ".join(r.errors[:2]) if r.errors else "no responses"
            mde_str = f"  MDE={r.mde:.3f}" if r.mde is not None else ""
            print(f"    [{r.cell:<14}] SKIPPED ({errs}){mde_str}")

    if shuffle_results:
        print("\n  CF-3=A SHUFFLE CONTROL McNemar (shuffled chains, GT=no):")
        for cell, m in shuffle_results.items():
            sig = "**SIG**" if m.significant else "n.s."
            print(
                f"    [{cell:<14}] n={m.n_chains:>5} "
                f"b={m.b:>4} c={m.c:>4} "
                f"p_corr={m.p_value_corrected:.4f} h={m.effect_size_h:+.4f} {sig}"
            )

        sig_shuffle = [cell for cell, m in shuffle_results.items() if m.significant]
        if sig_shuffle:
            print(f"\n  WARNING: CF-3=A significant for {sig_shuffle}.")
            for cell in sig_shuffle:
                cf3_m = shuffle_results[cell]
                primary_m = next(
                    (r.mcnemar for r in report.cells if r.cell == cell), None
                )
                primary_h = primary_m.effect_size_h if primary_m else 0.0
                ratio, suspected = _leakage_diagnosis(cf3_m.effect_size_h, primary_h)
                if ratio is None:
                    print(
                        f"    [{cell}] CF-3=A significant but primary effect near zero "
                        f"— investigate before claiming primary result valid."
                    )
                elif suspected:
                    print(
                        f"    [{cell}] FORMAT LEAKAGE SUSPECTED: "
                        f"|CF3 h|={abs(cf3_m.effect_size_h):.3f} is "
                        f"{ratio:.0%} of primary |h|={abs(primary_h):.3f} "
                        f"(>{_CF3_LEAKAGE_RATIO_THRESHOLD:.0%} threshold). "
                        f"Model may be using constraint-format cue, not causal reasoning."
                    )
                else:
                    print(
                        f"    [{cell}] Expected discrimination: "
                        f"|CF3 h|={abs(cf3_m.effect_size_h):.3f} is "
                        f"{ratio:.0%} of primary |h|={abs(primary_h):.3f} "
                        f"(within {_CF3_LEAKAGE_RATIO_THRESHOLD:.0%} tolerance)."
                    )

    agg = report.aggregate
    if agg:
        print(
            f"\n  Pooled primary: cells={agg.get('n_cells',0)} "
            f"significant={agg.get('n_cells_significant',0)} "
            f"pooled_p={agg.get('pooled_p', 1.0):.4f}"
        )
    print(f"{'='*70}\n")


def _save_report(report, shuffle_results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = report.to_dict()

    # Augment per-cell primary results with power/MDE
    cell_result_map = {r.cell: r for r in report.cells}
    for cell_dict in data.get("cells", []):
        cell = cell_dict.get("cell", "")
        r = cell_result_map.get(cell)
        if r:
            cell_dict["power"] = r.power
            cell_dict["mde"] = r.mde

    # CF-3=A shuffle control with leakage diagnosis
    cf3_data: dict = {}
    for cell, m in shuffle_results.items():
        primary_m = next((r.mcnemar for r in report.cells if r.cell == cell), None)
        primary_h = primary_m.effect_size_h if primary_m else 0.0
        ratio, suspected = _leakage_diagnosis(m.effect_size_h, primary_h)
        cf3_data[cell] = {
            "n_shuffled": m.n_chains,
            "b": m.b,
            "c": m.c,
            "statistic": m.statistic,
            "p_value": m.p_value,
            "p_value_corrected": m.p_value_corrected,
            "significant": m.significant,
            "effect_size_h": m.effect_size_h,
            "ci_lower": m.ci_lower,
            "ci_upper": m.ci_upper,
            "summary": m.summary(),
            "leakage_ratio": ratio,
            "leakage_suspected": suspected,
        }
    data["cf3_shuffle_control"] = cf3_data

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="v5 Phase D Evaluation")
    parser.add_argument(
        "--cells", nargs="+", default=ALL_CELLS,
        choices=ALL_CELLS, help="Cells to evaluate"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to save JSON report"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Use mock API responses (no ANTHROPIC_API_KEY needed, no cost)"
    )
    parser.add_argument(
        "--force-mock", action="store_true",
        help="Use mock event data even if real credentials are present"
    )
    parser.add_argument(
        "--no-shuffle", action="store_true",
        help="Skip CF-3=A shuffled-control evaluation"
    )
    args = parser.parse_args()

    if not args.dry_run:
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.error(
                "ANTHROPIC_API_KEY not set. "
                "Use --dry-run for a zero-cost integration check, "
                "or set the env var for real evaluation."
            )
            sys.exit(1)

    success = run_eval(
        cells=args.cells,
        output_path=args.output,
        dry_run=args.dry_run,
        force_mock=args.force_mock,
        include_shuffle=not args.no_shuffle,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
