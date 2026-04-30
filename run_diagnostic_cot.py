"""
Layer-2 Chain-of-Thought false-positive diagnostic.

Per the v3+v4 reviewer synthesis (ChatGPT's "CoT FP diagnostic" idea
crystallized in DECISION_LOG D-43): when the model says YES on a CLEAN
chain (no planted violation), we don't know WHY. The labels alone tell
us "the model thinks something is wrong" but not "what specifically."

This script takes the per-cell results from a prior diagnostic run
(produced by run_diagnostic_violations.py), filters to clean chains
where the model said YES (false positives), and asks Haiku to explain:

  "You previously evaluated this sequence as containing a rule violation.
   Identify (a) which specific listed rule is violated, and (b) which
   event number caused the violation."

The model's free-text response tells us:
- For NBA: maybe "Event 4 has 7 fouls" (correct) or "shot clock not specified" (rendering bug)
- For CS:GO: maybe "Bomb plant has no site" (compliance marker missing)
- For Poker: maybe "Action proceeds clockwise can't be verified" (positional info missing)

Output is a per-cell breakdown of the most common rules the model claims
were violated on clean chains. That tells us exactly what renderer fix
each cell needs.

Usage:
    python run_diagnostic_cot.py --input RESULTS/diagnostic_violations_v5.json \\
        --batch --output RESULTS/diagnostic_cot.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_diagnostic_cot")


COT_QUESTION_TEMPLATE = (
    "You evaluated this sequence and concluded it contains a rule violation.\n\n"
    "From the constraint context above, identify:\n"
    "(a) Which specific rule (quote it directly from the constraint context)\n"
    "(b) Which event number(s) (0-indexed) caused the violation\n\n"
    "Reply in this exact format on two lines:\n"
    "RULE: <quoted rule>\n"
    "EVENTS: <comma-separated event numbers>"
)


def main():
    parser = argparse.ArgumentParser(description="Chain-of-Thought FP diagnostic")
    parser.add_argument("--input", type=Path, required=True,
                        help="Prior diagnostic JSON (e.g. RESULTS/diagnostic_violations_v5.json)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch", action="store_true",
                        help="Use Anthropic Batches API (50%% discount)")
    parser.add_argument("--cells", nargs="+", default=None,
                        help="Restrict to specific cells (default: all in input)")
    parser.add_argument("--max-fps-per-cell", type=int, default=20,
                        help="Cap on number of FP chains to ask about per cell")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # We need to re-fetch the FP chains' actual prompts. The prior diagnostic
    # didn't save raw chain text — only summary stats. So we re-run the
    # diagnostic pipeline up to the prompt-rendering step, identify which
    # custom_ids in the prior batch were FPs, and ask CoT about those.
    #
    # Simpler path: the prior batch IDs are available via Anthropic API.
    # We fetch raw responses, identify FPs (clean chain + YES response),
    # rebuild their prompts, and submit follow-up batch.
    import anthropic
    import re
    client = anthropic.Anthropic()

    # Find the prior log next to the input JSON to extract batch IDs.
    log_path = args.input.with_suffix(".log")
    if not log_path.exists():
        logger.error(f"Need log file at {log_path} to extract batch IDs")
        sys.exit(1)

    log_text = log_path.read_text()
    pat = re.compile(r"\[(\w+)\] Submitted batch (msgbatch_\w+)")
    batches: dict[str, list[str]] = {}
    for cell, bid in pat.findall(log_text):
        batches.setdefault(cell, []).append(bid)

    cells_to_run = args.cells or list(batches.keys())

    # Load + re-run the diagnostic up to PromptPair generation so we can
    # rebuild full prompts for FP chains.
    from src.cells.csgo.pipeline import CSGOPipeline
    from src.cells.nba.pipeline import NBAPipeline
    from src.cells.poker.pipeline import PokerPipeline
    from src.cells.pubg.pipeline import PUBGPipeline
    from src.cells.rocket_league.pipeline import RocketLeaguePipeline
    from src.common.config import load_cell_configs, load_harness_config
    from src.harness.actionables import compute_retention_rate
    from src.harness.model_evaluator import HAIKU_MODEL, ModelEvaluator
    from src.harness.prompts import PER_CELL_PROMPT_BUILDERS, PromptPair
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
        model=HAIKU_MODEL, dry_run=False,
        allowed_predictions=None,  # we want free-form CoT responses
        use_batch=args.batch,
    )

    domain_names = {
        "pubg": "PUBG", "nba": "NBA basketball", "csgo": "Counter-Strike",
        "rocket_league": "Rocket League", "poker": "poker",
    }

    cot_results: dict = {}

    for cell in cells_to_run:
        bids = batches.get(cell, [])
        if len(bids) < 1:
            logger.warning(f"[{cell}] no batches in log")
            continue

        # Identify FP chains: clean chains where model said YES.
        # Clean is the first batch per cell; adversarial is the second.
        clean_bid = bids[0]
        fp_chain_ids: set[str] = set()
        for r in client.messages.batches.results(clean_bid):
            if r.result.type != "succeeded":
                continue
            text = r.result.message.content[0].text.strip().lower() if r.result.message.content else ""
            first = text.splitlines()[0].strip().rstrip(".!?,") if text else ""
            if first == "yes":
                # custom_id is "<chain_id>__baseline" or "<chain_id>__intervention"
                cid = r.custom_id.rpartition("__")[0]
                fp_chain_ids.add(cid)

        if not fp_chain_ids:
            logger.info(f"[{cell}] no FPs in clean chains — skipping CoT")
            cot_results[cell] = {"n_fps": 0, "rule_counter": {}, "samples": []}
            continue

        logger.info(f"[{cell}] {len(fp_chain_ids)} FP chains identified")

        # Re-run pipeline + T to regenerate the chains. We need only
        # the chains whose chain_ids are in fp_chain_ids.
        cfg = cell_configs.get(cell)
        if cfg is None:
            continue
        pipeline = pipeline_map[cell](config=cfg)
        streams = pipeline.run(force_mock=False)
        t_fn = DOMAIN_T_STUBS.get(cell)
        candidates = []
        for stream in streams:
            candidates.extend(t_fn.translate(stream))
        chains = chain_builder.build_from_candidates(candidates, cell=cell)
        compute_retention_rate(chains, floor=harness_config.gate2_retention_floor)
        chains_passed = [c for c in chains if c.is_actionable]
        # Get just the FPs.
        fp_chains = [c for c in chains_passed if c.chain_id in fp_chain_ids]
        fp_chains = fp_chains[: args.max_fps_per_cell]
        if not fp_chains:
            logger.warning(f"[{cell}] could not regenerate FP chains; "
                           f"chain_id namespace may have shifted")
            continue

        # Build CoT prompts.
        builder = PER_CELL_PROMPT_BUILDERS[cell]()
        domain = domain_names[cell]
        cot_pairs = []
        chain_id_to_cot_id = {}
        for c in fp_chains:
            chain_block = builder.format_chain(c)
            constraint_block = builder.format_constraint_context(c)
            # The CoT prompt mirrors the intervention prompt shape but with
            # the CoT question instead of the YES/NO question.
            cot_prompt = builder._compose(
                chain_block=chain_block,
                constraint_block=constraint_block,
                question=COT_QUESTION_TEMPLATE,
            )
            # Reuse the PromptPair shape so evaluator accepts it.
            cot_id = f"{c.chain_id}_cot"
            cot_pairs.append(PromptPair(
                chain_id=cot_id, cell=cell,
                baseline_prompt=cot_prompt,  # same prompt for both
                intervention_prompt=cot_prompt,
                metadata={"cot": True},
            ))
            chain_id_to_cot_id[c.chain_id] = cot_id

        logger.info(f"[{cell}] submitting {len(cot_pairs)} CoT prompts")
        results, b_responses, _ = evaluator.evaluate_pairs(cot_pairs)

        # Parse the rule + events from each response.
        rule_counter: Counter = Counter()
        samples = []
        for result in results:
            text = result.baseline_raw or ""
            rule_m = re.search(r"^\s*RULE:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
            events_m = re.search(r"^\s*EVENTS?:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
            rule = rule_m.group(1).strip() if rule_m else "(unparseable)"
            events = events_m.group(1).strip() if events_m else "(unparseable)"
            rule_counter[rule[:120]] += 1
            samples.append({
                "chain_id": result.chain_id,
                "rule": rule[:200],
                "events": events[:120],
                "raw_excerpt": text[:300],
            })

        cot_results[cell] = {
            "n_fps_analyzed": len(samples),
            "rule_distribution": dict(rule_counter.most_common(10)),
            "samples": samples[:5],  # 5 illustrative samples per cell
        }

    # Print summary
    print("\n" + "=" * 80)
    print("CHAIN-OF-THOUGHT FP DIAGNOSTIC")
    print("=" * 80)
    for cell, data in cot_results.items():
        print(f"\n[{cell}] {data.get('n_fps_analyzed', 0)} FP chains analyzed")
        for rule, count in data.get("rule_distribution", {}).items():
            print(f"  {count:>3}× {rule}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(cot_results, f, indent=2)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
