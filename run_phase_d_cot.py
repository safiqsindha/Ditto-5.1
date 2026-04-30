"""
Phase D Layer-2 CoT diagnostic on residual FPs.

NBA and CS:GO are the only Phase D cells with non-trivial intervention FP
rates (NBA 9.9%, CS:GO 29.8%). For the writeup we want to know *which
rule* the model thinks was violated on each FP, so we can characterize
the failure mode mechanistically.

For each cell:
  1. Read the archived clean-batch JSONL
  2. Find FP chains: clean chains where intervention text == "yes"
  3. Cap at --max-fps and re-run the pipeline + T to regenerate prompts
  4. Submit a CoT batch asking "RULE: <quoted>  EVENTS: <indices>"
  5. Parse + count the rule distribution

Output: RESULTS/phase_d_cot_residual_fps.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
logger = logging.getLogger("run_phase_d_cot")


COT_QUESTION = (
    "You evaluated this sequence and concluded it contains a rule violation.\n\n"
    "From the constraint context above, identify:\n"
    "(a) Which specific rule (quote it directly from the constraint context)\n"
    "(b) Which event number(s) (0-indexed) caused the violation\n\n"
    "Reply in this exact format on two lines:\n"
    "RULE: <quoted rule>\n"
    "EVENTS: <comma-separated event numbers>"
)

# Map cell -> (label-prefix in archive)
ARCHIVES = {
    "nba":  ("nba_clean", "msgbatch_01NFDJnX98DCEgzd6hzzo6Fp"),
    "csgo": ("csgo_clean", "msgbatch_01QgQdJGqEdSTtFngdDa2REX"),
}


def parse_yn(text: str) -> str:
    if not text:
        return "abstain"
    first = text.strip().splitlines()[0].strip().rstrip(".!?,").lower()
    if first.startswith("yes"):
        return "yes"
    if first.startswith("no"):
        return "no"
    return "abstain"


def find_intervention_fp_chain_ids(jsonl_path: Path) -> set[str]:
    """Return chain_ids where intervention response was 'yes' on a clean chain."""
    fps: set[str] = set()
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "succeeded":
                continue
            cid = rec["custom_id"]
            # custom_id format: <chain_id>__baseline | <chain_id>__intervention
            # (PUBG/NBA used pre-fix format; CSGO post-fix has positional prefix.
            # Both end with __intervention or __baseline, so rpartition works.)
            chain_id, _, variant = cid.rpartition("__")
            if variant != "intervention":
                continue
            if parse_yn(rec.get("text", "")) == "yes":
                # If chain_id has a positional prefix (000123__abcd...), strip it
                # to match the chain_builder's chain_id namespace.
                if "__" in chain_id and chain_id.split("__")[0].isdigit():
                    chain_id = chain_id.split("__", 1)[1]
                fps.add(chain_id)
    return fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", default=["nba", "csgo"])
    parser.add_argument("--max-fps-per-cell", type=int, default=50)
    parser.add_argument("--output", type=Path,
                        default=Path("RESULTS/phase_d_cot_residual_fps.json"))
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    sys.path.insert(0, ".")
    from src.cells.csgo.pipeline import CSGOPipeline
    from src.cells.nba.pipeline import NBAPipeline
    from src.common.config import load_cell_configs, load_harness_config
    from src.harness.actionables import compute_retention_rate
    from src.harness.model_evaluator import HAIKU_MODEL, ModelEvaluator
    from src.harness.prompts import PER_CELL_PROMPT_BUILDERS, PromptPair
    from src.interfaces.chain_builder import FixedPerCellChainBuilder
    from src.interfaces.translation import DOMAIN_T_STUBS

    cell_configs = load_cell_configs()
    harness_config = load_harness_config()
    pipeline_map = {"nba": NBAPipeline, "csgo": CSGOPipeline}
    chain_lengths = {"pubg": 8, "nba": 5, "csgo": 10, "rocket_league": 12, "poker": 8}
    chain_builder = FixedPerCellChainBuilder(per_cell_chain_length=chain_lengths)
    evaluator = ModelEvaluator(model=HAIKU_MODEL, dry_run=False,
                               allowed_predictions=None, use_batch=True)
    domain_names = {"nba": "NBA basketball", "csgo": "Counter-Strike"}

    archive_dir = Path("RESULTS/phase_d_raw_batches")

    cot_results: dict = {}

    for cell in args.cells:
        if cell not in ARCHIVES:
            logger.warning(f"[{cell}] no archive entry; skipping")
            continue
        label, batch_id = ARCHIVES[cell]
        jsonl = archive_dir / f"{label}__{batch_id}.jsonl"
        if not jsonl.exists():
            logger.error(f"[{cell}] archive missing: {jsonl}")
            continue

        fp_chain_ids = find_intervention_fp_chain_ids(jsonl)
        logger.info(f"[{cell}] {len(fp_chain_ids)} intervention FPs in clean batch")
        if not fp_chain_ids:
            cot_results[cell] = {"n_fps": 0, "rule_distribution": {}, "samples": []}
            continue

        # Re-run pipeline to regenerate the chains with matching chain_ids
        cfg = cell_configs.get(cell)
        pipeline = pipeline_map[cell](config=cfg)
        streams = pipeline.run(force_mock=False)
        t_fn = DOMAIN_T_STUBS.get(cell)
        candidates = []
        for stream in streams:
            candidates.extend(t_fn.translate(stream))
        chains = chain_builder.build_from_candidates(candidates, cell=cell)
        compute_retention_rate(chains, floor=harness_config.gate2_retention_floor)
        chains_passed = [c for c in chains if c.is_actionable]

        fp_chains = [c for c in chains_passed if c.chain_id in fp_chain_ids]
        fp_chains = fp_chains[: args.max_fps_per_cell]
        if not fp_chains:
            logger.warning(f"[{cell}] could not regenerate any FP chains")
            cot_results[cell] = {
                "n_fps": len(fp_chain_ids), "regenerated": 0,
                "rule_distribution": {}, "samples": [],
                "note": "FP chain_ids identified but could not be regenerated; "
                        "chain_id namespace may have shifted.",
            }
            continue

        builder = PER_CELL_PROMPT_BUILDERS[cell]()
        cot_pairs = []
        for c in fp_chains:
            chain_block = builder.format_chain(c)
            constraint_block = builder.format_constraint_context(c)
            cot_prompt = builder._compose(
                chain_block=chain_block,
                constraint_block=constraint_block,
                question=COT_QUESTION,
            )
            cot_pairs.append(PromptPair(
                chain_id=f"{c.chain_id}_cot", cell=cell,
                baseline_prompt=cot_prompt,
                intervention_prompt=cot_prompt,
                metadata={"cot": True},
            ))

        logger.info(f"[{cell}] submitting {len(cot_pairs)} CoT prompts")
        results, _, _ = evaluator.evaluate_pairs(cot_pairs)

        rule_counter: Counter = Counter()
        samples = []
        for r in results:
            text = r.baseline_raw or ""
            rule_m = re.search(r"^\s*RULE:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
            events_m = re.search(r"^\s*EVENTS?:\s*(.+)$", text, re.MULTILINE | re.IGNORECASE)
            rule = rule_m.group(1).strip() if rule_m else "(unparseable)"
            events = events_m.group(1).strip() if events_m else "(unparseable)"
            rule_counter[rule[:120]] += 1
            samples.append({
                "chain_id": r.chain_id, "rule": rule[:200],
                "events": events[:120], "raw_excerpt": text[:300],
            })

        cot_results[cell] = {
            "n_fps_total": len(fp_chain_ids),
            "n_fps_analyzed": len(samples),
            "rule_distribution": dict(rule_counter.most_common(10)),
            "samples": samples[:10],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cot_results, indent=2))

    print("\n" + "=" * 80)
    print("PHASE D — Layer-2 CoT diagnostic on residual intervention FPs")
    print("=" * 80)
    for cell, data in cot_results.items():
        print(f"\n[{cell}] {data.get('n_fps_analyzed', 0)}/"
              f"{data.get('n_fps_total', 0)} FP chains analyzed")
        for rule, count in data.get("rule_distribution", {}).items():
            print(f"  {count:>3}x  {rule}")

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
