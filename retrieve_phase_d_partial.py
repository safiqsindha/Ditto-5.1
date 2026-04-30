"""
Retrieve and score the PUBG + NBA Phase D batch results that completed
before the Poker custom_id collision crashed the run at 02:18:47.

The four completed batches sit on Anthropic's side ready to be pulled.
We fetch results, compute per-cell stats matching the run_diagnostic_violations
report shape, and write them to a partial JSON so synthesis can resume once
Poker/RL/CS:GO complete.

Saved batch IDs (from RESULTS/phase_d.log):
  pubg clean:        msgbatch_01VcHLdLj8zwBPjtgrQFw3Lp
  pubg adversarial:  msgbatch_01QxaNgJFjDjJdargiuZbUpF
  nba clean:         msgbatch_01NFDJnX98DCEgzd6hzzo6Fp
  nba adversarial:   msgbatch_01WmT9EJ3fzbcot4yDMx4FG6

Within each batch, custom_ids are <chain_id>__baseline and <chain_id>__intervention.
Clean batches → GT="no" (no violation); adversarial batches → GT="yes".
"""
from __future__ import annotations

import json
import logging
import os
import sys
from collections import defaultdict
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
logger = logging.getLogger("retrieve_phase_d_partial")

BATCHES = {
    "pubg": {
        "clean": "msgbatch_01VcHLdLj8zwBPjtgrQFw3Lp",
        "adversarial": "msgbatch_01QxaNgJFjDjJdargiuZbUpF",
    },
    "nba": {
        "clean": "msgbatch_01NFDJnX98DCEgzd6hzzo6Fp",
        "adversarial": "msgbatch_01WmT9EJ3fzbcot4yDMx4FG6",
    },
}


def parse_yn(text: str) -> str:
    """Lower-case first token; map to 'yes' / 'no' / 'abstain'."""
    if not text:
        return "abstain"
    first = text.strip().splitlines()[0].strip().rstrip(".!?,").lower()
    if first.startswith("yes"):
        return "yes"
    if first.startswith("no"):
        return "no"
    return "abstain"


def fetch_batch(client, batch_id: str) -> dict[str, str]:
    """Return custom_id -> raw text for all succeeded results in a batch."""
    out: dict[str, str] = {}
    n_total = 0
    n_succ = 0
    for r in client.messages.batches.results(batch_id):
        n_total += 1
        if r.result.type != "succeeded":
            continue
        n_succ += 1
        msg = r.result.message
        text = ""
        if msg.content:
            block = msg.content[0]
            text = getattr(block, "text", "") or ""
        out[r.custom_id] = text
    logger.info(f"  {batch_id}: {n_succ}/{n_total} succeeded")
    return out


def score_cell(clean_results: dict[str, str], adv_results: dict[str, str]) -> dict:
    """Compute the same metrics shape that run_diagnostic_violations.py emits."""
    # Re-pair by chain_id within each batch.
    def pair(results: dict[str, str]) -> dict[str, dict[str, str]]:
        # custom_id is "<chain_id>__baseline" or "<chain_id>__intervention".
        # Some chain_ids may contain underscores so use rpartition.
        paired: dict[str, dict[str, str]] = defaultdict(dict)
        for cid, text in results.items():
            chain_id, _, variant = cid.rpartition("__")
            paired[chain_id][variant] = parse_yn(text)
        return paired

    clean_pairs = pair(clean_results)
    adv_pairs = pair(adv_results)

    n_clean = len(clean_pairs)
    n_adv = len(adv_pairs)

    # Clean: GT=no → "yes" is a false positive
    clean_fp_b = sum(1 for v in clean_pairs.values() if v.get("baseline") == "yes")
    clean_fp_i = sum(1 for v in clean_pairs.values() if v.get("intervention") == "yes")
    clean_b_correct = sum(1 for v in clean_pairs.values() if v.get("baseline") == "no")
    clean_i_correct = sum(1 for v in clean_pairs.values() if v.get("intervention") == "no")

    # Adversarial: GT=yes → "yes" is correct detection
    adv_b_correct = sum(1 for v in adv_pairs.values() if v.get("baseline") == "yes")
    adv_i_correct = sum(1 for v in adv_pairs.values() if v.get("intervention") == "yes")

    # McNemar on adversarial detection: did intervention catch more violations?
    b_only = c_only = both_yes = both_no = 0
    for v in adv_pairs.values():
        bv = v.get("baseline") == "yes"
        iv = v.get("intervention") == "yes"
        if bv and iv:
            both_yes += 1
        elif bv and not iv:
            b_only += 1
        elif iv and not bv:
            c_only += 1
        else:
            both_no += 1

    # Also compute McNemar on the CLEAN batch (FP discordances) for completeness.
    clean_b_only = clean_c_only = 0
    for v in clean_pairs.values():
        bv = v.get("baseline") == "yes"
        iv = v.get("intervention") == "yes"
        if bv and not iv:
            clean_b_only += 1
        elif iv and not bv:
            clean_c_only += 1

    return {
        "n_clean": n_clean,
        "n_adversarial": n_adv,
        "clean_baseline_correct_no": clean_b_correct,
        "clean_intervention_correct_no": clean_i_correct,
        "adv_baseline_correct_yes": adv_b_correct,
        "adv_intervention_correct_yes": adv_i_correct,
        "violation_detection_rate_baseline": adv_b_correct / n_adv if n_adv else 0.0,
        "violation_detection_rate_intervention": adv_i_correct / n_adv if n_adv else 0.0,
        "false_positive_rate_baseline": clean_fp_b / n_clean if n_clean else 0.0,
        "false_positive_rate_intervention": clean_fp_i / n_clean if n_clean else 0.0,
        "mcnemar_b_baseline_only": b_only,
        "mcnemar_c_intervention_only": c_only,
        "mcnemar_both_yes": both_yes,
        "mcnemar_both_no": both_no,
        "clean_mcnemar_b_baseline_only": clean_b_only,
        "clean_mcnemar_c_intervention_only": clean_c_only,
    }


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.error("ANTHROPIC_API_KEY not set")
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic()

    out: dict = {"cells": {}}
    for cell, ids in BATCHES.items():
        logger.info(f"[{cell}] retrieving batches...")
        clean = fetch_batch(client, ids["clean"])
        adv = fetch_batch(client, ids["adversarial"])
        scored = score_cell(clean, adv)
        scored["batch_ids"] = ids
        out["cells"][cell] = scored
        logger.info(
            f"[{cell}] FP_b={scored['false_positive_rate_baseline']:.1%} "
            f"FP_i={scored['false_positive_rate_intervention']:.1%} "
            f"Det_b={scored['violation_detection_rate_baseline']:.1%} "
            f"Det_i={scored['violation_detection_rate_intervention']:.1%} "
            f"McNemar b={scored['mcnemar_b_baseline_only']} "
            f"c={scored['mcnemar_c_intervention_only']}"
        )

    output_path = Path("RESULTS/phase_d_partial_pubg_nba.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    logger.info(f"Saved → {output_path}")

    print("\n" + "=" * 78)
    print("PHASE D PARTIAL RESULTS — PUBG + NBA")
    print("=" * 78)
    print(f"{'Cell':<8} {'Det@Base':>10} {'Det@Int':>10} {'Δ':>8} "
          f"{'FP@Base':>9} {'FP@Int':>9} {'b':>5} {'c':>5}")
    print("-" * 78)
    for cell, r in out["cells"].items():
        det_b = r["violation_detection_rate_baseline"]
        det_i = r["violation_detection_rate_intervention"]
        delta = det_i - det_b
        fp_b = r["false_positive_rate_baseline"]
        fp_i = r["false_positive_rate_intervention"]
        print(f"{cell:<8} {det_b:>10.1%} {det_i:>10.1%} {delta:>+8.1%} "
              f"{fp_b:>9.1%} {fp_i:>9.1%} "
              f"{r['mcnemar_b_baseline_only']:>5} "
              f"{r['mcnemar_c_intervention_only']:>5}")


if __name__ == "__main__":
    main()
