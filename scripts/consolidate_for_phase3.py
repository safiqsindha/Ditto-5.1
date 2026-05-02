"""
Consolidate v5.1 raw JSONL outputs across all run directories into one
analysis-ready dataset for Phase 3.

Why this exists
---------------
Phase 2 ran in 7 different output directories due to FD-limit retries,
DeepSeek off-peak scheduling, and Google sequential restarts. Each
output dir has its own raw/<model>_results.jsonl files. Phase 3 needs
a single canonical dataset with provenance tracking.

Semantic note (CRITICAL — corrected 2026-05-02 after FP diagnostic)
-------------------------------------------------------------------
The classify question is: "Is the sequence consistent with the rules
of {domain}?"
  YES = consistent → no violation
  NO  = not consistent → violation present

Detection metrics (matching v5 phase D framing):
  Sensitivity (true positive rate on adversarial chain):
      P(intervention_parsed == 'no')
  Specificity (true negative rate on clean chain):
      P(baseline_parsed == 'yes')
  False positive rate:
      P(baseline_parsed == 'no')   = 1 - specificity

Output
------
RESULTS/v5_1/phase3_consolidated/
  consolidated.jsonl         one line per (model, chain, condition, cell) row
  per_model_summary.csv      sensitivity, specificity, parse rate per model
  per_condition_summary.csv  sensitivity, specificity per (model, condition)
  per_cell_summary.csv       sensitivity, specificity per (model, cell)
  manifest.json              source dir for each model + run inventory
"""
from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("consolidate")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "v5_1"
OUT_DIR = RESULTS_DIR / "phase3_consolidated"

# §7.3 inclusion floor — model must have strict-parse rate >= 90% to enter
# the primary regression.
PARSE_RATE_FLOOR = 0.90


def find_all_raw_files() -> list[Path]:
    """Find every raw/<model>_results.jsonl across all output dirs (including
    smoke and calibration are excluded — only `full_*` runs)."""
    return sorted(RESULTS_DIR.glob("full_*/raw/*.jsonl"))


def pick_best_per_model(files: Iterable[Path]) -> dict[str, tuple[Path, int]]:
    """For each model_id, pick the raw file with the highest row count."""
    best: dict[str, tuple[Path, int]] = {}
    for f in files:
        n = sum(1 for _ in open(f))
        if n == 0:
            continue
        # Reconstruct model_id from filename (the runner replaces '/' with '_')
        # We can't perfectly invert that, but for analysis the filename slug
        # is unique enough. The JSONL rows themselves carry the canonical
        # model_id field which we use as the dictionary key.
        with open(f) as fh:
            first = json.loads(fh.readline())
            model_id = first["model_id"]
        if model_id not in best or n > best[model_id][1]:
            best[model_id] = (f, n)
    return best


def load_rows(path: Path) -> list[dict]:
    rows = []
    src = path.parent.parent.name  # full_<TS> dir
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        r["_source_dir"] = src
        rows.append(r)
    return rows


def write_consolidated(best: dict[str, tuple[Path, int]], out_path: Path) -> int:
    """Write one big JSONL with all rows + source_dir provenance. Return total
    rows written."""
    n_total = 0
    with open(out_path, "w") as fout:
        for model_id, (path, _) in sorted(best.items()):
            for r in load_rows(path):
                fout.write(json.dumps(r) + "\n")
                n_total += 1
    return n_total


def compute_per_model_summary(consolidated_path: Path) -> list[dict]:
    """One row per model_id with overall metrics."""
    by_model: dict[str, dict] = defaultdict(lambda: {
        "n_rows": 0,
        "n_baseline_yes": 0,    # % YES on clean = specificity
        "n_baseline_no":  0,
        "n_baseline_ab":  0,
        "n_intervn_yes":  0,    # % YES on adv = miss rate
        "n_intervn_no":   0,    # % NO  on adv = sensitivity
        "n_intervn_ab":   0,
        "n_b_retry":      0,
        "n_iv_retry":     0,
        "cost_total":     0.0,
    })
    for line in open(consolidated_path):
        r = json.loads(line)
        m = r["model_id"]
        e = by_model[m]
        e["n_rows"] += 1
        bp = r.get("baseline_parsed", "")
        ip = r.get("intervention_parsed", "")
        e[f"n_baseline_{'yes' if bp=='yes' else ('no' if bp=='no' else 'ab')}"] += 1
        e[f"n_intervn_{'yes' if ip=='yes' else ('no' if ip=='no' else 'ab')}"] += 1
        if r.get("baseline_needed_retry"):     e["n_b_retry"] += 1
        if r.get("intervention_needed_retry"): e["n_iv_retry"] += 1
        e["cost_total"] += float(r.get("baseline_cost_usd", 0)) + float(r.get("intervention_cost_usd", 0))

    out = []
    for m, e in sorted(by_model.items()):
        n = e["n_rows"]
        b_total = e["n_baseline_yes"] + e["n_baseline_no"]
        i_total = e["n_intervn_yes"]  + e["n_intervn_no"]
        b_clf_total = b_total + e["n_baseline_ab"]
        i_clf_total = i_total + e["n_intervn_ab"]
        # Strict-parse rate: fraction of calls returning yes/no (not abstain)
        parse_rate = (b_total + i_total) / max(b_clf_total + i_clf_total, 1)
        # Sensitivity = % NO on adversarial chain (correct violation flag)
        sensitivity = e["n_intervn_no"] / max(i_total, 1)
        # Specificity = % YES on clean chain (correct "no violation" call)
        specificity = e["n_baseline_yes"] / max(b_total, 1)
        # FP rate = % NO on clean chain (incorrect violation flag)
        fp_rate = e["n_baseline_no"] / max(b_total, 1)
        accuracy = (e["n_baseline_yes"] + e["n_intervn_no"]) / max(b_total + i_total, 1)
        out.append({
            "model_id":       m,
            "n_rows":         n,
            "parse_rate":     round(parse_rate, 4),
            "meets_§7.3":     parse_rate >= PARSE_RATE_FLOOR,
            "sensitivity":    round(sensitivity, 4),
            "specificity":    round(specificity, 4),
            "fp_rate":        round(fp_rate, 4),
            "accuracy":       round(accuracy, 4),
            "n_baseline_yes": e["n_baseline_yes"],
            "n_baseline_no":  e["n_baseline_no"],
            "n_baseline_ab":  e["n_baseline_ab"],
            "n_intervn_yes":  e["n_intervn_yes"],
            "n_intervn_no":   e["n_intervn_no"],
            "n_intervn_ab":   e["n_intervn_ab"],
            "stage2_b_pct":   round(100*e["n_b_retry"] /max(n,1), 2),
            "stage2_iv_pct":  round(100*e["n_iv_retry"]/max(n,1), 2),
            "cost_total_usd": round(e["cost_total"], 4),
        })
    return out


def compute_per_condition_summary(consolidated_path: Path) -> list[dict]:
    """One row per (model_id, condition) with sens/spec metrics."""
    by_key: dict[tuple, dict] = defaultdict(lambda: {
        "n_baseline_yes": 0, "n_baseline_no": 0, "n_baseline_ab": 0,
        "n_intervn_yes": 0,  "n_intervn_no": 0,  "n_intervn_ab": 0,
        "n_rows": 0,
    })
    for line in open(consolidated_path):
        r = json.loads(line)
        key = (r["model_id"], r["condition"])
        e = by_key[key]
        e["n_rows"] += 1
        bp = r.get("baseline_parsed", "")
        ip = r.get("intervention_parsed", "")
        e[f"n_baseline_{'yes' if bp=='yes' else ('no' if bp=='no' else 'ab')}"] += 1
        e[f"n_intervn_{'yes' if ip=='yes' else ('no' if ip=='no' else 'ab')}"] += 1

    out = []
    for (m, c), e in sorted(by_key.items()):
        b_total = e["n_baseline_yes"] + e["n_baseline_no"]
        i_total = e["n_intervn_yes"]  + e["n_intervn_no"]
        out.append({
            "model_id":    m,
            "condition":   c,
            "n_rows":      e["n_rows"],
            "sensitivity": round(e["n_intervn_no"]  / max(i_total, 1), 4),
            "specificity": round(e["n_baseline_yes"] / max(b_total, 1), 4),
            "fp_rate":     round(e["n_baseline_no"]  / max(b_total, 1), 4),
            "n_baseline_yes": e["n_baseline_yes"],
            "n_baseline_no":  e["n_baseline_no"],
            "n_intervn_yes":  e["n_intervn_yes"],
            "n_intervn_no":   e["n_intervn_no"],
        })
    return out


def compute_per_cell_summary(consolidated_path: Path) -> list[dict]:
    """One row per (model_id, cell)."""
    by_key: dict[tuple, dict] = defaultdict(lambda: {
        "n_baseline_yes": 0, "n_baseline_no": 0,
        "n_intervn_yes": 0,  "n_intervn_no": 0,
        "n_rows": 0,
    })
    for line in open(consolidated_path):
        r = json.loads(line)
        key = (r["model_id"], r["cell"])
        e = by_key[key]
        e["n_rows"] += 1
        bp = r.get("baseline_parsed", "")
        ip = r.get("intervention_parsed", "")
        if bp == "yes": e["n_baseline_yes"] += 1
        elif bp == "no": e["n_baseline_no"] += 1
        if ip == "yes": e["n_intervn_yes"] += 1
        elif ip == "no": e["n_intervn_no"] += 1

    out = []
    for (m, c), e in sorted(by_key.items()):
        b_total = e["n_baseline_yes"] + e["n_baseline_no"]
        i_total = e["n_intervn_yes"]  + e["n_intervn_no"]
        out.append({
            "model_id":    m,
            "cell":        c,
            "n_rows":      e["n_rows"],
            "sensitivity": round(e["n_intervn_no"]  / max(i_total, 1), 4),
            "specificity": round(e["n_baseline_yes"] / max(b_total, 1), 4),
            "fp_rate":     round(e["n_baseline_no"]  / max(b_total, 1), 4),
        })
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = find_all_raw_files()
    logger.info(f"Found {len(files)} raw JSONL files across full_* output dirs")

    best = pick_best_per_model(files)
    logger.info(f"Deduped to {len(best)} unique models (best coverage per model)")

    # Manifest
    manifest = {
        "models": {m: {"source": str(p.relative_to(REPO_ROOT)), "n_rows": n}
                   for m, (p, n) in sorted(best.items())},
        "n_models": len(best),
        "parse_rate_floor": PARSE_RATE_FLOOR,
        "semantic_note": (
            "YES = consistent (no violation), NO = not consistent (violation present). "
            "Sensitivity = % NO on adversarial chain. "
            "Specificity = % YES on clean chain. "
            "FP rate = % NO on clean chain."
        ),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Consolidated dataset
    consolidated_path = OUT_DIR / "consolidated.jsonl"
    n_total = write_consolidated(best, consolidated_path)
    logger.info(f"Wrote {n_total:,} rows → {consolidated_path}")

    # Per-model summary
    per_model = compute_per_model_summary(consolidated_path)
    write_csv(per_model, OUT_DIR / "per_model_summary.csv")
    logger.info(f"Wrote per-model summary ({len(per_model)} models)")

    # Per-condition summary
    per_cond = compute_per_condition_summary(consolidated_path)
    write_csv(per_cond, OUT_DIR / "per_condition_summary.csv")
    logger.info(f"Wrote per-condition summary ({len(per_cond)} model×condition combos)")

    # Per-cell summary
    per_cell = compute_per_cell_summary(consolidated_path)
    write_csv(per_cell, OUT_DIR / "per_cell_summary.csv")
    logger.info(f"Wrote per-cell summary ({len(per_cell)} model×cell combos)")

    # Print headline table
    print()
    print("=" * 100)
    print("PER-MODEL HEADLINE — sensitivity (% NO on adversarial), specificity (% YES on clean)")
    print("=" * 100)
    print(f"{'model':<42s} {'rows':>5s} {'parse%':>6s} {'sens%':>6s} {'spec%':>6s} "
          f"{'fp%':>5s} {'§7.3':>5s} {'cost':>7s}")
    print("-" * 100)
    for r in per_model:
        flag = "✓" if r["meets_§7.3"] else "✗"
        print(f"  {r['model_id'][:40]:<40s} {r['n_rows']:5d} "
              f"{100*r['parse_rate']:5.1f}% {100*r['sensitivity']:5.1f}% "
              f"{100*r['specificity']:5.1f}% {100*r['fp_rate']:4.1f}% "
              f"{flag:>5s} ${r['cost_total_usd']:6.2f}")

    # Floor stats
    pass_count = sum(1 for r in per_model if r["meets_§7.3"])
    print()
    print(f"§7.3 PARSE-RATE FLOOR (≥{PARSE_RATE_FLOOR*100:.0f}%): "
          f"{pass_count}/{len(per_model)} models qualify for primary regression")
    print(f"OUTPUT DIR: {OUT_DIR}")


if __name__ == "__main__":
    main()
