#!/usr/bin/env python3
"""
Recompute smoke report from raw results (post-hoc).

The smoke run wrote per-model raw JSONL results before crashing in the
report writer. Rather than re-run the smoke ($3-5), recompute the report
directly from raw_*.jsonl files. Re-parses with both strict and lenient
parsers since both are computable from baseline_raw / intervention_raw.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.harness.runner_native import _parse_response, _parse_response_lenient


def recompute(run_dir: Path) -> dict:
    raw_dir = run_dir / "raw"
    if not raw_dir.exists():
        raise SystemExit(f"raw/ not found in {run_dir}")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_models": 0,
        "models": {},
    }

    for raw_file in sorted(raw_dir.glob("*_results.jsonl")):
        model_id = raw_file.stem.replace("_results", "")
        # Convert mangled filename back to slashed model_id where applicable
        # e.g., "moonshotai_kimi-k2-thinking" -> "moonshotai/kimi-k2-thinking"
        # We'll keep the filename form and let downstream re-key as needed.
        with open(raw_file) as f:
            results = [json.loads(line) for line in f if line.strip()]
        if not results:
            report["models"][model_id] = {"n_results": 0}
            continue

        by_condition: dict = {}
        for r in results:
            cond = r["condition"]
            entry = by_condition.setdefault(cond, {
                "total": 0,
                "b_yes": 0, "b_no": 0, "b_abstain": 0,
                "iv_yes": 0, "iv_no": 0, "iv_abstain": 0,
                "b_lenient_yes": 0, "b_lenient_no": 0, "b_lenient_abstain": 0,
                "iv_lenient_yes": 0, "iv_lenient_no": 0, "iv_lenient_abstain": 0,
                "b_empty": 0, "iv_empty": 0,
                "total_cost_usd": 0.0,
            })
            entry["total"] += 1
            # Re-parse from raw text to get both strict + lenient consistently.
            b_text  = r.get("baseline_raw", "") or ""
            iv_text = r.get("intervention_raw", "") or ""
            if not b_text:
                entry["b_empty"] += 1
            if not iv_text:
                entry["iv_empty"] += 1
            entry[f"b_{_parse_response(b_text)}"]                += 1
            entry[f"iv_{_parse_response(iv_text)}"]              += 1
            entry[f"b_lenient_{_parse_response_lenient(b_text)}"]  += 1
            entry[f"iv_lenient_{_parse_response_lenient(iv_text)}"] += 1
            entry["total_cost_usd"] += (
                r.get("baseline_cost_usd", 0) + r.get("intervention_cost_usd", 0)
            )

        parse_rates = {}
        for cond, e in by_condition.items():
            n = e["total"]
            parse_rates[cond] = {
                "baseline_strict":  round(1 - e["b_abstain"]  / n, 4) if n else 0,
                "intervention_strict":  round(1 - e["iv_abstain"] / n, 4) if n else 0,
                "baseline_lenient":  round(1 - e["b_lenient_abstain"]  / n, 4) if n else 0,
                "intervention_lenient":  round(1 - e["iv_lenient_abstain"] / n, 4) if n else 0,
                "baseline_empty_rate":  round(e["b_empty"]  / n, 4) if n else 0,
                "intervention_empty_rate":  round(e["iv_empty"] / n, 4) if n else 0,
            }
        total_cost = sum(
            r.get("baseline_cost_usd", 0) + r.get("intervention_cost_usd", 0)
            for r in results
        )
        resolved = list({r.get("resolved_provider", "") for r in results
                         if r.get("resolved_provider")})
        report["models"][model_id] = {
            "n_results": len(results),
            "total_cost_usd": round(total_cost, 6),
            "parse_rates_by_condition": parse_rates,
            "resolved_providers": resolved,
        }

    report["n_models"] = len(report["models"])
    return report


def main():
    if len(sys.argv) > 1:
        run_dir = Path(sys.argv[1])
    else:
        # Default: most recent smoke
        smoke_root = ROOT / "results" / "v5_1"
        run_dirs = sorted(smoke_root.glob("smoke_*"), reverse=True)
        if not run_dirs:
            raise SystemExit("No smoke runs found")
        run_dir = run_dirs[0]

    print(f"Recomputing report for: {run_dir.name}")
    report = recompute(run_dir)

    out_dir = run_dir / "smoke_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = out_dir / f"v5_1_smoke_{date_str}_recomputed.json"
    out_path.write_text(json.dumps(report, indent=2))

    # Pretty per-model summary
    print()
    print(f"{'Model':35s}  {'N':>3s}  {'cost':>7s}  {'strict':>10s}  {'lenient':>10s}  {'empty':>7s}")
    print("-" * 90)
    for mid, info in sorted(report["models"].items()):
        if info.get("n_results", 0) == 0:
            print(f"  {mid:33s}  -  (no data)")
            continue
        rates = info["parse_rates_by_condition"]
        # average across conditions
        n_cond = len(rates)
        avg_strict = sum(
            (r["baseline_strict"] + r["intervention_strict"]) / 2
            for r in rates.values()
        ) / n_cond
        avg_lenient = sum(
            (r["baseline_lenient"] + r["intervention_lenient"]) / 2
            for r in rates.values()
        ) / n_cond
        avg_empty = sum(
            (r["baseline_empty_rate"] + r["intervention_empty_rate"]) / 2
            for r in rates.values()
        ) / n_cond
        flag = "✓" if avg_strict >= 0.95 else ("~" if avg_strict >= 0.50 else "✗")
        print(f"{flag} {mid:33s}  {info['n_results']:>3d}  ${info['total_cost_usd']:>6.4f}  "
              f"{avg_strict:>9.0%}  {avg_lenient:>9.0%}  {avg_empty:>6.0%}")

    print()
    total_cost = sum(m.get("total_cost_usd", 0) for m in report["models"].values())
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Report → {out_path}")


if __name__ == "__main__":
    main()
