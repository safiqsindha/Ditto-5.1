"""
Archive all Phase D batch responses locally.

Anthropic retains batch results for 60 days. After that, re-scoring or
re-analysis would be impossible without the raw responses. This script
pulls every batch into RESULTS/phase_d_raw_batches/<batch_id>.jsonl with
one JSON line per request (custom_id, status, raw_text).
"""
from __future__ import annotations

import json
import os
import sys
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

BATCHES = {
    "pubg_clean":          "msgbatch_01VcHLdLj8zwBPjtgrQFw3Lp",
    "pubg_adversarial":    "msgbatch_01QxaNgJFjDjJdargiuZbUpF",
    "nba_clean":           "msgbatch_01NFDJnX98DCEgzd6hzzo6Fp",
    "nba_adversarial":     "msgbatch_01WmT9EJ3fzbcot4yDMx4FG6",
    "poker_clean":         "msgbatch_019CdSVP7bL3vjjiAxoogzoX",
    "poker_adversarial":   "msgbatch_01NSGy2V2ZayjVPGQd2qgLuC",
    "rl_clean":            "msgbatch_01BPB6U1sYABgv41arrnVPLb",
    "rl_adversarial":      "msgbatch_0186kapuWTHCPLkNcZHfmSgF",
    "csgo_clean":          "msgbatch_01QgQdJGqEdSTtFngdDa2REX",
    "csgo_adversarial":    "msgbatch_019Zn6WjnTHU1Ca3FKrL8NCi",
}


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY required", file=sys.stderr)
        sys.exit(1)

    import anthropic
    client = anthropic.Anthropic()

    out_dir = Path("RESULTS/phase_d_raw_batches")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for label, batch_id in BATCHES.items():
        out_path = out_dir / f"{label}__{batch_id}.jsonl"
        if out_path.exists():
            print(f"[skip] {out_path} exists ({out_path.stat().st_size} bytes)")
            with open(out_path) as f:
                summary[label] = sum(1 for _ in f)
            continue
        print(f"[fetch] {batch_id} -> {out_path}")
        n = 0
        n_succ = 0
        with open(out_path, "w") as f:
            for r in client.messages.batches.results(batch_id):
                rec = {
                    "custom_id": r.custom_id,
                    "type": r.result.type,
                }
                if r.result.type == "succeeded":
                    msg = r.result.message
                    text = ""
                    if msg.content:
                        block = msg.content[0]
                        text = getattr(block, "text", "") or ""
                    rec["text"] = text
                    rec["usage"] = {
                        "input_tokens": msg.usage.input_tokens,
                        "output_tokens": msg.usage.output_tokens,
                        "cache_creation_input_tokens": getattr(msg.usage, "cache_creation_input_tokens", 0),
                        "cache_read_input_tokens": getattr(msg.usage, "cache_read_input_tokens", 0),
                    }
                    n_succ += 1
                else:
                    rec["error"] = str(r.result)
                f.write(json.dumps(rec) + "\n")
                n += 1
        print(f"   {n_succ}/{n} succeeded; saved {out_path.stat().st_size} bytes")
        summary[label] = n

    summary_path = out_dir / "MANIFEST.json"
    summary_path.write_text(json.dumps({
        "batches": BATCHES,
        "request_counts": summary,
        "note": (
            "Phase D raw batch responses, archived locally because Anthropic "
            "retains batches for 60 days. Each line in <label>__<batch_id>.jsonl "
            "is one request: {custom_id, type, text, usage}. Re-score with "
            "synthesize_phase_d.py — it falls through to fetch_batch which can "
            "be modified to read these files instead of hitting the API."
        ),
    }, indent=2))
    print(f"\nManifest: {summary_path}")
    print(f"Total: {sum(summary.values())} request records archived")


if __name__ == "__main__":
    main()
