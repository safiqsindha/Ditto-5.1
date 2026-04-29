#!/usr/bin/env python3
"""
PUBG cell pipeline smoke test.

Runs the new src/cells/pubg/ pipeline end-to-end against real data,
reports per-match outcomes plus aggregate stats and chain projections.

Usage:
    python3 scripts/pubg_pipeline_smoke_test.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cells.pubg.pipeline import PUBGPipeline  # noqa: E402
from src.common.config import load_cell_configs  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> int:
    cfg = load_cell_configs().get("pubg")
    if cfg is None:
        print("'pubg' missing from config/cells.yaml — re-check yaml.")
        return 1

    pipeline = PUBGPipeline(config=cfg)
    print(f"PUBG smoke test (sample_target={cfg.sample_target}, shard={pipeline.shard})\n")

    raw_paths = pipeline.fetch()
    print(f"\n--- Fetch summary ---")
    print(f"Match files on disk: {len(raw_paths)}")
    if not raw_paths:
        print("No matches downloaded. Likely causes: missing PUBG_API_KEY, network, or quota.")
        return 1

    records = pipeline.parse(raw_paths)
    print(f"Records parsed: {len(records)}")

    streams = pipeline.extract_events(records)
    print(f"\n--- Extraction summary ---")
    print(f"EventStreams produced: {len(streams)}")
    if not streams:
        print("Extraction yielded 0 streams.")
        return 1

    nonzero = [s for s in streams if s.events]
    zeros = len(streams) - len(nonzero)
    total = sum(len(s.events) for s in streams)
    print(f"Streams with at least 1 event: {len(nonzero)}/{len(streams)}")
    if zeros:
        print(f"  ({zeros} streams had 0 events)")
    print(f"Total post-filter events: {total:,}")

    if nonzero:
        avg = total / len(nonzero)
        mn = min(len(s.events) for s in nonzero)
        mx = max(len(s.events) for s in nonzero)
        print(f"Events per non-empty match: avg={avg:.0f}  min={mn}  max={mx}")

    print(f"\n--- First 5 streams (sample) ---")
    for s in streams[:5]:
        types = sorted({e.event_type for e in s.events})
        meta = s.metadata
        print(
            f"  {s.game_id}: {len(s.events)} events  "
            f"({meta.get('game_mode')}, {meta.get('map_name')}, "
            f"{meta.get('duration_s')}s)"
        )
        if types:
            print(f"    types: {types}")

    n_matches = len(nonzero)
    if not n_matches:
        return 1
    avg_evt = total / n_matches
    chain_len = 8
    chains_full = (n_matches * avg_evt) / chain_len
    chains_gate2 = (n_matches * avg_evt * 0.5) / chain_len
    print(f"\n--- Chain projection (chain_length={chain_len}) ---")
    print(f"At 100% retention: {chains_full:.0f} chains")
    print(f"At 50% Gate-2 floor: {chains_gate2:.0f} chains")
    print(f"SPEC target: 1,200 chains/cell")
    if chains_gate2 == 0:
        print("  ✗ No chains projectable")
    elif chains_gate2 >= 1200:
        print("  ✓ Sample is sufficient at the Gate-2 floor")
    elif chains_full >= 1200:
        print("  ~ Sufficient only if real retention >50%")
    else:
        scale = 1200 / chains_gate2
        print(
            f"  ✗ Falls short — would need ~{scale:.1f}× more matches "
            f"(~{int(n_matches * scale)} total to hit 1,200)"
        )

    print(f"\n--- Saving streams to {pipeline.events_dir} ---")
    pipeline._save_streams(streams)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
