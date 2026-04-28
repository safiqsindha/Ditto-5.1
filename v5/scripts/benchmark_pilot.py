"""
v5 performance benchmark — measure timing + memory of the pilot pipeline.

Runs each cell's mock-data pipeline + pilot validation, captures:
  - wall-clock time per cell
  - peak memory per cell (resource.getrusage)
  - chains/second throughput

Run with: python -m v5.scripts.benchmark_pilot
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import resource
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

ALL_CELLS = ["fortnite", "nba", "csgo", "rocket_league", "hearthstone"]


def _memory_mb() -> float:
    """Return current process RSS in MB."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # On Linux ru_maxrss is KB; on macOS it's bytes
    if sys.platform == "darwin":
        return rss / 1024 / 1024
    return rss / 1024


def benchmark_cell(cell: str) -> dict:
    """Benchmark one cell's pipeline + pilot end-to-end."""
    from v5.src.common.config import load_cell_configs
    from v5.src.pilot.mock_t import MockT
    from v5.src.pilot.validator import PilotValidator

    pipeline_map = {}
    if cell == "fortnite":
        from v5.src.cells.fortnite.pipeline import FortnitePipeline
        pipeline_map[cell] = FortnitePipeline
    elif cell == "nba":
        from v5.src.cells.nba.pipeline import NBAPipeline
        pipeline_map[cell] = NBAPipeline
    elif cell == "csgo":
        from v5.src.cells.csgo.pipeline import CSGOPipeline
        pipeline_map[cell] = CSGOPipeline
    elif cell == "rocket_league":
        from v5.src.cells.rocket_league.pipeline import RocketLeaguePipeline
        pipeline_map[cell] = RocketLeaguePipeline
    elif cell == "hearthstone":
        from v5.src.cells.hearthstone.pipeline import HearthstonePipeline
        pipeline_map[cell] = HearthstonePipeline

    configs = load_cell_configs()
    config = configs[cell]
    pipeline = pipeline_map[cell](config=config)

    gc.collect()
    mem_before = _memory_mb()

    # Phase 1: generate mock data
    t0 = time.perf_counter()
    streams = pipeline.generate_mock_data()
    t1 = time.perf_counter()
    n_streams = len(streams)
    n_events = sum(len(s) for s in streams)

    # Phase 2: pilot validation (T translate + Gate 2 + reporting)
    validator = PilotValidator(gate2_floor=0.50)
    validator.register_cell(cell, MockT(cell=cell))
    t2 = time.perf_counter()
    report = validator.run({cell: streams})
    t3 = time.perf_counter()

    cell_report = next(r for r in report.cells if r.cell == cell)
    mem_after = _memory_mb()

    return {
        "cell": cell,
        "n_streams": n_streams,
        "n_events": n_events,
        "n_chains_post_gate2": cell_report.n_chains_post_gate2,
        "mock_gen_seconds": round(t1 - t0, 3),
        "pilot_seconds": round(t3 - t2, 3),
        "total_seconds": round(t3 - t0, 3),
        "events_per_second": round(n_events / (t1 - t0), 0) if t1 - t0 > 0 else 0,
        "chains_per_second": round(cell_report.n_chains_post_gate2 / max(t3 - t2, 0.001), 0),
        "mem_before_mb": round(mem_before, 1),
        "mem_after_mb": round(mem_after, 1),
        "mem_delta_mb": round(mem_after - mem_before, 1),
    }


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# v5 Pilot Performance Benchmark",
        "",
        f"Run on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "| Cell | Streams | Events | Chains | Mock-gen (s) | Pilot (s) | Total (s) | Events/s | Chains/s | Mem Δ (MB) |",
        "|------|---------|--------|--------|--------------|-----------|-----------|----------|----------|------------|",
    ]
    total_s = 0.0
    total_events = 0
    total_chains = 0
    for r in results:
        lines.append(
            f"| {r['cell']} | {r['n_streams']} | {r['n_events']:,} "
            f"| {r['n_chains_post_gate2']:,} | {r['mock_gen_seconds']} "
            f"| {r['pilot_seconds']} | {r['total_seconds']} "
            f"| {r['events_per_second']:,.0f} | {r['chains_per_second']:,.0f} "
            f"| {r['mem_delta_mb']} |"
        )
        total_s += r["total_seconds"]
        total_events += r["n_events"]
        total_chains += r["n_chains_post_gate2"]

    lines.extend([
        "",
        f"**Total wall-clock:** {total_s:.2f}s for {total_events:,} events "
        f"and {total_chains:,} chains across all 5 cells.",
        "",
        "## Implications for real-data acquisition",
        "",
        "These numbers are mock-data only. Real-data acquisition will be bound by:",
        "- Network I/O (HLTV demo downloads, NBA API, BallChasing API, HSReplay API)",
        "- Parser CPU (awpy, carball, hslog)",
        "- Disk I/O for raw replays (~18GB total estimated, see REAL_DATA_GUIDE.md)",
        "",
        "Pilot validation itself (T + Gate 2) is fast — well under 1 minute even at",
        "production scale (1,200 chains/cell × 5 cells = 6,000 chains).",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="v5 pilot performance benchmark")
    parser.add_argument("--cells", nargs="+", default=ALL_CELLS, choices=ALL_CELLS)
    parser.add_argument("--output", type=Path, default=Path("v5/RESULTS/benchmark.md"))
    parser.add_argument("--json", type=Path, default=Path("v5/RESULTS/benchmark.json"))
    args = parser.parse_args()

    print(f"Benchmarking {len(args.cells)} cell(s)...\n")
    results = []
    for cell in args.cells:
        print(f"  [{cell}] running...")
        r = benchmark_cell(cell)
        results.append(r)
        print(f"  [{cell}] {r['n_streams']} streams, "
              f"{r['n_events']:,} events, "
              f"{r['n_chains_post_gate2']:,} chains "
              f"in {r['total_seconds']}s "
              f"({r['events_per_second']:,.0f} ev/s)")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"results": results, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"\nJSON: {args.json}")

    md = render_markdown(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(md)
    print(f"Markdown: {args.output}")


if __name__ == "__main__":
    main()
