"""
Render a pilot JSON report to human-readable markdown.

Usage:
    python -m v5.src.pilot.render_report v5/RESULTS/pilot_report.json
    python -m v5.src.pilot.render_report v5/RESULTS/pilot_report.json --output v5/RESULTS/pilot_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def render(report: dict) -> str:
    cells = report.get("cells", [])
    all_passed = report.get("all_passed", False)

    lines = []
    lines.append("# v5 Pilot Validation Report\n")
    lines.append(f"**Aggregate result:** {'ALL PASS' if all_passed else 'SOME FAILURES'}\n")

    lines.append("\n## Per-Cell Summary\n")
    lines.append("| Cell | Streams | Events | Raw Chains | Post-Gate2 | Retention | Result |")
    lines.append("|------|---------|--------|------------|------------|-----------|--------|")
    for c in cells:
        status = "PASS" if c.get("passed") else "FAIL"
        retention = f"{c.get('retention_rate', 0):.1%}"
        lines.append(
            f"| {c['cell']} "
            f"| {c.get('n_streams', 0)} "
            f"| {c.get('n_events_total', 0):,} "
            f"| {c.get('n_chains_raw', 0):,} "
            f"| {c.get('n_chains_post_gate2', 0):,} "
            f"| {retention} "
            f"| {status} |"
        )

    for c in cells:
        lines.append(f"\n---\n\n## [{c['cell']}]\n")
        lines.append(f"- **Streams:** {c.get('n_streams', 0)}")
        lines.append(f"- **Total events:** {c.get('n_events_total', 0):,}")
        lines.append(f"- **Raw chains:** {c.get('n_chains_raw', 0):,}")
        lines.append(f"- **Post-Gate2 chains:** {c.get('n_chains_post_gate2', 0):,}")
        lines.append(
            f"- **Retention:** {c.get('retention_rate', 0):.1%} "
            f"(floor={c.get('gate2_floor', 0):.0%})"
        )
        lines.append(f"- **Gate 2:** {'PASS' if c.get('gate2_pass') else 'FAIL'}")

        lines.append("\n### Chain length distribution")
        lines.append(f"- mean: {c.get('chain_length_mean', 0):.2f}")
        lines.append(f"- min: {c.get('chain_length_min', 0)}")
        lines.append(f"- max: {c.get('chain_length_max', 0)}")
        lines.append(f"- median: {c.get('chain_length_median', 0):.1f}")

        lines.append("\n### Actionable fraction distribution")
        lines.append(f"- mean: {c.get('actionable_frac_mean', 0):.1%}")
        lines.append(f"- min: {c.get('actionable_frac_min', 0):.1%}")
        lines.append(f"- max: {c.get('actionable_frac_max', 0):.1%}")

        etypes = c.get("event_type_distribution", {})
        if etypes:
            lines.append("\n### Top 10 event types")
            lines.append("| Type | Count |")
            lines.append("|------|-------|")
            for etype, count in sorted(etypes.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"| {etype} | {count:,} |")

        warnings = c.get("warnings", [])
        if warnings:
            lines.append("\n### Warnings")
            for w in warnings:
                lines.append(f"- ⚠ {w}")

        errors = c.get("errors", [])
        if errors:
            lines.append("\n### Errors")
            for e in errors:
                lines.append(f"- ❌ {e}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Render pilot JSON report to markdown")
    parser.add_argument("input", type=Path, help="Path to pilot_report*.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output markdown path (default: input with .md extension)")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        report = json.load(f)

    md = render(report)
    out = args.output or args.input.with_suffix(".md")
    with open(out, "w") as f:
        f.write(md)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
