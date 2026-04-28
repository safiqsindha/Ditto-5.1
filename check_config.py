"""
v5 config sanity check CLI.

Reports per-cell setup state — what env vars are set, whether mock fallback
would activate, sample target, time range, and a green/yellow/red light.

Run with: python -m v5.check_config

Designed to be run before any real-data acquisition to catch missing
credentials or misconfigured cells without making API calls.
"""

from __future__ import annotations

import argparse
import os
import sys

from src.common.config import load_cell_configs, load_harness_config

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def status_for_cell(cfg) -> tuple[str, str, str]:
    """
    Return (color, light_emoji, summary).

    Light meanings:
      🟢 green  — credentials satisfied; real-data acquisition ready
      🟡 yellow — credentials missing but mock fallback enabled
      🔴 red    — credentials missing and no mock fallback (would fail)
    """
    if cfg.env_satisfied():
        return (GREEN, "🟢", "credentials satisfied — real-data acquisition ready")
    if cfg.mock_fallback:
        return (YELLOW, "🟡", "credentials missing — mock fallback would activate")
    return (RED, "🔴", "credentials missing and mock fallback disabled")


def report_cell(cfg, color_output: bool = True) -> None:
    color, light, status = status_for_cell(cfg)
    if not color_output:
        color = ""
        reset = ""
    else:
        reset = RESET

    print(f"{color}{BOLD}{light}  [{cfg.cell_id}] {cfg.display_name}{reset}")
    print(f"     status:        {status}")
    print(f"     data source:   {cfg.data_source}")
    print(f"     sample target: {cfg.sample_target:,}")
    print(f"     time range:    {cfg.time_range_start} → {cfg.time_range_end}")
    if cfg.env_vars:
        print("     env vars required:")
        for var in cfg.env_vars:
            val = os.getenv(var, "")
            mark = "✓" if val else "✗"
            preview = (val[:8] + "...") if val and len(val) > 8 else val
            print(f"       {mark} {var}{(' = ' + preview) if val else ''}")
    else:
        print("     env vars required: none (public source)")
    if cfg.stratification:
        print("     stratification:")
        for s in cfg.stratification:
            print(f"       - {s}")
    print()


def report_harness(harness_cfg) -> None:
    print(f"{BOLD}Harness configuration:{RESET}")
    print(f"  alpha:                {harness_cfg.alpha}")
    print(f"  Bonferroni divisor:   {harness_cfg.bonferroni_divisor} (Q2 LOCKED)")
    print(f"  alpha corrected:      {harness_cfg.alpha / harness_cfg.bonferroni_divisor}")
    print(f"  Gate 2 floor:         {harness_cfg.gate2_retention_floor:.0%}")
    print(f"  bootstrap iters:      {harness_cfg.bootstrap_iterations:,}")
    print()


def overall_status(cell_configs: dict) -> tuple[str, str]:
    n_green = sum(1 for c in cell_configs.values() if c.env_satisfied())
    n_total = len(cell_configs)
    n_red = sum(
        1 for c in cell_configs.values()
        if not c.env_satisfied() and not c.mock_fallback
    )
    if n_red:
        return RED, f"❌ {n_red}/{n_total} cell(s) blocked (no credentials, no mock)"
    if n_green == n_total:
        return GREEN, f"✅ all {n_total} cells ready for real-data acquisition"
    n_yellow = n_total - n_green
    return YELLOW, f"⚠ {n_green}/{n_total} cells ready; {n_yellow} would use mock fallback"


def main():
    parser = argparse.ArgumentParser(description="v5 config sanity check")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    parser.add_argument("--cell", help="Report only one cell")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero if any cell would use mock fallback (yellow). "
                             "Useful for CI gating real-data acquisition. (I4 fix)")
    args = parser.parse_args()

    color_output = not args.no_color and sys.stdout.isatty()
    if not color_output:
        # Disable color codes
        global GREEN, YELLOW, RED, BOLD, RESET
        GREEN = YELLOW = RED = BOLD = RESET = ""

    cell_configs = load_cell_configs()
    harness_cfg = load_harness_config()

    print(f"\n{BOLD}=== v5 Configuration Sanity Check ==={RESET}\n")
    report_harness(harness_cfg)

    if args.cell:
        if args.cell not in cell_configs:
            print(f"Unknown cell '{args.cell}'. Known: {list(cell_configs)}")
            sys.exit(2)
        report_cell(cell_configs[args.cell], color_output)
    else:
        for cell_id in sorted(cell_configs):
            report_cell(cell_configs[cell_id], color_output)

    color, summary = overall_status(cell_configs)
    print(f"{BOLD}Overall:{RESET} {color}{summary}{RESET}")
    print()

    # Exit code 0 if all green (or yellow when not strict), 2 if any red
    n_red = sum(
        1 for c in cell_configs.values()
        if not c.env_satisfied() and not c.mock_fallback
    )
    n_yellow = sum(
        1 for c in cell_configs.values()
        if not c.env_satisfied() and c.mock_fallback
    )
    if n_red:
        sys.exit(2)
    if args.strict and n_yellow:
        print(f"--strict: {n_yellow} cell(s) would use mock fallback; exiting nonzero")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
