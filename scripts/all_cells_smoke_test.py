#!/usr/bin/env python3
"""
Generic per-cell smoke test.

Runs the real-data pipeline for each requested cell and reports per-cell:
  - whether dependencies are importable
  - whether env vars are satisfied
  - whether fetch returned anything
  - events/match after extraction
  - chain projection vs the SPEC's 1,200/cell target

Defaults to running the four cells we haven't manually verified yet:
  nba, csgo, rocket_league, poker

PUBG is excluded from defaults because it was already verified separately
via scripts/pubg_pipeline_smoke_test.py. Pass it explicitly to re-test.

Usage:
    python3 scripts/all_cells_smoke_test.py
    python3 scripts/all_cells_smoke_test.py nba poker
    python3 scripts/all_cells_smoke_test.py pubg            # re-test PUBG
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_dotenv_into_environ(env_path: Path) -> int:
    """Load KEY=VALUE pairs from .env into os.environ. Returns count loaded."""
    if not env_path.exists():
        return 0
    n = 0
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if not k or not v:
            continue
        # Don't override values already in environ (shell takes precedence)
        if k not in os.environ:
            os.environ[k] = v
            n += 1
    return n


# Load .env BEFORE importing pipeline classes / loading configs so that
# env_satisfied() and the per-pipeline os.environ.get() lookups see real values.
_n_loaded = _load_dotenv_into_environ(PROJECT_ROOT / ".env")
print(f"Loaded {_n_loaded} env var(s) from {PROJECT_ROOT / '.env'}")

from src.common.config import load_cell_configs  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")

CELL_TO_PIPELINE: dict[str, tuple[str, str]] = {
    "pubg": ("src.cells.pubg.pipeline", "PUBGPipeline"),
    "nba": ("src.cells.nba.pipeline", "NBAPipeline"),
    "csgo": ("src.cells.csgo.pipeline", "CSGOPipeline"),
    "rocket_league": ("src.cells.rocket_league.pipeline", "RocketLeaguePipeline"),
    "poker": ("src.cells.poker.pipeline", "PokerPipeline"),
    "fortnite": ("src.cells.fortnite.pipeline", "FortnitePipeline"),
}

CHAIN_LEN_DEFAULT = {
    "pubg": 8, "nba": 5, "csgo": 10, "rocket_league": 12, "poker": 8, "fortnite": 8,
}

DEFAULT_CELLS = ["nba", "csgo", "rocket_league", "poker"]


def _try_import(module_path: str, class_name: str):
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except ImportError as e:
        return ("import_error", str(e))
    except AttributeError as e:
        return ("attr_error", f"{module_path} has no '{class_name}': {e}")


def smoke_one_cell(cell_name: str) -> dict:
    """Run pipeline for one cell, return summary dict."""
    summary = {"cell": cell_name, "stage_failed": None, "details": ""}

    if cell_name not in CELL_TO_PIPELINE:
        summary["stage_failed"] = "unknown_cell"
        summary["details"] = f"Unknown cell '{cell_name}'"
        return summary

    print(f"\n{'=' * 70}\n  CELL: {cell_name}\n{'=' * 70}")

    # 1. Import pipeline class
    module_path, class_name = CELL_TO_PIPELINE[cell_name]
    pipeline_cls = _try_import(module_path, class_name)
    if isinstance(pipeline_cls, tuple):
        kind, msg = pipeline_cls
        print(f"  ✗ Pipeline import failed ({kind}): {msg}")
        summary["stage_failed"] = "import"
        summary["details"] = msg
        return summary
    print(f"  ✓ Imported {module_path}.{class_name}")

    # 2. Load config
    try:
        cfg = load_cell_configs().get(cell_name)
    except Exception as e:
        print(f"  ✗ Config load failed: {e}")
        summary["stage_failed"] = "config"
        summary["details"] = str(e)
        return summary
    if cfg is None:
        print(f"  ✗ '{cell_name}' missing from config/cells.yaml")
        summary["stage_failed"] = "config_missing"
        return summary
    print(f"  ✓ Config loaded ({cfg.display_name}, sample_target={cfg.sample_target})")

    # 3. Env satisfied?
    env_ok = cfg.env_satisfied()
    print(f"  Env vars required: {cfg.env_vars or '(none)'}; satisfied: {env_ok}")
    if cfg.env_vars and not env_ok:
        print(
            f"  ⏭ Skipping real-data fetch — env vars not set. "
            f"Set {cfg.env_vars} in .env (with real values, not placeholders) "
            "and re-run to test the real-data path."
        )
        summary["stage_failed"] = "env_not_satisfied"
        summary["details"] = f"missing env: {cfg.env_vars}"
        return summary

    # 4. Instantiate pipeline
    try:
        pipeline = pipeline_cls(config=cfg)
    except Exception as e:
        print(f"  ✗ Pipeline instantiation failed: {e}")
        traceback.print_exc()
        summary["stage_failed"] = "instantiate"
        summary["details"] = str(e)
        return summary
    print(f"  ✓ Pipeline instantiated")

    # 5. Fetch (may use real or fall back)
    print(f"  ▶ Running pipeline.fetch() ...")
    try:
        raw_paths = pipeline.fetch()
    except Exception as e:
        print(f"  ✗ fetch() raised: {e}")
        traceback.print_exc()
        summary["stage_failed"] = "fetch"
        summary["details"] = str(e)
        return summary
    print(f"  ✓ Fetch returned {len(raw_paths)} raw file paths")

    if not raw_paths:
        print("  ⚠ Fetch returned no files — likely missing creds or no real-data path.")
        summary["stage_failed"] = "fetch_empty"
        return summary

    # 6. Parse + extract
    try:
        records = pipeline.parse(raw_paths)
        streams = pipeline.extract_events(records)
    except Exception as e:
        print(f"  ✗ parse/extract failed: {e}")
        traceback.print_exc()
        summary["stage_failed"] = "extract"
        summary["details"] = str(e)
        return summary

    print(f"  ✓ Parsed {len(records)} records → {len(streams)} EventStreams")

    nonzero = [s for s in streams if s.events]
    total_events = sum(len(s.events) for s in streams)
    if nonzero:
        avg = total_events / len(nonzero)
    else:
        avg = 0
    print(f"  Streams with at least 1 event: {len(nonzero)}/{len(streams)}")
    print(f"  Total post-filter events: {total_events:,}")
    if nonzero:
        print(
            f"  Events per non-empty: avg={avg:.0f}  "
            f"min={min(len(s.events) for s in nonzero)}  "
            f"max={max(len(s.events) for s in nonzero)}"
        )

    chain_len = CHAIN_LEN_DEFAULT.get(cell_name, 8)
    chains_full = (len(nonzero) * avg) / chain_len if nonzero else 0
    chains_gate2 = chains_full * 0.5
    print(f"\n  Chain projection (chain_length={chain_len}):")
    print(f"    100% retention: {chains_full:.0f} chains")
    print(f"    50% Gate-2 floor: {chains_gate2:.0f} chains")
    print(f"    Target: 1,200")
    if chains_gate2 >= 1200:
        print(f"    ✓ Sufficient")
    elif chains_full >= 1200:
        print(f"    ~ Sufficient only at high retention")
    else:
        scale = (1200 / chains_gate2) if chains_gate2 > 0 else float("inf")
        print(
            f"    ✗ Falls short — would need ~{scale:.1f}× more "
            f"(~{int(len(nonzero) * scale) if chains_gate2 > 0 else '?'} matches)"
        )

    summary["streams"] = len(streams)
    summary["nonzero_streams"] = len(nonzero)
    summary["total_events"] = total_events
    summary["avg_events"] = avg
    summary["chains_gate2"] = chains_gate2
    summary["stage_failed"] = None
    return summary


def main() -> int:
    cells = sys.argv[1:] or DEFAULT_CELLS
    summaries = []
    for c in cells:
        try:
            summaries.append(smoke_one_cell(c))
        except Exception as e:
            print(f"\n✗ Unexpected exception running {c}: {e}")
            traceback.print_exc()
            summaries.append({"cell": c, "stage_failed": "unexpected", "details": str(e)})

    print("\n" + "=" * 70)
    print("  AGGREGATE SUMMARY")
    print("=" * 70)
    print(f"  {'Cell':<18}{'Status':<22}{'Streams':<10}{'Events':<12}{'Chains@G2':<12}")
    print(f"  {'-' * 18}{'-' * 22}{'-' * 10}{'-' * 12}{'-' * 12}")
    for s in summaries:
        status = "✓ ok" if s.get("stage_failed") is None else f"✗ {s['stage_failed']}"
        streams = str(s.get("streams", "-"))
        events = f"{s.get('total_events', '-'):,}" if s.get("total_events") else "-"
        chains = f"{s.get('chains_gate2', 0):.0f}" if s.get("chains_gate2") else "-"
        print(f"  {s['cell']:<18}{status:<22}{streams:<10}{events:<12}{chains:<12}")

    return 0 if all(s.get("stage_failed") is None for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
