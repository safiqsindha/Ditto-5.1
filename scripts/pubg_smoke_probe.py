#!/usr/bin/env python3
"""
PUBG API smoke probe.

Verifies the PUBG developer API works end-to-end for our pipeline pattern:
  1. Auth with PUBG_API_KEY from .env
  2. Fetch sample matches from /shards/<shard>/samples
  3. Pull one match's full record from /shards/<shard>/matches/<id>
  4. Walk match -> included[] -> asset URL for telemetry
  5. Download gzipped telemetry from CDN
  6. Parse + event-type histogram

If this is green, we're cleared to swap Fortnite -> PUBG in the codebase.

Usage:
    python3 scripts/pubg_smoke_probe.py [<shard>]   # default shard: steam
"""
from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
BASE_URL = "https://api.pubg.com"
DEFAULT_SHARD = "steam"


def load_env(path: Path) -> dict:
    env: dict = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def main() -> int:
    shard = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SHARD

    env = load_env(ENV_PATH)
    api_key = env.get("PUBG_API_KEY", "").strip()
    if not api_key:
        print(f"PUBG_API_KEY not found in {ENV_PATH}", file=sys.stderr)
        return 1
    print(f"Loaded PUBG_API_KEY ({len(api_key)} chars)")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/vnd.api+json",
    }

    # ─── Step 1: sample matches ────────────────────────────────────────────
    print(f"\n--- Step 1: GET /shards/{shard}/samples ---")
    r = requests.get(f"{BASE_URL}/shards/{shard}/samples", headers=headers, timeout=20)
    print(f"  Status: {r.status_code}")
    if r.status_code != 200:
        print(f"  Body preview: {r.text[:400]}")
        return 1

    sample_data = r.json()
    matches_rel = (
        sample_data.get("data", {})
        .get("relationships", {})
        .get("matches", {})
        .get("data", [])
    )
    sample_ids = [m["id"] for m in matches_rel if m.get("type") == "match"]
    print(f"  Sample matches available: {len(sample_ids)}")
    if not sample_ids:
        print(f"  No sample matches — full body:\n{json.dumps(sample_data, indent=2)[:1500]}")
        return 1
    print(f"  First few: {sample_ids[:3]}")

    # ─── Step 2: fetch one match ───────────────────────────────────────────
    match_id = sample_ids[0]
    print(f"\n--- Step 2: GET /shards/{shard}/matches/{match_id} ---")
    r = requests.get(
        f"{BASE_URL}/shards/{shard}/matches/{match_id}", headers=headers, timeout=20
    )
    print(f"  Status: {r.status_code}")
    if r.status_code != 200:
        print(f"  Body preview: {r.text[:400]}")
        return 1

    match = r.json()
    attrs = match.get("data", {}).get("attributes", {})
    print(f"  Game mode: {attrs.get('gameMode')}")
    print(f"  Map: {attrs.get('mapName')}")
    print(f"  Duration: {attrs.get('duration')}s")
    print(f"  Match type: {attrs.get('matchType')}")
    print(f"  Created: {attrs.get('createdAt')}")
    print(f"  Telemetry available?: {attrs.get('isCustomMatch') is not None}")

    # ─── Step 3: telemetry asset URL ───────────────────────────────────────
    included = match.get("included", [])
    asset_url = None
    for item in included:
        if (
            item.get("type") == "asset"
            and item.get("attributes", {}).get("name", "").lower() == "telemetry"
        ):
            asset_url = item["attributes"].get("URL")
            break
    if not asset_url:
        # Fallback: any asset URL
        for item in included:
            if item.get("type") == "asset":
                asset_url = item.get("attributes", {}).get("URL")
                if asset_url:
                    break

    if not asset_url:
        print("  No telemetry asset found in included[]")
        types_in_included = Counter(i.get("type") for i in included)
        print(f"  included[] types: {dict(types_in_included)}")
        return 1
    print(f"  Asset URL: {asset_url[:120]}...")

    # ─── Step 4: download + decompress ─────────────────────────────────────
    print(f"\n--- Step 3: download telemetry from CDN ---")
    r = requests.get(asset_url, timeout=60)
    print(f"  Status: {r.status_code}, body: {len(r.content):,} bytes")
    if r.status_code != 200:
        print(f"  Body preview: {r.text[:400]}")
        return 1

    raw = r.content
    try:
        decompressed = gzip.decompress(raw)
        print(f"  Decompressed: {len(decompressed):,} bytes")
    except (gzip.BadGzipFile, OSError):
        decompressed = raw
        print(f"  (Body was not gzipped — using raw)")

    # ─── Step 5: parse + analyze ───────────────────────────────────────────
    print(f"\n--- Step 4: parse + event histogram ---")
    events = json.loads(decompressed)
    if not isinstance(events, list):
        print(f"  Unexpected telemetry shape: {type(events).__name__}")
        return 1
    print(f"  Total events: {len(events):,}")

    type_counter = Counter(e.get("_T") for e in events)
    print(f"  Distinct event types: {len(type_counter)}")
    print(f"\n  Top 25 event types:")
    for evt_type, count in type_counter.most_common(25):
        print(f"    {count:>7,}  {evt_type}")

    interesting = [
        "LogPlayerKillV2",
        "LogPlayerKill",
        "LogPlayerPosition",
        "LogGameStatePeriodic",
        "LogPlayerTakeDamage",
        "LogVehicleRide",
    ]
    print("\n  Sample events of interest:")
    for et in interesting:
        sample = next((e for e in events if e.get("_T") == et), None)
        if sample:
            print(f"\n  --- {et} ---")
            print(json.dumps(sample, indent=2, default=str)[:500])

    print("\n=== Summary ===")
    print(f"  ✓ Auth")
    print(f"  ✓ Sample-match discovery ({len(sample_ids)} ids)")
    print(f"  ✓ Match metadata ({attrs.get('gameMode')}, {attrs.get('duration')}s)")
    print(f"  ✓ Telemetry download ({len(decompressed):,} bytes)")
    print(f"  ✓ Parsed {len(events):,} events across {len(type_counter)} types")
    print()
    print("PUBG API works end-to-end. Cleared to proceed with Fortnite -> PUBG swap.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
