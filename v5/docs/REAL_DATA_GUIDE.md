# Real-Data Acquisition Guide

How to switch from mock to real-data acquisition, per cell. **Run only after SPEC sign-off.**

---

## Token / Credential Sources

| Cell | Env Var(s) | Where to request |
|------|------------|------------------|
| Fortnite | `EPIC_ACCOUNT_ID`, `EPIC_ACCESS_TOKEN` | Epic Dev Portal: https://dev.epicgames.com/portal/ — create app, generate OAuth client; per D-15, use a **burner account** (signup at https://www.epicgames.com/id/register) |
| NBA | _(none)_ | NBA Stats API is public — no key. See the `nba_api` docs: https://github.com/swar/nba_api |
| CS:GO / CS2 | `FACEIT_API_KEY` | FACEIT Developer Portal: https://developers.faceit.com → My Apps → Create App |
| Rocket League | `BALLCHASING_TOKEN` | Free signup → token on profile page: https://ballchasing.com/upload (login first); API docs: https://ballchasing.com/doc/api |
| Hearthstone | `HSREPLAY_API_KEY` | Account + API key: https://hsreplay.net/account/api/ ; bulk access docs: https://hsreplay.net/api/v1/ |

Create `.env` in repo root or export in your shell:

```bash
# Fortnite (Epic CDN replays) — request at https://dev.epicgames.com/portal/
export EPIC_ACCOUNT_ID=...
export EPIC_ACCESS_TOKEN=...

# Rocket League — request at https://ballchasing.com/upload (profile)
export BALLCHASING_TOKEN=...

# Hearthstone — request at https://hsreplay.net/account/api/
export HSREPLAY_API_KEY=...

# CS:GO/CS2 — request at https://developers.faceit.com
export FACEIT_API_KEY=...

# NBA: no key required (public API)
```

---

## Per-Cell Setup

### Fortnite

**Source:** Epic CDN replays for FNCS Chapter 5 Season 1 + Cash Cup 2024.

**Token request:**
- Burner account signup: https://www.epicgames.com/id/register
- Epic Developer Portal (OAuth app for `EPIC_ACCOUNT_ID` + `EPIC_ACCESS_TOKEN`): https://dev.epicgames.com/portal/

**Tooling required:**
- Node.js 18+ with `xnocken/replay-downloader` installed globally:
  ```bash
  npm install -g xnocken/replay-downloader
  ```
  Repo: https://github.com/xNocken/replay-downloader
- .NET 8 runtime + `FortniteReplayDecompressor.dll`:
  ```bash
  dotnet tool install --global FortniteReplayDecompressor
  ```
  Repo: https://github.com/SL-x-TnT/FortniteReplayDecompressor

**Credentials caveat:** Epic CDN access requires an authenticated account. **Per Decision D-15 in DECISION_LOG.md, use a burner Epic account + proxy/rate-limiter** to avoid main-account flagging. The user's Node.js async queue snippet (when provided) goes in `v5/src/cells/fortnite/fetch_queue.js`.

**Run:**
```bash
python -c "
from v5.src.common.config import load_cell_configs
from v5.src.cells.fortnite.pipeline import FortnitePipeline
config = load_cell_configs()['fortnite']
streams = FortnitePipeline(config=config).run()
print(f'Fetched {len(streams)} matches')
"
```

---

### NBA

**Source:** NBA Stats API (public, no key). 2023-24 season play-by-play via `PlayByPlayV3` endpoint.

**Token request:** None — public API.
- API package: https://github.com/swar/nba_api
- Endpoint reference: https://github.com/swar/nba_api/tree/master/docs/nba_api/stats/endpoints

**Tooling required:**
- `pip install nba_api>=1.4.0`

**Rate limit:** NBA Stats API has unofficial throttling. Pipeline includes a 0.6s delay between requests. For 300 games, expect ~5–10 minutes.

**Run:**
```bash
python -c "
from v5.src.common.config import load_cell_configs
from v5.src.cells.nba.pipeline import NBAPipeline
config = load_cell_configs()['nba']
streams = NBAPipeline(config=config).run()
print(f'Fetched {len(streams)} games')
"
```

**Note:** Spatial tracking data (Second Spectrum) is not publicly available. Pipeline uses play-by-play only (Decision D-10).

---

### CS:GO/CS2

**Source:** FACEIT API v4 (JSON). Level 5+ skill championships, CS2. No local demo tooling required.

**Token request:**
- FACEIT Developer Portal: https://developers.faceit.com → My Apps → Create App → copy API Key
- Set `FACEIT_API_KEY` in `.env`

**Tooling required:** None beyond `pip install -r v5/requirements.txt`. The pipeline fetches
championship → match IDs → per-match JSON stats via `Authorization: Bearer {FACEIT_API_KEY}`.

**Run:**
```bash
python -c "
from v5.src.common.config import load_cell_configs
from v5.src.cells.csgo.pipeline import CSGOPipeline
config = load_cell_configs()['csgo']
streams = CSGOPipeline(config=config).run()
print(f'Fetched {len(streams)} matches')
"
```

**Note:** Round-level granularity synthesized from per-player kill/assist/entry/flash stats (Decision D-7).

---

### Rocket League

**Source:** BallChasing.com API (free tier with key). RLCS 2024 replays.

**Token request:**
- BallChasing signup: https://ballchasing.com/login (free)
- Generate `BALLCHASING_TOKEN` on your profile page after login: https://ballchasing.com/upload (token shown in the upload section / profile dropdown)
- API documentation: https://ballchasing.com/doc/api
- Parsers:
  - `carball` (primary): https://github.com/SaltieRL/carball
  - `rrrocket` (fallback, Rust): https://github.com/nickbabcock/rrrocket

**Tooling required:**
- `pip install carball` (primary parser) — falls back to `rrrocket` subprocess

**Rate limit:** Free tier ≈ 2 requests/second. Pipeline waits 1.0s between requests.

**Run:**
```bash
python -c "
from v5.src.common.config import load_cell_configs
from v5.src.cells.rocket_league.pipeline import RocketLeaguePipeline
config = load_cell_configs()['rocket_league']
streams = RocketLeaguePipeline(config=config).run()
print(f'Fetched {len(streams)} replays')
"
```

**Note:** Hit-level event granularity by default (Decision D-RL2). [REQUIRES SIGN-OFF Q7] before changing.

---

### Hearthstone

**Source:** HSReplay.net API. 2024 Legend-rank ladder replays.

**Token request:**
- HSReplay account signup: https://hsreplay.net/account/signup/
- API key request page: https://hsreplay.net/account/api/
- API v1 docs: https://hsreplay.net/api/v1/
- HearthSim hslog parser: https://github.com/HearthSim/python-hearthstone

**Tooling required:**
- `pip install hearthstone>=2.2.0` (HearthSim hslog parser)
- HSReplay API key: bulk replay access requires authenticated key

**Fallback:** If API key absent, the `hslog` parser can also process locally collected `Power.log` files exported from the Hearthstone client. (Power.log is automatically generated when running Hearthstone in logging mode — see HearthSim/python-hearthstone README.)

**Run:**
```bash
python -c "
from v5.src.common.config import load_cell_configs
from v5.src.cells.hearthstone.pipeline import HearthstonePipeline
config = load_cell_configs()['hearthstone']
streams = HearthstonePipeline(config=config).run()
print(f'Fetched {len(streams)} games')
"
```

**Note:** Per-action granularity by default (Decision D-9). [REQUIRES SIGN-OFF Q8] before changing.

---

## Sanity Checks After Real-Data Acquisition

```bash
# Verify event stream sizing per cell matches expectations
for cell in fortnite nba csgo rocket_league hearthstone; do
  count=$(find v5/data/events/$cell -name '*.jsonl' | wc -l)
  echo "$cell: $count streams"
done

# Run pilot validator against real data
python -m v5.run_pilot --output v5/RESULTS/pilot_report_real.json

# Render to markdown for human review
python -m v5.src.pilot.render_report v5/RESULTS/pilot_report_real.json
```

If any cell's Gate 2 retention falls below 50%:
1. Check the event-type distribution (look for non-actionable types dominating)
2. Consider whether `CELL_ACTIONABLE_OVERRIDES` in `actionables.py` needs cell-specific additions (requires sign-off — see SPEC §8.1)
3. Do **not** lower the Gate 2 floor without pre-registration override

---

## Expected Costs

Real data acquisition is API/CDN-bound — minimal API spend:
- NBA: free (rate-limited public endpoints)
- CS:GO: free (FACEIT API v4; JSON responses, no demo download)
- Rocket League: free tier (BallChasing free key)
- Hearthstone: free tier or API key (depending on volume)
- Fortnite: free (Epic account; account-flagging risk per D-15)

**No per-call API spend until Haiku evaluation runs** (which require SPEC sign-off).

---

## Storage Estimates

| Cell | Raw | Processed | Events |
|------|-----|-----------|--------|
| Fortnite | ~10 GB (200 .replay) | ~1 GB (JSON) | ~100 MB (JSONL) |
| NBA | ~50 MB (300 JSON) | same | ~50 MB |
| CS:GO | ~50 MB (150 JSON via FACEIT API) | same | ~50 MB |
| Rocket League | ~500 MB (250 .replay) | ~250 MB | ~80 MB |
| Hearthstone | ~100 MB (300 XML) | same | ~80 MB |

**Total:** ~11 GB raw, ~1.8 GB processed, ~400 MB events.

All data directories are gitignored.
