# v5 Build Plan

**Written:** 2026-04-27  
**Author:** Claude (autonomous build session)  
**Scope:** v5 infrastructure — five-cell parallel detection experiment  
**Subject model:** Claude Haiku  
**Domains:** Fortnite, NBA, CS:GO/CS2, Rocket League, Hearthstone  
**Reference experiments:** v3 (Chess/Checkers), v4 (single-cell methodology characterization)

---

## Overview

v5 is a five-cell parallel replication of v3's constraint-chain detection methodology across five new game domains. Today's session builds the infrastructure layer: scaffolding, data acquisition pipelines, event extraction, statistical harness, and pilot validation tooling. Translation functions (T) and chain construction are explicitly **out of scope** and left as stubs pending both-author review.

---

## Dependency Graph

```
Task 0: Repo scaffolding
    └── Task 1: Common schema & config
            ├── Task 2: ACTIONABLE_TYPES / harness modules
            │       └── Task 3: Statistical harness (McNemar, scoring, variance)
            │               └── Task 7: Pilot validation harness
            ├── Task 4: Interface stubs (T, chain builder)
            │       └── Task 7: Pilot validation harness
            └── Task 5a-5e: Per-cell pipelines (parallel, independent)
                    └── Task 6: Event extraction (depends on schema)
                            └── Task 7: Pilot validation harness

Task 8: SPEC.md (can begin after schema is defined, no code deps)
Task 9: DECISION_LOG.md (populated throughout)
Task 10: STATUS.md (written last)
```

---

## Tasks

### Task 0: Repo Scaffolding
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 20 min  
**Dependencies:** None  
**Description:**  
Create the full v5/ directory tree matching v3/v4 pattern. Subdirectories: `src/`, `src/cells/` (one per domain), `src/harness/`, `src/interfaces/`, `src/pilot/`, `src/common/`, `data/raw/`, `data/processed/`, `data/events/` (one per domain each), `RESULTS/`, `notebooks/`, `tests/`, `config/`.

Create `requirements.txt`, `config/cells.yaml`, `config/harness.yaml`, `__init__.py` files throughout.

**Deliverable:** Complete directory tree with all stub files.

---

### Task 1: Common Schema & Config
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 30 min  
**Dependencies:** Task 0  
**Description:**  
Define the normalized `GameEvent` dataclass used by all five cells:
```
timestamp: float          # seconds from game start
event_type: str           # normalized type string (phase_-stripped)
actor: str                # player/team identifier  
location_context: dict    # domain-specific spatial/context data
raw_data_blob: dict       # full raw event record
cell: str                 # domain identifier
game_id: str              # match/game unique identifier
sequence_idx: int         # ordinal position within game
```

Config dataclasses for cell parameters, harness parameters, and pipeline parameters.

**Deliverable:** `src/common/schema.py`, `src/common/config.py`

---

### Task 2: ACTIONABLE_TYPES & Harness Modules
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 30 min  
**Dependencies:** Task 1  
**Description:**  
Port v4's ACTIONABLE_TYPES whitelist with v1.1 amendments:
- Include ResourceBudget in actionables
- Apply phase_ prefix strip when comparing event types
- Gate 2 floor: 50% retention rate

Create `src/harness/actionables.py` implementing the gate logic.

**Deliverable:** `src/harness/actionables.py` with ACTIONABLE_TYPES dict, Gate 2 check.

---

### Task 3: Statistical Evaluation Harness
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 60 min  
**Dependencies:** Task 2  
**Description:**  
Adapt v4's McNemar pipeline to support five parallel cells:
- `src/harness/mcnemar.py`: McNemar's test implementation for paired binary outcomes
- `src/harness/scoring.py`: Chain-level scoring logic (correct/incorrect per chain)
- `src/harness/variance.py`: Bootstrap confidence interval computation
- `src/harness/cell_runner.py`: Orchestrates per-cell runs, collects results, produces cell-level + aggregate reports

Cell runner must accept a T implementation via dependency injection so cells are pluggable.

**NOTE:** Bonferroni divisor (5 vs 1 for multi-cell) is NOT pre-registered. Cell runner will expose this as a configurable parameter. The default (divisor=5) will be logged in DECISION_LOG but flagged as REQUIRES SIGN-OFF.

**Deliverable:** `src/harness/mcnemar.py`, `scoring.py`, `variance.py`, `cell_runner.py`

---

### Task 4: Interface Stubs (T, Chain Builder)
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 20 min  
**Dependencies:** Task 1  
**Description:**  
Define abstract base classes / protocols for:
- `TranslationFunction` (T): accepts event stream, returns chain candidates
- `ChainBuilder`: accepts T output + parameters, constructs chains

Both must raise `NotImplementedError` in their base implementations. Domain-specific T implementations are explicitly **out of scope**.

**Deliverable:** `src/interfaces/translation.py`, `src/interfaces/chain_builder.py`

---

### Task 5a: Fortnite Data Acquisition Pipeline
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 45 min  
**Dependencies:** Task 1  
**Description:**  
Build pipeline using:
- xNocken/replay-downloader (Node.js CLI) for downloading FNCS/Cash Cup tournament replays from Epic CDN
- SL-x-TnT/FortniteReplayDecompressor (C#/.NET) for binary replay parsing

Pipeline architecture: download → decompress → normalize to `GameEvent` stream.

**Access constraint:** Requires Epic account credentials (environment variable `EPIC_ACCOUNT_ID`, `EPIC_ACCESS_TOKEN`). Pipeline degrades gracefully to mock data when credentials absent. This will be logged as Decision D-F1.

**Sample target:** 200 tournament matches from FNCS Chapter 5 Season 1 (2024) or most recent available. Rationale and stratification in DECISION_LOG.

**Deliverable:** `src/cells/fortnite/pipeline.py`, `src/cells/fortnite/extractor.py`

---

### Task 5b: NBA Data Acquisition Pipeline
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 45 min  
**Dependencies:** Task 1  
**Description:**  
Build pipeline using:
- `nba_api` Python package for NBA Stats API (publicly accessible, no key required for play-by-play)
- Play-by-play endpoint: `PlayByPlayV3`
- Tracking data (Second Spectrum): NOT publicly accessible — log as Decision D-N1

**Sample target:** 300 games stratified across 2023-24 regular season (240) + 2024 playoffs (60). Rationale in DECISION_LOG.

**Deliverable:** `src/cells/nba/pipeline.py`, `src/cells/nba/extractor.py`

---

### Task 5c: CS:GO/CS2 Data Acquisition Pipeline
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 60 min  
**Dependencies:** Task 1  
**Description:**  
Build pipeline using:
- HLTV demo archive (publicly accessible tournament demos)
- `awpy` Python package (wraps markus-wa/demoinfocs-golang) for .dem file parsing

**Sample target:** 150 S-tier tournament maps from 2024 (IEM Katowice, BLAST, ESL One). Rationale in DECISION_LOG.

**Note on CS2 migration:** HLTV serves CS2 demos since August 2023. Pipeline must handle both CS:GO and CS2 demo formats; awpy handles this transparently. Log as Decision D-C1.

**Deliverable:** `src/cells/csgo/pipeline.py`, `src/cells/csgo/extractor.py`

---

### Task 5d: Rocket League Data Acquisition Pipeline
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 45 min  
**Dependencies:** Task 1  
**Description:**  
Build pipeline using:
- BallChasing.com public API (free, key required via env var `BALLCHASING_TOKEN`)
- `carball` Python package or direct rrrocket subprocess for .replay parsing

**Sample target:** 250 RLCS replays from 2024 season. BallChasing has a `/replays` endpoint with `playlist=rlcs` filter.

**Access constraint:** BallChasing API token required (free registration). Graceful mock fallback when absent.

**Deliverable:** `src/cells/rocket_league/pipeline.py`, `src/cells/rocket_league/extractor.py`

---

### Task 5e: Hearthstone Data Acquisition Pipeline
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 45 min  
**Dependencies:** Task 1  
**Description:**  
Build pipeline using:
- HSReplay.net API or HearthSim open-source replay files
- `hearthstone` Python package (HearthSim) for gamestate protocol log parsing

**Sample target:** 300 high-rank (Legend) ladder games from 2024. HSReplay has public replay sharing; bulk access via API requires account.

**Access constraint:** HSReplay API key required (env var `HSREPLAY_API_KEY`). The HearthSim `hslog` package can parse locally available log files without API access. Mock fallback when credentials absent.

**Deliverable:** `src/cells/hearthstone/pipeline.py`, `src/cells/hearthstone/extractor.py`

---

### Task 6: Normalized Event Extraction
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 60 min (across all cells, built into Task 5a-5e)  
**Dependencies:** Task 1, Tasks 5a-5e  
**Description:**  
Each cell's extractor.py must produce a normalized `GameEvent` stream matching the schema from Task 1. Target: upstream data sized for ~1,200 chains per cell after expected filtering loss (assume 40-60% retention through Gate 2 → need ~2,000-2,400 raw chain candidates → need ~6,000-12,000 raw events per cell).

Schema decisions (event_type normalization, actor representation) logged in DECISION_LOG.

**Deliverable:** Extractors in each `src/cells/*/extractor.py`

---

### Task 7: Pilot Validation Harness
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 45 min  
**Dependencies:** Tasks 3, 4  
**Description:**  
Build `src/pilot/validator.py`:
- Accepts 50-100 sample chains per cell (or event streams when T is stubbed)
- Computes: retention rate vs Gate 2 floor (50%), sample chain inspection output, basic distributional sanity checks
- T is a pluggable interface — validator accepts any T implementation; default is MockT

Build `src/pilot/mock_t.py`:
- `MockT` class implementing `TranslationFunction` interface
- Returns plausible-looking chain candidates from event streams using deterministic heuristics (not real constraint-chain detection)
- Enables full pilot harness testing without a real T

**Deliverable:** `src/pilot/validator.py`, `src/pilot/mock_t.py`

---

### Task 8: SPEC.md
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 60 min  
**Dependencies:** Tasks 1-7 (informed by implementation)  
**Description:**  
Draft v5/SPEC.md with:
- Domain list and justification
- Statistical methodology (McNemar, effect size, power)
- Harness description
- References to v3/v4 methodology
- ACTIONABLE_TYPES whitelist (v1.1)
- All pre-registration decisions marked as [REQUIRES SIGN-OFF] with candidate options

**Sign-off questions to surface (minimum):**
1. Bonferroni divisor: 5 (one per cell) vs 1 (single aggregate test)?
2. Chain length consistency: fixed-length vs variable?
3. Granularity for CS:GO: round-level vs tick-level events?
4. Granularity for NBA: possession-level vs play-level?
5. Continuous-state handling for Rocket League (physics engine state)?
6. Hearthstone: per-turn vs per-action granularity?

**Deliverable:** `v5/SPEC.md`

---

### Task 9: DECISION_LOG.md (ongoing)
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 30 min (distributed throughout session)  
**Dependencies:** None (populated throughout)  
**Description:** Log every non-pre-specified decision throughout the build. Format per instructions. Log reference-repo-inaccessibility as Decision D-0.

**Deliverable:** `v5/DECISION_LOG.md`

---

### Task 10: STATUS.md
**Status:** IN SCOPE — TODAY  
**Effort estimate:** 20 min  
**Dependencies:** All other tasks  
**Description:** End-of-day summary: what's done, what's pending, decisions needing review, SPEC sign-off blockers.

**Deliverable:** `v5/STATUS.md`

---

## Out of Scope (Do Not Build)

| Item | Reason |
|------|--------|
| Translation functions T (per domain) | Requires both-author review; load-bearing methodological work |
| Chain construction from events | Depends on unregistered chain-length/granularity decisions |
| SPEC pre-registration decisions (Bonferroni, chain length, granularity, continuous-state) | Must be locked by both authors |
| Real Haiku API evaluation runs | No evaluation until SPEC is signed off |
| Fortnite Mew transfer-training preparation | Separate scope from detection methodology cell |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Fortnite Epic CDN access requires credentials | High | Medium | Mock fallback built into pipeline |
| HSReplay bulk API requires paid account | Medium | Medium | HearthSim hslog parsing as fallback; mock fallback |
| awpy requires Go toolchain | Medium | Low | pip install handles; document in requirements |
| NBA tracking data not publicly available | Certain | Low | Play-by-play is sufficient; tracking noted as enhancement |
| BallChasing API rate limits | Low | Low | Respectful pagination; exponential backoff |
| rrrocket requires Rust binary | Medium | Low | carball Python package as alternative |
| V3/V4 reference repos inaccessible from this session | Certain | Medium | Build from prompt description; log as D-0 |

---

## Execution Order for Today

1. BUILD_PLAN.md ← *you are here*
2. DECISION_LOG.md initial entry (D-0: reference repos inaccessible)
3. Task 0: Repo scaffolding
4. Task 1: Common schema & config
5. Task 2: ACTIONABLE_TYPES
6. Task 3: Statistical harness
7. Task 4: Interface stubs
8. Tasks 5a-5e: All five pipelines (can be implemented sequentially; logically parallel)
9. Task 7: Pilot harness + mock T
10. Task 8: SPEC.md
11. Task 10: STATUS.md (final)

---

## Success Criteria

**Minimum (must hit):**
- [x] All five `src/cells/*/pipeline.py` files exist and runnable with mock data
- [x] All five `src/cells/*/extractor.py` produce `GameEvent` streams
- [x] `src/harness/` fully functional (McNemar, scoring, variance, cell runner)
- [x] `src/interfaces/` stubs raise `NotImplementedError` cleanly (verified by test_stub_t_raises_noted_not_crashes)
- [x] `src/pilot/validator.py` runs against mock T output (5/5 cells PASS)
- [x] `v5/SPEC.md` with all [REQUIRES SIGN-OFF] sections marked (Q1–Q8)
- [x] `v5/DECISION_LOG.md` populated throughout (D-0 through D-13)
- [x] `v5/STATUS.md` written at session end

**Aspirational (if time permits):**
- [x] Basic distributional sanity checks per cell using mock data (chain length, actionable fraction, event-type distribution per cell in PilotValidator)
- [x] Tests in `tests/` for harness and pilot modules (29/29 PASS)
- [x] Event stream sizing estimates per cell based on mock runs (1,200 streams persisted: 200/300/150/250/300)
