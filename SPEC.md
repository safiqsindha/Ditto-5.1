# v5 Experiment Specification

**Status:** ✅ FULLY PRE-REGISTERED (both-author sign-off complete, 2026-04-27)  
**Version:** 1.0 (signed)  
**Authors:** Lead Author + Myriam (both signatures recorded 2026-04-27)  
**Subject model:** Claude Haiku (claude-haiku-4-5-20251001)  
**Reference experiments:** v3 (Chess/Checkers), v4 (single-cell methodology characterization)

**Sign-off status:** ✅ ALL EIGHT QUESTIONS LOCKED
- Q1: ✅ LOCKED — (B) two separate API calls per chain
- Q2: ✅ LOCKED — Bonferroni divisor = 5
- Q3: ✅ LOCKED — n = 1,200 chains/cell (matches v3's design intent at same scale)
- Q4: ✅ LOCKED — (B) fixed length per cell; (C) variable-bounded flagged for post-hoc micro-experiment
- Q5: ✅ LOCKED — (A) CS:GO round-level; (C) tick-sampled flagged for post-hoc micro-experiment
- Q6: ✅ LOCKED — (A) NBA possession-level; (B) play-level flagged for post-hoc micro-experiment
- Q7: ✅ LOCKED — (C) RL boost-enriched hit-level
- Q8: ✅ LOCKED — (A) HS per-action

---

## 1. Research Question

Does the constraint-chain detection methodology from v3 — which demonstrated [result from v3] on Chess and Checkers — generalize across five new game domains when applied identically?

Specifically: does Claude Haiku's rate of actionable-chain detection differ significantly from chance across five diverse game domains (Fortnite, NBA, CS:GO/CS2, Rocket League, Hearthstone)?

---

## 2. Experimental Design

### 2.1 Design Type

Five-cell parallel replication. Each cell is an independent application of the v3 detection methodology to a different game domain. Cells share the statistical harness but have domain-specific data acquisition pipelines and translation functions (T).

### 2.2 Subject Model

Claude Haiku (`claude-haiku-4-5-20251001`). Evaluation will run against the API once this SPEC is pre-registered. **No Haiku calls before sign-off.**

### 2.3 Conditions

**Baseline:** Model given a chain without constraint context → generates a response.  
**Intervention:** Model given same chain with constraint context injected → generates a response.

**Q1 LOCKED: (B) Two separate API calls per chain.** Each chain is evaluated in two independent API calls — one baseline (no constraint context), one intervention (with constraint context). This costs 2× per chain but eliminates the within-prompt contamination risk where the model's response to one condition leaks into how it answers the other within the same context window. See cost estimator (`v5/src/harness/cost_estimator.py`) for projected spend at this design.

### 2.4 Cells

| Cell | Domain | Data Source | Sample Target |
|------|---------|-------------|---------------|
| fortnite | Fortnite (FNCS/Cash Cup) | Epic CDN replay archive | 200 matches |
| nba | NBA 2023-24 | NBA Stats API (play-by-play) | 300 games |
| csgo | CS:GO/CS2 2024 S-tier | HLTV demo archive | 150 maps |
| rocket_league | RLCS 2024 | BallChasing.com API | 250 replays |
| hearthstone | HS 2024 Legend ladder | HSReplay.net API | 300 games |

---

## 3. Statistical Methodology

### 3.1 Primary Test

McNemar's test on paired binary outcomes (correct / incorrect per chain), with continuity correction.

```
H₀: P(intervention correct | baseline wrong) = P(baseline correct | intervention wrong)
H₁: P(intervention correct | baseline wrong) ≠ P(baseline correct | intervention wrong)
```

### 3.2 Multiple Comparisons

**Q2 LOCKED: (A) Bonferroni divisor = 5.** Each of the five cells receives a McNemar test at α_corrected = 0.05/5 = 0.01. This controls family-wise error rate across the five tests and provides a per-cell verdict — the actual research question is whether the v3 methodology generalizes across domains, which requires cell-level results, not a single global aggregate. Locked in `config/harness.yaml`.

### 3.3 Effect Size

Cohen's h for proportion differences. Reported per cell and aggregate.

### 3.4 Confidence Intervals

Bootstrap (10,000 iterations, seed=42) at 95% confidence on the proportion difference (P(intervention) - P(baseline)).

### 3.5 Minimum Sample

n_chains per cell after Gate 2 filtering: aiming for ~1,200. Power analysis:

With Bonferroni divisor=5, α_corrected=0.01, and n=1,200 chains:
- MDE ≈ 0.09 (can detect effects of 9+ percentage points with 80% power)

**Q3 LOCKED: n = 1,200 chains/cell.** Justified by alignment with v3's pre-registered design (target = 1,200 real chains per cell, calibrated to 90% power at gap = 0.06 with α = 0.0125). v5's Bonferroni divisor=5 vs v3's divisor=4 is marginally more conservative (α=0.01 vs 0.0125) — power reduction is ~3–5pp, well within v3's calibrated headroom.

**Contingency:** If real-data Gate 2 retention falls below 50% in any cell, that cell's upstream acquisition target scales up to maintain 1,200 post-filter. This is built into the run protocol — see `docs/REAL_DATA_GUIDE.md`.

**Cost projection (Q1 + Q3 combined):** 1,200 chains/cell × 2 API calls/chain (Q1=B) × 5 cells = 12,000 Haiku calls Phase 1. At v3's observed rate (~$0.0003/call for Haiku), expected spend ≈ $4 for v5 evaluation (much less than v3 because we're not also running shuffled controls in v5 Phase 1). See `src/harness/cost_estimator.py`.

**v3 reference numbers** (from v3 SPEC, retrieved 2026-04-27):
- n_target_per_cell: 1,200 real chains pre-filter, 1,000 post-filter
- Bonferroni divisor: 4 (4 cells in v3)
- α_corrected: 0.0125
- Pre-registered effect thresholds: moderate ≥ 0.05, strong ≥ 0.08
- Power calibration: ~90% at gap = 0.06
- Phase 2 not yet run; observed effect size not yet measured (v3 paused at Gate 8)

---

## 4. Data Acquisition

### 4.1 General Principles

- Tournament-tier data only for all cells (no casual/unranked matches)
- 2024 data preferred (most recent available at time of acquisition)
- Stratification by competition tier and/or season phase (see `config/cells.yaml`)
- Mock fallback when credentials unavailable (pilot and infrastructure testing only)

### 4.2 Per-Cell Sources

**Fortnite:** Epic CDN replay archive via xNocken/replay-downloader + SL-x-TnT/FortniteReplayDecompressor. FNCS Chapter 5 Season 1 (70%) + Cash Cup 2024 (30%).

**NBA:** NBA Stats API (`nba_api` package, no key required). PlayByPlayV3 endpoint. 2023-24 season: 80% regular season, 20% playoffs. Note: Second Spectrum tracking data (spatial) is NOT publicly available; this cell uses play-by-play events only.

**CS:GO/CS2:** HLTV public demo archive. CS2 format (post-August 2023). 2024 S-tier events: IEM Katowice (~33%), BLAST Premier (~34%), ESL One (~33%). Parsed with `awpy` (demoinfocs-golang wrapper).

**Rocket League:** BallChasing.com API (free tier, key required). RLCS 2024: World Championship (30%), Majors (40%), Regionals (30%). Parsed with `carball` Python package.

**Hearthstone:** HSReplay.net API (key required for bulk access). Legend-rank ladder 2024: legend 1-100 (40%), legend 101-1000 (60%). Parsed with HearthSim `hslog` package.

---

## 5. Event Extraction

### 5.1 Normalized Event Schema

All cells produce `GameEvent` objects with fields:
```
timestamp        : float  (seconds from game start, monotonic)
event_type       : str    (normalized; phase_ prefix stripped — v1.1 amendment)
actor            : str    (player/team identifier, consistent within domain)
location_context : dict   (domain-specific spatial/situational data)
raw_data_blob    : dict   (full source record)
cell             : str    (domain identifier)
game_id          : str    (unique match/game identifier)
sequence_idx     : int    (0-based ordinal within game)
```

### 5.2 Sizing Target

Each cell targets upstream event volumes sufficient for ~1,200 chains after filtering. Assuming 40-60% retention through Gate 2:
- Raw chain candidates needed: ~2,000-3,000 per cell
- Raw events needed: ~6,000-15,000 per cell (depending on chain length)
- Mock data validates this sizing; real data acquisition may require adjustment.

---

## 6. Translation Function (T)

### 6.1 Interface

T is a `TranslationFunction` that accepts an `EventStream` and returns `List[ChainCandidate]`. Interface defined in `src/interfaces/translation.py`. All domain T implementations inherit from this base class.

### 6.2 Out of Scope for Infrastructure Build

All five T implementations (`FortniteT`, `NBAT`, `CSGOT`, `RocketLeagueT`, `HearthstoneT`) are **stubs** that raise `NotImplementedError`. They require both-author review per the v6-discipline upgrade. **Do not implement T until SPEC is signed off.**

**Q4 LOCKED: (B) Fixed chain length per cell, varying across cells.** Each cell's T pre-registers a single chain length N appropriate to its domain's natural decision granularity. Per-cell N values to be locked at T-design time (joint authoring session). Variant **(C) variable bounded** is flagged for a post-hoc micro-experiment after primary v5 results — see `MICRO_EXPERIMENTS.md`.

**Q5 LOCKED: (A) CS:GO round-level.** ~300 events per map. Each kill, grenade throw, bomb event, and buy-phase summary is one event. Variant **(C) tick-sampled** is flagged for a post-hoc micro-experiment.

**Q6 LOCKED: (A) NBA possession-level.** ~85 possessions per game, ~2–3 events per possession (≈ 200 events/game). Plays are grouped into possessions (offensive trip + defensive set as one unit, with shot clock and possession-change boundaries). Variant **(B) play-level** is flagged for a post-hoc micro-experiment. *Note:* This requires updating `nba/extractor.py` from its current play-level default to possession-level grouping.

**Q7 LOCKED: (C) Rocket League boost-enriched hit-level.** Hits + boost pickups + low-boost (resource_budget) events interleaved chronologically. ~250–400 events per 5-min game. Captures both ball-contact decisions and boost-economy decisions in one stream. *Note:* This requires extending `rocket_league/extractor.py` from hit-only to boost-enriched.

**Q8 LOCKED: (A) Hearthstone per-action.** ~80 events/game. Each card play, attack, hero power, and battlecry trigger is a separate event. Preserves intra-turn decision sequencing (combos, trades, removal sequencing).

---

## 7. Chain Construction

### 7.1 Interface

`ChainBuilder` defined in `src/interfaces/chain_builder.py`. Stub only — raises `NotImplementedError`.

### 7.2 Pre-Registered Parameters (per Q4 sign-off)

Chain length is **fixed per cell**; per-cell N values to be locked at T-design time. Overlap policy and sampling strategy are pre-registered as:
- **Overlap policy:** non-overlapping by default (sliding window with step = chain length)
- **Sampling strategy:** sequential within game; cells subsample chains uniformly to hit the per-cell n_chains target.

`DefaultChainBuilder` is implemented as `FixedPerCellChainBuilder` accepting a per-cell `chain_length` map (see `src/interfaces/chain_builder.py`). Per-cell N values stay TBD until T design.

---

## 8. Evaluation Harness

### 8.1 Gate 2 Filter

Before evaluation, chains pass through Gate 2: at least 50% of events in the chain must be actionable under the v1.1 ACTIONABLE_TYPES whitelist.

**v1.1 ACTIONABLE_TYPES whitelist** (see `src/harness/actionables.py`):
- Core types: engage_decision, position_commit, resource_gain/spend/budget, rotation_commit, zone_enter/exit, ability_use, item_use, target_select, team_coordinate, objective_contest/capture/abandon, timing_commit, draft_pick/ban, strategy_adapt, risk_accept/reject, concede, delay_action, force_action

**v1.1 Amendments:**
- `resource_budget` added (was absent in v1.0)
- `phase_` prefix stripped from event_type before comparison

### 8.2 Scoring

Binary: correct (1) / incorrect (0) / abstain (-1). Abstains excluded from McNemar input by default.

### 8.3 Cell Runner

`CellRunner` in `src/harness/cell_runner.py` orchestrates per-cell runs:
1. Translate event streams to chains via T
2. Gate 2 filter
3. Score chains against model responses
4. McNemar test per cell
5. Variance analysis per cell
6. Aggregate report

---

## 9. Pilot Validation

Pilot harness (`src/pilot/validator.py`) runs 50-100 sample chains per cell through the full pipeline using `MockT` (deterministic sliding-window heuristic, not a real T). Validates:
- Event stream generation
- Gate 2 retention rate (>= 50% floor)
- Chain length and actionable-fraction distributions
- Event type distribution sanity

**Pilot status as of infrastructure build:** ALL FIVE CELLS PASS with mock data.
- Fortnite: 200 streams, 24,000 events, 8,000 chains (100% retention)
- NBA: 300 streams, 55,200 events, 18,420 chains (100% retention)
- CS:GO: 150 streams, 45,000 events, 15,000 chains (100% retention)
- Rocket League: 250 streams, 50,000 events, 16,750 chains (100% retention)
- Hearthstone: 300 streams, 23,100 events, 7,740 chains (100% retention)

Note: 100% retention with mock data is expected because MockT only selects actionable events. Real T may have lower retention.

---

## 10. Pre-Registration Sign-Off Required

The following decisions are **BLOCKING** on pre-registration. They must be signed by both authors before any T implementation or evaluation run.

| # | Question | Status | Locked Choice |
|---|----------|--------|---------------|
| Q1 | Baseline/intervention prompt structure | ✅ LOCKED | (B) two separate calls |
| Q2 | Bonferroni divisor | ✅ LOCKED | (A) divisor = 5 |
| Q3 | Is n=1,200/cell sufficient? | ✅ LOCKED | n = 1,200 (matches v3 design intent; contingency in §3.5) |
| Q4 | Chain length consistency | ✅ LOCKED | (B) fixed per cell; (C) flagged for micro-experiment |
| Q5 | CS:GO granularity | ✅ LOCKED | (A) round-level; (C) flagged for micro-experiment |
| Q6 | NBA granularity | ✅ LOCKED | (A) possession-level; (B) flagged for micro-experiment |
| Q7 | Rocket League state handling | ✅ LOCKED | (C) boost-enriched hit-level |
| Q8 | Hearthstone granularity | ✅ LOCKED | (A) per-action |

### Future micro-experiments (flagged but not in v5 primary scope)

These are pre-noted as "would be interesting if v5 results justify follow-up." They are NOT pre-registered for v5 and must NOT be implemented as part of the primary v5 run.

- Q4-C: Variable-bounded chain length (3–15 events) as alternative to fixed-per-cell
- Q5-C: CS:GO tick-sampled granularity (every 256 ticks)
- Q6-B: NBA play-level granularity (~220 events/game)

See `MICRO_EXPERIMENTS.md` for full list and pre-registration status.

---

## 11. References

- v3 Experiment: safiqsindha/Project-Ditto-V3 (Chess & Checkers)
- v4 Experiment: safiqsindha/Project-Ditto-V4 (single-cell methodology characterization)
- McNemar (1947): "Note on the Sampling Error of the Difference Between Correlated Proportions"
- Cohen (1988): "Statistical Power Analysis for the Behavioral Sciences" (h effect size)
- nba_api: github.com/swar/nba_api
- awpy: github.com/pnxenopoulos/awpy
- BallChasing API: ballchasing.com/api
- HearthSim: github.com/HearthSim
