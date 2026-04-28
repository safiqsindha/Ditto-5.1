# v5 Decision Log

**Session:** Infrastructure build, 2026-04-27  
**Format:** Each decision logged per instructions with Default/Alternative/Reversibility framing.

---

## Decision D-0: Reference Repos Inaccessible

**Context:** Beginning the build. User specified "Reference experiments are v3 and v4 main, both in the repo."

**Question:** v3 (safiqsindha/Project-Ditto-V3) and v4 (safiqsindha/Project-Ditto-V4) are in separate GitHub repositories. The MCP session is scoped only to safiqsindha/Ditto-V5. How do we handle the lack of reference code?

**Default chosen:** Build infrastructure from first principles using the detailed description in the build prompt (ACTIONABLE_TYPES, McNemar pipeline, v1.1 amendments, Gate 2 floor, event schema) as the authoritative specification. No speculative inference beyond what's described.

**Reasoning:** The prompt is sufficiently detailed to reconstruct the harness. Attempting to infer undescribed v3/v4 specifics risks introducing artifacts. Building from the explicit specification is safer than building from inferred code.

**Alternatives considered:**
- Clone v3/v4 repos via git: Would require external network access not available in session.
- Ask user to wait: Violates the "don't block, pick a defensible default" instruction.

**Reversibility:** Moderate — if v3/v4 have subtly different McNemar implementations or ACTIONABLE_TYPES, individual harness files will need adjustment. File-level changes, not architectural.

**Needs review before SPEC sign-off:** Yes — please review `src/harness/` against v3/v4 implementations to confirm alignment.

---

## Decision D-1: Sample Size — Fortnite

**Context:** Setting sample target for Fortnite data acquisition pipeline.

**Question:** How many tournament matches to pull for Fortnite? No specific number given beyond "~1,200 chains worth of upstream data per cell."

**Default chosen:** 200 tournament matches. Breakdown: 140 FNCS Chapter 5 Season 1 (70%) + 60 Cash Cup 2024 (30%).

**Reasoning:** At ~120 events/match and window_size=5/step=3, a 200-match corpus produces ~8,000 chain candidates (4 chains per stream segment), well above the 2,000-3,000 needed to hit 1,200 post-Gate2. This leaves headroom for real T's lower extraction rate.

**Alternatives considered:**
- 100 matches: Would produce ~4,000 candidates, borderline if real T retains <50%.
- 500 matches: Overkill for infrastructure phase; acquisition takes longer and storage is larger.

**Reversibility:** Easy — config/cells.yaml `sample_target_matches` is a single integer.

**Needs review before SPEC sign-off:** Yes — confirm adequacy of 200 once T extraction rate is estimated.

---

## Decision D-2: Sample Size — NBA

**Context:** Setting sample target for NBA pipeline.

**Question:** How many games? NBA season has 1,230 regular season games + 82-105 playoff games.

**Default chosen:** 300 games: 240 regular season (80%) + 60 playoffs (20%). Season: 2023-24.

**Reasoning:** NBA play-by-play has ~220 rows per game, of which ~100-150 are decision events (types 1-10). 300 games × 120 events = 36,000 events upstream. At window_size=5/step=3, ~12,000 chains pre-filter. Well above target.

**Alternatives considered:**
- Use both 2022-23 and 2023-24 seasons: More data but broader time range introduces roster/strategy changes that complicate interpretation.
- 500 games: More representative but acquisition time is proportionally longer.

**Reversibility:** Easy — `sample_target_games` in config.

**Needs review before SPEC sign-off:** Yes — confirm 2023-24 single-season scope is appropriate. Also: [REQUIRES SIGN-OFF] whether to include All-Star game (excluded by default).

---

## Decision D-3: Sample Size — CS:GO/CS2

**Context:** Setting sample target for CS:GO pipeline.

**Question:** How many maps? Each HLTV best-of-3 series has 2-3 maps.

**Default chosen:** 150 maps from 2024 S-tier events: IEM Katowice 2024 (~50), BLAST Premier 2024 (~50), ESL One 2024 (~50).

**Reasoning:** 150 maps × ~300 events (round-level granularity) = 45,000 events. At window_size=5, ~15,000 chain candidates. Well-sized for the target. Three tournaments provides stylistic diversity (different meta phases of 2024 CS2).

**Alternatives considered:**
- All 2024 S-tier maps (~300+): More comprehensive but demo download time becomes a bottleneck.
- Single tournament (IEM Katowice): Lower diversity; one tournament's meta may not generalize.

**Reversibility:** Easy — config stratification fractions and `sample_target_maps`.

**Needs review before SPEC sign-off:** Yes — confirm tournament tier definition (S-tier per HLTV ranking).

---

## Decision D-4: Sample Size — Rocket League

**Context:** Setting sample target for Rocket League pipeline.

**Question:** How many RLCS replays?

**Default chosen:** 250 replays. Breakdown: 75 World Championship (30%) + 100 Majors (40%) + 75 Regionals (30%). Season: RLCS 2024.

**Reasoning:** RL replays average ~200 hit-level events per 5-minute game. 250 × 200 = 50,000 events. At window_size=5, ~16,750 chain candidates. Good headroom. World Championship overrepresented slightly relative to volume (fewer games played) because highest competitive quality.

**Alternatives considered:**
- 500 replays: More comprehensive but BallChasing free tier rate limits would make acquisition slower.
- Only World Championship: Too few games total (~40-50 in 2024 WC) for statistical adequacy.

**Reversibility:** Easy — config.

**Needs review before SPEC sign-off:** Yes — confirm RLCS 2024 season dates and BallChasing playlist filter string.

---

## Decision D-5: Sample Size — Hearthstone

**Context:** Setting sample target for Hearthstone pipeline.

**Question:** How many games? HSReplay tracks millions of games but Legend-rank games are rarer.

**Default chosen:** 300 Legend-rank games: 120 legend 1-100 (40%) + 180 legend 101-1000 (60%). Year: 2024.

**Reasoning:** HS games average ~15 turns per player × 5-6 decisions/turn = ~80 decision events per game. 300 × 80 = 24,000 events. At window_size=5, ~7,740 chain candidates. Adequate. Top-legend (1-100) oversampled relative to population to capture highest-skill decision-making.

**Alternatives considered:**
- Standard ladder (non-Legend): Decision quality lower; harder to attribute decisions to strategic constraint application.
- All-legend (not stratified): Less control over rank distribution; top-100 decisions may be qualitatively different.

**Reversibility:** Easy — config.

**Needs review before SPEC sign-off:** Yes — confirm "Legend" is the right rank tier (vs Diamond 1+ or Top 500).

---

## Decision D-6: Time Range — 2024 Calendar Year

**Context:** Setting time range for all five data sources.

**Question:** Should we use 2024 calendar year or most recent 12 months from date of acquisition?

**Default chosen:** 2024 calendar year (Jan 1 – Dec 31, 2024) for all cells.

**Reasoning:** Aligns all five cells to the same temporal scope, avoiding cross-cell confounds from different meta/patch environments. Calendar year is a clean, pre-specifiable boundary.

**Alternatives considered:**
- "Most recent N months": Risk of inconsistent cut-offs if acquisition happens over multiple days.
- 2023-2024 combined: More data but introduces two-season variation in strategy/meta.

**Reversibility:** Easy — date parameters in config.

**Needs review before SPEC sign-off:** Yes — flag any cell where 2024 data is sparse (Hearthstone rotation, RL off-season periods).

---

## Decision D-7: CS:GO Granularity Default (Round-Level)

**Context:** CS2 demo files are 128Hz tick-rate. Must choose event granularity.

**Question:** Operate at round-level, tick-sampled, or clutch-level?

**Default chosen:** Round-level — kill events, grenade events, bomb events, and buy-phase events. ~300 events per map.

**Reasoning:** Round-level provides semantically clear decision units (each kill/utility represents a discrete agent decision). Tick-level would produce ~100k events per map which is incompatible with chain-length constraints and T's expected operating scale. This default is flagged in SPEC.md Q5 as [REQUIRES SIGN-OFF].

**Alternatives considered:**
- Tick-sampled (every 256 ticks = 2 seconds): ~6,750 events/map; more granular but generates mostly non-decision frames.
- Clutch-level: Semantically meaningful but adds complexity in defining "clutch" situation without T review.

**Reversibility:** Moderate — extractor.py would need reworking; event schema is unchanged.

**Needs review before SPEC sign-off:** YES — marked BLOCKING in SPEC.md.

---

## Decision D-8: Rocket League Parser — carball Primary, rrrocket Fallback

**Context:** Two viable parsers exist: carball (Python) and rrrocket (Rust binary).

**Question:** Which parser to use as primary?

**Default chosen:** `carball` as primary (pip-installable Python package), `rrrocket` subprocess as fallback.

**Reasoning:** carball is directly pip-installable in the Python environment, produces structured Python objects, and provides hit-level abstraction out of the box. rrrocket produces raw JSON requiring more post-processing.

**Alternatives considered:**
- rrrocket primary: Better performance for large replay sets but requires Rust compilation or pre-built binary.
- Use only carball: Carball may not be maintained for all replay versions; rrrocket fallback provides robustness.

**Reversibility:** Easy — `_parse_replay()` method in pipeline.py; parser output schema is normalized in extractor.

**Needs review before SPEC sign-off:** No — this is a tooling choice, not a methodological one.

---

## Decision D-9: Hearthstone Granularity Default (Per-Action)

**Context:** HS hslog provides per-packet granularity (card plays, attacks, hero power, battlecries).

**Question:** Per-action or per-turn granularity?

**Default chosen:** Per-action — each card play, attack, hero power = one GameEvent. Produces ~80 events/game.

**Reasoning:** Per-action preserves the most decision information. A single turn may contain multiple strategically significant decisions (e.g., trading before playing a threat). Per-turn would collapse this into one event and lose sequencing information.

**Alternatives considered:**
- Per-turn: Simpler chains; each chain = one game turn context. But loses intra-turn decision ordering.

**Reversibility:** Moderate — extractor.py tree walker would be reworked. Schema unchanged.

**Needs review before SPEC sign-off:** YES — marked BLOCKING in SPEC.md (Q8).

---

## Decision D-10: NBA Tracking Data Not Available

**Context:** NBA Stats API is public; Second Spectrum spatial tracking data is not.

**Question:** Should NBA cell be built with play-by-play only (no spatial tracking) or should we flag this as a limitation?

**Default chosen:** Build with play-by-play only. Log explicitly that spatial tracking is unavailable. This limits the `location_context` field to play description strings and score, not x/y court coordinates.

**Reasoning:** Play-by-play is sufficient for event extraction at the play-level or possession-level. The location_context field can be populated with period/clock/score as a reasonable substitute. Spatial tracking would require a partnership with Second Spectrum or another licensed provider — out of scope.

**Alternatives considered:**
- Use shot chart API (public, limited): Only covers shots, not all play types.
- Partner data source: Out of scope for infrastructure build.

**Reversibility:** Easy — extractor.py could be extended with spatial data if a source becomes available.

**Needs review before SPEC sign-off:** Yes — flag this limitation in SPEC.md final version. [Already noted in §4.2]

---

## Decision D-11: ACTIONABLE_TYPES Whitelist — v1.1 Adopted as-Is

**Context:** Building the harness. The prompt specifies "Reuse v4's ACTIONABLE_TYPES whitelist and v1.1 amendments (ResourceBudget in actionables, phase_ prefix strip) as defaults."

**Question:** Should any cell's data suggest the whitelist needs adaptation, should we adapt now or flag for review?

**Default chosen:** Adopt v1.1 whitelist exactly as described. Flag in DECISION_LOG (this entry). Do not adapt for any specific cell without sign-off.

**Reasoning:** The pilot run with mock data showed 100% retention across all cells, suggesting the whitelist is broad enough to capture all mock event types. With real data, a narrower retention rate may emerge — if any cell falls below 50% Gate 2 floor with real data, that warrants a sign-off discussion, not a unilateral whitelist change.

**Alternatives considered:**
- Add domain-specific types now: Premature without T review; risk of widening the whitelist beyond what v3 used.

**Reversibility:** Easy — `CELL_ACTIONABLE_OVERRIDES` dict in actionables.py is ready to receive per-cell additions.

**Needs review before SPEC sign-off:** Yes — confirm v1.1 whitelist matches v4's implemented version.

---

## Decision D-12: Chain Construction Interfaces — Not Defaulted

**Context:** Chain construction requires pre-registration of chain length and granularity.

**Question:** Should we set default chain length parameters in the stub to avoid "empty interface" risk?

**Default chosen:** No defaults set. Both `ChainBuilder.build()` and all domain `TranslationFunction.translate()` raise `NotImplementedError` with explicit messages referencing SPEC.md sign-off sections.

**Reasoning:** Setting defaults — even "just for infrastructure" — risks those defaults becoming de facto choices if sign-off is delayed. The user's instructions are explicit: "don't default and proceed" for chain construction.

**Alternatives considered:**
- Set chain_length=5 as placeholder: Violates explicit out-of-scope instruction.

**Reversibility:** N/A — stubs contain no defaults to reverse.

**Needs review before SPEC sign-off:** YES — BLOCKING.

---

## Decision D-13: Mock Data Sizing

**Context:** Mock data must be "sized for ~1,200 chains downstream." MockT uses window_size=5, step_size=3.

**Question:** How many events per mock game to generate?

**Default chosen:**
- Fortnite: 120 events/match (200 matches → 24,000 events)
- NBA: 180-200 events/game (300 games → ~55,200 events)
- CS:GO: 300 events/map (150 maps → 45,000 events)
- Rocket League: 200 events/replay (250 replays → 50,000 events)
- Hearthstone: 75-80 events/game (300 games → ~23,100 events)

Pilot run confirmed 7,740–18,420 chains per cell (all well above 1,200 target).

**Reasoning:** Per-domain estimates based on real expected event rates at chosen granularity. Slightly conservative to account for real T's lower extraction rate.

**Alternatives considered:**
- Uniform 100 events/game across all cells: Doesn't reflect domain-specific event rates.

**Reversibility:** Easy — `n_events` parameter in `generate_mock_data()` per pipeline.

**Needs review before SPEC sign-off:** No — mock sizing doesn't affect pre-registration decisions.

---

## Decision D-14: `force_mock` Parameter on Pipelines

**Context:** End-of-day audit caught that NBA and CS:GO pipelines have `env_vars: []` in their config. Python's `all([])` is vacuously True, so `env_satisfied()` returned True, so `should_use_mock()` returned False. The pilot run path was bypassing this by calling `generate_mock_data()` directly, but the bug was invisible until I tried to call `pipeline.run()` for stream persistence — at which point NBA pipeline started attempting real API calls.

**Question:** Should we (a) fix the `env_satisfied()` semantics to require explicit credentials, (b) add a `force_mock` parameter to make pilot/infrastructure runs unambiguous, or (c) require all cells to have at least one declared env var?

**Default chosen:** (b) — added `force_mock=False` parameter to `BasePipeline.run()`. When True, skips fetch/parse and uses mock data regardless of credential state. `run_pilot.py` updated to use `force_mock=True`. Test added (`test_force_mock_overrides_credentials`).

**Reasoning:** Option (a) would change the meaning of "no credentials required" in cells like NBA/CS:GO that legitimately have public APIs. Option (c) is unnatural — NBA Stats API genuinely needs no key. Option (b) makes intent explicit at the call site without restricting the underlying logic.

**Alternatives considered:**
- Auto-detect "infrastructure" mode by environment variable: too magical; obscures which path actually ran.
- Always default to mock for cells with `env_vars=[]`: would make NBA/CS:GO real-data acquisition awkward (would require explicit override).

**Reversibility:** Easy — single optional parameter, default preserves prior behavior.

**Needs review before SPEC sign-off:** No — this is a correctness fix for the existing infrastructure, not a methodological choice.

---

## Decision D-15: Fortnite Pipeline — Burner Account + Proxy/Rate-Limiter

**Context:** During the build session, the user noted that FortniteTracker (TRN) and Replay.io are easier-auth alternatives to the current xNocken + FortniteReplayDecompressor pipeline, but they provide aggregated stats (placement, kill counts) rather than raw spatial telemetry.

**Question:** Should we switch to TRN/Replay.io for simpler authentication, or stick with the current pipeline despite credential complexity?

**Default chosen:** Stick with xNocken/replay-downloader + FortniteReplayDecompressor. User to provide a Node.js async queue snippet that uses a burner Epic account + proxy/rate-limiter to avoid main-account flagging.

**Reasoning:** The current pipeline produces `GameEvent` streams with full spatial telemetry (player positions, build events with x/y/z coordinates, eliminations with attacker/victim positions). This is what the `FortniteExtractor` is designed to consume and what makes Fortnite's chain-detection methodologically informative. Aggregated-stats sources would force us to either (a) drop spatial features or (b) add a separate spatial source — neither attractive.

**Alternatives considered:**
- TRN API: Aggregated only; loses spatial decision context.
- Replay.io: Same limitation; depends on event being indexed.
- Pure mock for v5 evaluation: Defeats the experiment's purpose for the Fortnite cell.

**Reversibility:** Moderate — switching parsers later would require rewriting `FortniteExtractor` and re-running acquisition.

**Needs review before SPEC sign-off:** No — this is a tooling and operational-security decision, not a methodological one. The user's Node.js queue snippet, when provided, will be added to `v5/src/cells/fortnite/fetch_queue.js` and invoked from `pipeline.fetch()` via `subprocess.run(["node", "fetch_queue.js", ...])`.

**Pending dependency:** Awaiting user-provided Node.js async queue snippet.

---

## Decision D-16: SPEC Q1 Sign-Off — Two-Call Prompt Structure

**Context:** SPEC sign-off round 1 (2026-04-27). Q1 asked: same API call vs two separate calls for baseline/intervention.

**Question:** How should baseline and intervention be presented to the subject model?

**Default chosen:** (B) Two separate API calls per chain.

**Reasoning:** User-confirmed. Eliminates within-prompt contamination risk where the model's response to one condition could leak into how it answers the other when both share a context window. Costs 2× per chain but the experiment is small enough (~$4 Phase 1) that the cost premium is negligible.

**Alternatives considered:** (A) Same call with constraint segment as a flag — cheaper but contamination risk; rejected.

**Reversibility:** Easy — single parameter in evaluation harness.

**Needs review before SPEC sign-off:** Done — locked.

---

## Decision D-17: SPEC Q2 Sign-Off — Bonferroni Divisor = 5

**Context:** SPEC sign-off round 1.

**Question:** Multiple-comparisons correction across the five cells.

**Default chosen:** (A) Divisor = 5; α_corrected = 0.01 per cell.

**Reasoning:** User-confirmed. The actual research question is whether the v3 methodology generalizes — which requires per-cell verdicts, not a single global aggregate. Conservative but correct. Aligned with v3's approach (v3 used divisor = 4 for its 4 cells).

**Alternatives considered:** (B) divisor=1 treats v5 as one global test, loses cell-level resolution; (C) mixed adds interpretive complexity for marginal benefit.

**Reversibility:** Easy — single config value.

**Needs review before SPEC sign-off:** Done — locked.

---

## Decision D-18: SPEC Q3 Sign-Off — n = 1,200 chains/cell

**Context:** SPEC sign-off round 1. v3 effect-size review via WebFetch on public v3 SPEC and SESSION_LOG.

**Question:** Is n=1,200 per cell sufficient given v3's anticipated effect size?

**Default chosen:** Lock at n=1,200/cell with contingency: if real-data Gate 2 retention < 50%, scale upstream acquisition to maintain 1,200 post-filter.

**Reasoning:** v3 designed at the same scale (1,200 real chains pre-filter, 1,000 post-filter) calibrated to 90% power at gap=0.06 with α=0.0125. v5's Bonferroni=5 vs v3's =4 reduces α from 0.0125 to 0.01 — minor power loss (~3-5pp). v3 has not yet run Phase 2, so observed effect size is unknown; v5 inherits v3's design intent.

**Alternatives considered:**
- Bump to 1,500 for headroom against tighter Bonferroni: defensible but costs ~25% more API; not necessary unless results come in null.
- Bump to 2,500: only justified if v3 had observed gap < 0.05, which it hasn't.

**Reversibility:** Easy in code (config integer); harder in practice (would need additional acquisition).

**Needs review before SPEC sign-off:** Done — locked.

---

## Decision D-19: SPEC Q4 Sign-Off — Fixed Chain Length Per Cell

**Context:** SPEC sign-off round 1.

**Question:** Chain length: fixed global, fixed per cell, or variable bounded.

**Default chosen:** (B) Fixed length per cell, varying across cells. Per-cell N values to be locked at T-design time (joint authoring session). Variant (C) variable-bounded flagged as a post-hoc micro-experiment ME-1.

**Reasoning:** User-confirmed. Domain decision granularities differ substantially (CS round vs HS turn vs NBA possession) and forcing one global N would either fragment some cells' decisions or merge multiple decisions into one chain unnaturally. Per-cell N respects domain structure while still preserving McNemar's per-cell validity (the test doesn't require equal n across cells).

**Alternatives considered:** (A) Fixed global treats domains as identical when they aren't; (C) variable bounded gives most flexibility but interpretation harder. (C) preserved as ME-1 micro-experiment for after primary run.

**Reversibility:** Moderate — changing N per cell requires regenerating chains.

**Needs review before SPEC sign-off:** Done — locked.

---

## Decision D-20: SPEC Q5 Sign-Off — CS:GO Round-Level Granularity

**Context:** SPEC sign-off round 1.

**Question:** CS:GO event granularity.

**Default chosen:** (A) Round-level (~300 events/map). (C) tick-sampled flagged as micro-experiment ME-2.

**Reasoning:** User-confirmed. Rounds are the natural decision boundary in CS — each round is one buy decision, one execute, one retake. Tick-sampled gives more events but mostly captures non-decision frames. Round-level matches existing extractor; no code change needed.

**Reversibility:** Moderate.

**Needs review before SPEC sign-off:** Done — locked.

---

## Decision D-21: SPEC Q6 Sign-Off — NBA Possession-Level Granularity

**Context:** SPEC sign-off round 1.

**Question:** NBA event granularity.

**Default chosen:** (A) Possession-level (~85 possessions/game). (B) play-level flagged as micro-experiment ME-3.

**Reasoning:** User-confirmed. Possession is the natural NBA decision unit — one offensive trip + one defensive set. Play-level fragments possessions across multiple events.

**Code impact:** REQUIRES update to `nba/extractor.py` — current default emits one event per PBP row (play-level). New behavior: group plays into possessions using shot-clock and possession-change boundaries; emit one event per possession with possession metadata.

**Reversibility:** Moderate — extractor change, fully backwards-compatible since events still satisfy the GameEvent schema.

**Needs review before SPEC sign-off:** Done — locked.

---

## Decision D-22: SPEC Q7 Sign-Off — RL Boost-Enriched Hit-Level

**Context:** SPEC sign-off round 1.

**Question:** Rocket League event granularity / state handling.

**Default chosen:** (C) Boost-enriched hit-level — interleaves ball-contact events with boost-economy events (pickups, low-boost decisions).

**Reasoning:** User-confirmed. Boost economy is approximately half the decision space in RL; ignoring it (option A hit-only) loses critical decision context. Possession-level (option B) is too coarse for hit-by-hit decisions.

**Code impact:** REQUIRES update to `rocket_league/extractor.py`. Current implementation extracts hits + boost events separately; needs to merge them into one chronologically-sorted stream with consistent event_type normalization. Already partially done — boost events are extracted but as a separate pass; the new requirement is integrating them as first-class events in the same stream with appropriate event_type tags (resource_gain for pickup, resource_spend for use, resource_budget for low-boost decisions).

**Reversibility:** Moderate — extractor change.

**Needs review before SPEC sign-off:** Done — locked.

---

## Decision D-23: SPEC Q8 Sign-Off — Hearthstone Per-Action Granularity

**Context:** SPEC sign-off round 1.

**Question:** Hearthstone event granularity.

**Default chosen:** (A) Per-action (~80 events/game). Each card play, attack, hero power, battlecry trigger = one event.

**Reasoning:** User-confirmed. Per-action preserves intra-turn decision sequencing (combo plays, trade-then-play, removal sequencing). Per-turn (option B) collapses these into single-event blobs and loses the actual decision structure. Matches existing extractor; no code change needed.

**Reversibility:** Easy.

**Needs review before SPEC sign-off:** Done — locked.

---

## Decision D-24: Phase B CF-1 — Constraint Expression Format

**Context:** Phase B joint authoring session 2026-04-27 (lead author + Myriam).

**Question:** How is constraint expressed in the prompt?

**Decision:** (A) Natural-language description of rule structure.

**Reasoning:** Matches v3 framing; easiest to keep consistent across 5 cells. B (formal predicate list) and C (in-context examples) noted as interesting variants for future study.

**Code impact:** `format_constraint_context()` in each PromptBuilder subclass now returns natural-language text. See prompts.py.

---

## Decision D-25: Phase B CF-2 — Prediction Target

**Context:** Phase B joint authoring session 2026-04-27.

**Question:** What is the model asked to predict?

**Decision:** (D) Binary classify: is this chain constraint-respecting or constraint-violating? YES / NO.

**Reasoning:** Directly tests the detection methodology; matches v3's framing. Options A (next event type), B (actor), C (domain outcome) flagged as micro-experiment variants.

**Code impact:** `format_question()` in each PromptBuilder returns "Is this sequence consistent with [domain] rules? Reply YES or NO." `parse_model_response` already handles binary tokens.

---

## Decision D-26: Phase B CF-3 — Shuffled Controls

**Context:** Phase B joint authoring session 2026-04-27.

**Question:** Do we generate shuffled-chain controls?

**Decision:** (A) Yes — 1× shuffled chains per cell as controls (random event reordering within game). Matches v3 design.

**Reasoning:** v5 is intended to replicate v3's detection methodology across new domains. Shuffled controls are the primary comparison; without them, we have no within-experiment null distribution.

**Code impact:** ChainBuilder will need a `shuffle_chains()` method (Phase D implementation). Budget impact: 2× chains = 2× API calls = ~$3.90 total (within cap).

---

## Decision D-27: Phase B CF-4 — Chain Provenance

**Context:** Phase B joint authoring session 2026-04-27.

**Question:** What provenance does the model see?

**Decision:** (B) Domain only — model sees "this is a CS:GO chain" but no team/player/tournament identifiers.

**Reasoning:** Prevents model leaning on memorized tournament facts. Option A (anonymous) flagged as a micro-experiment — interesting to see if domain label helps or hurts detection.

**Code impact:** PromptBuilder formats events with cell name visible but strips actor names where they would reveal specific players. Implemented via format_event() (using generic actor IDs from the schema).

---

## Decision D-28: Phase B F-1–F-6 — Fortnite T Design

**Context:** Phase B joint authoring session 2026-04-27.

**Decisions:**
- F-1: Constraint = storm-zone boundary + elimination causality. Build-cost rule noted for future testing.
- F-2: Chain = storm-rotation phase (zone_enter / zone_exit / position_commit trigger window)
- F-3: N = 8
- F-4: Binary classify (follows CF-2)
- F-5: "In Fortnite, players must remain within the safe storm zone or take damage over time. An eliminated player cannot generate further actions. Building structures consumes exactly one material per piece."
- F-6: Add storm_rotation, build_decision to CELL_ACTIONABLE_OVERRIDES["fortnite"]

**Code impact:** FortniteT implemented in translation.py. FortnitePromptBuilder updated. harness.yaml fortnite: 8. actionables.py overrides populated.

---

## Decision D-29: Phase B N-1–N-6 — NBA T Design

**Context:** Phase B joint authoring session 2026-04-27.

**Decisions:**
- N-1: shot-clock (24s) + foul-out (6 fouls = ejection)
- N-2: consecutive possessions from single quarter
- N-3: N = 5
- N-4: Binary classify (follows CF-2)
- N-5: (see prompts.py NBAPromptBuilder)
- N-6: Add shot_selection, clutch_decision to overrides

**Code impact:** NBAT implemented. NBAPromptBuilder updated. harness.yaml nba: 5.

---

## Decision D-30: Phase B C-1–C-6 — CS:GO T Design

**Context:** Phase B joint authoring session 2026-04-27.

**Decisions:**
- C-1: Respawn-disabled + bomb-objective (buy-phase economy is pre-round, less useful chain-level)
- C-2: Full round (buy phase + tactical phase + outcome)
- C-3: N = 10
- C-4: Binary classify (follows CF-2)
- C-5: (see prompts.py CSGOPromptBuilder)
- C-6: Add buy_decision, utility_deploy, bombsite_commit to overrides

**Code impact:** CSGOT implemented. CSGOPromptBuilder updated. harness.yaml csgo: 10.

---

## Decision D-31: Phase B R-1–R-6 — Rocket League T Design

**Context:** Phase B joint authoring session 2026-04-27.

**Decisions:**
- R-1: Boost-economy (max 100, depletes, replenishes) + goal-causality
- R-2: Play = team-possession sequence to next team's possession or goal (boost events interleaved)
- R-3: N = 12
- R-4: Binary classify (follows CF-2)
- R-5: (see prompts.py RocketLeaguePromptBuilder)
- R-6: Add aerial_commit, boost_steal, rotation_back to overrides
- NOTE: per-player chain variant flagged as ME-RL-1 for v6 deep T analysis

**Code impact:** RocketLeagueT implemented. RocketLeaguePromptBuilder updated. harness.yaml rocket_league: 12.

---

## Decision D-32: Phase B H-1–H-6 — Hearthstone T Design

**Context:** Phase B joint authoring session 2026-04-27.

**Decisions:**
- H-1: Mana-cost rule + turn-alternation + board-state (minion HP → removal)
- H-2: All actions within one player's turn
- H-3: N = 6
- H-4: Binary classify (follows CF-2)
- H-5: (see prompts.py HearthstonePromptBuilder)
- H-6: Add card_play, lethal_lining_up, mana_curve_choice to overrides

**Code impact:** HearthstoneT implemented. HearthstonePromptBuilder updated. harness.yaml hearthstone: 6.

---

## Decision D-33: Phase B insight — Fortnite + RL T variability

**Context:** Lead author observation during Phase B session.

**Observation:** "T structure for Fortnite and Rocket League can vary heavily — deep T analysis in v6."

**Decision:** Flag Fortnite and RL as priority cells for v6 T-design refinement. Both cells have multi-dimensional constraint structure (spatial + resource for RL, spatial + elimination + temporal for Fortnite) that a single T may compress too aggressively.

**Follow-up:** ME-FN-1 (build-cost constraint variant), ME-RL-1 (per-player chains), and ME-RL-2 (possession-level vs play-level) are pre-registered for v6 consideration.

---

## Decision D-34: CF-4=B actor anonymisation in prompt rendering

**Date:** 2026-04-27 (bug sweep post Phase B)

**Context:** Bug sweep Layer 2 found that `PromptBuilder.format_event()` was emitting the raw `actor` field (e.g. `"LeBron_James"`, `"player_12345"`) verbatim in both baseline and intervention prompts. CF-4=B ("domain-only provenance") requires that no player or team identity appear in the prompt — the model must reason from constraint rules alone, not from domain knowledge about specific athletes.

**Decision:** Anonymise actors to chain-local positional slots: first actor seen in the chain becomes `Player_0`, second becomes `Player_1`, etc. The mapping is per-chain (not per-game) so different chains do not share slot assignments.

**Implementation:** `PromptBuilder.format_chain()` builds the `actor_map` dict and passes it to `format_event(actor_map=...)`. A module-level `_build_actor_map()` helper keeps the logic reusable. Location context (x/y coordinates, round numbers, period numbers) is retained — those are structural game state, not player identity.

**Code impact:** `v5/src/harness/prompts.py` — `format_chain`, `format_event`, `_build_actor_map` updated. Two tests added to `test_prompts.py` verifying absence of real names and presence of anonymised slots.
