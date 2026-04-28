# v5 Micro-Experiments — Flagged for Post-Hoc Investigation

**Status:** Pre-noted, NOT pre-registered for v5 primary run. These are variants the lead author flagged as "interesting if v5 results justify follow-up."

**Critical rule:** None of these may be implemented as part of the v5 primary run. They are post-v5 work, dependent on v5 results and a separate pre-registration sign-off if pursued.

---

## ME-1: Variable Chain Length (Q4-C)

**Source:** Q4 sign-off — locked at (B) fixed per cell; (C) flagged as variable-bounded chain length variant.

**Premise:** The primary v5 fixes chain length per cell. ME-1 would re-run v5 with chain length variable in [3, 15] events, T choosing per-chain.

**Contingency:** Run only if v5 primary results show significant per-cell effects in ≥ 3 of 5 cells. If primary results are null, ME-1 doesn't add interpretive value.

**Cost estimate:** Same Haiku spend as v5 Phase 1 (~$4) since chain count and number of API calls is unchanged.

**Pre-registration required:** Yes, before any data analysis on this variant.

---

## ME-2: CS:GO Tick-Sampled Granularity (Q5-C)

**Source:** Q5 sign-off — locked at (A) round-level for primary; (C) tick-sampled flagged.

**Premise:** Primary uses round-level events (~300/map). ME-2 would re-acquire CS:GO data at tick-sampled granularity (every 256 ticks ≈ 2 seconds, ~6,750 events/map) and re-run.

**Contingency:** Run only if (a) primary CS:GO cell shows a significant effect, AND (b) we want to test whether finer granularity preserves or strengthens the effect.

**Cost estimate:** Re-acquisition is free (same HLTV demos); re-evaluation = ~$1 in Haiku calls.

**Pre-registration required:** Yes.

---

## ME-3: NBA Play-Level Granularity (Q6-B)

**Source:** Q6 sign-off — locked at (A) possession-level for primary; (B) play-level flagged.

**Premise:** Primary groups plays into possessions (~85 possessions/game × 2-3 events). ME-3 would use raw play-level (~220 events/game).

**Contingency:** Run only if primary NBA cell shows a significant effect AND we want to compare granularities. Conversely, if primary NBA cell is null, ME-3 might be worth running to test whether granularity choice is the cause.

**Cost estimate:** ~$1 re-evaluation.

**Pre-registration required:** Yes.

---

## How These Get Promoted

If v5 primary results justify a micro-experiment:
1. Update this document with the specific question being tested.
2. Pre-register the micro-experiment in a new SPEC version (`v5/SPEC_v0.3_ME-N.md`).
3. Implement the variant in a side-branch (`claude/me-{N}-{shortname}`).
4. Run, score, report — same harness, just the parameter swapped.
5. Cross-reference results in v5/RESULTS/ME_{N}_results.md.

---

## ME-4: CF-1 Variants — Formal Predicate (B) and In-Context Examples (C)

**Source:** Phase B session — CF-1 locked at A (natural language); B and C noted as interesting.

**Premise:** Would a formal predicate list (e.g., "Invariant 1: eliminated player event_count = 0") or in-context examples of constraint violations improve detection accuracy over natural-language description?

**Contingency:** Run if v5 primary shows ≥ 3/5 significant cells. Variant compares A vs B vs C on the same chains.

**Pre-registration required:** Yes.

---

## ME-5: CF-2 Variants — Prediction Targets A, B, C

**Source:** Phase B session — CF-2 locked at D (binary); A/B/C noted.

**Premise:**
- ME-5A: Predict next event type (A) — tests sequential constraint knowledge
- ME-5B: Predict actor of next event (B) — tests player-tracking constraint knowledge
- ME-5C: Predict domain outcome (C) — tests terminal constraint prediction

**Contingency:** Run if v5 primary D (binary) shows ≥ 3/5 significant. Variants test whether detection generalizes across prediction formulations.

**Pre-registration required:** Yes per variant.

---

## ME-6: CF-4 Variant — Anonymous Chains (A)

**Source:** Phase B session — CF-4 locked at B (domain label only); A noted.

**Premise:** Does removing the domain label ("this is a CS:GO chain") change detection accuracy? Would the model rely purely on structural sequence patterns?

**Contingency:** Run in any cell that shows a significant effect. Low cost — only prompt format changes.

**Pre-registration required:** Yes.

---

## ME-RL-1: Rocket League Per-Player Chains

**Source:** Phase B session user suggestion — "could you make chains per player?"

**Premise:** Instead of per-play chains (team-level), extract per-player chains: for each player, extract their actions within a play (hits + boost decisions). This tests whether individual player boost-economy detection is stronger than team-level.

**Contingency:** Run if RL primary shows significant effect. Per-player variant is the main v6 Rocket League T design candidate (D-33).

**Cost estimate:** ~3× chains per play (3 players); ~3× API cost for RL cell only (~$0.80 extra).

**Pre-registration required:** Yes.

---

## ME-FN-1: Fortnite Build-Cost Constraint

**Source:** Phase B F-1 decision — build-cost rule noted for future testing.

**Premise:** Add "building consumes one material" as a testable constraint. Generate build-decision sub-chains and test whether the model detects phantom builds (build without material deduction).

**Contingency:** Run if primary Fortnite shows significant effect AND we have build-event-level data from the FortniteReplayDecompressor.

**Pre-registration required:** Yes.

---

## ME-RL-2: Rocket League Possession-Level vs Play-Level

**Source:** D-33 — RL T variability noted as high.

**Premise:** Re-run RL with possession-level chains (Q7-B, not currently primary) to compare against play-level (Q7-C, primary). Tests whether boost-economy detection is stronger at possession or play granularity.

**Pre-registration required:** Yes.
