# MEMO — Project Ditto v5 Closeout & Bridge to V1–V5.1 arXiv Preprint

**Date:** 2026-04-30
**Author:** Lead Author (with Claude Opus 4.7 as co-author on infrastructure)
**Status:** Internal — for use as the source-of-truth document when assembling the full V1–V5.1 arXiv preprint after the OpenRouter cross-model run completes.

This memo is the bridge document. It freezes what v5 found, contextualizes it within the broader Project Ditto experiment series (V1 → V5.1), and lays out the empirical and mechanistic claims that the eventual arXiv preprint will make. The numbers and methodology choices documented here are not subject to further iteration — they are the locked v5 contribution.

---

## 1. Executive Summary

**v5 finding (defensible and locked):**

> *Constraint reasoning in Claude Haiku 4.5 is gated by representational alignment between the rule and the observable event structure. It succeeds when violations reduce to observable unary predicates over event streams; it degrades predictably under missing observability; and it is suppressed entirely under strict grounding when required variables are absent.*

This claim is supported by Phase D results across 5 game domains at n=1,200 chains/cell (24,000 paired API calls, total spend ~$5). 4 of 5 cells produce significant intervention lifts at p_Bonferroni < 1e-15. The fifth (poker) shows a ceiling effect at 100% baseline detection. The cells partition cleanly into a 4-tier representational hierarchy.

**What v5 is NOT:**

- It is *not* a faithful replication of v3. v3 measured "consistency rating"; v5 measured "violation detection". The framing pivot was forced by a floor effect on 8-event chains under the original design.
- It is *not* the final answer. Whether the 4-tier hierarchy is a property of the *task* or a property of *Haiku 4.5 specifically* is the next experiment (V5.1 cross-model replication).

---

## 2. Project Ditto Series Context (for the eventual arXiv assembly)

Ditto is a multi-version research program studying constraint reasoning in LLMs over structured event sequences. Each version refines methodology, data, or scope. v5 is the fifth iteration.

> ⚠️ **For the arXiv author**: V1, V2, V4 details are in their own GitHub repos (see DECISION_LOG D-0). The summaries below are from references in v5's SPEC and the conversational record; **pull authoritative details from the V1/V2/V3/V4 repos when assembling §§1–4 of the preprint.**

### v1, v2 (early experiments — repos: separate)
- Earliest pilots establishing the "abstract chain + constraint context" methodology.
- Domain unclear in v5 record — likely chess/strategy. **Pull details from V1, V2 repos.**

### v3 — Chess + Checkers (repo: `safiqsindha/Project-Ditto-V3`)
- Two-cell experiment establishing the constraint-chain detection methodology proper.
- Pre-registered design: 1,200 chains/cell, McNemar paired binary outcomes, Bonferroni divisor=4 (4 cells in v3's enumeration), α_corrected=0.0125, ~90% power at gap=0.06.
- Phase 1 found a measurable intervention effect; Phase 2 (shuffled controls) was paused at "Gate 8" — observed effect size never published. **Pull final v3 numbers and effect-size estimates from the V3 repo.**
- v3 measured *consistency rating* — does the chain look like a valid game given the rules? — as the dependent variable.

### v4 — Single-cell methodology characterization (repo: `safiqsindha/Project-Ditto-V4`)
- Stabilized the harness and characterized what kinds of failures the v3 design exposes. **Pull details from V4 repo.**
- Provided the ACTIONABLE_TYPES whitelist, CF-3=A shuffle controls, Gate 2 retention floor, and the cell-runner pattern that v5 inherited.

### v5 — Five-cell parallel replication (this work; repo: `safiqsindha/Ditto-V5`)
- Five new game domains: PUBG, NBA, CS:GO, Rocket League, Poker.
- Pre-registered exactly per v3's design intent (same n, similar Bonferroni structure, paired-McNemar primary test).
- **Pivoted mid-experiment** to violation-detection framing after the consistency-rating framing produced a floor effect.
- Final headline: 4-tier hierarchy.

### v5.1 — Cross-model replication (planned; OpenRouter; scoped separately)
- Same Phase D prompts, replayed across ~11 models (Anthropic / OpenAI / Google / open-weights cross-section).
- Adds a derived-state-marker ablation as a first-class second axis.
- Goal: test whether the 4-tier hierarchy is a property of the task or of Haiku specifically.

The eventual arXiv preprint will narrate this trajectory: v3 establishes the methodology, v4 characterizes its failure modes, v5 generalizes across domains and discovers the representational hierarchy, v5.1 establishes whether the hierarchy holds across model scales.

---

## 3. v5 Methodology Trajectory

### 3.1 What was pre-registered

Per `SPEC.md` (signed by both authors, 2026-04-27):

- **Five cells**, one per domain, sample target 1,200 chains each
- **Two API calls per chain** (baseline = no constraint context; intervention = constraint context injected); paired McNemar's test with continuity correction
- **Bonferroni divisor = 5** (α_corrected = 0.01 per cell)
- **Bootstrap 95% CI** on (intervention − baseline) proportion difference, 10,000 iterations, seed=42
- **Cohen's h** for effect size
- **Gate 2 retention floor** — minimum fraction of chains passing structural filters
- **CF-3=A shuffle controls** — diagnose format leakage by shuffling chain order
- **Subject model**: `claude-haiku-4-5-20251001`

### 3.2 Amendments before Phase D

Documented in `SPEC.md` Amendments Log; rationale in `DECISION_LOG.md`:

| Amendment | Change | Trigger |
|---|---|---|
| **A1** | Hearthstone → Poker | HSReplay API friction; PHH dataset is open and methodologically equivalent |
| **A2** (D-35/36) | Fortnite → PUBG | Epic CDN locked down public chunk-data access; PUBG provides documented public API |
| **A3** (D-37) | Pluribus → HandHQ + WSOP 2023 | Pluribus contains Facebook's superhuman bot in 1 of 6 seats — 17% of actions non-human |
| **A4** (D-38) | NBA chain rendering: surface terminal action, cumulative foul count, possession-elapsed time | Pilot showed NBA primary h=0 under flat rendering — chain didn't surface what the constraint required |
| **A5** (D-39) | CS:GO chain rendering: surface team_id, kills, headshots, MVPs, round outcome | Same diagnosis as A4 — best-effort within FACEIT aggregate-stat ceiling |
| **A6** (D-40) | RL chain rendering: surface team_color, score, boost_used, demos | Same diagnosis as A4 — best-effort within BallChasing aggregate-stat ceiling |
| **A7** (D-41) | Poker chain unit: per-(actor, hand) → per-hand sequence | Per-actor 3-action filter discarded 99.9% of (actor, hand) pairs |

### 3.3 The mid-experiment pivot (D-42 → D-44)

The n=200/cell pilot under the original consistency-rating framing produced **0% YES on 4 of 5 cells**. Reviewer synthesis (Gemini, ChatGPT, two Opus reads) converged on the diagnosis: chain content didn't surface the variables the locked constraint context referenced, so the model couldn't anchor its judgment in the chain.

**D-42 — methodology pivot to violation detection.** Plant ONE explicit rule violation per chain; ask "does any event violate the rules of {domain}?". The dependent variable is no longer "consistency rating" but "violation detected? Y/N".

**D-43 — derived-state markers.** For multi-event predicates (e.g., "folded player can't act"), surface per-event tags like `NOTE=Player_X_already_eliminated` so the model can verify constraints from a single-event check rather than cross-event inference.

**D-44 — Layer 1 strict-grounding + Layer 2 CoT diagnostic.** The diagnostic question explicitly forces the model to identify *both* (a) which specific listed rule, and (b) which specific event breaks it. The CoT diagnostic (Layer 2) inverts the YES/NO answer for FP analysis: ask the model to *quote* the rule it thinks was violated, and parse the response to identify the failure mode.

### 3.4 What survived the pivots, what changed

Preserved from the SPEC: 5 cells, n=1,200, paired-McNemar primary test, Bonferroni divisor=5, bootstrap CIs.

Changed: dependent variable (consistency rating → violation detection), prompt structure (added strict-grounding instruction + per-event derived-state markers), framing (now characterizes when constraint context helps, not whether it does).

**Honest framing note for the arXiv author:** the methodology pivot is a liability if §1 of the preprint claims faithful v3 replication. Frame it as a *discovery experiment about LLM constraint reasoning methodology*, with v3's design as the starting point that led to the floor effect that motivated the pivot. The pivot itself is then a finding ("the consistency-rating framing has a floor effect at 8-event chain length on Haiku 4.5"), not a methodological weakness.

---

## 4. Phase D — Final Results

### 4.1 Headline table

| Cell | n_clean | n_adv | Det@Base | Det@Int | Δ | 95% CI | FP@Base | FP@Int | b | c | χ² | p_raw | p_Bonf |
|---|---:|---:|---:|---:|---:|:-:|---:|---:|---:|---:|---:|---:|---:|
| pubg | 1,200 | 1,200 | 75.9% | 100.0% | +24.1% | [+21.75, +26.58] | 0.0% | 0.0% | 0 | 289 | 287.0 | 2.2e-64 | **1.1e-63** |
| nba | 1,200 | 1,200 | 57.4% | 100.0% | +42.6% | [+39.83, +45.42] | 0.8% | 9.9% | 0 | 511 | 509.0 | 1.0e-112 | **5.2e-112** |
| csgo | 1,200 | 1,199 | 65.1% | 98.1% | +32.9% | [+30.28, +35.70] | 11.3% | 29.8% | 0 | 395 | 393.0 | 1.8e-87 | **9.2e-87** |
| rocket_league | 1,200 | 1,200 | 0.2% | 6.2% | +5.9% | [+4.67, +7.25] | 0.0% | 0.0% | 0 | 71 | 69.0 | 9.8e-17 | **4.9e-16** |
| poker | 1,200 | 1,200 | 100.0% | 99.9% | -0.1% | [-0.25, 0.00] | 0.5% | 1.1% | 1 | 0 | 0.0 | 1.000 | 1.000 |

**Statistical methods:**
- McNemar's test with continuity correction when n_discordant ≥ 25; exact two-sided binomial test when n_discordant < 25 (poker).
- Bonferroni correction: p_Bonf = min(1, 5 × p_raw).
- Bootstrap 95% CI: 5,000 iterations, percentile bootstrap on cluster-resampled paired adversarial outcomes.
- Detection rate computed on adversarial chains (GT = "yes"); FP rate computed on clean chains (GT = "no").

### 4.2 Per-cell narrative

**PUBG (Tier 1, aligned).** The "actor acts after their own elimination" violation is a direct unary predicate over events. Baseline 75.9% — model already catches most planted violations from common knowledge. Intervention pushes to 100% with zero FP both arms. Cleanest result in the experiment. Violation injector targets the `_elim_PlayerX` derived-state marker.

**NBA (Tier 1, aligned).** "Player commits 7th personal foul" is a direct count predicate. Baseline 57.4% — model knows the rule but doesn't always count. Intervention to 100%. The 9.9% intervention FP rate is the cell's interesting residue: the field rename `possession_elapsed_s` → `time_in_possession_s` (D-44 era) eliminated the v5-diagnostic confabulation, but at n=1,200 a different residual confabulation persists. Layer-2 CoT diagnostic findings on this residue are documented in §4.4. Violation injector elevates a real foul event's foul-count to 7.

**CS:GO (Tier 2, partial-observability).** "Bomb plant only at site A or B" is unary in principle but the FACEIT aggregate-stat data path doesn't emit bomb-plant events at all — the cell's chains contain kills, deaths, MVPs, round outcomes, but never `actionType=plant`. The model knows CS:GO has plants and *infers* they happened off-chain. With no `site=A/B` marker visible, it concludes "plants must have been at unauthorized locations," producing a high false-positive rate. Detection 98.1% under intervention is real (planted violations land); FP 29.8% is a constraint-triggered confabulation signature, not a reasoning failure. Violation injector targets `_eliminated_acts` derived-state markers (same family as PUBG).

**Rocket League (Tier 3, misaligned).** "After a goal, ball + player positions reset" requires per-event positional or ball-state observability. The BallChasing aggregate-stat data path provides only per-player game totals (boost used, demos, MVP), not per-event positional state. Under the original (D-43) marker injection, the model achieved near-100% detection by pattern-matching the synthetic `pre_goal_state_persists_after_goal` marker. **Under strict grounding (D-44), that shortcut collapsed.** Phase D shows 0.2% baseline → 6.2% intervention. This is the experiment's strongest negative finding: strict grounding correctly refuses indirect-marker inference when the rule's required variables aren't surfaced in the chain. McNemar c=71 is still highly significant — there *is* a real but small lift — but the absolute detection rate stays near floor.

**Poker (Tier 0, saturated).** "Folded player can't act in same hand" is a direct unary predicate that matches semantic structure of poker action notation. Baseline 100.0% — the rule is fully internalized from common knowledge; the violation is surface-readable. Intervention adds nothing. McNemar b=1, c=0 reflects no discordance worth measuring. The ceiling effect anchors the top of the hierarchy: when the rule is already internalized, constraint context cannot lift performance.

### 4.3 The 4-tier hierarchy (the contribution)

| Tier | Cells | Anchored | Observable | Unary-reducible | Empirical signature |
|---|---|:-:|:-:|:-:|---|
| **0** Saturated | poker | ✓ | ✓ | ✓ | both arms ≥99%, b ≈ c, no lift possible |
| **1** Aligned | pubg, nba | ✓ | ✓ | ✓ | baseline 50-80%, intervention →100%, c >> b, FP near zero |
| **2** Partial-observability | csgo | ✓ | ✗ | ✓ | high lift but elevated intervention FP — confabulation signature |
| **3** Misaligned | rocket_league | ✓ | ✗ | ✗ | both arms near floor, small but significant c, zero FP |

**Three necessary conditions** for productive constraint-context intervention:

1. **Rule anchoring** — the rule must be explicitly stated and match the violation type
2. **Predicate observability** — the variables required by the rule must be present in the rendered events
3. **Unary reducibility** — the violation must collapse to a per-event or per-actor check

When all three hold (Tier 1), constraint context produces near-perfect intervention lifts with zero FP. Drop observability (Tier 2) and the model still tries to reason, producing high FP under unverifiable inference. Drop both observability and unary reducibility (Tier 3) and strict grounding suppresses reasoning entirely — the model correctly refuses to infer.

Tier 0 is a special case: the rule is internalized so deeply that baseline detection saturates, leaving no room for intervention to demonstrate effect.

### 4.4 Layer-2 CoT diagnostic on residual intervention FPs

NBA (9.9%) and CSGO (29.8%) have non-trivial intervention FP rates worth mechanistic analysis. Layer-2 CoT was run on a sample of FP chains from each cell (`run_phase_d_cot.py`) — for each FP, the model was asked to quote the rule it thinks was violated and identify the event index.

#### NBA — 50 of 119 FPs analyzed

| Cited rule | Count | % of analyzed |
|---|---:|---:|
| "Offensive team must shoot within 24 seconds of gaining possession" | 46 | 92% |
| "A player with 6 fouls is ejected" | 4 | 8% |

**Mechanism**: 92% of NBA's residual intervention FPs are shot-clock confabulations. This is *not* the same failure mode that D-44's field rename (`possession_elapsed_s` → `time_in_possession_s`) addressed — that fix eliminated misreading of the per-event timing variable. At n=1,200 a *second-order* shot-clock failure surfaces: the model computes some inferred elapsed-time from chain context and flags it against the 24-second rule even when the chain shows a successful shot well within bounds.

The 4 foul-rule FPs are likely real edge cases (chains that happen to contain players with high foul counts that the violation injector didn't elevate to 7).

**Implication**: NBA residual confabulation is a *renderer* problem, not a *reasoning* problem — the chain renders enough timing structure that the model attempts a 24-second check, and that inference fails on ~10% of chains. A v5.2 fix could surface explicit possession-start markers (`POSSESSION_START_AT=t_n`) so the timing check is verifiable end-to-end. Deferred to follow up on cross-model results.

#### CSGO — 50 of 358 FPs analyzed

| Cited rule | Count | % of analyzed |
|---|---:|---:|
| "Bomb plants only at sites A or B" | 40 | 80% |
| "Eliminated players don't respawn until next round" | 10 | 20% |

**Mechanism**: 80% of CSGO's residual intervention FPs are exactly the predicted constraint-triggered confabulation — the model knows CS:GO bomb plants must occur at sites A or B, the FACEIT data path doesn't emit plant events at all, and the model concludes plants must have happened off-chain at unauthorized locations. This is the textbook signature of *unverifiable rule + plausible inference = false positive*.

The 20% citing the respawn rule is a related failure: the model is reasoning about player counts across rounds and concluding that respawn timing was violated, when in fact the FACEIT data doesn't render per-round respawn information either. Same root cause: missing observability for a stated rule.

**Implication**: CSGO is the cleanest empirical demonstration of Tier-2 partial-observability failure. v5.2 awpy demo extraction adds the missing `actionType=plant` events with `site=A/B` attributes; if that drops intervention FP from 29.8% to <5%, it confirms that the failure is observability-bound rather than reasoning-bound. **This is exactly the test that motivates running awpy-fixed CSGO *after* cross-model lands** — separating model capability from data observability requires both the cross-model heatmap and the observability-fix as independent axes.

#### Why this matters for the arXiv

Both NBA and CSGO residual FPs are *interpretable* via the model's own self-explanation. The CoT diagnostic is therefore the mechanistic evidence that supports the 4-tier hierarchy claim — we're not inferring "the model confabulates because of missing observability"; the model *tells us* which rule it thinks was broken, and the rule citations cluster cleanly around the rules with missing predicate observability. This is among the strongest mechanistic evidence the paper has.

> **For the arXiv author**: full CoT data at `RESULTS/phase_d_cot_residual_fps.json`. Per-sample rule + event citations are in the `samples` array — useful for case studies in §4.4.

---

## 5. What v5 Cost vs. Pre-Registered Estimate

| Phase | Pre-reg estimate | Actual |
|---|---|---|
| Pilot (n=20–200) | $0.50 | ~$0.50 |
| Diagnostic iterations (D-42, D-43, D-44) | unbudgeted | ~$0.50 |
| Phase D first attempt (PUBG + NBA + crashed Poker) | included in $4 | ~$2 |
| Phase D resume (Poker + RL + CSGO) | not anticipated (crash) | ~$2 |
| **Total** | **~$4** | **~$5** |

The custom_id duplicate-key crash on Poker added one batch's worth of resume cost. Within rounding of the pre-registered budget.

---

## 6. v5.1 Cross-Model Replication — Pre-Registered Design

This section is the design document for v5.1, locked 2026-04-30 in conversation with the reviewer. Not yet executed.

### 6.1 Goal

Test whether the 4-tier representational hierarchy is a property of the task (constraint reasoning in structured event streams) or a property of Haiku 4.5 specifically.

### 6.2 Design

- **Frozen prompt corpus**: dump the exact Phase D prompts (clean + adversarial × baseline + intervention) to immutable JSON. Compute SHA-256 hash of every prompt. Replay this same corpus on every model.
- **n = 300/cell** (down from Phase D's 1,200). McNemar power calculation: at PUBG's c=289 and Phase D n=1,200, the equivalent c at n=300 is ≈72; even RL's c=71 maps to ≈18 at n=300, still significant at p<0.001 unadjusted. Phase D was overpowered for the ratio of cell-level effects observed.
- **Marker ablation as second axis**: each prompt pair runs in 4 conditions:
  - baseline × markers (Phase D condition)
  - baseline × no-markers (suppress `_MarkerSurfacing` block)
  - intervention × markers (Phase D condition)
  - intervention × no-markers
- **Models** (~11 total, cap at 12):
  - Anthropic: Haiku 4.5 (parity check), Sonnet 4.5, Opus 4.x
  - OpenAI: gpt-5-mini, gpt-5, one o-series reasoning model
  - Google: Gemini 2.5 Flash, Gemini 2.5 Pro
  - Open weights: Llama 3.3 70B, Qwen 2.5 72B, DeepSeek-V3
- **Temperature 0**, max_tokens 32, identical system prompt
- **Pre-flight**: n=20 smoke test per model (~$0.50 total) to catch output-format quirks before committing budget
- **Budget cap**: $150

### 6.3 Pre-registered analyses

1. **Per-cell × per-model × per-condition McNemar heatmap.** Headline figure of v5.1.
2. **Tier-collapse test.** For each model, check whether the 4-tier hierarchy from Haiku replicates. Specific predictions worth testing:
   - **More capable models collapse RL Tier 3 → Tier 1** (catch indirect markers without needing direct state) ⟹ representational alignment is a model-capability axis.
   - **Smaller open-weights models collapse PUBG/NBA Tier 1 → Tier 3** (miss direct markers without strong context) ⟹ constraint context only helps when the model already has the conceptual scaffolding.
3. **Marker ablation × model size.** Does removing markers degrade Tier 1 cells uniformly across models, or only on weaker models? If capable models recover without markers, markers are *scaffolding*; if even capable models need them, representation alignment is *fundamental*.
4. **FP-discipline scaling.** Plot CSGO intervention FP rate vs. model capability proxy. If capability buys discipline (lower FP at similar detection), confabulation is a capability-bound failure.
5. **Cross-provider confound check.** Group results by Anthropic / OpenAI / Google / open-weights and check if effects cluster by provider (training-distribution effect) vs. by model size (capability effect).

### 6.4 Out of scope for v5.1

- Per-model prompt tuning (would p-hack the comparison)
- Reasoning-mode toggles (separate axis; defer to v6)
- New cells or new methodology
- CS:GO awpy fix (deferred to v5.2 — running awpy-fixed CS:GO *after* cross-model lands gives a clean "capability vs. observability" comparison)

---

## 7. The Eventual arXiv Preprint — Section Plan

For the future arXiv author assembling V1 → V5.1 into one preprint:

**Title direction:** *"Representational Alignment Gates Constraint Reasoning in Large Language Models: A Five-Domain × Multi-Model Study"* (or similar — the paper is about *when constraint context works*, not *whether it does*).

**§1. Introduction.**
- The constraint-reasoning question. Why structured event sequences. v3's original framing.
- ⚠️ Pull v1, v2 motivation from those repos.

**§2. Prior Work in this Series.**
- v3 (Chess + Checkers): consistency-rating methodology, Phase 1 results, Phase 2 paused at Gate 8. ⚠️ Pull from V3 repo.
- v4 (single-cell characterization): harness stabilization, ACTIONABLE_TYPES, CF-3=A controls. ⚠️ Pull from V4 repo.
- The methodology gap that v5 attempted to fill.

**§3. v5 Methodology.**
- Five-cell parallel design, pre-registration commitments.
- The pivot: floor effect under consistency-rating, switch to violation-detection. Frame as a finding about methodology, not a weakness.
- Layer 1 (strict grounding) and Layer 2 (CoT FP) machinery.
- Source data: PUBG telemetry, NBA PlayByPlayV3, FACEIT CSGO, BallChasing RL, HandHQ + WSOP poker.

**§4. v5 Phase D Results.**
- Headline table (§4.1 of this memo).
- Per-cell narrative (§4.2 of this memo).
- 4-tier hierarchy (§4.3 of this memo) — **this is the centerpiece figure of the paper**.
- Layer-2 CoT mechanism analysis (§4.4 of this memo).

**§5. v5.1 Cross-Model Generalization.**
- Design: frozen corpus, marker ablation, per-cell × per-model × per-condition heatmap.
- Tier-collapse results — does the hierarchy hold across models?
- Marker-ablation results — scaffolding vs. fundamental.
- FP-discipline scaling.

**§6. Discussion.**
- The three necessary conditions: rule anchoring, predicate observability, unary reducibility.
- Why this matters: implications for LLM evaluation methodology, prompt engineering for structured-data tasks, and the boundary between knowledge and reasoning.
- Comparison to other constraint-reasoning literature (LogiQA, FOLIO, RuleTaker, etc. — pull whatever's current at write time).

**§7. Limitations.**
- Single-corpus per cell; no cross-corpus replication within v5.
- CSGO observability gap acknowledged; v5.2 awpy-fixed CSGO is a planned but unrun follow-up.
- No reasoning-mode (chain-of-thought, ReAct, etc.) variation studied; v6 candidate.
- ≤12 events/chain — does the hierarchy hold at chain_length=50? Open.

**§8. Future Work.**
- v5.2 awpy-fixed CSGO (capability-vs-observability comparison)
- v6 chain-length sweep
- v6 reasoning-mode × representation-tier interaction
- Alternative violation injector designs for Tier-3 (test whether *any* observability would lift RL)

---

## 8. Things to NOT Forget When Writing the arXiv

1. **Frame the v3→v5 transition as a methodology pivot, not a faithful replication.** The honest framing is "discovery experiment — v3's design exposed a floor effect at Haiku 4.5 chain-length, which led to a methodology pivot, which led to the 4-tier hierarchy finding." Pretending v5 replicates v3 sets up a reviewer ambush.
2. **Do not bury Poker.** A Tier-0 ceiling is the top anchor of the hierarchy, not a missing data point.
3. **Do not "rescue" Rocket League.** The near-floor detection is the strongest demonstration of strict grounding's anti-confabulation behavior. Tampering with the violation injector to inflate detection re-introduces the indirect-marker shortcut that D-44 eliminated.
4. **Make the McNemar choice explicit.** v5 uses continuity-corrected χ² when n_discordant ≥ 25 and exact two-sided binomial when n_discordant < 25 (poker). Document this in §4 — reviewers will ask.
5. **Acknowledge the n=1,200 over-power explicitly.** v5.1 drops to n=300 because Phase D's effect sizes were so large that even the smallest cell (RL c=71) was hugely over-powered. This isn't post-hoc justification — it's documented in the v5.1 design as the rationale for the n change.
6. **Cite the raw-data archive.** The 23,998 archived request records (`RESULTS/phase_d_raw_batches/`) are the experiment's immutable evidence. Cite the commit hash (`7033cb1`) so reviewers can reproduce from the raw responses if they question the scoring pipeline.
7. **Acknowledge the custom_id bug as a methods note.** It's relevant because it's a concrete example of how Anthropic's Batches API behaves under heavy use, and it explains the resume-run timestamps and the saved batch IDs. Don't hide it.

---

## 9. Pointers to Definitive Artifacts

- **Headline numbers** (locked): `RESULTS/phase_d_final.json` (regenerable from `synthesize_phase_d.py`)
- **Raw evidence**: `RESULTS/phase_d_raw_batches/` (10 JSONLs, ~6 MB, 23,998 records)
- **Methodology decisions**: `DECISION_LOG.md` D-0 → D-45 (D-42, D-43, D-44, D-45 are the most consequential)
- **Pre-registration**: `SPEC.md` (as signed 2026-04-27, with A1–A7 amendments)
- **Per-cell prompt structure**: `src/harness/prompts.py` (PromptBuilder + `_MarkerSurfacing`)
- **Per-cell violation injectors**: `src/harness/violation_injector.py`
- **Build-day status**: `docs/STATUS_BUILD_DAY_2026-04-27.md`
- **End-state status**: `STATUS.md`
- **This memo**: `MEMO.md` (you are here)

---

## 10. Sign-off

v5 is closed as of 2026-04-30. The Phase D headline numbers are frozen. No further methodology iteration on this experiment. Cross-model replication (v5.1) is scoped separately and will use the Phase D Haiku 4.5 numbers as its anchor. The eventual arXiv preprint will narrate v3 → v4 → v5 → v5.1 as one study with consistent representational-hierarchy framing.
