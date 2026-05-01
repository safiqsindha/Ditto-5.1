# PRE_MORTEM.md

**Project Ditto v5.1 — Risk Register**

**Filed alongside SPEC.md before any model is hit at full-run scale.**

---

## Purpose

This is the full risk register for v5.1. It expands the headline 10 risks in SPEC §15 with trigger conditions, detection signals, mitigations, and escalation paths. It exists so that risks are surfaced *prospectively* rather than discovered during the run, and so that response to risk events is pre-decided rather than improvised under time pressure.

This document is filed on OSF alongside SPEC.md as evidence of pre-registration discipline. It does not modify the SPEC. If a risk event triggers a methodology change, both authors must re-sign the SPEC before resuming.

---

## Risk register format

Each risk includes:

- **Trigger**: what conditions would cause this risk to materialize
- **Likelihood**: low / medium / high (subjective, pre-data)
- **Impact**: cost / methodology / scope (and severity within each)
- **Detection**: how we will know the risk has materialized
- **Mitigation**: pre-registered response if detected
- **Escalation**: when both authors must convene before continuing

---

## 1. Pre-flight phase risks (Tasks 1–4 in BUILD_PLAN)

### R-PF-1. Harness usage-logging fix introduces a regression in v5's existing pipeline

**Trigger**: Task 1 modifies `src/harness/model_evaluator.py`. If the modification breaks v5's existing tests or changes the response-extraction behavior in subtle ways, downstream code that consumes `EvaluationResult` may behave differently than v5.

**Likelihood**: medium. The change is small but touches a hot path.

**Impact**: methodology — could invalidate v5's reproducibility if existing tests no longer pass. Cost: trivial.

**Detection**: v5's existing test suite must pass after Task 1. If any existing test fails, the regression is real.

**Mitigation**:
- Run v5's full test suite as the acceptance gate for Task 1
- Compare a 100-call sample of Task 1's output against v5's archived JSONL on identical inputs; outputs must be byte-identical except for the new usage fields
- Both-author code review before Task 1 commits

**Escalation**: any v5 test failure pauses Task 1 until both authors review.

### R-PF-2. Provider pinning unavailable for some OR-routed models

**Trigger**: `/v1/models/{id}/endpoints` returns only one provider for a model, or returns providers with very different quantizations (e.g., only fp8 and int4 available, no fp16).

**Likelihood**: medium. Some Chinese-origin models may have only one upstream provider on OR.

**Impact**: scope — if a model has no high-quantization provider available, results from that model may not be comparable to other models. Methodology: documented but not invalidating.

**Detection**: Task 2 surfaces this when building `provider_pinning.json`.

**Mitigation**:
- For each affected model, log the available providers and their quantizations in `provider_pinning.json`
- If only int4 quantization is available, flag the model as "low-quantization" in the panel and report results with that caveat
- If no provider is available at all (extremely unlikely), drop the model and document the drop with both-author sign-off

**Escalation**: dropping any panel model requires both-author sign-off and a numbered DECISION_LOG entry.

### R-PF-3. Cluster definitions don't extract cleanly from v5's chain_ids

**Trigger**: Task 3's regex extraction logic for source-game IDs fails on some chain_ids, especially adversarial-side `_adv` suffixes.

**Likelihood**: medium. The token audit caught this exact issue (`_adv` suffix) and adjusted; other edge cases may exist.

**Impact**: methodology — incorrect clustering biases CSGO and RL standard errors.

**Detection**: Task 3 acceptance criterion requires every CSGO chain to map to one of 364 cluster IDs and every RL chain to map to one of 148 cluster IDs. Coverage check catches misses.

**Mitigation**:
- Manual inspection of 20 random chain_ids per cell during Task 3
- Cluster size distribution from v5 audit (CSGO mean 3.3, max 8; RL mean 8.1, max 23) is the validation target — Task 3's output must match
- `clusters.json` is committed and hash-recorded in DECISION_LOG.md as D-45 before any analysis runs

**Escalation**: any mismatch between Task 3's cluster sizes and v5 audit's cluster sizes requires both-author review before locking.

### R-PF-4. Strict-grounding instruction wording leaks information about v5's hypothesis

**Trigger**: Task 4's strict-grounding text could theoretically prime the model toward a specific answer pattern.

**Likelihood**: low. The text is symmetric across cells (same instruction regardless of whether variables are present).

**Impact**: methodology — would confound the strict-grounding contrast.

**Detection**: 8-variant prompt verification on 10 random chains in Task 4. The strict variant must differ from non-strict only in the appended instruction; no other text changes.

**Mitigation**:
- Use the verbatim instruction text from BUILD_PLAN Task 4 (no paraphrasing)
- Acceptance check: token count of strict-variant prompts is exactly `non-strict + ~70 tokens` (the appended instruction)
- Manual inspection of 5 strict-vs-non-strict pairs to confirm only the instruction differs

**Escalation**: any structural difference beyond the appended instruction triggers redesign of Task 4.

---

## 2. Smoke test phase risks (Task 6 in BUILD_PLAN)

### R-SM-1. Models fail parseability on strict grounding specifically

**Trigger**: smoke test shows a model that parses fine on non-strict but fails parseability under strict grounding, particularly on RL or CSGO.

**Likelihood**: medium-high. Strict grounding is a more constraining instruction; some models may produce non-binary responses ("I cannot determine without more information") that fail first-token parsing.

**Impact**: scope — affects per-model results on the strict-grounding contrast specifically.

**Detection**: smoke test parse-rate-per-condition reporting. A model with >5% parse failures in strict-conditions but <5% in non-strict-conditions has a strict-grounding-specific issue.

**Mitigation**:
- Parse failure ternary outcome handling (SPEC §8) treats this as a logged outcome, not exclusion
- Dual-treatment reporting (SPEC §8.3) reports results both ways
- If aggregate parse rate <90%, model is excluded; if 90–98%, flagged
- Format-adherence-under-constraint becomes a secondary capability dimension

**Escalation**: if 4+ models fail strict-grounding parseability, pause and re-evaluate the strict-grounding instruction wording before full run.

### R-SM-2. Smoke cost exceeds prediction by >10%

**Trigger**: smoke test ground-truth cost (from API responses) exceeds predicted cost by more than 10%.

**Likelihood**: medium. Token audit was on v5's exact prompt; v5.1's longer prompts and additional axes could push tokens higher than predicted.

**Impact**: cost — full run could exceed $300 realistic ceiling.

**Detection**: Task 6 acceptance criterion: actual cost ≤110% of predicted cost.

**Mitigation**:
- If overage is in input tokens: recalibrate worst-case from 950 to measured-actual + 30% buffer
- If overage is in output tokens (cap-binding more often than predicted): consider raising 64-token cap to 96 for affected models
- If overage is on a specific reasoning model (256-cap binding fully): recompute that model's worst-case at measured rate
- Recompute full-run budget with new estimates; if new estimate >$300, escalate

**Escalation**: if recomputed full-run estimate exceeds $300 ceiling, both authors decide whether to (a) raise ceiling, (b) drop high-cost models, (c) reduce n on RL.

### R-SM-3. OpenRouter silent provider routing observed

**Trigger**: smoke test reveals that for some OR-routed models, `response["provider"]` does not match the pinned provider in some calls, despite `allow_fallbacks: false`.

**Likelihood**: low (the `allow_fallbacks: false` should prevent this), but high impact if it occurs.

**Impact**: methodology — confounds cross-model variance with provider variance.

**Detection**: Task 6 acceptance criterion explicitly checks every OR-routed model's resolved provider matches its pin across all 80 smoke calls.

**Mitigation**:
- Per-call resolved-provider logging in the harness (Task 1 + Task 5)
- If discrepancy detected, immediately halt the affected model, investigate
- If routing variance is unavoidable for a model (e.g., pinned provider went down), document and either (a) accept variance with caveat, or (b) drop that model

**Escalation**: any routing variance requires both-author review and DECISION_LOG entry before that model proceeds to full run.

### R-SM-4. Smoke fails on a tier required for within-provider hierarchy test

**Trigger**: smoke test results show that ≥1 model in a capability tier fails parseability, breaking that provider's ladder.

**Likelihood**: medium. Some smaller or older models may have lower parseability rates than expected.

**Impact**: methodology — within-provider hierarchy test (SPEC §7.1.3) requires intact ladders. A broken ladder reduces N from 6 to 5, raising the proportional bar for the conjunctive headline.

**Detection**: per-provider ladder integrity check after smoke results.

**Mitigation**:
- Replication threshold (SPEC §2.3) is *proportional* (5 of 6); if N drops to 5, threshold becomes 4 of 5
- If a critical ladder breaks (e.g., DeepSeek loses V3.2 — the cheap mid-tier anchor), document and accept reduced ladder
- Within-provider hierarchy test still runs with available ladder depth

**Escalation**: if ≥2 ladders break, headline conjunctive framing may need pre-publication adjustment; both authors decide.

---

## 3. Full run phase risks (Task 8 in BUILD_PLAN)

### R-FR-1. Per-model cost kill-switch trips mid-run

**Trigger**: a single model's cumulative cost exceeds 110% of its mean allocation during the full run.

**Likelihood**: medium for reasoning-mode models specifically (worst-case is 3-4× mean for those).

**Impact**: cost — depending on which model and how far over.

**Detection**: kill-switch checks cumulative cost every 1,000 calls per model.

**Mitigation**:
- Kill-switch pauses the affected model only, not the whole run
- Both authors review: continue if remaining model budget covers expected overage; drop if not
- If dropped, document in DECISION_LOG and report panel as 31 models in writeup
- If continued with raised allocation, draw from buffer (10% of mean budget)

**Escalation**: any kill-switch trip requires both-author sign-off before resuming that model. SPEC amendment required if drop affects within-provider hierarchy ladder (e.g., dropping GPT-5.5 leaves OpenAI at 4 tiers, still valid; dropping Sonnet 4.5 leaves Anthropic at 3 tiers, still valid).

### R-FR-2. DeepSeek run window is missed or misconfigured

**Trigger**: DeepSeek calls execute outside UTC 16:30–00:30 window despite scheduler.

**Likelihood**: low (the scheduler is a hard check), but high cost impact (~4× DeepSeek slice cost).

**Impact**: cost — DeepSeek slice goes from ~$3 mean to ~$12 mean.

**Detection**: every DeepSeek API call's timestamp is logged. Per-call cost from response is compared against off-peak vs. standard rates; mismatch indicates window violation.

**Mitigation**:
- Scheduler hard-check before every DeepSeek API call (BUILD_PLAN Task 5)
- Calls attempted outside window are deferred, not failed
- Run pauses overnight US-time and resumes next window
- Cost ledger flags DeepSeek calls that show standard pricing as anomalies

**Escalation**: any DeepSeek standard-pricing call triggers immediate halt of DeepSeek slice for both-author review.

### R-FR-3. Anthropic Batches API submission fails or times out beyond 24h SLA

**Trigger**: Anthropic's batch processing exceeds 24-hour SLA, or a batch upload fails entirely.

**Likelihood**: low. Batches typically complete within 2–6 hours; SLA failures are rare.

**Impact**: schedule — delays full run completion. Methodology: none if batch eventually completes; if batch fails, must rerun.

**Detection**: batch status polling every 30 minutes after submission. SLA timer at 24 hours.

**Mitigation**:
- Smaller batch sizes (e.g., 5,000-call chunks rather than full 10,000) reduce risk of single-batch failure
- Failed batches retry once automatically; second failure escalates
- If Anthropic SLA is breached, cancel batch and resubmit
- All Anthropic models can theoretically run synchronously as fallback (loses 50% Batches discount; ~$15 cost increase)

**Escalation**: 2+ batch failures on the same model triggers fallback to synchronous mode with both-author sign-off.

### R-FR-4. OpenAI/Google batch API behaves differently than documented

**Trigger**: OpenAI Batches or Google AI Studio Batch produces results with different rate limits, output structure, or cost calculation than documented.

**Likelihood**: low (these are mature APIs).

**Impact**: variable — could be cost, schedule, or methodology.

**Detection**: per-call usage logging (Task 1) compares actual vs. predicted cost. Output structure validation in Task 5 runner.

**Mitigation**:
- Smoke test catches structural differences before full run
- Per-call cost logging detects cost-calculation differences in real-time during full run
- Per-batch retry logic for transient failures

**Escalation**: persistent unexpected behavior triggers both-author review.

### R-FR-5. Reasoning models exceed 256-token cap on every adversarial-intervention call

**Trigger**: smoke test showed reasoning-model output cap-binding at 256 in some conditions.

**Likelihood**: medium-high. Reasoning models are designed to use thinking tokens.

**Impact**: methodology — output truncated mid-reasoning. Parsing logic uses first-token-after-whitespace, so this should not produce false negatives, but it could produce parse failures if the response structure breaks.

**Detection**: smoke test parse rate per condition for reasoning models specifically. Full-run parse failure log monitored in real time.

**Mitigation**:
- 256-cap is the budget for thinking + final answer combined. If responses truncate, the final answer may not appear
- Per-model parse-rate gate (90% aggregate, 98% per condition) catches systematic failures
- Worst-case scenario: a reasoning model produces 0 parseable responses and is excluded from panel (reduces panel to 31)
- Q12 dual-treatment reporting (failures-as-misses + failures-excluded) preserves analyzability for partial parseability

**Escalation**: if any reasoning model has aggregate parse rate <90%, exclude from panel.

### R-FR-6. Provider rate-limits unexpectedly during full run

**Trigger**: An OpenRouter provider, native API, or DeepSeek hits unexpected rate limits during full run, slowing or halting model progress.

**Likelihood**: medium. Especially for less-mature providers (Xiaomi, Z.AI, Moonshot via OR).

**Impact**: schedule — extends full run wall time.

**Detection**: 429 / rate-limit errors in API responses, logged per call.

**Mitigation**:
- Pre-registered exponential backoff with jitter on rate-limit errors
- Per-provider concurrency limits respected in BUILD_PLAN Task 5 runner
- Rate-limit budget allocation: each provider gets up to 4 hours of rate-limit-induced pause before escalation

**Escalation**: 4+ hours of cumulative rate-limit-induced pause for any single provider triggers both-author review.

### R-FR-7. Network or infrastructure failure mid-run

**Trigger**: harness machine reboots, network outage, or other infrastructure failure during full run.

**Likelihood**: low.

**Impact**: schedule — must resume run from checkpoint.

**Detection**: deterministic request ID system (BUILD_PLAN Task 5) tracks every (model, cell, chain_pair, condition) tuple. On resume, harness checks "have I logged a successful response for this ID?" before retrying.

**Mitigation**:
- Idempotent request structure: re-running a request that already succeeded is a no-op
- Cost ledger persists to disk after every batch completion
- Run can resume from any point by replaying the ID table

**Escalation**: only if checkpoint recovery fails (very unlikely with deterministic IDs).

---

## 4. Analysis phase risks (Task 9 in BUILD_PLAN)

### R-AN-1. Cluster-robust SE computation produces unrealistically wide CIs on RL

**Trigger**: cluster-resampling bootstrap on RL produces CIs that are wider than expected given the predicted detection lift (+5.9% in v5 was barely significant; v5.1's effective n is similar).

**Likelihood**: medium. RL has 148 sources × n=500 chain pairs / 8.1 mean cluster size ≈ 62 effective sources. McNemar power is borderline.

**Impact**: methodology — load-bearing strict-grounding contrast on RL may not reach significance, weakening the mechanism claim's third predicted failure mode.

**Detection**: bootstrap CI on RL strict-vs-non-strict contrast. If 95% CI includes 0 or has lower bound < +2pp, the contrast is underpowered.

**Mitigation**:
- Pre-registered: report results with cluster-robust SEs as primary, IID SEs as robustness
- Discordance between SE methods is itself a finding
- If contrast is underpowered, reframe finding as "strict-grounding effect on RL is consistent with mechanism prediction but not powered to detect at 95%"
- v5.2 follow-up with larger n on RL specifically would address this

**Escalation**: if all RL contrasts are underpowered, write-up reframes mechanism claim's third failure mode as "predicted but not confirmed at this sample size."

### R-AN-2. 3-way interaction test produces unexpected significant interactions

**Trigger**: 3-way interaction (intervention × marker × strict) shows significant non-zero coefficients on cells where SPEC predicted null (poker, PUBG, NBA).

**Likelihood**: medium. The pre-registered predictions are theory-grounded but not theoretically guaranteed.

**Impact**: methodology — could refine the mechanism claim or invalidate it.

**Detection**: 3-way interaction tests run per cell × per model. Results are pre-registered as expected to be null on poker/PUBG/NBA.

**Mitigation**:
- All 3-way interactions are reported regardless of significance, per pre-registration
- Unexpected interactions are reported as "discordance with pre-registered prediction" — itself a finding, not a failure
- The mechanism claim's three predicted failure modes are pre-registered with clear conditions; results that contradict them refine the claim

**Escalation**: if 3+ cells show unexpected significant interactions, the mechanism claim itself is reframed as "predicted but not exclusively supported."

### R-AN-3. Within-provider hierarchy fails on 3+ ladders

**Trigger**: the conjunctive headline test (SPEC §2.3) shows fewer than 5 of 6 ladders preserving the moderate hierarchy criterion.

**Likelihood**: medium. The cross-model panel in v5.1 is intentionally diverse; some ladders may exhibit tier-collapse on RL or unexpected behavior on CSGO.

**Impact**: scope — replication threshold not met.

**Detection**: within-provider analysis per ladder.

**Mitigation**:
- Pre-registered fallback framings:
  - 5–6 of 6 → "v5 replicates as a general phenomenon"
  - 3–4 of 6 → "model-class-dependent finding"
  - ≤2 of 6 → "v5 may have been Haiku-specific"
- All three framings are publishable; the experiment is informative regardless of outcome
- Within-provider results are reported individually, preserving partial-replication interpretability

**Escalation**: no escalation needed; pre-registered fallbacks handle all cases.

### R-AN-4. FDR and Bonferroni produce sharply different conclusions

**Trigger**: many contrasts are significant under FDR but not Bonferroni, suggesting reliance on FDR's permissive control.

**Likelihood**: medium-high. With 640+ contrasts and Bonferroni divisor in the hundreds, this is the expected case for any moderate-effect contrast.

**Impact**: methodology — interpretation depends on which method is treated as primary.

**Detection**: per-contrast significance under both methods, reported in heatmap.

**Mitigation**:
- Pre-registered: FDR is primary, Bonferroni is robustness check
- Discordance is reported, not hidden
- The mechanism claim does not depend on Bonferroni-strict significance for secondary contrasts

**Escalation**: no escalation needed; pre-registered framing handles this.

### R-AN-5. Output-length-tier correlation (Q19) does not replicate

**Trigger**: v5's striking finding that RL output mean is 4 tokens vs. 31.5 cap-bound on other cells does not replicate in v5.1's panel.

**Likelihood**: low. v5's pattern was extreme; partial replication is more likely than total failure.

**Impact**: scope — secondary confirmatory analysis not confirmed.

**Detection**: per-cell × per-model output-token mean reported in analysis.

**Mitigation**:
- Q19 is a secondary confirmatory analysis, not a primary finding
- Failure to replicate is itself a finding (suggests v5's output-length pattern may be Haiku-specific)
- Does not affect headline conjunctive replication test

**Escalation**: no escalation needed.

---

## 5. Writeup phase risks (Task 10 in BUILD_PLAN)

### R-WR-1. Mid-flight methodology amendments not properly logged

**Trigger**: an issue arose during build, smoke, or full run that was resolved with a tweak that wasn't formally logged with both-author re-signature.

**Likelihood**: low — explicit prohibition in SPEC §11 and BUILD_PLAN.

**Impact**: methodology — undermines the value of pre-registration.

**Detection**: writeup-phase audit of DECISION_LOG against actual build/run actions. Every D-XX entry should correspond to a signed-off decision.

**Mitigation**:
- Discrepancies log is a required Task 10 deliverable
- If found, report transparently in writeup with caveat
- Pre-registration value is not lost; just appropriately scoped

**Escalation**: any unlogged amendment is reported transparently in writeup.

### R-WR-2. Replication threshold met but mechanism claim inadequately supported

**Trigger**: hierarchy replicates in 5+ ladders, but the strict-grounding contrast on RL (mechanism claim's third failure mode) is not significantly different from zero.

**Likelihood**: medium. The hierarchy ordering is a *structural* property; the mechanism claim is a *causal-adjacent* claim. They can come apart.

**Impact**: scope — headline replicates but underlying explanation remains underdetermined.

**Detection**: writeup must reconcile ladder-level hierarchy results with cell-level mechanism contrasts.

**Mitigation**:
- Pre-registered scope of mechanism claim (SPEC §2.4) explicitly hedges on causal identification
- Writeup separates the structural finding (hierarchy replicates) from the mechanistic finding (variable availability is associated with detection rate)
- v5.2 follow-up scoped if mechanism claim needs synthetic-domain validation

**Escalation**: no escalation needed; scope hedging handles this.

### R-WR-3. External reviewer challenges training-distribution alternative

**Trigger**: reviewer argues that "CSGO is less represented in pretraining than poker" or similar, undermining the variable-availability mechanism.

**Likelihood**: high. This is the most defensible alternative explanation and we expect it.

**Impact**: scope — does not invalidate findings, but bounds claim strength.

**Detection**: anticipated, addressed in SPEC §10.

**Mitigation**:
- SPEC §10 explicitly acknowledges training-distribution as a not-ruled-out alternative
- Writeup names v5.2 (synthetic constraint domains controlled for training exposure) as the appropriate follow-up
- Publication framing: "v5.1 establishes structural replication; v5.2 will address mechanistic specificity"

**Escalation**: no escalation needed; this critique is anticipated and pre-registered against.

---

## 6. Cross-cutting risks

### R-CC-1. Total cost exceeds $300 realistic ceiling

**Trigger**: cumulative cost across all phases exceeds $300.

**Likelihood**: medium. The realistic ceiling has 60% buffer over mean, but reasoning-mode worst-case is 3–4× mean.

**Impact**: cost — depends on overage size.

**Detection**: per-model kill-switch + aggregate cost ledger updated in real time.

**Mitigation**:
- Kill-switch on each model at 110% of mean prevents single-model runaway
- Aggregate ledger has hard ceiling at $300 with both-author sign-off required to exceed
- If ceiling is hit, options: (a) raise ceiling, (b) drop most expensive remaining models, (c) reduce remaining n on RL

**Escalation**: $300 ceiling cannot be exceeded without both-author re-signature.

### R-CC-2. Anthropic / OpenAI / Google API pricing changes mid-run

**Trigger**: any provider changes pricing during the experiment window.

**Likelihood**: low. Major providers don't typically change prices mid-month.

**Impact**: cost — if increase; methodology: results are still valid but cost ledger may be off.

**Detection**: per-call ground-truth cost from API response detects pricing changes immediately.

**Mitigation**:
- Per-call usage logging (Task 1) captures actual cost regardless of pricing changes
- Cost ledger flags any per-call cost outside expected range (>10% deviation from prediction)
- If pricing change is detected, recompute remaining-budget impact and decide to continue or pause

**Escalation**: any unexpected cost-per-call deviation >10% triggers both-author review.

### R-CC-3. v5's frozen prompt corpus is corrupted or modified between v5 and v5.1

**Trigger**: an accidental change to v5's `src/harness/prompt_corpus/` JSON files between Phase D completion and v5.1 build start.

**Likelihood**: very low. The repo is version-controlled.

**Impact**: methodology — invalidates the "v5.1 reuses v5's frozen corpus" claim.

**Detection**: hash check of v5's prompt corpus against v5's archived hash from Phase D completion.

**Mitigation**:
- Pre-flight gate: hash all prompt corpus files; compare to v5 archive hash from Phase D
- If hashes match: proceed
- If hashes don't match: identify which files differ, restore from v5 archive, document in DECISION_LOG

**Escalation**: any hash mismatch requires both-author review and DECISION_LOG entry before proceeding.

### R-CC-4. OSF DOI not generated before full run starts

**Trigger**: SPEC and appendices uploaded to OSF, but DOI assignment is delayed.

**Likelihood**: low. OSF DOI assignment is typically immediate.

**Impact**: methodology — pre-registration timestamp not established before full run.

**Detection**: Task 7 acceptance criterion: OSF DOI exists and is recorded in SPEC.md before Task 8 begins.

**Mitigation**:
- Task 7 is a hard prerequisite for Task 8 in BUILD_PLAN
- If DOI is delayed, full run is held until it arrives
- OSF support contacted if delay exceeds 24 hours

**Escalation**: full run does not begin without OSF DOI.

### R-CC-5. Both-author sign-off discipline breaks down

**Trigger**: time pressure or convenience leads to one-author decisions on items requiring both authors.

**Likelihood**: medium. This is a discipline risk, not a technical risk.

**Impact**: methodology — undermines pre-registration value.

**Detection**: every signed decision has both author signatures in DECISION_LOG. Missing signatures are visible.

**Mitigation**:
- Pre-registered: both-author sign-off required for SPEC, smoke approval, OSF filing, full run start, drops/amendments, writeup
- Synchronous communication channel maintained between authors during build/smoke/full run phases
- If sign-off cannot be obtained within 24 hours of need, run pauses

**Escalation**: any single-author decision on a both-author item is treated as a methodology violation requiring DECISION_LOG entry and remediation.

---

## 7. Headline risk summary (matches SPEC §15)

The 10 headline risks from SPEC §15, with mapping to this document:

| # | SPEC §15 risk | This doc reference |
|---:|---|---|
| 1 | Token count higher than audit predicts | R-SM-2, R-CC-1 |
| 2 | Anthropic cache fragmentation | R-CC-1 (subset) |
| 3 | Models fail strict-grounding parseability | R-SM-1, R-FR-5 |
| 4 | Cross-model results don't replicate | R-AN-3 |
| 5 | Reasoning models exceed 256-cap | R-FR-5 |
| 6 | OpenRouter silent provider routing | R-SM-3 |
| 7 | DeepSeek off-peak window misses | R-FR-2 |
| 8 | 3-way interaction more complex than predicted | R-AN-2 |
| 9 | Within-provider hierarchy holds <5 of 6 ladders | R-AN-3 |
| 10 | Capability-distribution confound | R-WR-3 |

---

## 8. Escalation matrix

Pre-registered escalation triggers. Both authors must convene before continuing if any of these occur:

| Phase | Trigger | Action |
|---|---|---|
| Pre-flight | Any v5 test failure after Task 1 | Pause Task 1, both-author review |
| Pre-flight | Cluster size mismatch with audit | Pause Task 3, both-author review |
| Smoke | Aggregate cost >110% of prediction | Recompute, both-author decision on continue |
| Smoke | 4+ models fail strict-grounding parseability | Pause, re-evaluate instruction wording |
| Smoke | OpenRouter routing variance observed | Halt affected model, investigate |
| Smoke | 2+ critical ladders broken | Headline framing review |
| Full run | Per-model kill-switch trip | Both-author sign-off to resume |
| Full run | DeepSeek standard-pricing call observed | Halt DeepSeek slice |
| Full run | Aggregate $300 ceiling reached | Both-author re-signature to exceed |
| Full run | Any single-call cost >10% above prediction | Both-author review |
| Full run | 4+ hours rate-limit pause on any provider | Both-author review |
| Analysis | All RL contrasts underpowered | Reframe mechanism claim 3rd mode |
| Analysis | 3+ unexpected interaction effects | Reframe mechanism claim |
| Writeup | Unlogged amendment discovered | Transparent reporting in writeup |
| Cross-cutting | Hash mismatch on v5 prompt corpus | Halt all work |
| Cross-cutting | OSF DOI delay >24 hours | Hold full run start |
| Cross-cutting | Single-author decision on dual item | DECISION_LOG remediation |

---

## 9. Sign-off

This pre-mortem is filed alongside SPEC.md as a required component of v5.1's pre-registration. It does not modify the SPEC. It documents anticipated risks and pre-registered responses, so that risk events during build/smoke/full run trigger pre-decided actions rather than improvised ones.

Risks not listed here may still occur. If they do, the response is: pause, document in DECISION_LOG, both-author review, decide whether the response constitutes a SPEC amendment (requiring re-signature) or fits within an existing pre-registered category. Default to caution.

**Author 1 (Safiq):** ___________________ Date: ___________

**Author 2 (Myriam):** ___________________ Date: ___________

---

*This document is the canonical risk register for v5.1. It supersedes any prior pre-mortems. It does not modify any of v5's frozen artifacts or v5.1's SPEC.*
