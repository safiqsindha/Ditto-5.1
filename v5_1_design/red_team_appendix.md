# red_team_appendix.md

**Project Ditto v5.1 — Red-Team Appendix**

**Pre-registered responses to anticipated critiques. Filed alongside SPEC.md before any model is hit at full-run scale.**

---

## Purpose

This appendix documents methodological gaps identified during pre-registration review and the changes made to close them. It is filed *before* the experiment runs, not after, so reviewers can verify that critiques were addressed prospectively rather than post-hoc.

Each section names a potential critique, evaluates its severity, documents the SPEC change (or non-change with justification) that addresses it, and references the SPEC section where the resolution lives.

This document is not a substitute for the SPEC. It is evidence that the SPEC closes specific known gaps. If a reviewer raises an unanticipated critique post-publication, the response is honest acknowledgment, not retrofit.

---

## Origin of these critiques

The first 10 items (R1–R10) come from a structured pre-registration review by an external skeptical reviewer who read the v5.1 design draft. The reviewer was instructed to identify methodological gaps that could be exploited by an adversarial peer reviewer.

Items R11 and R12 are surfaced by a token audit of v5's archived API responses (n=23,997 calls). They were not anticipated by the original review but materialized once we examined v5's empirical behavior at the call level.

All 12 items are addressed by SPEC changes filed before any v5.1 model is hit.

---

## R1. Independence assumption across chain pairs

**Critique**: chain pairs generated from the same templates within a cell are not independent. Effective n is lower than nominal n, which inflates McNemar power and narrows confidence intervals misleadingly. If true effective n is ½ of nominal, McNemar p-values are roughly 4× too small.

**Severity**: Real. Plausibly halves effective n on cells with high cluster ratios.

**Token audit refinement**: the audit revealed cluster sizes vary sharply by cell:
- PUBG, NBA, Poker: 1:1 chains-to-sources (no clustering needed)
- CSGO: 1,200 chains over 364 sources, mean 3.3 chains/source
- Rocket League: 1,200 chains over 148 sources, mean 8.1 chains/source

This is a *better* situation than the original critique anticipated, but it changes where clustering matters.

**Pre-registered response**:
- Cluster-robust SEs applied to CSGO and Rocket League (clusters: source-game)
- IID SEs valid for PUBG, NBA, Poker (1:1 chains-to-sources)
- For McNemar specifically, paired analyses use a clustered variant where cluster sizes vary across cells
- Bootstrap CIs on RL detection lift use cluster-resampling (resample clusters with replacement, then chains within cluster)
- Cluster definitions locked in `clusters.json` from v5's frozen prompt corpus before any analysis runs

**SPEC reference**: §5.3 (cluster sizes), §7.4 (cluster-robust SE policy), §14 (pre-flight gate 4)

---

## R2. Three-way interaction between added axes

**Critique**: the two new axes (marker ablation, strict grounding) are unlikely to be orthogonal in practice. Strict grounding probably suppresses detection most strongly when markers are absent. Without modeling the 3-way interaction `intervention × marker × strict`, the mechanism claim looks cleaner than it is.

**Severity**: High. This interaction directly tests the load-bearing third predicted failure mode of the mechanism claim.

**Pre-registered response**:
- The 3-way interaction is a *primary* analysis, not a robustness check
- Fit per cell × per model
- Pre-registered predictions:
  - Poker: no significant 3-way interaction (saturated)
  - PUBG/NBA: marker × intervention significant, strict main effect ~null
  - CSGO: full 3-way interaction expected
  - RL: strict × intervention significant, marker × intervention ~null (markers cannot scaffold absent variables)
- Discordance between predictions and data is reportable as mechanism refinement, not failure

**SPEC reference**: §7.1.2 (primary analysis 2)

---

## R3. Parseability handling and selection bias

**Critique**: the original 95% parse-failure exclusion threshold was too lenient and biased the panel toward well-behaved APIs. Format adherence under constraint is itself a capability signal that gets discarded.

**Severity**: Real. Especially dangerous under strict grounding, where parse failures may be condition-conditional rather than random.

**Pre-registered response**:
- Parse status is logged as a ternary outcome `{parsed_yes, parsed_no, parse_failure}` for every call
- Parse failures enter the analysis as a logged outcome dimension, not as missing data
- Inclusion threshold raised: a model is excluded only if aggregate parse rate falls below 90% across all conditions (was 95% in original draft)
- Models between 90–98% are retained but flagged
- Condition-conditional parse rates (e.g., parse rate under strict vs. non-strict) are reported per-model as a secondary capability dimension
- Dual treatment in McNemar contrasts: failures-as-misses (conservative for detection-rate gains) and failures-excluded (capability-charitable). Discordance between treatments is itself a finding.

**Downstream impact**: the harness output schema is now ternary, requiring updates to `synthesize_cross_model.py`. Documented as a non-trivial change vs. v5.

**SPEC reference**: §8 (parse-failure handling, full section)

---

## R4. Output token cap conditions all claims

**Critique**: capping `max_completion_tokens` at 64 systematically handicaps models that compress reasoning into output tokens. Some APIs compress reasoning differently than others. Results measure constrained reasoning, not unconstrained reasoning.

**Severity**: Moderate. Doesn't invalidate findings; does scope them.

**Token audit refinement**: 38.2% of v5 responses hit the 32-token cap. v5.1 raised the cap to 64 (or 256 for thinking-mode models) as a result. The cap is still binding in some conditions, just less frequently.

**Pre-registered response**:
The output cap is preserved (cost reasons), and all claims are explicitly scoped:

> *All v5.1 findings are conditional on: fixed prompt structure (no per-model tuning), constrained output budgets (`max_completion_tokens=64` for non-reasoning, 256 for thinking-mode), and fixed domain distributions across the 5 cells from v5. This is not a measure of unconstrained constraint-reasoning capacity. Findings about reasoning-mode models are particularly conditional on the 256-token cap.*

Reasoning-mode models receive a per-model annotation in the results table flagging that their results are most sensitive to this scope condition.

**SPEC reference**: §10 (scope of claims), §4.5 (output-token cap policy)

---

## R5. OpenRouter routing variability

**Critique**: OpenRouter may silently route to different provider backends under load. Token accounting can drift from native APIs. Retries may double-bill.

**Severity**: Real but addressable.

**Pre-registered response**:
- Provider pinning via OpenRouter `provider.order` parameter for all 14 OR-routed models
- `allow_fallbacks: false` set on every call: if pinned provider is down, the call fails loudly rather than silently substituting
- The pinned backend is selected before smoke test and locked in `provider_pinning.json`
- Per-call resolved-provider metadata logged from the OR response
- Deterministic request IDs of the form `{model}_{cell}_{chain_pair_id}_{condition}_{run_id}`. The harness checks "have I logged a successful response for this ID?" before retrying.
- All OR responses' `usage` blocks are logged and audited against expected token counts post-run; >10% drift between expected and actual triggers manual review

**Why not full idempotency keys**: not all OR-routed providers honor them consistently. Harness-layer dedup is more reliable than provider-layer idempotency for this experimental context.

**SPEC reference**: §4.3 (routing decisions), §13 (new for v5.1, harness logging)

---

## R6. Multiple comparisons control

**Critique**: Bonferroni at 640+ cells × multiple contrasts is overly conservative. Will produce false negatives, especially for tier-3 (RL) signals where effects are small.

**Severity**: Real.

**Pre-registered response**:
- **Primary**: Benjamini–Hochberg FDR at q=0.05, applied across all cells × models × pre-registered contrasts
- **Robustness check**: Bonferroni at α=0.05/k for the same set, reported alongside
- **Discordance reporting**: any contrast significant under FDR but not Bonferroni is reported as such, not as "robust" or "not robust." Both are informative.
- The contrast list `k` is finalized before the full run begins; no post-hoc additions.

**SPEC reference**: §7.3 (multiple comparisons control)

---

## R7. Mechanism claim wording

**Critique**: "performance limited by availability and explicitness of rule-relevant variables" is vulnerable to alternative explanations like training-distribution familiarity. CSGO might be less represented in training than poker.

**Severity**: Moderate. Doesn't refute the claim; weakens its specificity.

**Pre-registered response**:

The claim is hedged with explicit conditioning:

> *Conditional on fixed prompt structure and the domain distributions in v5's 5 cells, constraint-reasoning performance in LLMs varies systematically with the explicit availability of rule-relevant variables in input representation.*

Three failure modes are stated as predictions, not assertions:
1. Full observability + intervention → near-perfect detection
2. Partial observability → systematic confabulation false positives
3. Absent observability + strict grounding → suppressed detection

The claim explicitly does NOT assert:
- That models cannot constraint-reason in domains where variables are absent (only that they don't, under v5's prompt regime)
- That the failure modes are causally identified (only systematically associated)
- That the hierarchy generalizes beyond the specific cells tested

The training-distribution alternative explanation is acknowledged as not ruled out by this experiment. A v5.2 follow-up using synthetic constraint domains controlled for training-distribution exposure is named as the appropriate next step if this becomes a load-bearing critique.

**SPEC reference**: §2.4 (mechanism claim), §10 (scope of claims)

---

## R8. Capability-distribution confound

**Critique**: open-weight models cluster in mid-tier capability bands; frontier closed models dominate tier-3 expectations. "Hierarchy holds" might reflect capability distribution rather than representational structure.

**Severity**: High. This is the most attackable claim in the panel-level finding.

**Pre-registered response**:

The within-provider hierarchy test is *primary*, not robustness:

> *Does the 4-tier ordering (poker tier-0 → PUBG/NBA tier-1 → CSGO tier-2 → RL tier-3) hold along each provider's capability ladder, holding provider constant?*

Tested independently for 6 ladders:
- Anthropic 4-tier: Haiku → Sonnet 4.5 → Sonnet 4.6 → Opus 4.7
- OpenAI 5-tier: GPT-5 Mini → GPT-5 → GPT-5.4 Mini → GPT-5.4 → GPT-5.5
- Google 5-tier: Flash-Lite → 2.5 Flash → 2.5 Pro → 3 Flash → 3.1 Pro
- xAI 3-tier: Grok 4 Fast → Grok 4 → Grok 4.20
- DeepSeek 4-tier: V3.2 → V4 Flash → R1 → V4 Pro
- Moonshot 2-tier: K2.6 → K2 Thinking

The headline finding is **conjunctive**: "the hierarchy holds within ≥5 of 6 provider ladders." Cross-panel results are reported as additional context, not the headline.

If within-provider hierarchies do *not* hold while cross-panel does, the panel result is reframed as capability-distribution-confounded. Pre-registered, not post-hoc.

**SPEC reference**: §2.2 (operational definition of "replicates"), §2.3 (replication thresholds), §7.1.3 (within-provider hierarchy test)

---

## R9. No per-model prompt tuning

**Critique anticipated** (not raised in original review but expected from external reviewers): "Why not optimize prompts per model? Some models follow instructions better with different framing."

**Severity**: Would invalidate cross-model comparability if relaxed.

**Pre-registered response**:

Prompt corpus is frozen from v5. No per-model variants. No prompt engineering per model. Models that perform poorly under v5's prompts perform poorly — that is a finding about the prompt's portability, not a flaw to engineer around.

If a reviewer requests per-model tuning post-publication, the response is: that experiment is v5.2, with its own pre-registration, designed to disentangle prompt-portability from representation-portability.

**SPEC reference**: §11 (what v5.1 does NOT do), §12 (reuse from v5)

---

## R10. Reasoning-mode confound

**Critique**: comparing Grok 4 (reasoning baked in) to Llama 3.3 (no reasoning) confounds model identity with reasoning mode.

**Severity**: Real and acknowledged. Cannot be cleanly separated within v5.1.

**Pre-registered response**:
- Reasoning-mode is logged per model as a categorical metadata variable: `{always_on, toggleable_off, none}`
- Where toggleable, reasoning is set to off (e.g., Grok 4 Fast `reasoning: false`, DeepSeek V4 Pro thinking off)
- Where built-in and non-disable-able (Grok 4, GPT-5.4+, GPT-5.5, Grok 4.20, DeepSeek R1, K2 Thinking), the model is included with its reasoning mode flagged
- Analysis reports results stratified by reasoning-mode category. If reasoning-mode-on models systematically differ from reasoning-mode-off models in patterns *uncorrelated with provider*, that is a finding (and a v5.2 hypothesis)
- Within-provider reasoning-mode pairs (4 of them: OpenAI, xAI, DeepSeek, Moonshot) are used for the cleanest reasoning-vs-non-reasoning contrast
- The mechanism claim does not depend on reasoning-mode being held constant; it is conditional on whatever each model's default reasoning mode is

**SPEC reference**: §4.4 (reasoning-mode policy), §2.5 (secondary hypotheses, item 3)

---

## R11. v5 had no prefix caching (audit-surfaced)

**Critique not anticipated; surfaced by token audit**: v5's brief implied 90% prefix caching from v4.5 carryover. Audit of archived Phase D responses revealed `cache_creation_input_tokens=0` and `cache_read_input_tokens=0` for every single call. **v5 had 0% prefix cache hit rate.** The 90% number was from v4.5; v5's harness never enabled caching at all.

**Severity**: Methodological honesty issue. Doesn't invalidate v5, but means v5.1's "we now enable caching" change is more substantive than v5's brief suggested.

**Pre-registered response**:
- v5.1 explicitly enables Anthropic prompt caching on the system + cell prefix
- Documented in MEMO addendum and SPEC §1.4 as a methodological correction
- v5.1's budget calculations do not assume cache hits (caching is logged but treated as a bonus, not a baseline)
- Per-call cache_read_tokens and cache_creation_tokens are logged for empirical measurement

The framing in the writeup is: "v5 did not cache; v5.1 enables Anthropic prompt caching as an efficiency improvement; this does not affect the experimental contrast since cached and uncached calls produce identical model outputs."

**SPEC reference**: §1.4 (token audit findings, item 3), §5.2 (cache rate)

---

## R12. RL output mean of 4 tokens is itself a mechanism observation (audit-surfaced)

**Observation not anticipated; surfaced by token audit**: across all RL variants in v5, output token mean is 4–6 tokens. Across PUBG/NBA/CSGO/Poker adversarial-intervention, output mean is 31.5 (cap-bound). The model is producing terse "NO" responses on RL because it does not see violations to explain.

**Severity**: Confirmatory observation, not a critique. Worth pre-registering so it is not later cited as post-hoc.

**Pre-registered response**:

Output-length-tracks-tier added as a pre-registered secondary confirmatory analysis.

> *Per-cell output-token mean is predicted to track tier — saturated/aligned cells produce verbose YES explanations; misaligned cell (RL) produces terse NO responses regardless of intervention. v5 baseline: RL mean 4 tokens vs. 31.5 cap-bound on PUBG/NBA/CSGO/Poker adversarial-intervention.*

If v5.1 replicates this output-length-tier correlation across providers, it is an additional brick in the mechanism wall (and is reported as such). If it does not replicate, the secondary finding does not affect the primary headline replication test.

**SPEC reference**: §2.5 (secondary hypotheses, item 4), §7.2 (secondary analysis 10)

---

## Summary table

| # | Critique | Severity | Resolution | SPEC reference |
|---:|---|---|---|---|
| R1 | Independence assumption | Real | Cluster-robust SEs (CSGO + RL) | §5.3, §7.4 |
| R2 | 3-way interaction hidden | High | Promoted to primary analysis | §7.1.2 |
| R3 | Parse-failure exclusion bias | Real | Ternary outcome + dual treatment | §8 |
| R4 | Output cap conditions claims | Moderate | Explicit scope hedging | §10, §4.5 |
| R5 | OpenRouter routing variance | Real | Provider pinning + `allow_fallbacks: false` | §4.3, §13 |
| R6 | Bonferroni overconservative | Real | FDR primary, Bonferroni robustness | §7.3 |
| R7 | Mechanism wording attackable | Moderate | Hedged with conditioning | §2.4, §10 |
| R8 | Capability-distribution confound | High | Within-provider conjunctive headline | §2.2, §2.3, §7.1.3 |
| R9 | Per-model prompt tuning gap | N/A | Held firm; v5.2 if challenged | §11, §12 |
| R10 | Reasoning-mode confound | Real | Logged as metadata, stratified | §4.4, §2.5 |
| R11 | v5 had no caching (audit-surfaced) | Methodological | v5.1 enables caching, documented | §1.4, §5.2 |
| R12 | RL output mean 4 tokens (audit-surfaced) | Confirmatory | Pre-registered as secondary | §2.5, §7.2 |

---

## What this appendix does not do

This appendix does not anticipate every possible critique. Reviewers post-publication may raise concerns we did not foresee. The response to such concerns is:

1. **Honest acknowledgment** that the concern was not pre-registered against
2. **Evaluation** of whether the concern affects the headline finding, secondary findings, or scope
3. **Decision** on whether the concern requires (a) errata, (b) v5.2 follow-up, or (c) reframing in writeup

We do not retrofit responses to critiques that were not anticipated. We document them honestly.

---

## Sign-off

This red-team appendix is a required component of v5.1's pre-registration. It is filed alongside SPEC.md on OSF before any model is hit at full-run scale. It documents 12 known methodological concerns and the SPEC changes made to address them.

The appendix is evidence of pre-registration discipline: critiques caught before data collection, not after.

**Author 1 (Safiq):** ___________________ Date: ___________

**Author 2 (Myriam):** ___________________ Date: ___________

---

*This document is the canonical red-team appendix for v5.1. It supersedes any prior critique-response logs. It does not modify v5.1's SPEC.*
