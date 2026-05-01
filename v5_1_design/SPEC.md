# SPEC.md

**Project Ditto v5.1 — Cross-Model Replication of a 4-Tier Representational Hierarchy in Constraint-Reasoning**

**Pre-registration document**

---

## Document metadata

| Field | Value |
|---|---|
| Project | Ditto v5.1 |
| Type | Cross-model empirical replication of v5 |
| Authors | Safiq, Myriam |
| Status | Draft pending sign-off |
| OSF DOI | TBD (Task 7 in BUILD_PLAN) |
| Filed | TBD |
| Supersedes | — |
| Sign-off required before | Smoke test (Task 6 in BUILD_PLAN) |

---

## Abstract

Project Ditto v5 established a 4-tier representational hierarchy in Claude Haiku 4.5's constraint-reasoning behavior across five game domains (poker, PUBG, NBA, CSGO, Rocket League). v5.1 tests whether this hierarchy replicates across 32 models from 12 providers, using the same frozen prompt corpus extended along two new axes — marker ablation and strict-grounding — that were identified by skeptical review as load-bearing for the mechanism claim. The headline finding is conjunctive: the hierarchy is considered to replicate if it holds within at least 5 of 6 capability-ladder-deep providers, controlling for capability-distribution confounds. v5.1 is pre-registered prospectively on OSF before any model is hit at full-run scale. The mechanism claim is tightened relative to v5 to condition on prompt structure and domain distributions explicitly.

---

## 1. Background and motivation

### 1.1 v5's findings (being replicated)

v5 ran 24,000 paired McNemar evaluations on Claude Haiku 4.5, n=1,200 per cell, with Bonferroni correction (divisor=5). Phase D produced this 4-tier hierarchy:

| Cell | Baseline detection | Intervention detection | Δ | Tier |
|---|---:|---:|---:|---|
| Poker | 100.0% | 99.9% | -0.1% | 0 — saturated (rule internalized) |
| PUBG | 75.9% | 100.0% | +24.1% | 1 — aligned |
| NBA | 57.4% | 100.0% | +42.6% | 1 — aligned |
| CSGO | 65.1% | 98.1% | +32.9% | 2 — partial (29.8% confabulation FPs) |
| Rocket League | 0.2% | 6.2% | +5.9% | 3 — misaligned (strict grounding suppresses detection) |

The 4-tier hierarchy is gated on three claimed conditions: rule anchoring, predicate observability, and unary reducibility.

### 1.2 The skeptical critique

A reviewer-grade critique identified three load-bearing gaps in v5's mechanism claim:

1. **The strict-grounding instruction was a confound.** v5's prompts include "only reply YES if you can identify the specific rule and the specific event." Rocket League's 0.2% baseline detection rate could be explained by either (a) the model genuinely cannot represent the rule, or (b) strict grounding suppresses detection when required variables are absent. Without a strict/non-strict contrast, the mechanism is underdetermined.

2. **Derived-state markers were doing unmeasured work.** v5's adversarial chains include markers like `NOTE=Player_X_already_eliminated`. These are close to the violation itself. Without a marker-ablation contrast, the SPEC cannot distinguish "the model reasons about the constraint" from "the marker tells the model the answer."

3. **Single-model results may be Haiku-specific.** A representational hierarchy in one model is not a property of LLMs; it is a property of one model. v5.1's cross-model panel addresses this directly.

### 1.3 What v5.1 adds

v5.1 extends v5 with three changes:

- **Cross-model panel** of 32 models from 12 providers (Western: Anthropic, OpenAI, Google, xAI, Meta. Chinese-origin: DeepSeek, Xiaomi, Alibaba, Z.AI, MiniMax, Moonshot)
- **2×2×2 conditions per chain pair**: baseline/intervention × marker/no-marker × strict/non-strict
- **Conjunctive within-provider hierarchy test** as primary headline finding (controls for capability-distribution confound)

v5.1 does NOT modify v5's frozen prompt corpus, chain construction, violation injectors, or per-cell pipelines. v5.1 reuses these directly and adds only the two new axes as wrappers around existing v5 prompts.

### 1.4 v5's methodological gaps surfaced by token audit

A separate audit of v5's archived API responses surfaced five facts that affect v5.1 design:

1. **Mean input tokens per call: 673** (range 450–822 across cells). Worst-case at full-condition multiplication: ~950.
2. **Mean output tokens per call: 14.5** with 38% of responses cap-bound at 32 tokens. v5.1 raises the output cap to 64 (or 256 for thinking-mode models).
3. **v5 had 0% prefix cache hit rate.** Caching was not enabled. v5.1 enables Anthropic prompt caching as an efficiency improvement; documented as a methodological change relative to v5.
4. **Template/scenario family clustering varies sharply by cell:** PUBG, NBA, Poker are 1:1 chains-to-sources (no clustering). CSGO has mean 3.3 chains/source. Rocket League has mean 8.1 chains/source. v5.1 applies cluster-robust SEs to CSGO and RL only.
5. **RL output token mean is 4–6 across all variants** (vs. 31.5 cap-bound on PUBG/NBA/CSGO/Poker adversarial-intervention). The model is not attempting to explain RL violations because it is not detecting them. v5.1 pre-registers this output-length-tier correlation as a confirmatory test.

---

## 2. Research questions and hypotheses

### 2.1 Primary research question

Does the 4-tier representational hierarchy observed in Claude Haiku 4.5 (v5) replicate across a panel of 32 models from 12 providers?

### 2.2 Operational definition of "replicates"

The headline finding is **conjunctive**, evaluated within-provider rather than across the aggregate panel. Specifically:

- For each provider with a capability ladder ≥3 models deep, evaluate whether the hierarchy ordering (poker tier-0 → PUBG/NBA tier-1 → CSGO tier-2 → RL tier-3) holds within that ladder
- Apply the **moderate criterion** for "ladder holds": 3 of 4 cell-tier transitions preserve ordering with significant separations within the ladder
- Six ladders qualify: Anthropic (4 models), OpenAI (5), Google (5), xAI (3), DeepSeek (4), Moonshot (2)*

*Moonshot's 2-tier ladder uses a single-transition check.

### 2.3 Replication thresholds

Pre-registered:

- **≥5 of 6 ladders** show moderate-criterion hierarchy → "v5 replicates as a general phenomenon"
- **3–4 of 6 ladders** → "model-class-dependent finding; v5 generalizes partially"
- **≤2 of 6 ladders** → "v5 may have been Haiku-specific or limited to a subset of model families"

These thresholds are pre-registered before any model is hit at full-run scale and do not change based on data.

### 2.4 Mechanism claim

The mechanism claim is tightened relative to v5 to condition on the experimental regime:

> **Conditional on fixed prompt structure and the domain distributions in v5's 5 cells, constraint-reasoning performance in LLMs varies systematically with the explicit availability of rule-relevant variables in input representation.**

Three predicted failure modes:

1. **Full observability + intervention → near-perfect detection** (poker, PUBG, NBA when intervention is applied)
2. **Partial observability → systematic confabulation false positives** (CSGO under intervention)
3. **Absent observability + strict grounding → suppressed detection** (RL — load-bearing for the mechanism claim, tested directly via strict-vs-non-strict contrast)

The claim explicitly does NOT assert:
- That models cannot constraint-reason in domains where variables are absent (only that they don't, under v5's prompt regime)
- That the failure modes are causally identified (only that they are systematically associated)
- That the hierarchy generalizes beyond the specific cells tested
- That training-distribution familiarity is ruled out as an alternative explanation (this requires synthetic-domain follow-up in v5.2 if it becomes load-bearing)

### 2.5 Secondary hypotheses

1. **Tier-collapse**: a provider's frontier model catches RL's indirect markers when its mid-tier model does not. Tested via within-provider ladder analysis on RL specifically.
2. **FP-discipline scaling**: capability buys resistance to confabulation. Tested via CSGO false-positive-rate trends within each provider's capability ladder.
3. **Reasoning-mode contribution**: reasoning-trained models perform better on tier-3 (RL) than non-reasoning variants of similar capability. Tested via 4 within-provider reasoning-mode pairs (OpenAI, xAI, DeepSeek, Moonshot).
4. **Output-length tracks tier**: v5 showed RL responses average 4 tokens vs. 31.5 (cap-bound) on other adversarial-intervention cells. Pre-registered prediction: this output-length-tier correlation replicates across providers, with absent-variable cells (RL) producing terse responses regardless of intervention condition.

---

## 3. Design

### 3.1 Conditions per chain pair

8 conditions in a fully-crossed 2×2×2 design:

| Condition ID | Intervention | Marker | Strict |
|---|---|---|---|
| C1 | baseline | marker | strict |
| C2 | baseline | marker | non-strict |
| C3 | baseline | no-marker | strict |
| C4 | baseline | no-marker | non-strict |
| C5 | intervention | marker | strict |
| C6 | intervention | marker | non-strict |
| C7 | intervention | no-marker | strict |
| C8 | intervention | no-marker | non-strict |

All 8 conditions are run on every cell. Uniform design; no per-cell asymmetry.

### 3.2 Sample sizes per cell

Variable n by cell, concentrating samples where the predicted effect is small (per cluster audit, RL has effective n ≈ n / 8.1 due to source-game clustering):

| Cell | n chain pairs | Total evals/model |
|---|---:|---:|
| Poker | 150 | 1,200 |
| PUBG | 150 | 1,200 |
| NBA | 150 | 1,200 |
| CSGO | 300 | 2,400 |
| Rocket League | 500 | 4,000 |
| **Total per model** | | **10,000** |

### 3.3 Total experimental scale

- 32 models × 10,000 evals/model = **320,000 total evaluations**
- 8 conditions × 5 cells × variable n × 32 models, fully crossed

### 3.4 What constitutes the paired observation for McNemar

McNemar pairs are within-condition, within-cell, within-model. For each chain pair × condition × cell × model combination, we compare detection of the clean variant against the adversarial variant. The 4 binary outcomes for each pair determine the 2×2 McNemar contingency table.

This is the same pairing structure as v5.

---

## 4. Models and routing

### 4.1 Final 32-model panel

| # | Model | Provider | Tier | Route | Reasoning |
|---:|---|---|---|---|---|
| 1 | Haiku 4.5 | Anthropic | cheap | native (Batches+cache) | off |
| 2 | Sonnet 4.5 | Anthropic | mid (gen-prior) | native (Batches+cache) | off |
| 3 | Sonnet 4.6 | Anthropic | mid (current) | native (Batches+cache) | off |
| 4 | Opus 4.7 | Anthropic | frontier | native (Batches+cache) | off |
| 5 | GPT-5 Mini | OpenAI | cheap | native (Batches+cache) | off |
| 6 | GPT-5 | OpenAI | mid | native (Batches+cache) | off |
| 7 | GPT-5.4 Mini | OpenAI | mid (reasoning) | native (Batches+cache) | always-on, 256-cap |
| 8 | GPT-5.4 | OpenAI | frontier (reasoning) | native (Batches+cache) | always-on, 256-cap |
| 9 | GPT-5.5 | OpenAI | true-frontier | native (Batches+cache) | always-on, 256-cap |
| 10 | Gemini 3.1 Flash-Lite Preview | Google | cheap | native (Batches+cache) | off |
| 11 | Gemini 2.5 Flash | Google | cheap-mid | native (Batches+cache) | off |
| 12 | Gemini 2.5 Pro | Google | mid | native (Batches+cache) | off |
| 13 | Gemini 3 Flash Preview | Google | mid (current) | native (Batches+cache) | off |
| 14 | Gemini 3.1 Pro Preview | Google | true-frontier | native (Batches+cache) | off |
| 15 | Grok 4 Fast | xAI | cheap | OpenRouter (provider-pinned) | reasoning=false |
| 16 | Grok 4 | xAI | frontier (gen-prior) | OpenRouter (provider-pinned) | always-on, 256-cap |
| 17 | Grok 4.20 | xAI | true-frontier | OpenRouter (provider-pinned) | always-on, 256-cap |
| 18 | Llama 3.3 70B Instruct | Meta (open) | mid | OpenRouter (provider-pinned) | off |
| 19 | Llama 4 Maverick | Meta (open) | frontier | OpenRouter (provider-pinned) | off |
| 20 | DeepSeek V3.2 | DeepSeek (open) | mid | native (off-peak) | off |
| 21 | DeepSeek V4 Flash | DeepSeek (open) | cheap-mid | native (off-peak) | thinking=off |
| 22 | DeepSeek R1 | DeepSeek (open) | mid (reasoning) | native (off-peak) | always-on, 256-cap |
| 23 | DeepSeek V4 Pro | DeepSeek (open) | frontier | native (off-peak) | thinking=off |
| 24 | MiMo-V2-Pro | Xiaomi | frontier (gen-prior) | OpenRouter (provider-pinned) | off |
| 25 | MiMo-V2.5-Pro | Xiaomi | frontier (current) | OpenRouter (provider-pinned) | off |
| 26 | Qwen3.6 Plus | Alibaba | mid | OpenRouter (provider-pinned) | off |
| 27 | Qwen3.6 Max-Preview | Alibaba | true-frontier | OpenRouter (provider-pinned) | off |
| 28 | GLM-4.7 | Z.AI (open) | mid | OpenRouter (provider-pinned) | off |
| 29 | GLM-5 | Z.AI (open) | frontier | OpenRouter (provider-pinned) | off |
| 30 | MiniMax M2.5 | MiniMax | cheap | OpenRouter (provider-pinned) | off |
| 31 | Kimi K2.6 | Moonshot | frontier (current) | OpenRouter (provider-pinned) | off |
| 32 | Kimi K2 Thinking | Moonshot | frontier (reasoning) | OpenRouter (provider-pinned) | always-on, 256-cap |

### 4.2 Coverage summary

- 12 providers
- 20 Western / 12 Chinese-origin
- 9 open-weight (Llama 3.3, Llama 4 Maverick, DeepSeek V3.2/R1/V4 Flash/V4 Pro, GLM-4.7, GLM-5, K2 Thinking)
- 6 capability ladders ≥3 deep for within-provider hierarchy testing
- 4 within-provider reasoning vs. non-reasoning contrasts (OpenAI, xAI, DeepSeek, Moonshot)

### 4.3 Routing decisions

**Native APIs:**
- Anthropic, OpenAI, Google, DeepSeek
- Anthropic/OpenAI/Google: Batches API + prompt caching
- DeepSeek: synchronous, off-peak window only (UTC 16:30–00:30 = US Eastern 12:30 PM – 8:30 PM)

**OpenRouter:**
- All 14 third-party models
- Provider-pinned per model via `provider.order` parameter with `allow_fallbacks: false`
- Pinned provider selection: highest-quantization-at-acceptable-cost (Task 2 in BUILD_PLAN)
- Per-call resolved-provider metadata logged and audited against pin

### 4.4 Reasoning-mode policy

Reasoning mode is logged as model metadata, not manipulated as an experimental variable. Where toggleable, reasoning is set to off (e.g., Grok 4 Fast `reasoning: false`, DeepSeek V4 Pro thinking off). Where reasoning is built-in and non-disable-able (GPT-5.4+, Grok 4, Grok 4.20, DeepSeek R1, K2 Thinking), the model is included with its reasoning mode flagged in results.

The mechanism claim does not depend on reasoning-mode being held constant; it is conditional on whatever each model's default reasoning mode is.

### 4.5 Output-token cap policy

| Cap | Models |
|---|---|
| 64 tokens | All non-reasoning models (23 of 32) |
| 256 tokens | All reasoning-mode models (9 of 32): GPT-5.4 Mini, GPT-5.4, GPT-5.5, Grok 4, Grok 4.20, DeepSeek R1, K2 Thinking |

The 256-cap budgets thinking + final answer combined. Final-answer parsing uses first-token-after-whitespace logic; thinking content is logged but not parsed.

---

## 5. Token audit findings (informing design)

Per audit of v5's archived Phase D responses (n=23,997 succeeded calls):

### 5.1 Input-token distribution per cell

| Cell | Mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| pubg | 769 | 774 | 845 | 866 |
| nba | 450 | 450 | 484 | 498 |
| csgo | 624 | 626 | 665 | 682 |
| rocket_league | 822 | 820 | 905 | 920 |
| poker | 701 | 711 | 739 | 755 |

**Global: mean 673 input, p99 ≈ 920 tokens.**

Per-variant detail (matters for v5.1 cost):
- Intervention prompts add ~50 tokens (constraint context)
- Adversarial chains add ~30–60 tokens (derived-state markers)
- v5.1's strict-grounding instruction adds another 50–100 tokens

**v5.1 worst-case input estimate: 950 tokens** (used in worst-case budget calculations).

### 5.2 Cache rate

v5 had **0% prefix cache hit rate** in Phase D. Caching was not enabled. v5.1 enables Anthropic prompt caching as an efficiency improvement; budget calculations do not assume cache hits (no caching for OpenRouter routes).

### 5.3 Cluster sizes per cell

| Cell | n_chains (clean) | n_sources | Mean chains/src | Singletons | Cluster-robust SEs? |
|---|---:|---:|---:|---:|---|
| pubg | 1,200 | 1,200 | 1.00 | 1,200 | No (IID) |
| nba | 1,200 | 1,200 | 1.00 | 1,200 | No (IID) |
| poker | 1,046 | 1,046 | 1.00 | 1,046 | No (IID) |
| **csgo** | 1,200 | 364 | 3.30 | 64 | **Yes** |
| **rocket_league** | 1,200 | 148 | 8.11 | 2 | **Yes** |

For RL specifically, effective n ≈ n_chains / 8.1. At nominal n=500, effective n ≈ 62 sources — bordering on what cluster-robust McNemar can detect with c≈18 discordants. This is the load-bearing power constraint.

### 5.4 Output-token distribution

38.2% of v5 responses hit the 32-token cap, almost entirely on adversarial calls. The model wants to explain its YES answer with structured reasoning; the cap truncates mid-sentence. For parsing (first-token-after-whitespace), this does not produce false negatives, but v5.1 raises the cap to 64 tokens for clean insurance.

RL output mean is **4–6 tokens** across all variants — the model is not attempting to explain RL violations because it is not detecting them. v5.1 pre-registers this output-length-tier correlation as a confirmatory secondary analysis.

---

## 6. Budget

### 6.1 Headline budget

**Mean-case: $189 API spend / $225 with overhead** (smoke + 10% buffer + OR fee).

Computed at audited mean tokens (673 input × 14.5 output) and current verified pricing across all 32 models.

### 6.2 Worst-case ceiling

**Theoretical worst-case: $417 API / $483 total.** Realistic worst-case: $300, set as the absolute spend ceiling.

The realistic ceiling models:
- Worst-case input tokens (950 — adversarial-intervention-marker-strict condition)
- 64-token output for non-reasoning models
- 128-token output for reasoning models (realistic upper bound, not 256-cap theoretical max)

### 6.3 Per-model kill-switch

Triggered if any single model's cumulative cost exceeds 110% of its mean allocation. Run pauses, both authors review, decide to continue / drop the model / abort.

The kill-switch is per-model, not panel-wide, so failure of one model does not blow the whole run.

### 6.4 Per-model cost allocation

At measured mean tokens (673 in, 14.5 out) and verified prices:

| Model | Source | Mean cost | Worst-case | Kill-switch (110% mean) |
|---|---|---:|---:|---:|
| Haiku 4.5 | Anthropic Batches | $3.73 | $6.35 | $4.10 |
| Sonnet 4.5 | Anthropic Batches | $11.18 | $19.06 | $12.30 |
| Sonnet 4.6 | Anthropic Batches | $11.18 | $19.06 | $12.30 |
| Opus 4.7 | Anthropic Batches | $18.63 | $31.76 | $20.49 |
| GPT-5 Mini | OpenAI Batches | $0.99 | $1.83 | $1.09 |
| GPT-5 | OpenAI Batches | $4.93 | $9.14 | $5.42 |
| GPT-5.4 Mini | OpenAI Batches (256-cap) | $2.62 | $9.32 | $2.88 |
| GPT-5.4 | OpenAI Batches (256-cap) | $8.78 | $31.07 | $9.66 |
| GPT-5.5 | OpenAI Batches (256-cap) | $17.55 | $62.15 | $19.31 |
| Gemini 3.1 Flash-Lite | Google Batch | $0.95 | $1.67 | $1.04 |
| Gemini 2.5 Flash | Google Batch | $1.19 | $2.23 | $1.31 |
| Gemini 2.5 Pro | Google Batch | $4.93 | $9.14 | $5.42 |
| Gemini 3 Flash Preview | Google Batch | $1.90 | $3.34 | $2.09 |
| Gemini 3.1 Pro Preview | Google Batch | $7.60 | $13.34 | $8.36 |
| Grok 4 Fast | OR (reasoning=off) | $1.42 | $2.22 | $1.56 |
| Grok 4 | OR (256-cap) | $22.36 | $66.90 | $24.60 |
| Grok 4.20 | OR (256-cap) | $14.34 | $34.36 | $15.77 |
| Llama 3.3 70B | OR | $0.93 | $1.49 | $1.02 |
| Llama 4 Maverick | OR | $1.10 | $1.81 | $1.21 |
| DeepSeek V3.2 | DeepSeek off-peak | $0.99 | $1.63 | $1.09 |
| DeepSeek V4 Flash | DeepSeek off-peak | $0.25 | $0.38 | $0.28 |
| DeepSeek R1 | DeepSeek off-peak (256-cap) | $1.10 | $2.74 | $1.21 |
| DeepSeek V4 Pro | DeepSeek off-peak | $3.05 | $4.69 | $3.36 |
| MiMo-V2-Pro | OR | $7.16 | $11.42 | $7.88 |
| MiMo-V2.5-Pro | OR | $7.16 | $11.42 | $7.88 |
| Qwen3.6 Plus | OR | $2.47 | $4.34 | $2.72 |
| Qwen3.6 Max-Preview | OR | $7.90 | $13.88 | $8.69 |
| GLM-4.7 | OR | $3.92 | $6.40 | $4.31 |
| GLM-5 | OR | $7.19 | $11.55 | $7.91 |
| MiniMax M2.5 | OR | $1.18 | $2.16 | $1.30 |
| Kimi K2.6 | OR | $5.59 | $9.37 | $6.15 |
| Kimi K2 Thinking | OR (256-cap) | $5.07 | $11.10 | $5.58 |
| **Total** | | **$189.42** | **$417.32** | **— per-model** |

---

## 7. Pre-registered analyses

### 7.1 Primary analyses

1. **Per-cell × per-model × per-condition McNemar with appropriate SEs**
   - Cluster-robust SEs for CSGO and Rocket League (clusters: source-game)
   - IID SEs for PUBG, NBA, Poker (1:1 chains-to-sources per audit)

2. **3-way interaction**: `performance ~ intervention × marker × strict`, fit per cell × per model.
   - Predictions:
     - Poker: no significant 3-way interaction (saturated)
     - PUBG/NBA: marker × intervention significant, strict main effect ~null
     - CSGO: full 3-way interaction expected
     - RL: strict × intervention significant, marker × intervention ~null (markers cannot scaffold absent variables)

3. **Within-provider hierarchy test (CONJUNCTIVE HEADLINE)**: does the 4-tier ordering hold within each provider's capability ladder?
   - Tested per ladder: Anthropic, OpenAI, Google, xAI, DeepSeek, Moonshot
   - Moderate criterion: 3 of 4 cell-tier transitions preserve ordering (Moonshot 2-tier uses single transition)
   - Headline: "the hierarchy holds within ≥N of 6 provider ladders"
   - Replication threshold: ≥5 of 6 → "v5 replicates"; 3–4 of 6 → "model-class-dependent"; ≤2 of 6 → "v5 may have been Haiku-specific"

4. **Cross-panel hierarchy replication** (supporting evidence for primary 3): does the ordering hold across the 32-model aggregate panel?

### 7.2 Secondary analyses

5. **Tier-collapse test**: does any frontier model catch RL's indirect markers? Examines the strongest model in each ladder for tier-collapse.

6. **FP-discipline scaling on CSGO**: does capability buy resistance to confabulation FPs? Within-provider trend test.

7. **Strict-vs-non-strict on RL** (load-bearing for mechanism claim): isolates the strict-grounding contrast on the cell where the mechanism prediction is most exposed.

8. **Marker-ablation contrast**: distinguishes scaffolding (markers help) from fundamental (markers don't help). Run per cell × per model.

9. **Provider × tier ANOVA**: tests whether provider explains residual variance after capability tier is controlled.

10. **Output-length-tier correlation (Q19 confirmatory)**: per-cell output-token mean is predicted to track tier — saturated/aligned cells produce verbose YES explanations; misaligned cell (RL) produces terse NO responses regardless of intervention. v5 baseline: RL mean 4 tokens vs. 31.5 cap-bound on PUBG/NBA/CSGO/Poker adversarial-intervention. Pre-registered prediction: this output-length-tier correlation replicates across providers.

### 7.3 Multiple comparisons control

- **Primary**: Benjamini–Hochberg FDR at q=0.05, applied across all cells × models × pre-registered contrasts
- **Robustness**: Bonferroni at α=0.05/k for the same contrast set
- **Discordance reporting**: any contrast significant under FDR but not Bonferroni is reported as such, not as "robust" or "not robust." Both are informative.

The contrast list `k` is finalized before the full run begins; no post-hoc additions.

### 7.4 Cluster-robust SE policy

| Cell | SE type | Cluster definition |
|---|---|---|
| pubg | IID | n/a (1:1) |
| nba | IID | n/a (1:1) |
| poker | IID | n/a (1:1) |
| csgo | Cluster-robust | source-game (mean cluster size 3.3, 364 clusters) |
| rocket_league | Cluster-robust | source-game (mean cluster size 8.1, 148 clusters) |

Cluster definitions are locked in `clusters.json` from v5's frozen prompt corpus before any analysis runs (Task 3 in BUILD_PLAN). No cluster definition changes are permitted post-data.

For McNemar specifically, paired analyses use a clustered variant (Obuchowski-Rockette or equivalent) where cluster sizes vary across cells. Bootstrap CIs on RL detection lift use cluster-resampling (resample clusters with replacement, then chains within cluster).

---

## 8. Parse-failure handling

### 8.1 Ternary outcome variable

Every call's parse status is logged as `{parsed_yes, parsed_no, parse_failure}`. Parse failures are not treated as missing data. They are a separate outcome dimension.

### 8.2 Inclusion thresholds

- **≥98% aggregate parse rate**: model passes cleanly
- **90–98%**: model included but flagged in results
- **<90%**: model excluded from the panel
- **>5% parse failures in any single condition**: that condition flagged for the model, results retained but caveated

### 8.3 Dual treatment in analysis

For McNemar contrasts, parse failures are reported under two treatments:

1. **Failures-as-misses** (conservative for detection-rate gains): parse failures count as false negatives. This is the conservative direction for an experiment claiming detection-rate improvements under intervention.
2. **Failures-excluded** (capability-charitable): parse failures are excluded from the McNemar count.

Discordance between treatments is itself a finding — it indicates the model has condition-conditional format-following issues.

### 8.4 Format-adherence as capability signal

Per-model condition-conditional parse rates (e.g., parse rate under strict vs. non-strict) are reported per-model as a secondary capability dimension. A model with structurally-skewed parse failures (e.g., higher failure rate under strict grounding) reveals a format-adherence-under-constraint signal that is itself informative.

---

## 9. Smoke test gate

### 9.1 Smoke configuration

- 32 models × 8 conditions × 1 cell (CSGO) × 10 chain pairs = 5,120 calls
- Synchronous APIs (no batch); we need fast feedback
- Estimated cost: ~$10–15

CSGO is selected as the smoke cell because it has moderate complexity and both-direction effects (intervention raises detection AND introduces FPs). It's the most diagnostic single cell for catching parseability and routing issues.

### 9.2 Pre-registered evaluation criteria

- **Per-model parse rate**: ≥98% across all 8 conditions ⇒ proceed; 90–98% ⇒ flag and review; <90% ⇒ exclude
- **Per-model token count**: must be within ±20% of token audit predictions (CSGO mean 624 input; worst-case 780)
- **Per-model cost ground truth**: actual cost ≤110% of predicted cost; if higher, recalibrate budget before full run
- **Provider pin verification**: for each OR-routed model, all 80 calls returned by the pinned provider (no silent re-routing)
- **DeepSeek off-peak scheduler test**: verify calls outside window are deferred, not failed

### 9.3 Decision gate (both-author sign-off required)

Smoke test must satisfy:
- All models pass parseability threshold (or are pre-registered as excluded)
- Cost is within budget projection
- No silent provider routing observed
- ≥1 model in each capability tier per provider passes (so we don't lose entire ladder)

If any of these fail: pause, document, decide whether to adjust SPEC and re-sign, or proceed with reduced panel. SPEC amendments require both-author re-signature.

### 9.4 Smoke-test failure budget

Pre-registered: if 3 or fewer models fail the 90% parse rate gate, drop those models and proceed. If 4 or more fail, pause and re-evaluate the design — this would indicate a systematic harness issue or a prompt-portability problem rather than per-model capability differences.

---

## 10. Scope of claims

All v5.1 findings are conditional on:

- **Fixed prompt structure** (no per-model prompt tuning)
- **Constrained output budgets** (`max_completion_tokens=64` for non-reasoning, 256 for thinking-mode)
- **Fixed domain distributions** across the 5 cells from v5
- **Specific reasoning-mode settings** per model (off where toggleable, default where built-in)
- **One-shot evaluation** (not multi-turn, not iterative, not tool-using)

This is not a measure of unconstrained constraint-reasoning capacity. It is a measure of constraint-reasoning under the input/output constraints that v5 established and v5.1 holds fixed for cross-model comparability.

Findings about reasoning-mode models (Grok 4, GPT-5.4+, GPT-5.5, Grok 4.20, DeepSeek R1, K2 Thinking) are particularly conditional on the 256-token output cap, which forces compression of reasoning that those models would otherwise emit.

The mechanism claim does not assert causal identification of variable-availability as the driver — only systematic association. Alternative explanations (training-distribution familiarity, domain frequency in pretraining corpora) are not ruled out. A v5.2 follow-up using synthetic constraint domains controlled for training-distribution exposure is the appropriate next step if this becomes a load-bearing critique.

---

## 11. What v5.1 explicitly does NOT do

- **No new cells/domains** beyond v5's 5
- **No reasoning-mode toggling** as an experimental variable (logged, not manipulated)
- **No per-model prompt tuning** (would p-hack the comparison; preserved firm)
- **No CSGO awpy observability fix** (deferred per v5 brief; capability-vs-observability axis preserved for the heatmap)
- **No mid-flight methodology amendments** without both-author re-signature
- **No free-tier OpenRouter models** in the actual run (smoke only; production-stable pinning required)
- **No Anthropic prompt caching changes** beyond enabling it (the prefix structure is unchanged from v5)
- **No reduced-n on RL** below 500 chain pairs (this is the load-bearing strict-grounding contrast; cutting it is not permitted)

---

## 12. Reuse from v5 (frozen, unchanged)

These v5 components are reused as-is. No modifications permitted:

- Prompt corpus (5 cells, all per-cell baseline + intervention prompts)
- Chain construction logic (`src/interfaces/chain_builder.py`)
- Chain-to-prompt translation (`src/interfaces/translation.py`)
- Per-cell data pipelines (`src/cells/{cell}/pipeline.py`)
- Per-cell evidence extractors (`src/cells/{cell}/extractor.py`)
- Violation injectors (`src/harness/violation_injector.py`)
- Statistical analysis core (`synthesize_phase_d.py` extends to `synthesize_cross_model.py`)
- Decision log discipline (continues from D-44)

---

## 13. New for v5.1

- 4 native-API integrations (Anthropic Batches, OpenAI Batches, Google AI Studio Batch, DeepSeek)
- OpenRouter integration with per-model provider pinning
- Async parallel runner across providers
- Strict-grounding prompt variant generator (third axis)
- No-marker variant generator (second axis)
- Parse-failure ternary outcome logging
- Cluster-robust SE computation for CSGO + RL
- Deterministic request IDs for retry safety
- Per-call usage + routing metadata logging
- Anthropic prompt caching (improvement vs. v5's 0% cache hit rate)
- DeepSeek off-peak scheduler

---

## 14. Pre-flight gates (must complete before full run)

1. Token audit on v5's logs — **complete** (Q1–Q5 resolved)
2. Pricing verification on all 32 models — **complete**
3. Provider pinning configuration locked per OR-routed model — Task 2 in BUILD_PLAN
4. Cluster definitions locked in `clusters.json` — Task 3 in BUILD_PLAN
5. Smoke test passes pre-registered criteria — Task 6 in BUILD_PLAN
6. Both authors sign SPEC + red-team appendix — required pre-smoke
7. OSF DOI generated and recorded — required pre-full-run

---

## 15. Pre-mortem (top risks)

The full risk register is in PRE_MORTEM.md. Headline risks:

1. **Token count higher than audit predicts** → cost overrun. Mitigated by per-model kill-switch at 110% of mean allocation.
2. **Anthropic cache fragmentation under 8 conditions** → reduced cache hit rate vs. expected. Mitigated by 10% buffer in budget.
3. **Some models fail parseability on strict grounding** → drop via smoke gate.
4. **Cross-model results don't replicate v5's hierarchy** → publishable as model-specific finding via the conjunctive within-provider headline framing, not failure.
5. **Reasoning-mode models exceed 256-token cap** → output truncated mid-reasoning. Mitigated by first-token parse logic.
6. **OpenRouter silent provider routing** → confounds cross-model variance. Mitigated by `allow_fallbacks: false` + per-call resolved-provider logging.
7. **DeepSeek off-peak window misses** → 4× cost overrun on DeepSeek slice. Mitigated by hard scheduler check in harness.
8. **3-way interaction more complex than predicted** → main-effect mechanism claim becomes misleading. Mitigated by pre-registering interaction test as primary, not robustness.
9. **Within-provider hierarchy holds in <5 of 6 ladders** → headline finding becomes "model-class-dependent" rather than "v5 replicates." This is itself publishable; pre-registered.
10. **Capability-distribution confound holds even with within-provider design** → cannot rule out training-corpus alternative. Acknowledged as scope limit; v5.2 with synthetic domains is the appropriate follow-up.

---

## 16. Sign-off

This SPEC is the pre-registration document for v5.1. It is filed prospectively on OSF before any model is hit at full-run scale. No mid-flight methodology amendments are permitted without both-author re-signature.

Companion documents (filed alongside this SPEC on OSF):
- BUILD_PLAN.md (build phase plan)
- PRE_MORTEM.md (full risk register)
- red_team_appendix.md (R1–R10 critique-and-resolution log)
- MEMO.md (token audit findings)
- DECISION_LOG.md (numbered decisions, continuing from v5's D-44)
- clusters.json (Task 3)
- provider_pinning.json (Task 2)

**Author 1 (Safiq):** ___________________ Date: ___________

**Author 2 (Myriam):** ___________________ Date: ___________

**OSF DOI** (assigned at filing): ___________________

---

*This document is the canonical pre-registration for v5.1. It supersedes any prior SPEC drafts. It does not modify any of v5's frozen artifacts.*
