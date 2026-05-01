# OSF Pre-Registration — Project Ditto v5.1

**Cross-model evaluation of constraint-reasoning behavior under suppressed-CoT inference, with provider-stratified disclosure of inference-regime heterogeneity.**

---

## Document metadata

| Field | Value |
|---|---|
| Project | Ditto v5.1 |
| Document | OSF pre-registration (v2 — addresses two-reviewer feedback 2026-05-01) |
| Authors | Safiq Sindha, Myriam Khalil (full CRediT roles per §12) |
| Status | Pending OSF filing after final revision sign-off |
| OSF DOI | _to be assigned at filing_ |
| Filed at | _date of OSF submission_ |
| Code commit hash | _to be inserted at filing time — `git rev-parse HEAD` after final commit_ |
| Prompt corpus SHA-256 | _to be inserted at filing — hash of the frozen `prompt_corpus_v5_1.jsonl` artifact_ |
| Supersedes | SPEC.md §2.2, §2.3, §2.5, §3.1, §4.1, §5.3, §6, §7.1, §7.4, §8.2 — see §0 below |

---

## §0. Deviations from SPEC.md (full disclosure)

This pre-registration **does not** simply operationalise SPEC.md. The smoke-test iteration cycles (2026-04-30 to 2026-05-01) surfaced empirical constraints that required substantive design changes. We enumerate them here so reviewers and replicators can compare the two documents side-by-side without inferring intent.

### 0.1 Headline finding changed

| | SPEC.md | OSF_PREREG (this doc) |
|---|---|---|
| Headline test | Conjunctive within-provider hierarchy: "v5 replicates if ≥5 of 6 capability ladders show the moderate-criterion 4-tier hierarchy" | Mixed-effects logistic regression of the condition main effect with model-stratified heterogeneity tests |
| Statistical model | Per-cell McNemar with cluster-robust SEs aggregated across cells | Multi-level GLM with crossed random effects on `model` and `chain_id` |
| Replication thresholds | Strict (6/6), moderate (5/6), weak (3-4/6) ladders | Pre-registered minimum-detectable-effect threshold for H1 (see §8.5); H2/H3 require interaction terms exceeding pre-specified Cohen's d |

**Why changed:** The within-provider ladder framing required ≥3 models per provider on a known capability gradient. The post-exclusion panel (§2.3) leaves only 1-3 providers with ≥3 models on a credible gradient (Anthropic 3, Google 4, OpenAI 3). A conjunctive 6-ladder test is no longer testable. We retain the spirit (cross-model heterogeneity matters) but operationalise via mixed-effects rather than ladder voting.

### 0.2 Sample-size allocation changed

| | SPEC.md | OSF_PREREG |
|---|---|---|
| Per-cell n | 150 (Poker, PUBG, NBA), 300 (CSGO), 500 (Rocket League) — varied by data ceiling and effective n | Flat 100 chain pairs per cell |
| Total chain pairs | 1,400 | 500 |
| Justification | RL needs effective n=63 due to ICC clustering; CSGO needs 200 effective; others need ~100 | See §5.2 — power calculation under crossed random effects |

**Why changed:** The flat-100 allocation is sufficient for the H1 effect under the new mixed-effects analysis (see power calc in §8.5). RL's higher ICC means RL specifically retains lower power for cell-stratified analysis; this is acknowledged in §8.6 as a sensitivity concern.

### 0.3 Model panel reduced 32 → 22

See §2.2 (retrospective exclusions) and §2.3 (final panel).

### 0.4 New methodological scope condition added

**Constrained-inference regime.** All non-reasoning models are run with provider-specific parameters that suppress hidden chain-of-thought (`reasoning:enabled:false` on OR routes, `reasoning_effort:minimal` on OpenAI, `thinking:type:disabled` on Moonshot direct). The empirical evidence for why this matters is in §6.3, with formal stratification by inference-regime in §8.4. SPEC.md does not contemplate this regime.

### 0.5 Cluster-robust SEs replaced with mixed-effects random terms

SPEC.md §7.4 specifies cluster-robust standard errors. We use mixed-effects random terms instead, since the two approaches are not compatible (cluster-robust SEs are post-hoc adjustments to fixed-effects models; mixed-effects models account for clustering at fit time via random effects).

### 0.6 Cost target dropped $189 → ~$70

The reduced panel (22 vs 32) and reduced per-cell sample (100 vs 150-500) drop projected cost. The $189 budget remains the cap, not the target.

### 0.7 Precedence order of definitions (binding)

When multiple sections define the same construct, the following precedence applies:

1. **§1 (Hypotheses)** — definitions here are the *inferential* authority. H1's estimand, MDE, and falsification rule cannot be overridden elsewhere.
2. **§8 (Statistical analysis plan)** — definitions here are the *operational* authority. Specific regression formulas, fallback hierarchy, and test statistics override less-specific descriptions in §1.
3. **§2 (Model panel) and §6 (Inference settings)** — definitions here are the *design* authority. Routing decisions and inference parameters override generic descriptions elsewhere.
4. **§7 (Parsing rule), §9 (Stopping rules)** — definitions here are the *execution* authority for outcome coding and run-time decisions.
5. **Appendices and SPEC.md** — informational only; do not override pre-registered text.

If a reviewer or replicator finds an apparent contradiction, the lower-numbered section wins for inferential matters and the higher-numbered section wins for operational matters at the same level.

---

## §1. Hypotheses (pre-specified, falsifiable, directional)

### 1.1 Outcome variable definitions

For every (model, chain, condition) triple, the model produces a response parsed (per §7) into one of `{yes, no, abstain}`. We define:

- `yes_rate(model, chain_class, condition) = P(parsed = "yes" | model, chain_class, condition)`
- `chain_class ∈ {clean, adversarial}` per the v5 chain construction (clean = injector-passing chain, adversarial = same chain with a violation injected by the v5 violation_injector module).
- `detection_rate(model, condition) = 1 - yes_rate(model, "adversarial", condition)` — i.e., the rate at which the model correctly says NO to adversarial chains.

**Primary outcome:** `detection_rate` on adversarial chains.
**Secondary outcome:** `yes_rate` on clean chains (specificity; should not change much under intervention).

### 1.2 Hypotheses

**H1 (primary, directional, one-sided).** The intervention condition (constraint context appended to the chain) **increases** `detection_rate` on adversarial chains compared to the baseline condition, **averaged across the regime mixture present in the panel** (see §2.3, §6.2 for regime composition). This is the v5 finding, replicated cross-model under the specific inference-regime mixture of this study. The pre-specified minimum effect size (MDE) is **log-odds = +0.20 on the marginal contrast scale**, where the marginal contrast is computed via `emmeans::emmeans(fit, ~intervention) |> contrast()` on the §8.1 fitted model — **not** on the raw fixed-effect coefficient scale. (In nonlinear mixed models with random slopes and interactions, marginal effects ≠ fixed-effect coefficients; aligning the MDE with the actual test statistic prevents misinterpretation.) The probability mapping at v5-observed baseline rates of 0.40–0.60 corresponds to approximately +0.04 to +0.05 in detection probability; this mapping is given for interpretability only.

**H1 carries a binding scope condition.** The estimand is the marginal intervention effect over the regime mixture, **not** the effect that would obtain if all models were run in their default deployment state. We acknowledge this explicitly and (a) include `intervention × regime_stratum` as a fixed-effect interaction in the primary regression so the mixture's regime-shift heterogeneity is not silently absorbed into the main effect, and (b) report regime-stratified estimates as primary supplementary tables (not just a sensitivity trigger).

**Primary-estimand precedence (binding).** The **only** primary estimand is the marginal intervention effect averaged over the empirical regime distribution, computed via `emmeans::emmeans(fit, ~intervention) |> contrast()` on the §8.1 fitted model. All regime-stratified analyses (§8.4) and routing-partition analyses (§8.7) are descriptive supplementary results that **cannot alter** the H1 primary inferential outcome. The §8.4 escalation rule changes *reporting priority* (which estimate appears first in the abstract and discussion), not *inferential authority* — H1's pass/fail status against its MDE and FDR threshold is determined solely by the marginal estimand.

**H2 (secondary, two-sided).** The intervention effect on `detection_rate` is heterogeneous across the model panel: the `intervention × model` interaction term has at least one **shrinkage-adjusted (BLUP)** coefficient with absolute log-odds ≥ 0.5 after **single-global-family** FDR correction at q = 0.05 (per §8.6). Because we use partial pooling, BLUPs shrink toward zero; the |log-odds| ≥ 0.5 threshold is therefore a **conservative** test of heterogeneity.

If the §8.1 random-effects fallback hierarchy reaches step 3 (drop `(intervention | model)`), H2 is instead evaluated via **unpooled per-model fixed-effect contrasts** (no shrinkage) using the same |log-odds| ≥ 0.5 + FDR threshold. This deviation will be explicitly noted in the manuscript's methods section.

**H3 (secondary, two-sided).** The intervention effect on `detection_rate` varies by domain cell: the `intervention × model × cell` three-way interaction has at least one shrinkage-adjusted BLUP coefficient with absolute log-odds ≥ 0.5 after single-global-family FDR correction at q = 0.05 (per §8.6). Same conservativeness caveat and same fallback handling as H2.

**Specificity null (S1).** Intervention does not increase `false_positive` rate (P(parsed = "yes")) on clean chains by more than log-odds = +0.20 (≈ +0.04 in probability at v5-observed baselines). Failure of S1 would qualify any H1 success as a possible response-bias artifact rather than improved constraint reasoning.

**ITT-style outcome (S2, parallel sensitivity to H1).** To address potential post-treatment bias from abstain-conditioning (since `abstain` is excluded from H1's primary regression and intervention may affect abstention rates), we pre-register an intent-to-treat outcome:

```text
detection_correct_itt = 1 if parsed = "no" else 0
                       (i.e., "yes" or "abstain" both count as not-detected)
```

**S2 is interpreted as a conservative lower bound on detection** under the assumption that any abstention is equivalent to a missed violation. It is **not** a behavioral-equivalence claim that abstentions are semantically identical to incorrect "yes" responses (abstentions could reflect uncertainty, refusal, or format failure — distinct latent constructs not separated here).

H1 is re-estimated with `detection_correct_itt` as the outcome on the full adversarial-chain dataset (no abstain exclusion). Pre-registered decision rule: if the H1 effect estimate using `detection_correct_itt` differs from the primary estimate by more than |log-odds| = 0.2, the ITT estimate becomes co-primary in reporting; otherwise the primary estimate stands and ITT is supplementary.

### 1.3 Falsification rules

- **H1 fails** if the panel-averaged effect on adversarial chains is < +0.05 OR the 95% confidence interval includes zero.
- **H2 / H3 fail** if no interaction term meets the |log-odds| ≥ 0.5 threshold post-FDR.
- **S1 fails** if intervention raises clean-chain `yes_rate` by ≥ +0.05.

We commit to reporting the test results as primary even if any combination of H1, H2, H3, or S1 fails.

---

## §2. Model panel

### 2.1 Within-family standard-vs-reasoning pairs preserved

**N = 2 pairs.** This is below the SPEC.md target of 4 pairs and limits any reasoning-class interaction inference to exploratory status.

- **OpenAI**: gpt-5 (standard) vs gpt-5.4-mini (reasoning)
- **xAI**: grok-4-fast (standard) vs grok-4.20 (reasoning)

**Lost pairs** (relative to SPEC.md):
- DeepSeek (deepseek-v3.2/v4 vs deepseek-r1) — all three deprecated by provider; see §2.2.
- MoonshotAI (kimi-k2.6 vs kimi-k2-thinking) — kimi-k2-thinking exceeds budget; see §2.2.

We explicitly **withdraw** SPEC.md §2.5's secondary hypothesis 3 (reasoning-trained vs non-reasoning models on tier-3 RL) as untestable in this panel.

### 2.2 Retrospective exclusions (10 models — exploratory, not pre-registered)

**These exclusions are post-hoc.** The smoke-test iteration cycles (5 cycles, 2026-04-30 to 2026-05-01) surfaced these failures and informed the prospective inclusion criteria below. We disclose this transparently rather than retrospectively framing these exclusions as pre-specified.

| Model | Smoke evidence | Exclusion reason |
|---|---|---|
| `claude-opus-4-7` | Per-call cost $0.006 → $77 projected on full run | Cost ceiling |
| `gpt-5.4` | Per-call cost $0.004 → $52 projected | Cost ceiling |
| `gpt-5.5` | Per-call cost $0.008 → $105 projected | Cost ceiling |
| `x-ai/grok-4` | Per-call cost $0.007 → $93 projected | Cost ceiling |
| `minimax/minimax-m2.5` | API returns 400 to `reasoning:enabled:false` ("Reasoning is mandatory for this endpoint and cannot be disabled"); empty content under all attempted configurations; per-call cost at 4096 tokens → $32 projected | Mandatory inference mode |
| `gemini-2.5-pro` | Google docs explicitly state "Cannot disable thinking" for 2.5 Pro; 96% empty content over 160 calls in smoke #4 (`results/v5_1/smoke_20260501_162949/raw/gemini-2.5-pro_results.jsonl`) | Mandatory inference mode |
| `moonshotai/kimi-k2-thinking` | Provider (Novita) does not honor `max_tokens`; observed median 1,784 / max 18,182 output tokens; per-call cost projection $55 over 8K calls; not exposed via direct Moonshot API | Cost ceiling |
| `meta-llama/llama-4-maverick` | Verbose preface format ("To determine if the sequence of events is consistent..."); 32% strict-parse rate with `reasoning:enabled:false`; no documented format-control parameter | Format-adherence floor |
| `deepseek-v3.2` | API returns HTTP 400: "supported model names are deepseek-v4-pro or deepseek-v4-flash" (deprecation 2026-04) | Provider deprecation |
| `deepseek-v4` | Same deprecation 400 | Provider deprecation |
| `deepseek-r1` | Same deprecation 400 | Provider deprecation |

### 2.3 Final 22-model panel (locked)

| # | Model | Lab | Routing | Reasoning class | Inference-regime stratum (§8.4) |
|---|---|---|---|---|---|
| 1 | claude-haiku-4-5-20251001 | Anthropic | Anthropic Batch API | standard | default (no flag) |
| 2 | claude-sonnet-4-5 | Anthropic | Anthropic Batch API | standard | default (no flag) |
| 3 | claude-sonnet-4-6 | Anthropic | Anthropic Batch API | standard | default (no flag) |
| 4 | gpt-5 | OpenAI | OpenAI sync (smoke) / Batch (full) | standard | reasoning_effort=minimal |
| 5 | gpt-5-mini | OpenAI | OpenAI sync / Batch | standard | reasoning_effort=minimal |
| 6 | gpt-5.4-mini | OpenAI | OpenAI sync / Batch | reasoning | reasoning_effort default (model is reasoning-class) |
| 7 | gemini-3.1-flash-lite-preview | Google | Gemini Batch API | standard | default (no flag) |
| 8 | gemini-2.5-flash | Google | Gemini Batch API | standard | default (no flag) |
| 9 | gemini-3-flash-preview | Google | Gemini Batch API | standard | default (no flag) |
| 10 | gemini-3.1-pro-preview | Google | Gemini Batch API | standard | default (no flag) |
| 11 | deepseek-v4-pro | DeepSeek | DeepSeek sync (off-peak) | standard | default (no flag) |
| 12 | deepseek-v4-flash | DeepSeek | DeepSeek sync (off-peak) | standard | default (no flag) |
| 13 | x-ai/grok-4-fast | xAI | OpenRouter → xAI | standard | reasoning:enabled:false |
| 14 | x-ai/grok-4.20 | xAI | OpenRouter → xAI | reasoning | default (model is reasoning-class) |
| 15 | meta-llama/llama-3.3-70b-instruct | Meta | OpenRouter → WandB | standard | reasoning:enabled:false (no-op per §6.3) |
| 16 | qwen/qwen3.6-plus | Alibaba | OpenRouter → Alibaba | standard | reasoning:enabled:false |
| 17 | qwen/qwen3.6-max-preview | Alibaba | OpenRouter → Alibaba | standard | reasoning:enabled:false |
| 18 | xiaomi/mimo-v2-pro | Xiaomi | OpenRouter → Xiaomi | standard | reasoning:enabled:false |
| 19 | xiaomi/mimo-v2.5-pro | Xiaomi | OpenRouter → Xiaomi | standard | reasoning:enabled:false |
| 20 | z-ai/glm-4.7 | Z-AI | OpenRouter → Cerebras | standard | reasoning:enabled:false |
| 21 | z-ai/glm-5 | Z-AI | OpenRouter → Baidu | standard | reasoning:enabled:false |
| 22 | kimi-k2.6 | MoonshotAI | Direct Moonshot API | standard | thinking:type:disabled |

**Model identifier verification.** _**MUST FILL BEFORE FILING**: Add an appendix table with provider model-card URL, retrieval timestamp, and observed per-token pricing for each of the 22 model identifiers above. Identifiers must be re-verified within 24 hours of OSF filing._

### 2.4 Prospective inclusion criteria (forward-looking, for replication or extension)

Researchers extending or replicating this study should apply these criteria to any candidate model:

1. **Cost cap.** Per-model projected cost ≤ 110% of the analyst's mean per-model budget allocation in smoke-to-full extrapolation.
2. **Documented or empirically-confirmed format compliance.** Either (a) provider documents that hidden inference (reasoning, thinking, internal CoT) can be disabled via API parameter AND empirical strict-parse rate ≥ 90% under the prompt regime in §3.1, OR (b) the model produces ≥ 90% strict-parse rate in default mode.
3. **Provider availability.** Model identifier returns successful API responses (HTTP 2xx) at the time of smoke testing.

We acknowledge that criteria (1)–(2) are *generalisations from* the smoke evidence in §2.2, not pre-registered specifications independent of the smoke iterations. They are offered as forward-looking guidance; we do not claim they were specified before observing the smoke data.

---

## §3. Prompts (locked, inlined)

### 3.1 Classification question (verbatim, all cells)

```
Is the sequence of events above consistent with the rules of {domain}?
Answer with exactly one token:
YES
NO
If you output anything else, the answer is invalid.
```

`{domain}` substitutions per cell:

| Cell | `{domain}` |
|---|---|
| poker | poker |
| pubg | PUBG |
| nba | NBA basketball |
| csgo | Counter-Strike |
| rocket_league | Rocket League |

### 3.2 Per-cell constraint context (intervention condition only)

The constraint-context strings are inlined verbatim below. They are also frozen in `src/harness/prompts.py` at the code commit hash referenced in metadata, and a SHA-256 hash of the rendered prompt corpus is recorded.

**poker:**
```
Rules: fold/check/call/bet/raise up to stack. Cannot wager more than held.
Action proceeds clockwise. Folded player can't act again. Best 5-card hand
wins at showdown.
```

**pubg:**
```
Players must stay inside the shrinking safe zone; remaining outside deals
damage that increases per phase. An eliminated player cannot act. Squads
consist of up to 4 players who can revive downed teammates.
```

**nba:**
```
Offensive team must shoot within 24 seconds of gaining possession. A player
with 6 fouls is ejected. Possession changes on made shots, turnovers,
defensive rebounds.
```

**csgo:**
```
T wins by detonating bomb or eliminating CTs. CT wins by defusing bomb,
eliminating Ts, or time expiring. Eliminated players don't respawn until
next round. Bomb plants only at sites A or B.
```

**rocket_league:**
```
Teams of 3 score by hitting the ball into the opposing goal. Boost meter
caps at 100; collected from pads. A goal resets ball and player positions
for a kickoff.
```

### 3.3 Chain rendering

Chain events are rendered with a per-chain-local actor anonymisation map (`Player_0`, `Player_1`, …) per CF-4=B in SPEC.md §3. The mapping is deterministic given a fixed RNG seed (see §10.2). No team identity, score, or post-game state appears on per-event lines.

### 3.4 Frozen prompt corpus

Before the full run is launched, all 500 chain pairs (5 cells × 100 chain pairs) × 8 conditions = 4,000 distinct prompt strings will be materialised as a single JSONL artifact `prompt_corpus_v5_1.jsonl`. Its SHA-256 hash is recorded in document metadata. Any deviation (even a whitespace change) between this artifact and the prompts sent to models invalidates the pre-registration.

---

## §4. Conditions (8, factorial 2 × 2 × 2)

The condition factorial is locked from SPEC.md §3.1:

| # | Condition label | intervention | marker | strict |
|---|---|---|---|---|
| 1 | baseline_marker_strict | 0 | 1 | 1 |
| 2 | baseline_marker_nonstrict | 0 | 1 | 0 |
| 3 | baseline_nomarker_strict | 0 | 0 | 1 |
| 4 | baseline_nomarker_nonstrict | 0 | 0 | 0 |
| 5 | intervention_marker_strict | 1 | 1 | 1 |
| 6 | intervention_marker_nonstrict | 1 | 1 | 0 |
| 7 | intervention_nomarker_strict | 1 | 0 | 1 |
| 8 | intervention_nomarker_nonstrict | 1 | 0 | 0 |

**H1 tests the marginal `intervention` main effect** averaging over `marker` and `strict`. The full 2×2×2 decomposition is reported in supplementary tables.

---

## §5. Sampling design

### 5.1 Quantities

| Quantity | Value |
|---|---|
| Cells | 5 (poker, pubg, nba, csgo, rocket_league) |
| Chain pairs per cell | 100 |
| Conditions per chain pair | 8 |
| Calls per (chain × condition) | 1 (one shot; baseline+intervention are different conditions) |
| Models | 22 |
| **Total API calls** | 5 × 100 × 8 × 22 = **88,000** |
| **Estimated cost (sync + batch mix)** | **~$70** (under $189 budget cap) |
| **Estimated wall time** | 2–4 hours (parallel across models) |

**Note on pre-registration v1 error:** earlier draft listed 176,000 calls assuming 2 calls per condition. The correct count is 88,000 (1 call per (chain, condition) tuple — baseline and intervention ARE the conditions, not separate calls within a condition).

### 5.2 Power justification

For H1 (panel-averaged intervention effect on `detection_rate`), we have:
- 22 models × 5 cells × 100 chains × 4 conditions per arm (baseline / intervention) = 44,000 observations per arm.
- Pre-specified MDE: +0.05 in `detection_rate` (panel-averaged).
- Under the v5 observed within-model intervention effect (median across cells: +0.30; range: +0.10 to +0.45), the smallest expected effect is well above MDE.

For H2 (per-model effects), each model contributes 5 × 100 × 4 = 2,000 observations per arm. With ICC ≈ 0.2 (estimated from v5 cross-cell residuals), effective n per model ≈ 250 per arm. Standard sample-size calculations give 80% power to detect log-odds = 0.5 at α = 0.05 with this n, which is the minimum interaction effect size we pre-register as meaningful.

For H3 (cell-by-model interaction), per-cell n per model = 400; effective n ≈ 50. Power for the 110-cell × model interactions is lower; we acknowledge this in §8.6 as a sensitivity concern. Replicators with budget for higher per-cell n could improve H3 power.

---

## §6. Inference settings (constrained-inference regime — disclosed)

### 6.1 Output-token caps

| Class | Stage 1 (sync, retry) | Stage 2 (sync recovery) | Batch | Reasoning models |
|---|---|---|---|---|
| Standard | `max_tokens=96` + `stop=["\n"]` | `max_tokens=1024` | `max_tokens=1024` | n/a |
| Reasoning | n/a | n/a | `max_tokens=384` | `max_tokens=384` |

### 6.2 Reasoning-disable parameters (per-routing)

| Routing | Parameter | Effect |
|---|---|---|
| OpenRouter | `extra_body={"reasoning":{"enabled":false}}` | Provider-side suppression of CoT block (varies by upstream) |
| OpenAI direct (sync + batch) | `reasoning_effort="minimal"` | Reduces but does not eliminate reasoning tokens |
| Moonshot direct (kimi-k2.6) | `extra_body={"thinking":{"type":"disabled"}}` | Disables thinking channel |
| Anthropic / Google batch / DeepSeek | none exposed | Default behavior, possibly with hidden CoT |

**Critical asymmetry:** these parameters are not equivalent operations and not all providers expose them. See §8.4 for the regime stratification this requires in analysis.

### 6.3 Reasoning-control validation experiment

#### 6.3.1 Pilot (4 OR models, 25 chains)

`results/v5_1/reasoning_control_20260501_192146.json`. 4 OR-routed models × 25 chains × 2 conditions (ON, OFF) × 2 calls (baseline, intervention) = 400 calls. Per-model strict-agreement rates between ON and OFF conditions:

| Model | Strict agreement |
|---|---|
| qwen/qwen3.6-plus | 76% |
| z-ai/glm-4.7 | 52% |
| xiaomi/mimo-v2-pro | 44% |
| meta-llama/llama-3.3-70b-instruct (control, no hidden reasoning) | 100% |

Three of four reasoning-capable OR models tested fall below the 95% agreement threshold for "neutral optimization." **This is preliminary evidence that the reasoning-disable flag changes the evaluated object on reasoning-capable models routed via OR.**

#### 6.3.2 Pre-filing expansion (executed before filing)

Re-ran the reasoning-control experiment on **all 13 flag-bearing panel models** at 25 chains each (650 chain-pair tests × 2 conditions × 2 calls = 2,600 calls). The 9 default-mode models (3 Anthropic + 4 Google + 2 DeepSeek) are excluded by design because their public APIs expose no reasoning-disable parameter — the §8.4 regression `lm(per_model_intervention_effect ~ agreement_rate, data = flag_models_only)` operates on the 13-model subset for this reason.

Per-model agreement rates from this expanded run are archived as `reasoning_control_full_panel_<timestamp>.json` and serve as the empirical foundation for §8.4. The pilot pattern from §6.3.1 (3 of 4 reasoning-capable OR models below 95% agreement) is the prior; the expanded run characterises the same property across the full flag-bearing panel.

#### 6.3.3 Scope statement (binding)

The full-run results characterise model behavior **under the constrained-inference regime described in §6.1–§6.2**, not default deployed behavior. We are explicit that:

1. For 9 of 22 models (Anthropic 3, Google 4, DeepSeek 2), no reasoning-disable parameter exists on the relevant API; these models run in their **default** state. Whether their default state includes hidden CoT is provider-dependent and not directly verifiable from outside.
2. For 13 of 22 models, a reasoning-disable parameter is set; the §6.3.1 pilot shows this materially changes answers on the OR subset tested.
3. The H1, H2, H3 results are valid as cross-model comparisons **within this regime mixture** but should not be interpreted as characterising default-deployment behavior of any individual model.

We commit to reporting agreement rates from §6.3.2 as a covariate in supplementary analysis (§8.4).

### 6.4 Two-stage retry policy (sync only)

Sync calls (OpenRouter, MoonshotRunner, OpenAI sync, DeepSeek) follow:

1. **Stage 1**: `max_tokens=96`, `stop=["\n"]`. If non-empty content AND `finish_reason ≠ "length"`, return.
2. **Stage 2**: `max_tokens=1024`, no stop sequences. Stage-2 response is the unit of analysis; Stage-1 response is logged for audit only and never enters the regression.

Cost from both stages is summed for accounting; **the parsed answer is from the Stage-2 response when Stage-2 fires**, otherwise from Stage 1.

Batch calls (Anthropic, OpenAI batch, Google batch) skip Stage 1 (cannot retry mid-batch) and submit directly at the larger cap.

**Retry indicator as covariate.** The boolean `needed_retry` is recorded per call and included as a fixed-effect covariate in the §8 regression to absorb any systematic retry-related response-distribution shifts.

### 6.5 Pathological-model auto-kill

Per `runner_native.PATHOLOGICAL_*`: rolling sliding window of the most recent 50 `_call()` outcomes; trigger model thread termination if the empty fraction ≥ 50% after a **20-call warmup independent of condition assignment** (i.e., the warmup is global per model, not per-condition; first 20 calls accumulate before the threshold can fire). The 50% threshold is set conservatively above the highest observed empty rate in surviving smoke-test models. This guardrail's smoke-test behavior is documented in Appendix A.

---

## §7. Parsing rule (locked)

### 7.1 Strict parser (primary)

```python
def _parse_response(text: str) -> str:
    if not text:
        return "abstain"
    # Exhaustively strip leading markdown wrappers — handles nested cases
    # like ```**YES**``` where a single-pass loop would leave residue.
    stripped = text.strip()
    while True:
        start_len = len(stripped)
        for wrapper in ("```", "``", "`", "**", "__", "*", "_"):
            if stripped.startswith(wrapper):
                stripped = stripped[len(wrapper):].lstrip()
        if len(stripped) == start_len:
            break
    parts = stripped.split()
    if not parts:
        return "abstain"
    # Strip trailing punctuation AND markdown closers from the first token.
    first = parts[0].lower().rstrip(".,!?:;\"'`*_")
    if first in ("yes", "y"):
        return "yes"
    if first in ("no", "n"):
        return "no"
    return "abstain"
```

First-token-anchored after markdown-wrapper stripping. Empty content or first-token-not-in-{yes, y, no, n} → "abstain". Format adherence is treated as a precondition for inclusion in the primary analysis. The implementation in `src/harness/runner_native.py::_parse_response` is the authoritative version; this snippet must match byte-for-byte at the filing commit hash.

### 7.2 Lenient parser (sensitivity)

```python
def _parse_response_lenient(text: str) -> str:
    # Whole-line YES/NO scan; first line of stripped text only.
    # Ambiguous (both YES and NO on the same line) → "abstain".
```

Used as a sensitivity check only. **Pre-registered contingency:** if strict and lenient parsers disagree by > 5 percentage points on the H1 effect estimate for any model, both are reported in the main results table; primary inference remains based on the strict parser.

### 7.3 Inclusion floor

A model is included in the primary regression if and only if its strict-parse rate (averaged across all conditions) is **≥ 90% (inclusive)**. Models below this floor are reported in supplementary tables but excluded from H1, H2, H3, S1, S2 inferential tests. Smoke #5 verified all 22 panel models meet this floor (deepseek-v4-pro at exactly 90%; the inclusive boundary is binding).

---

## §8. Statistical analysis plan

### 8.1 Primary regression (H1)

We fit a mixed-effects logistic regression in R using `lme4::glmer` with the `bobyqa` optimizer, max iterations 100,000, convergence tolerance 1e-7:

```r
glmer(
  detection_correct ~ intervention * marker * strict
                    + intervention * regime_stratum    # critical: regime moderation
                    + cell
                    + needed_retry
                    + (1 | model)
                    + (intervention | model)
                    + (1 | chain_id)
                    + (1 | model:chain_id),
  data    = adversarial_chains_only,
  family  = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa")
)
```

**Outcome.** `detection_correct` = 1 if model's parsed response on an adversarial chain is "no" (correct detection), 0 if "yes" (false positive). Abstain responses are excluded here; the parallel ITT specification (S2 in §1.2) refits with abstains coded as 0 to bound the abstain-conditioning bias.

**Fixed effects:**
- `intervention * marker * strict`: full 2×2×2 condition factorial.
- `intervention * regime_stratum`: explicit interaction so the panel-averaged H1 estimate is not silently absorbing regime-shift moderation. The marginal `intervention` main effect (averaged over `regime_stratum` levels via `emmeans`) is the H1 estimand.
- `cell`: 5-level factor (poker, pubg, nba, csgo, rocket_league).
- `needed_retry`: boolean covariate (see §6.4) — absorbs systematic differences induced by Stage-2 longer-generation fallback.
- `regime_stratum`: 6-level factor (default-anthropic, reasoning-effort-openai, default-google, default-deepseek, reasoning-disabled-or, thinking-disabled-moonshot).

**Random effects (full model — pre-specified simplification order if non-convergent):**
- `(1 | model)`: random intercept per model.
- `(intervention | model)`: random slope on intervention per model (the H2 quantity of interest, partial-pooled).
- `(1 | chain_id)`: random intercept per chain (item difficulty shared across models).
- `(1 | model:chain_id)`: model-specific deviation per chain.

**H1 test.** Estimated marginal mean (EMM) of the **intervention contrast** across `regime_stratum` levels via `emmeans::emmeans(fit, ~intervention) |> contrast()`. Reported as: log-odds estimate of the contrast, 95% CI from profile likelihood, z-test p-value (one-sided since H1 is directional).

**Non-convergence — pre-specified fallback hierarchy.** If `glmer` fails to converge after 100,000 iterations with `bobyqa`:

| Step | Action | Rationale |
|---|---|---|
| 1 | Try `nlminbwrap` optimizer with same model | Different optimizer may find the optimum |
| 2 | Drop `(1 \| model:chain_id)` (model × item interaction) | This is the heaviest random term; only 8 obs per cell |
| 3 | Drop random slope `(intervention \| model)` → keep `(1 \| model)` only | Removes H2's partial-pooled estimate; H2 instead estimated from per-model fixed-effect contrasts |
| 4 | Refit with `glmmTMB::glmmTMB` (Laplace approximation) | Different fitting backend |
| 5 | If all four fail: report convergence failure as primary result; H1 not estimated | Honest failure |

The **first** strategy that converges is the primary result. Which strategy was used is reported in the methods section. Earlier-step results from converged-but-warning fits are reported in supplementary tables.

### 8.2 H2 (per-model heterogeneity)

The random slopes from `(intervention | model)` provide the per-model effects. We test the model × intervention interaction by comparing log-likelihoods between the full model (with `(intervention | model)`) and a reduced model (with only `(1 | model)`) via likelihood-ratio test. If significant, we extract per-model BLUPs and test each against zero with profile-likelihood CIs at q = 0.05 BH-FDR.

### 8.3 H3 (cell-by-model interaction)

Add `(intervention | model:cell)` to the random structure and compare via LRT to the H2 model. Per-(model, cell) BLUPs are tested at q = 0.05 BH-FDR across the 110 (22×5) cells.

### 8.4 Regime stratification (primary supplementary, not just sensitivity)

The `intervention × regime_stratum` interaction in §8.1 explicitly models how the intervention effect varies across the 6 regime levels. Per-stratum intervention effects are computed via:

```r
emmeans(fit, ~ intervention | regime_stratum) |> contrast(method = "revpairwise")
```

These 6 stratum-specific estimates are reported as **primary supplementary tables** in the main results (not as sensitivity-on-trigger). The H1 panel-averaged estimate remains the primary headline, with the explicit caveat in §1.2 that its estimand is the regime-mixture marginal.

**Pre-registered escalation rule.** If the largest pairwise difference between regime-stratified intervention effects exceeds |log-odds| = 0.5, the abstract and discussion **must** report the stratified estimates first and the panel-averaged second.

**Per-model regime-shift coupling (formal exploratory test).** From the §6.3.2 full-panel reasoning-control run we obtain per-model ON-vs-OFF agreement rates `agreement_i` for the 13 models with a reasoning-disable flag. We pre-register the following exploratory regression of per-model H1 effects on these agreement rates:

```r
lm(per_model_intervention_effect ~ agreement_rate, data = flag_models_only)
```

This is exploratory (only 13 of 22 models have agreement rates; not powered for inference); a slope significantly different from zero is reported as evidence that regime-shift sensitivity moderates the intervention effect. A null finding does **not** rescue interpretability — it bounds, not eliminates, the regime confound.

### 8.5 Specificity test (S1) and ITT-style test (S2)

**S1.** Refit the primary regression on **clean chains only** (replacing `detection_correct` with `false_positive = 1{parsed = "yes"}`). Pre-registered failure threshold: panel-averaged increase ≥ log-odds +0.20 in `false_positive` rate (≈ +0.04 in probability at v5-observed baselines).

**S2 (ITT-style).** Refit the primary regression on **adversarial chains** with `detection_correct_itt` as outcome (1 if parsed = "no" else 0; abstains coded as 0 = not-detected). Same model spec, same regime-stratum interaction, same fallback hierarchy. Pre-registered escalation rule: if the H1 effect estimate from S2 differs from the primary H1 estimate by more than |log-odds| = 0.20, S2 becomes co-primary in reporting and abstract; otherwise S2 is supplementary.

Both S1 and S2 are members of the §8.6 single FDR family.

### 8.6 Multiple-comparisons control: single global FDR family

All pre-registered inferential tests in §1.2 are members of a single FDR family:

- H1 (1 test)
- H2 (22 model-specific tests)
- H3 (110 model×cell tests)
- S1 (1 test)
- S2 (1 test, ITT)

Total: 135 tests. We apply Benjamini-Hochberg FDR at **q = 0.05** across the full family. We additionally pre-register a **gatekeeping rule**: if H1 fails (after FDR), H2 and H3 are reported as exploratory only (no inferential threshold). S1 and S2 are evaluated independently of H1's gatekeeping outcome.

### 8.7 Routing-invariance sensitivity (composite system-level test)

Compare H1 effect-size estimates between two model partitions:

- **(a) Direct-API (13 models):** Anthropic 3 + OpenAI 3 + Google 4 + DeepSeek 2 + Moonshot 1 = 13.
- **(b) OpenRouter-routed (9 models):** xAI 2 (grok-4-fast, grok-4.20) + Meta 1 (llama-3.3-70b) + Qwen 2 + Xiaomi 2 + Z-AI 2 = 9.

Total: 13 + 9 = 22 ✓.

Test statistic: difference in regime-and-cell-adjusted `intervention` marginal effect between the two partitions, with bootstrap 95% CI (10,000 resamples clustered at chain level).

**Interpretation (binding).** The two partitions are not orthogonal to model family or provider ecosystem (e.g., direct-API contains all Anthropic + Google + OpenAI; OR contains all xAI + Qwen + Xiaomi + Z-AI + Meta). Therefore the routing-difference statistic is interpreted as a **composite system-level sensitivity test** capturing the joint effect of routing layer + model-family composition, **not** as a pure causal effect of routing per se. Stratification cannot disentangle these in this design.

Pre-registered decision rule: if the routing-difference 95% CI excludes zero **and** the absolute difference is ≥ log-odds 0.3, the primary H1 result is reported with and without OR-routed models, and the discordance is acknowledged in the headline narrative. This composite sensitivity test is descriptive, not part of the §8.6 FDR family, and will be executed regardless of the H1 outcome (not gated by the §8.6 gatekeeping rule).

### 8.8 OR-exclusion sensitivity for regime interaction

Because OR-routed models are the subset where reasoning-disable was empirically validated to change behavior (§6.3), we pre-register a sensitivity test isolating their contribution to regime heterogeneity:

```r
# Refit §8.1 with OR-routed models excluded
fit_no_or <- update(fit, subset = regime_stratum != "reasoning-disabled-or")
```

Pre-registered comparison: `intervention × regime_stratum` interaction estimates from the full model vs. the OR-excluded model. If the largest stratum-pair difference shrinks by > 50% when OR is removed, the regime heterogeneity is interpreted as primarily an OR-routing artifact rather than a global regime effect. This is descriptive (no inferential threshold; not in the §8.6 FDR family) and runs regardless of H1 outcome.

### 8.9 Software, versions, seeds

- R version: ≥ 4.3
- Packages: `lme4 ≥ 1.1-35`, `glmmTMB ≥ 1.1-9`, `boot ≥ 1.3`, `multcomp ≥ 1.4`, `tidyverse ≥ 2.0`
- Bootstrap RNG seed: **20260501** (locked)
- Pre-registration of analysis script: `analysis/synthesize_cross_model.R` at the same code commit hash as the data-collection harness.

---

## §9. Stopping rules and exclusion criteria

### 9.1 Per-model cost kill-switch

If cumulative cost for any single model exceeds **$9.45** (110% of $189/22 mean allocation), the model thread terminates. The model is **excluded from the primary regression** (re-run §8 without this model) and reported in supplementary tables with a note.

**Sensitivity bounds.** Re-run §8 three times with the excluded model's missing observations imputed under three explicit assumptions:

| Imputation label | Definition (applies only to adversarial chains; `detection_correct` is undefined elsewhere per §1.1) | Interpretation |
|---|---|---|
| best-case | All missing **adversarial-chain** observations imputed as `detection_correct = 1` (the excluded model would have correctly detected every adversarial chain it didn't reach) | Upper bound on H1 effect under "the model would have helped if completed" |
| worst-case | All missing **adversarial-chain** observations imputed as `detection_correct = 0` (the excluded model would have incorrectly answered "yes" to every adversarial chain) | Lower bound on H1 effect |
| abstain-as-missing | Missing observations dropped (no imputation); listwise deletion | Equivalent to dropping the model from the regression entirely |

The H1 estimate range across these three imputations bounds the impact of the kill-switch on the reported effect. If H1 passes its MDE under all three imputations, the result is robust to the kill-switch event; if it fails under any, that contingency is reported.

### 9.2 Pathological-model auto-kill

See §6.5. Same exclusion + sensitivity protocol as §9.1.

### 9.3 Parse-rate inclusion floor

See §7.3. A model is excluded from primary inferential tests if average strict-parse rate < 90%. Such models are reported in supplementary tables only.

### 9.4 No data-dependent stopping

We do not stop the run early based on observed effect sizes. The experiment runs to completion or to a kill-switch trigger.

### 9.5 No model-selection feedback

Once filed, **no model may be added or removed from the panel** based on subsequent observations. Failures of any kind are reported transparently with the panel as filed.

---

## §10. Code, data, reproducibility

### 10.1 Artifact list

| Artifact | Location | SHA-256 hash | Status |
|---|---|---|---|
| Pre-registration document | `OSF_PREREG.md` | _filled at filing_ | Frozen at OSF filing |
| Specification | `../Ditto V5.1/SPEC.md` | _filled at filing_ | Frozen at OSF filing |
| Pipeline + harness code | Repository at commit hash `<TBD>` | _git rev-parse HEAD_ | Frozen at OSF filing |
| Provider pinning configuration | `../Ditto V5.1/provider_pinning.json` | _filled at filing_ | Frozen at OSF filing |
| Frozen prompt corpus (4,000 prompts) | `prompt_corpus_v5_1.jsonl` | _filled before launch_ | Generated and hashed before full run |
| Smoke validation runs (1 final + per-iteration changelog) | `results/v5_1/smoke_*` + `results/v5_1/SMOKE_CHANGELOG.md` | n/a (timestamped) | Public on OSF |
| Reasoning-control pilot | `results/v5_1/reasoning_control_20260501_192146.json` | _filled at filing_ | Public on OSF |
| Reasoning-control full-panel run | `results/v5_1/reasoning_control_full_panel_<timestamp>.json` | _filled before filing_ | _MUST GENERATE BEFORE FILING_ |
| Full-run raw responses | `results/v5_1/full_*` | _filled at completion_ | Public on OSF post-run |
| Analysis script | `analysis/synthesize_cross_model.R` | _filled at filing_ | Frozen at OSF filing |

### 10.2 RNG seeds

| Component | Seed | Used for |
|---|---|---|
| Chain-pair selection (per cell) | 20260501 | Sampling 100 pairs from candidate set |
| Violation injector | 20260501 | Injecting rule violations into clean chains |
| Per-chain actor-anonymisation map | hash(chain_id) | Deterministic Player_N assignment |
| Bootstrap (analysis) | 20260501 | §8.7 routing-invariance bootstrap |

All seeds are integer-typed and pinned in code at the filing commit hash.

### 10.3 Smoke changelog

A `SMOKE_CHANGELOG.md` will be produced before filing, documenting each of the 5 smoke iterations: the model panel tested, the failure mode surfaced, the change made for the next iteration. This makes the exclusion-criteria origin §2.2 fully traceable and audit-able.

---

## §11. Timeline

| Step | Date / Status |
|---|---|
| SPEC.md drafted | 2026-04-27 |
| Smoke-test iteration (5 cycles, ~3 days) | 2026-04-30 — 2026-05-01 |
| Reasoning-control pilot (4 models) | 2026-05-01 |
| Two-reviewer pre-filing audit | 2026-05-01 |
| OSF_PREREG v2 (this document) | 2026-05-01 |
| Reasoning-control full-panel run (REQUIRED before filing) | _planned within 24 hours_ |
| Frozen prompt corpus generation + hashing | _planned within 24 hours_ |
| OSF pre-registration filing | _planned within 1–2 days_ |
| Full experiment run | _within 1 week of filing_ |
| Analysis | _within 2 weeks of full run_ |
| Manuscript draft | _within 4 weeks of full run_ |

---

## §12. Authorship and contributions (CRediT taxonomy)

| Contributor | Role(s) (CRediT) |
|---|---|
| **Safiq Sindha** | Conceptualisation, Methodology, Software, Investigation, Data Curation, Formal Analysis, Writing — Original Draft, Project Administration |
| **Myriam Khalil** | Conceptualisation, Methodology, Validation, Writing — Review & Editing, Supervision |

Both authors contributed to study design, prompt-corpus authoring (T-design review 2026-04-28), and final pre-registration sign-off. Software implementation, smoke-test iteration, and reasoning-control validation were executed by Sindha; methodological consultation, exclusion-criteria adjudication, and pre-registration review were led by Khalil. All authors approved this pre-registration before filing.

---

## Appendix A — Smoke validation summary

Five smoke-test cycles were run during 2026-04-30 to 2026-05-01. Each iteration's design changes are documented in `results/v5_1/SMOKE_CHANGELOG.md` (to be produced before filing). The final smoke (`smoke_20260501_180858`) demonstrated:

- 21 of 22 panel models at 100% strict-parse rate
- 1 panel model (deepseek-v4-pro) at 90% strict-parse rate
- 0 pathological-model auto-kills
- 0 Stage-2 retries on panel models (Stage 1 sufficient — note: Stage-2 logic retained as guardrail; would have fired in earlier smokes before §6.2 fixes)
- Total cost: $2.50 (smoke), extrapolated to ~$70 for full run

Earlier smoke iterations identified the reasoning-disable parameters per provider, the model panel reductions in §2.2, and the parsing rule in §7. The OSF pre-registration is filed after these design choices are locked but before the full run is executed.

---

## Appendix B — Reasoning-control experiment summary

Pilot experiment data: `results/v5_1/reasoning_control_20260501_192146.json`. 4 OR-routed models × 25 chains × 2 conditions (ON, OFF) × 2 calls (baseline, intervention) = 400 calls. See §6.3 for headline finding.

Full-panel expansion (22 models, same 25 chains × 2 conditions): _MUST RUN BEFORE FILING._ Will be archived as `reasoning_control_full_panel_<timestamp>.json` and referenced in §6.3.2.
