# BUILD_PLAN.md

**Project Ditto v5.1 — Cross-Model Replication of the 4-Tier Hierarchy**

**Build phase plan: design → SPEC → smoke → full run**

---

## Status

- Design phase: **complete**
- SPEC: drafting in parallel (do not block on SPEC for build tasks 1–4)
- Both-author sign-off: required before smoke test executes
- OSF pre-registration: required before full run executes

---

## Build phase principles

1. **Reuse v5's repo aggressively.** The harness, prompt corpus, chain construction, violation injectors, and analysis pipelines all exist. v5.1 extends v5; it does not replace v5.
2. **Land the harness logging gap before anything else.** v5's harness silently discarded API `usage` blocks. v5.1 cannot run cost-aware without this fix. This is the only blocking dependency for everything downstream.
3. **Smoke test gates the full run.** No full-run model is hit until smoke results are reviewed and both authors sign off.
4. **No mid-flight methodology changes.** Once SPEC is signed, any deviation requires re-signing before resuming.

---

## Reuse from v5 (do NOT rebuild)

These components exist in v5's repo and v5.1 reuses them as-is:

- `src/cells/{pubg,nba,csgo,rocket_league,poker}/pipeline.py` — per-cell data pipelines
- `src/cells/{cell}/extractor.py` — per-cell evidence extractors
- `src/interfaces/chain_builder.py` — chain construction
- `src/interfaces/translation.py` — chain → prompt translation
- `src/harness/violation_injector.py` — adversarial chain injection
- `src/harness/prompt_corpus/` — frozen JSON prompt corpus from Phase D
- `synthesize_phase_d.py` — base statistical analysis (extends to `synthesize_cross_model.py` in task 9)
- `DECISION_LOG.md` — continues with new entries D-45 onward

**Constraint:** the prompt corpus is FROZEN. v5.1 does not modify any prompt text from v5. v5.1 only adds the strict-grounding instruction variant and the no-marker variant — both as wrappers around the existing v5 prompts.

---

## Task sequence

### Task 1 — Harness usage-logging fix (BLOCKING)

**Objective:** v5's harness discards `usage` blocks from API responses. Fix this so every call's input/output token counts and cost are logged.

**Files to modify:**
- `src/harness/model_evaluator.py` — `_extract_batch_text` and any synchronous response handler

**Implementation:**

```python
# Replace the existing return-text-only pattern with:
def _extract_response(self, message):
    return {
        "text": getattr(block, "text", "") or "",
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "cache_creation_input_tokens": getattr(message.usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(message.usage, "cache_read_input_tokens", 0),
        "raw_usage": dict(message.usage),  # full block for provider-specific fields
    }
```

**Propagate through:**
- `EvaluationResult` dataclass — add `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_creation_tokens`, `cost_usd` fields
- `run_phase_d.py` (or v5.1 equivalent) — write usage to sidecar `<run>_usage.jsonl` automatically per call
- Cumulative cost ledger — maintain per-model running total in memory and persist after each batch completes

**OpenRouter-specific:**
- Send `"usage": {"include": true}` in every request body
- Read `response["usage"]["cost"]` (ground-truth cost from OR) and `response["provider"]` (resolved provider)
- Log both per call

**Native API specifics:**
- Anthropic: `message.usage.cache_creation_input_tokens` and `cache_read_input_tokens` are critical for Q18 caching effectiveness measurement
- OpenAI: `usage.prompt_tokens_details.cached_tokens` for cache hits
- Google: `usage_metadata.prompt_token_count` and `cached_content_token_count`
- DeepSeek: `usage.prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`

**Acceptance criteria:**
- Every call writes a usage record to JSONL with non-null token counts
- Cumulative cost ledger updates correctly across batched + synchronous calls
- Cache hit rates computable per (model, condition) combination from the JSONL
- Test: run 10 calls against Haiku 4.5, verify all 10 usage records present and totals match invoice from Anthropic console

**Estimated effort:** 2–3 hours

**Blocks:** all subsequent tasks

---

### Task 2 — OpenRouter provider pinning (Q8 resolution)

**Objective:** for each of the 14 OR-routed models, identify and pin the upstream provider that will serve all calls. Eliminates routing variance per R5 in red-team appendix.

**Models to pin** (in priority order — pin the most-used first):

1. Grok 4 Fast (xAI native upstream is the only meaningful provider)
2. Grok 4 (xAI native)
3. Grok 4.20 (xAI native)
4. Llama 3.3 70B Instruct
5. Llama 4 Maverick
6. MiMo-V2-Pro
7. MiMo-V2.5-Pro
8. Qwen3.6 Plus
9. Qwen3.6 Max-Preview
10. GLM-4.7
11. GLM-5
12. MiniMax M2.5
13. Kimi K2.6
14. Kimi K2 Thinking

**Implementation:**

```python
# For each model:
endpoints = requests.get(
    f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
).json()

# Select pin based on (in order of preference):
#   1. Highest quantization (fp16 > fp8 > int8 > int4)
#   2. Lowest acceptable price among highest-quantization options
#   3. Provider with throughput >= our smoke-test rate
#   4. Provider with documented uptime >= 99% in last 30 days

# Save to provider_pinning.json
```

**Output:** `provider_pinning.json` at repo root, format:

```json
{
  "x-ai/grok-4-fast": {
    "pinned_provider": "x-ai",
    "quantization": "fp16",
    "input_price": 0.20,
    "output_price": 0.50,
    "context_length": 2000000,
    "selected_at": "2026-04-30T...",
    "rationale": "only provider serving this model"
  }
}
```

**Per-call usage (in harness):**

```python
response = openrouter_client.chat.completions.create(
    model=model_id,
    messages=[...],
    max_tokens=64,  # or 256 for thinking-mode models
    extra_body={
        "provider": {
            "order": [pin_config["pinned_provider"]],
            "allow_fallbacks": False,  # CRITICAL: fail loud, don't silently substitute
        },
        "usage": {"include": True},
    },
)

# Validate that response["provider"] matches pinned_provider; warn loudly if not
```

**Acceptance criteria:**
- `provider_pinning.json` exists at repo root with all 14 entries
- Each entry has `pinned_provider`, `quantization`, `input_price`, `output_price`
- Test call to each model returns `response["provider"]` matching the pin
- Test call with deliberately-wrong pin fails (validates `allow_fallbacks: False` works)

**Estimated effort:** 1 hour

**Blocks:** Task 5 (smoke test) for OR-routed models

---

### Task 3 — Cluster definitions lock (Q13 resolution)

**Objective:** lock the cluster definitions for cluster-robust SE computation on CSGO and RL. Required before any analysis runs.

**Per token audit findings (Q3):**
- PUBG, NBA, Poker: each chain is its own source → IID SEs, no clustering needed
- CSGO: 364 sources, mean 3.3 chains/source, max 8 → moderate clustering
- RL: 148 sources, mean 8.1 chains/source, max 23 → meaningful clustering

**Implementation:**

```python
# Extract source IDs from chain_ids in v5's frozen prompt corpus
# CSGO chain_id format: "csgo_{source_id}_{idx}" (verify in repo)
# RL chain_id format: "rl_{source_id}_{idx}_adv" (note _adv suffix on adversarial side)

clusters = {
    "csgo": {chain_id: source_id_extraction_logic},
    "rocket_league": {chain_id: source_id_extraction_logic},
}
# poker, pubg, nba: chain_id == cluster_id (1:1)

# Persist as JSON, both clean and adversarial sides covered
```

**Output:** `clusters.json` at repo root.

**Acceptance criteria:**
- All 1,200 CSGO chain_ids map to one of 364 cluster IDs
- All 1,200 RL chain_ids map to one of 148 cluster IDs
- Both clean and adversarial sides covered (RL uses `_adv` suffix; handle this)
- File is committed to repo and signed-off before any analysis runs
- Manifest hash recorded in DECISION_LOG.md as D-45

**Estimated effort:** 30 min

**Blocks:** statistical analysis (Task 9)

---

### Task 4 — Strict-grounding and no-marker prompt variant generators

**Objective:** add the new axes' prompt variants to the corpus. Strict-grounding is the third axis; no-marker is the second. Both wrap existing v5 prompts without modifying them.

**Implementation:**

**Strict-grounding variant** (per intervention prompt):

```python
def add_strict_grounding(prompt: str, strict: bool) -> str:
    if strict:
        return prompt + "\n\nIMPORTANT: Only reply YES if you can identify the specific rule and the specific event in the input. If either the rule or the event is not present in the provided context, reply NO."
    else:
        return prompt + "\n\nReply YES if you believe a violation has occurred, NO otherwise."
```

**No-marker variant** (per adversarial chain):

```python
def strip_markers(chain: dict) -> dict:
    """Remove derived-state markers (e.g., NOTE=Player_X_already_eliminated) from chain."""
    chain_copy = deepcopy(chain)
    chain_copy["events"] = [
        {k: v for k, v in event.items() if k != "NOTE"}
        for event in chain_copy["events"]
    ]
    return chain_copy
```

**Verification:**
- Token count audit on strict-grounding addition: should add 50–100 tokens (per token audit Q1)
- Token count audit on no-marker variant: should remove 30–60 tokens for adversarial chains (per token audit Q1)
- Verify v5's original baseline-clean prompts are unchanged (do not modify the frozen corpus)

**Acceptance criteria:**
- All 8 condition variants generate correctly per chain pair: baseline×marker×strict, baseline×marker×non-strict, baseline×no-marker×strict, baseline×no-marker×non-strict, intervention×marker×strict, intervention×marker×non-strict, intervention×no-marker×strict, intervention×no-marker×non-strict
- Verification script: for 10 random chains, all 8 variants produce parseable, distinct prompts
- Token count distributions match expected ranges

**Estimated effort:** 2 hours

**Blocks:** smoke test

---

### Task 5 — Async parallel runner across providers

**Objective:** orchestrate calls across 4 native APIs + OpenRouter, respecting per-provider concurrency limits, batch-vs-sync routing, and DeepSeek off-peak window.

**Files to create:**
- `run_v5_1_replication.py`
- `src/harness/runner_native.py` (Anthropic, OpenAI, Google, DeepSeek)
- `src/harness/runner_openrouter.py`
- `src/harness/runner_orchestrator.py`

**Per-provider routing logic:**
- **Anthropic:** Batches API for full run (24h SLA, ~50% cheaper). Synchronous for smoke test.
- **OpenAI:** Batches API for full run, synchronous for smoke. Standard JSONL upload pattern.
- **Google AI Studio:** Batch API for full run (50% off, 24h SLA). Synchronous for smoke.
- **DeepSeek:** Synchronous (no batch). Schedule run to fit within UTC 16:30–00:30 off-peak window.
- **OpenRouter:** Synchronous with per-provider concurrency limits + provider pinning per Task 2.

**DeepSeek scheduling:**

```python
# Hard constraint: only call DeepSeek during off-peak window
# UTC 16:30 – 00:30 = 8.0 hours
# Run can be paused and resumed across multiple windows if needed
def is_deepseek_off_peak() -> bool:
    utc_now = datetime.now(timezone.utc)
    hour, minute = utc_now.hour, utc_now.minute
    # Window: 16:30:00 – 00:30:00 (next day)
    if hour > 16 or (hour == 16 and minute >= 30):
        return True
    if hour < 0 or (hour == 0 and minute < 30):
        return True
    return False
```

**Per-call cost ledger:**
- Maintain `cost_ledger.json` updated after every successful call
- Per model: cumulative cost, n_calls, mean tokens in/out, cache_hit_rate, max single-call cost
- Pre-registered kill-switch: if any model exceeds 110% of its mean allocation, pause and require manual sign-off to resume

**Acceptance criteria:**
- Smoke test fully runnable end-to-end (Task 5 produces results in Task 6)
- DeepSeek scheduler verified to skip calls outside off-peak window
- Cost ledger updates per call and triggers kill-switch warning when threshold approached
- All 4 native API integrations + OR runner pass independent unit tests

**Estimated effort:** 8–12 hours

**Blocks:** smoke test, full run

---

### Task 6 — Smoke test execution

**Objective:** run a small parameterized test across all 32 models and 8 conditions to validate (a) parseability, (b) provider pinning, (c) cost calibration, (d) token usage matches v5 audit.

**Configuration:**
- 32 models × 8 conditions × 1 cell × 10 chain pairs (paired) = 5,120 calls
- Single cell to validate: **CSGO** (it's the most balanced for parseability — moderate complexity, both-direction effects)
- Use synchronous APIs everywhere (no batch); we need fast feedback
- Estimated cost: ~$10–15

**Pre-registered evaluation criteria:**
- Per-model parse rate: ≥98% across all 8 conditions ⇒ proceed; 90–98% ⇒ flag and review; <90% ⇒ exclude
- Per-model token count: must be within ±20% of token audit predictions (mean 624 input for CSGO; worst-case 780)
- Per-model cost ground truth: actual cost ≤110% of predicted cost; if higher, recalibrate budget before full run
- Provider pin verification: for each OR-routed model, all 80 calls returned by the pinned provider (no silent re-routing)
- DeepSeek off-peak scheduler test: verify calls outside window are deferred, not failed

**Output:** `smoke_results/v5_1_smoke_<date>.json` with:
- Per-model parse rates per condition
- Per-model actual cost vs. predicted cost
- Per-model resolved provider (and discrepancies if any)
- Per-model parse failure modes (truncated, format error, refused, etc.)
- Aggregate cost vs. budget

**Decision gate (both-author sign-off required):**
- All models pass parseability threshold
- Cost is within budget projection
- No silent provider routing issues
- ≥1 model in each capability tier per provider passes (so we don't lose entire ladder)

If any of these fail: pause, document, decide whether to adjust SPEC and re-sign, or proceed with reduced panel.

**Estimated effort:** 1–2 hours wall time + ~30 min review

**Blocks:** full run

---

### Task 7 — SPEC + appendices on OSF

**Objective:** file pre-registration on OSF before any model is hit at full-run scale.

**Components to attach:**
- `SPEC.md` — main pre-registration document
- `MEMO.md` (with token audit addendum)
- `red_team_appendix.md`
- `clusters.json` (from Task 3)
- `provider_pinning.json` (from Task 2)
- `BUILD_PLAN.md` (this document)
- `PRE_MORTEM.md`
- `DECISION_LOG.md` (continued from D-44)

**OSF setup:**
- Project type: Standard Pre-Registration
- Title: "Ditto v5.1: Cross-Model Replication of a 4-Tier Representational Hierarchy in Constraint-Reasoning"
- License: CC-BY-4.0 for SPEC, MIT for code attachments
- Embargo:
    - SPEC.md, red_team_appendix.md, MEMO.md, BUILD_PLAN.md, PRE_MORTEM.md → public from filing date
    - clusters.json, provider_pinning.json → public (no embargo needed)
    - cost_ledger.json (post-run) → public after full run completes

**Acceptance criteria:**
- OSF DOI generated and recorded in SPEC.md and DECISION_LOG.md
- Both authors verify the OSF page contents match local files
- Full run does not start until OSF DOI exists

**Estimated effort:** 1–2 hours

**Blocks:** full run

---

### Task 8 — Full run execution

**Objective:** run 32 models × 8 conditions × 5 cells × 10,000 evals/model = 320,000 total evaluations.

**Sequencing strategy (parallel where safe, serial where required):**

**Phase A — Native APIs in parallel (asynchronous, batched):**
- Anthropic Batches: 4 models × 10,000 evals = 40,000 calls
- OpenAI Batches: 5 models × 10,000 evals = 50,000 calls
- Google Batches: 5 models × 10,000 evals = 50,000 calls
- All three submitted simultaneously; native batch APIs handle the queueing
- Expected wall time: 2–6 hours each (often <2 hours actual, 24h ceiling)

**Phase B — DeepSeek (synchronous, off-peak only):**
- 4 models × 10,000 evals = 40,000 calls
- Must run within UTC 16:30–00:30 windows (8 hours/day)
- At ~3 req/sec safe rate, 40,000 calls ≈ 4 hours wall time → fits in one window
- Run after Phase A submission so we can monitor live cost

**Phase C — OpenRouter (synchronous, throughout):**
- 14 models × 10,000 evals = 140,000 calls
- Provider-pinned, no auto-routing
- Async parallel across models; per-provider concurrency limits respected
- Expected wall time: 6–12 hours

**Live monitoring during run:**
- Cost ledger updates every 100 calls per model
- Kill-switch checked every 1,000 calls per model
- Provider drift detection (resolved provider ≠ pinned provider) triggers immediate halt
- DeepSeek off-peak boundary detection triggers automatic pause-and-resume

**Output structure:**

```
results/v5_1/
  raw/
    <model_id>/
      <cell>/
        <condition>/
          chain_<chain_id>.json   # full response + usage block
  derived/
    detection_results.parquet     # one row per evaluation
    cost_ledger_final.json
    parse_failure_log.jsonl
```

**Acceptance criteria:**
- All 320,000 calls complete or accounted for (failure logged with reason)
- Per-model parse rate ≥ smoke-test threshold
- Cumulative cost ≤ $300 (realistic ceiling)
- No model exceeded its kill-switch
- All provider pins held throughout

**Estimated effort:** 1–2 days wall time

**Blocks:** analysis (Task 9)

---

### Task 9 — Cross-model statistical analysis

**Objective:** extend v5's `synthesize_phase_d.py` into `synthesize_cross_model.py` with all v5.1 pre-registered analyses.

**Files to create:**
- `synthesize_cross_model.py` — main analysis pipeline
- `src/analysis/clustered_se.py` — cluster-robust SE computation for CSGO + RL
- `src/analysis/three_way_interaction.py` — performance ~ intervention × marker × strict, fit per cell × per model
- `src/analysis/within_provider_hierarchy.py` — Q10 conjunctive headline test
- `src/analysis/parse_failure_dual_treatment.py` — Q12 reporting both as-FN and excluded
- `src/analysis/output_length_tier_correlation.py` — Q19 confirmatory analysis
- `src/analysis/multiple_comparisons.py` — BH-FDR primary, Bonferroni secondary

**Primary analyses** (from SPEC):
1. Per-cell × per-model × per-condition McNemar with cluster-robust SEs (CSGO + RL only) or IID (PUBG/NBA/Poker)
2. 3-way interaction `performance ~ intervention × marker × strict`, fit per cell × per model
3. Within-provider hierarchy test (Q10): does the 4-tier ordering hold within each provider's ladder?
4. Cross-panel hierarchy replication

**Secondary analyses:**
- Tier-collapse test
- FP-discipline scaling on CSGO
- Provider × tier ANOVA
- Strict-vs-non-strict on RL (load-bearing)
- Marker-ablation contrast
- Output-length-tracks-tier (Q19)

**Multiple comparisons:**
- Primary: Benjamini–Hochberg FDR at q=0.05
- Robustness: Bonferroni at α=0.05/k (where k is the contrast count, finalized pre-run)

**Output:**
- `results/v5_1/analysis/headline_findings.json` — within-provider hierarchy conjunctive result, replication threshold check
- `results/v5_1/analysis/per_cell_per_model_heatmap.parquet` — full McNemar grid
- `results/v5_1/analysis/three_way_interactions.parquet` — interaction effects per cell × model
- `results/v5_1/analysis/parse_failure_dual_treatment.json` — Q12 dual reporting
- `results/v5_1/analysis/output_length_tier_correlation.json` — Q19 confirmatory

**Acceptance criteria:**
- All primary and secondary pre-registered analyses run successfully
- Replication threshold (≥5 of 6 ladders) evaluated and reported
- FDR + Bonferroni discordance reported per contrast
- Cluster-robust SEs applied correctly (CSGO + RL); IID elsewhere
- Output reproducible from raw results + frozen analysis code (commit hash recorded)

**Estimated effort:** 6–10 hours

**Blocks:** writeup

---

### Task 10 — Writeup + OSF results post

**Objective:** publish results back to OSF and produce a writeup ready for external review.

**Components:**
- Headline findings document (1–2 pages): replication outcome, within-provider hierarchy conjunctive result, mechanism claim verdict
- Full results document (5–10 pages): all primary + secondary analyses, with discussion
- Discrepancies log: any ways v5.1 deviated from pre-registered SPEC (must be empty if no mid-flight amendments occurred)

**OSF post-run attachments:**
- `cost_ledger_final.json`
- `analysis/headline_findings.json`
- `analysis/per_cell_per_model_heatmap.parquet`
- All other analysis outputs
- Final writeup PDF

**Acceptance criteria:**
- Both authors sign writeup before posting
- OSF DOI of pre-registration cited in writeup as prospective registration
- All analyses traceable to pre-registered SPEC sections (no post-hoc additions)

**Estimated effort:** 8–16 hours

---

## Critical path

```
Task 1 (harness logging)
    |
    v
Task 2 (provider pinning) --+-- Task 4 (prompt variants)
Task 3 (clusters)           |
                            v
                       Task 5 (runner)
                            |
                            v
                       Task 6 (smoke)  <-- both-author sign-off
                            |
                            v
                       Task 7 (OSF filing)  <-- both-author sign-off
                            |
                            v
                       Task 8 (full run)
                            |
                            v
                       Task 9 (analysis)
                            |
                            v
                       Task 10 (writeup)
```

**Bottleneck:** Task 5 (runner integration). Everything else is mechanical or fast.

---

## Pre-flight checklist before Task 6 (smoke test)

- [ ] Task 1 complete: harness logs `usage` per call to JSONL
- [ ] Task 2 complete: `provider_pinning.json` at repo root with all 14 entries
- [ ] Task 3 complete: `clusters.json` at repo root with CSGO + RL definitions
- [ ] Task 4 complete: 8-variant prompt generation verified on 10 random chains
- [ ] Task 5 complete: runner integration tested per-provider with single calls
- [ ] DeepSeek off-peak window scheduler verified
- [ ] Cost kill-switch verified to halt run on simulated overage
- [ ] Provider-pin failure mode verified (`allow_fallbacks: false` causes hard error)
- [ ] Smoke test budget approved (~$10–15)

## Pre-flight checklist before Task 8 (full run)

- [ ] Task 6 complete: smoke results show all 32 models pass parseability gate
- [ ] Smoke cost matches predicted within ±10%
- [ ] No silent provider routing observed in smoke
- [ ] Both authors sign smoke results
- [ ] Task 7 complete: OSF DOI generated and recorded
- [ ] Realistic ceiling ($300) confirmed against smoke-extrapolated full-run cost
- [ ] DeepSeek off-peak window confirmed for run start time
- [ ] Cost kill-switch armed at 110% of per-model mean allocation

---

## Time estimate

| Task | Effort | Wall time |
|---|---|---|
| 1 — Harness logging | 2–3 hours | 0.5 day |
| 2 — Provider pinning | 1 hour | 0.25 day |
| 3 — Clusters | 30 min | 0.1 day |
| 4 — Prompt variants | 2 hours | 0.5 day |
| 5 — Runner | 8–12 hours | 1–2 days |
| 6 — Smoke + review | 2 hours + sign-off | 1 day |
| 7 — OSF filing | 1–2 hours | 0.25 day |
| 8 — Full run | mostly automated | 1–2 days |
| 9 — Analysis | 6–10 hours | 1–2 days |
| 10 — Writeup | 8–16 hours | 2–3 days |

**Total: ~9–13 days from build start to published results.**

Tasks 1–4 can begin immediately (this build plan is the handoff). Tasks 5–10 depend on SPEC sign-off, which is in parallel drafting.

---

## What blocks SPEC sign-off (does NOT block early build tasks)

- SPEC.md final draft
- PRE_MORTEM.md (extending the v5.1 pre-mortem with audit-surfaced risks)
- Both-author review and sign-off

These are in parallel. Tasks 1–4 of this build plan are independent of SPEC and can begin now.

---

## Decision log entries to expect during build

- D-45: clusters.json locked, hash recorded
- D-46: provider_pinning.json locked
- D-47: smoke results approved
- D-48: OSF DOI generated
- D-49: full run started (timestamp + DeepSeek window confirmed)
- D-50: full run completed
- D-51: analysis results sign-off
- D-52: writeup posted to OSF

Any deviations during build require a numbered decision log entry with both-author sign-off before proceeding.

---

## Sign-off

This build plan is the handoff document for the build phase. It does not require SPEC sign-off to begin Tasks 1–4 (they are pure infrastructure and mechanical lookups). Tasks 5+ require SPEC sign-off and OSF filing.

**Author 1 (Safiq):** ___________________ Date: ___________

**Author 2 (Myriam):** ___________________ Date: ___________

---

*This document supersedes any prior build plans for v5.1. It does not modify any of v5's frozen artifacts.*
