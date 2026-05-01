# Smoke-Test Changelog — Project Ditto v5.1

This changelog documents the smoke-test iteration cycle (2026-04-30 to 2026-05-01) that informed the final 22-model panel and the inference settings locked in `OSF_PREREG.md`. It is the audit trail for the post-hoc nature of the model exclusions disclosed in OSF_PREREG.md §2.2.

The five "main" smoke iterations summarised here are the cycles that produced material design changes. Several intermediate debugging runs (timestamped before 2026-04-30 22:21 UTC) are not summarised individually as they were development-stage validations of the harness rather than design decisions.

---

## Smoke #1 — initial 32-model dispatch
**Run dir:** `smoke_20260501_030458` (the longest legacy run, b7w4x4p72)
**Wall time:** ~4 hours (killed by user)
**Trigger for next iteration:** `kimi-k2.6` empty-content failure on DeepInfra; minimax/glm-4.7/mimo×2 also 100% empty; gpt-5/gpt-5-mini hit 64-token cap and returned empty.

### Discoveries
- **`MAX_TOKENS_STANDARD=64` is too restrictive.** Models that secretly chain-of-thought (gpt-5, gpt-5-mini, llama-4-maverick, minimax-m2.5, mimo-v2-pro, mimo-v2.5-pro, glm-4.7, kimi-k2.6) burned the entire output budget on hidden reasoning and returned empty content with `finish_reason=length`.
- **`kimi-k2.6` 100% empty on DeepInfra** — DeepInfra capacity issue identified; thread died after ~22 calls.
- **gemini-2.5-pro 250 RPD daily quota** discovered — Google AI Studio Tier 1 enforces this even on paid tier for `gemini-3.1-pro-preview`.

### Design changes for smoke #2
- Switch `kimi-k2.6` from DeepInfra → Novita
- Begin investigating per-provider reasoning-disable parameters
- Begin porting Google batch path to use `google.genai` SDK (deprecation of `google.generativeai`)

---

## Smoke #2 — 28-model panel after 4 cost-driven drops
**Run dir:** `smoke_20260501_040429` (b6yogrzu8)
**Wall time:** ~80 min (killed at 24/28 ledgers)
**Trigger for next iteration:** still seeing 100% empty on multiple OR models; gemini-2.5-pro now blocked by daily quota.

### Discoveries
- **Dropped 4 models on cost ceiling** before this run launched: `claude-opus-4-7`, `gpt-5.4`, `gpt-5.5`, `x-ai/grok-4`.
- **Two-stage retry policy (96 → 1024)** added to avoid burning through cap on Stage 1.
- **Mega-batch architecture** (one batch per `(condition, cell)` submitted in parallel) added for native runners.
- **Per-call empty patterns confirmed**: minimax/mimo/glm-4.7/kimi-k2.6 all at 100% empty; first-token parser added to detect verbose-prefaced responses (llama-4-maverick).

### Design changes for smoke #3
- Add metadata logging (lenient parser, route, needed_retry, route_distribution)
- Update DeepSeek panel to v4-pro + v4-flash (deprecation of v3.2/v4/r1 by provider 2026-04-30)
- Build `MoonshotRunner` for direct Moonshot API access (kimi-k2.6 needs `thinking:type:disabled` which OR strips)

---

## Smoke #3 — 27-model panel with metadata logging + new pipeline
**Run dir:** `smoke_20260501_162949` (bfn5ul9uv)
**Wall time:** ~50 min (report writer crashed at end; recomputed via `scripts/recompute_smoke_report.py`)
**Trigger for next iteration:** 5 models still 100% empty (kimi-k2.6 [threading bug], minimax-m2.5, mimo×2, glm-4.7); 4 models partially empty (gpt-5/5-mini, gemini-2.5-pro, kimi-k2-thinking).

### Discoveries
- **`reasoning:enabled:false` parameter is OR's standardized reasoning-disable**. Empirically validated:
  - `qwen/qwen3.6-plus`: default 19s/1017 tokens → with disabled 1.1s/1 token (17× speedup)
  - `z-ai/glm-4.7`: empty → "YES" in 0.4s
  - `xiaomi/mimo-v2-pro`: empty → "YES" in 1.1s
  - `minimax/minimax-m2.5`: 400 "Reasoning is mandatory for this endpoint and cannot be disabled" → DROPPED
- **kimi-k2.6 needs `thinking:{type:"disabled"}` (different parameter family)** — empty everywhere via OR; only works via direct Moonshot API.
- **DeepInfra 200 concurrent natively allowed** — confirmed via DeepInfra docs.
- **`needed_retry` was 0% across the panel** — Stage 1 sufficient with the new prompt + reasoning-disable.

### Design changes for smoke #4
- Add `extra_body={"reasoning":{"enabled":false}}` to OR runner for non-reasoning models
- Build `MoonshotRunner` for direct Kimi (thinking-disable)
- Drop `minimax/minimax-m2.5` (mandatory-reasoning)
- Threading-bug fix in `MoonshotRunner._init_` (missing `import threading`)

---

## Smoke #4 — 24-model panel with reasoning-disable + MoonshotRunner
**Run dir:** `smoke_20260501_172043` (bnn12b19x)
**Wall time:** ~38 min (killed by user when gpt-5 last batch stuck at "in_progress" 37+ min)
**Trigger for next iteration:** OpenAI batch long-tail blocked completion; gemini-2.5-pro still 96% empty even with new pipeline; kimi-k2-thinking false-killed by old streak-based pathological detector.

### Discoveries
- **gpt-5 / gpt-5-mini still 25-49% empty** despite cap=1024 — secret reasoning consumes the full budget. Discovered OpenAI's `reasoning_effort: "minimal"` parameter:
  - gpt-5: default 14s/970 tokens → minimal 2s/10 tokens, "NO" ✓
  - gpt-5-mini: default 17s/1024 (empty!) → minimal 1.3s/10 tokens, "YES" ✓
- **gemini-2.5-pro: cannot disable thinking.** Google docs explicitly state this for 2.5 Pro. Confirmed empirically: 96% empty over 160 calls. → DROPPED.
- **Gemini batch text extraction** improved to iterate `candidates[0].content.parts` (fixes thought_signature dropping).
- **kimi-k2-thinking (Novita)** doesn't honor `max_tokens` — produces median 1,784 / max 18,182 tokens. Cost projection $55 over $7.88 per-model budget. Pathological-streak detector (5 consecutive empties → kill) was over-triggering due to chain parallelism converting independent ~20% empty rates into spurious streaks. → DROPPED kimi-k2-thinking; replaced streak-based detector with **rolling failure-rate guardrail** (window=50, threshold=50%, min-samples=20).
- **OpenAI batch long-tail problem** identified — single batch can stall in `in_progress` for 30+ min while others finish. Switch to sync OpenAI in smoke (batch reserved for full run).

### Design changes for smoke #5
- Add `reasoning_effort: "minimal"` for OpenAI non-reasoning models (sync + batch paths)
- Drop `gemini-2.5-pro` (mandatory thinking)
- Drop `meta-llama/llama-4-maverick` (verbose preface, no format toggle, 32% strict parse)
- Drop `moonshotai/kimi-k2-thinking` (cost ceiling — Novita ignores max_tokens)
- Replace pathological-streak detector with rolling failure-rate guardrail (per second-AI consultation)
- Switch OpenAI runner to sync in smoke mode (batch in full run)

---

## Smoke #5 — 23→22 model panel with all fixes (final, locked)
**Run dir:** `smoke_20260501_180858` (bc3mb9t0y)
**Wall time:** ~20 min
**Outcome:** Successful end-to-end smoke. **Spec locked from this run.**

Note: smoke #5 launched with 23 models including kimi-k2-thinking (the data was kept for documentation purposes). The final pre-registered panel is 22 models — kimi-k2-thinking was dropped after smoke #5 confirmed its cost trajectory was untenable. The spec change happened post-smoke; the data from this smoke for the other 22 models is the basis for the §6.3 evidence in the prereg.

### Results (per `v5_1_smoke_20260501.json`)
- **20 models at 100% strict parse rate** (claude haiku/sonnet-4-5/sonnet-4-6, gpt-5/mini/5.4-mini, gemini-flash×3, gemini-3.1-pro-preview, llama-3.3-70b, qwen×2, grok-4-fast/4.20, mimo×2, glm×2, deepseek-v4-flash, kimi-k2.6 via MoonshotRunner)
- **2 models at 98-99%** (claude-sonnet-4-5: 98%, gemini-3-flash-preview: 99%)
- **1 model at 90%** (deepseek-v4-pro — at the inclusive parse-rate floor)
- **1 model at 79%** (kimi-k2-thinking — subsequently dropped)
- 0 pathological-model auto-kills
- 0% Stage-2 retry rate across the panel
- Total cost: $2.50

### Design changes for full run
None. The pipeline is locked at the smoke #5 commit (post-drop of kimi-k2-thinking).

---

## Reasoning-control validation (post-smoke)

After smoke #5, a separate experiment (`scripts/reasoning_control_experiment.py`) tested whether the `reasoning:enabled:false` parameter materially changes classification answers:

- **4 OR models × 25 chains × 2 conditions × 2 calls = 400 calls**
- `qwen/qwen3.6-plus`: 76% strict agreement
- `z-ai/glm-4.7`: 52% strict agreement
- `xiaomi/mimo-v2-pro`: 44% strict agreement
- `meta-llama/llama-3.3-70b-instruct` (control): 100% strict agreement

**Three of four reasoning-capable models tested fall below the 95% agreement threshold for "neutral optimization."** This finding became the empirical foundation for the §6.3 disclosure in the prereg (the constrained-inference regime is an experimental condition, not just a cost optimization).

A pre-filing expansion to all 13 flag-bearing panel models was committed in §6.3.2; results in `reasoning_control_full_panel_<timestamp>.json`.

---

## Summary of post-hoc design decisions

| Decision | Justified by | Pre-registered as |
|---|---|---|
| Drop `claude-opus-4-7`, `gpt-5.4`, `gpt-5.5`, `x-ai/grok-4` | Cost projections in smoke #1-#2 | Cost ceiling criterion (§2.4) |
| Drop `minimax/minimax-m2.5` | Smoke #3 — explicit "Reasoning is mandatory" 400 from API | Mandatory-inference-mode criterion (§2.4) |
| Drop `gemini-2.5-pro` | Smoke #4 — Google docs + 96% empty | Mandatory-inference-mode criterion (§2.4) |
| Drop `meta-llama/llama-4-maverick` | Smoke #4 — 32% strict parse, no format toggle | Format-adherence-floor criterion (§2.4) |
| Drop `moonshotai/kimi-k2-thinking` | Smoke #4-#5 — Novita ignores max_tokens, $55 over budget | Cost ceiling criterion (§2.4) |
| Drop `deepseek-v3.2`, `v4`, `r1` | Smoke #3 — provider deprecation 400 | Provider-deprecation criterion (§2.4) |
| Two-stage retry policy | Smoke #1-#2 cap-empty failures | §6.4 |
| Rolling failure-rate guardrail | Smoke #4 — chain parallelism made streak-based detector over-trigger | §6.5 |
| `reasoning:enabled:false` for OR | Smoke #3 — 5+ models converted from 100% empty to 100% parse | §6.2 (with disclosure §6.3) |
| `reasoning_effort:minimal` for OpenAI | Smoke #4 — gpt-5/mini converted from 25-49% empty to 100% | §6.2 |
| `thinking:type:disabled` for kimi-k2.6 | Smoke #4 testing — only direct Moonshot API exposes this | §6.2 + MoonshotRunner |
| OpenAI sync in smoke (batch in full run) | Smoke #4 — batch long-tail | §6.2 |

All exclusions and parameter choices are exploratory in origin. The §2.4 inclusion criteria are forward-looking generalisations from this smoke evidence, not pre-specifications independent of it. This distinction is preserved transparently in the prereg.
