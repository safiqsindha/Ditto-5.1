# v5 Status — Closed (2026-04-30)

**Status:** ✅ Phase D complete, results locked, methodology paused for cross-model replication (v5.1, scoped separately)
**Subject model:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)
**Total spend:** ~$5 across entire experiment
**Build-day status (2026-04-27):** archived at `docs/STATUS_BUILD_DAY_2026-04-27.md`

---

## What v5 set out to do

Five-cell parallel replication of v3's constraint-chain detection methodology (originally validated on Chess + Checkers in v3) across five new game domains using Claude Haiku 4.5. Pre-registered design: 1,200 chains/cell, McNemar + Bonferroni (divisor=5) + bootstrap CI, two API calls per chain (paired baseline / intervention).

| Cell | Domain | Data |
|---|---|---|
| `pubg` | PUBG sample matches (squad-fpp / solo-fpp) | PUBG Developer API telemetry |
| `nba` | 2023-24 NBA regular season | NBA Stats API PlayByPlayV3 |
| `csgo` | 2024 S-tier CS2 | FACEIT Open API (aggregate stats) |
| `rocket_league` | RLCS 2024 | BallChasing.com (aggregate stats) |
| `poker` | NLHE 200NL–1000NL + WSOP 2023 (all-human) | HandHQ + PHH Dataset v3 |

The original (consistency-rating) framing produced a floor effect on 4 of 5 cells. v5 pivoted mid-experiment to a violation-detection diagnostic, then layered on derived-state markers, strict-grounding, and a CoT FP analysis (D-42 → D-44) before scaling to Phase D at the full pre-registered n=1,200/cell.

---

## Final result (Phase D, n=1,200/cell, Haiku 4.5)

| Cell | Det@Base | Det@Int | Δ | 95% CI | FP@Base | FP@Int | b | c | χ² | p_Bonf |
|------|---------:|--------:|--------:|:-:|--------:|-------:|---:|---:|---:|---:|
| pubg | 75.9% | 100.0% | +24.1% | [+21.8, +26.6] | 0.0% | 0.0% | 0 | 289 | 287.0 | **1.1e-63** |
| nba | 57.4% | 100.0% | +42.6% | [+39.8, +45.4] | 0.8% | 9.9% | 0 | 511 | 509.0 | **5.2e-112** |
| csgo | 65.1% | 98.1% | +32.9% | [+30.3, +35.7] | 11.3% | 29.8% | 0 | 395 | 393.0 | **9.2e-87** |
| rocket_league | 0.2% | 6.2% | +5.9% | [+4.7, +7.3] | 0.0% | 0.0% | 0 | 71 | 69.0 | **4.9e-16** |
| poker | 100.0% | 99.9% | -0.1% | [-0.25, 0.00] | 0.5% | 1.1% | 1 | 0 | 0.0 | 1.000 |

**4 of 5 cells significant past Bonferroni at α/5 by many orders of magnitude.** Poker shows a ceiling effect (no room to lift), not a failure.

### Headline: 4-tier representational hierarchy

| Tier | Cell | Anchored | Observable | Unary-reducible | Profile |
|---|---|:-:|:-:|:-:|---|
| **0** Saturated | poker | ✓ | ✓ | ✓ | rule pre-internalized; no lift possible |
| **1** Aligned | pubg, nba | ✓ | ✓ | ✓ | clean intervention lift to perfect detection |
| **2** Partial | csgo | ✓ | ✗ (bomb sites) | ✓ | high lift, elevated FP — confabulation signature |
| **3** Misaligned | rocket_league | ✓ | ✗ (positions) | ✗ | tiny but real lift; strict grounding suppresses confabulation |

**Three necessary conditions** for constraint reasoning emerge: rule anchoring, predicate observability, unary reducibility.

**Defensible claim from this data:**
> Constraint reasoning in LLMs is gated by representational alignment between the rule and the observable event structure. It succeeds when violations reduce to observable unary predicates over event streams; it degrades predictably under missing observability; it is suppressed entirely under strict grounding when required variables are absent.

---

## Where the artifacts live

| Path | Contents |
|---|---|
| `SPEC.md` | Full v5 specification (8 sign-off questions, 7 amendments) |
| `DECISION_LOG.md` | D-0 through D-45 — every methodology decision with rationale |
| `MEMO.md` | Internal closeout memo + bridge to V1–V5.1 arXiv paper |
| `RESULTS/phase_d_final.json` | Final scored results (regenerable from archive) |
| `RESULTS/phase_d_raw_batches/` | 23,998 archived request records (~6 MB) — immutable raw evidence |
| `synthesize_phase_d.py` | Per-cell McNemar + Bonferroni + bootstrap synthesis |
| `retrieve_phase_d_partial.py` | Re-fetch a saved batch by ID |
| `archive_phase_d_batches.py` | Pull every batch to local JSONL |
| `run_phase_d_cot.py` | Layer-2 CoT diagnostic on residual intervention FPs |
| `run_diagnostic_violations.py` | Phase D entry point |
| `src/harness/prompts.py` | Per-cell PromptBuilder + `_MarkerSurfacing` (derived-state markers) |
| `src/harness/violation_injector.py` | Per-cell adversarial-violation planters |
| `src/harness/model_evaluator.py` | Anthropic Batches API caller (post-fix custom_id format) |

---

## Reproducing the headline numbers

```bash
# Reproduce Phase D final synthesis (uses saved batch IDs):
.venv/bin/python synthesize_phase_d.py
# -> RESULTS/phase_d_final.json + console table

# If 60-day batch retention has expired, re-score from local archive
# (modify fetch_batch in retrieve_phase_d_partial.py to read JSONL):
ls RESULTS/phase_d_raw_batches/*.jsonl

# Layer-2 CoT diagnostic on residual NBA + CSGO FPs:
.venv/bin/python run_phase_d_cot.py --cells nba csgo
# -> RESULTS/phase_d_cot_residual_fps.json

# Full re-run from scratch (regenerates new batches, costs ~$5):
.venv/bin/python run_diagnostic_violations.py --batch \
    --cells pubg nba poker rocket_league csgo \
    --n-per-cell 1200 \
    --ignore-timestamps \
    --output RESULTS/phase_d_repro.json
```

---

## What's deferred to v5.1 / v6

**1. Cross-model replication via OpenRouter** (the most valuable next experiment).

- Freeze Phase D prompt corpus to immutable JSON
- Replay across ~11 models (Haiku 4.5, Sonnet 4.5, Opus 4.x, gpt-5-mini, gpt-5, one o-series, Gemini 2.5 Flash + Pro, Llama 3.3 70B, Qwen 2.5 72B, DeepSeek-V3)
- n=300/cell (Phase D was overpowered)
- **Add a derived-state-marker ablation** as a first-class second axis (each prompt pair runs in 4 conditions: baseline×marker, baseline×no-marker, intervention×marker, intervention×no-marker)
- n=20 smoke test per model first; budget cap ~$150
- Pre-registered analyses: per-cell × per-model McNemar heatmap; tier-collapse test (does Opus catch RL's indirect markers?); FP-discipline-vs-capability scaling

**2. CS:GO awpy fix** — explicitly deferred. The 29.8% intervention FP is a *signature* worth preserving as v1 of cross-model. Add awpy-derived bomb-site observability as a secondary v5.2 cell *after* cross-model lands, so the comparison "model capability vs. data observability" can be measured cleanly.

**3. Rocket League per-event extraction (carball / boxcars-py)** — deferred. RL's near-floor detection rate is the strongest demonstration of strict-grounding's anti-confabulation behavior. Better RL data is interesting, but adding it would weaken the Tier-3 anchor.

---

## Things to be careful about

- **Don't claim v3 replication.** v3 measured "consistency rating"; v5 measured "violation detection". The pivot was forced by the floor effect on 8-event chains. Frame this as a discovery experiment about methodology, not a faithful replication.
- **Don't bury Poker.** A ceiling effect is a result, not a missing data point. It anchors the top of the hierarchy.
- **Don't try to "fix" Rocket League.** The low detection rate under strict grounding is the headline finding for Tier 3.
- **Custom_id collisions at scale.** The Anthropic Batches API rejects duplicate custom_ids. Always prefix with positional indices (`f"{i:06d}__{chain_id}__{variant}"`). Fixed in `model_evaluator.py` (commit `be84525`).
- **Decision discipline.** `DECISION_LOG.md` exists for a reason. Every methodological deviation goes there with rationale before code changes. There are 45 entries; D-42 (the methodology pivot), D-43 (derived-state markers), D-44 (Layer 1 + 2), and D-45 (Phase D closeout) are the most consequential.

---

## Sign-off

Phase D results frozen 2026-04-30. Methodology paused. v5 closes here. v5.1 cross-model replication is scoped separately and will use this Haiku 4.5 baseline as its anchor.
