# Phase 3 Launch Prompt — paste this entire file into a fresh Claude Code session

---

I'm continuing **Project Ditto v5.1** in a fresh session. Phase 2 (the full cross-model experiment, 22 frontier LLMs evaluating 100 chain-pairs × 8 conditions × 5 cells) is complete. I need you to implement **Phase 3 (statistical analysis)** per the locked OSF pre-registration.

## Working directory
`/Users/safiqsindha/Desktop/Project Ditto/Ditto V5.0`

## Inputs already on disk
- **`OSF_PREREG.md`** — the authoritative analysis plan. Treat as immutable. §1 has hypotheses (H1, H2, H3, S1, S2), §7 the parsing rule, §8 the full statistical plan, §9 the exclusion rules.
- **`results/v5_1/phase3_consolidated/`** — pre-consolidated, deduplicated dataset (one model per file, best-coverage version selected from across 7 output dirs). Contents:
  - `consolidated.jsonl` — every (model × chain × condition × cell) row with parsed answers, retry flags, source-dir provenance
  - `per_model_summary.csv` — sensitivity / specificity / FP rate / parse rate per model
  - `per_condition_summary.csv` — same broken out by 8 conditions
  - `per_cell_summary.csv` — same broken out by 5 cells
  - `manifest.json` — provenance + semantic-note documentation

## CRITICAL semantic note (do not get this wrong)

The classification question is: **"Is the sequence of events above consistent with the rules of {domain}?"**
- **YES = "consistent"** = no violation present
- **NO = "not consistent"** = violation present

In `consolidated.jsonl` each row has `baseline_parsed` (answer to the **clean** chain) and `intervention_parsed` (answer to the **adversarial** chain — same chain pair with violation injected).

**Detection metrics:**
- **Sensitivity** (true positive on adversarial) = `P(intervention_parsed == 'no')`
- **Specificity** (true negative on clean) = `P(baseline_parsed == 'yes')`
- **FP rate** (incorrect violation flag) = `P(baseline_parsed == 'no')`
- **Within-pair flip indicating detection**: `intervention_parsed == 'no' AND baseline_parsed == 'yes'` (flagged adv but not clean)

This was inverted in earlier exploratory analysis. The corrected interpretation is what goes in the paper.

## Pre-launch context (findings from exploratory peek)

These are NOT registered analyses but inform what to look for:

1. **Cross-model calibration is the dominant variance**, not the warning-prompt main effect (H1). Models split into:
   - **Aggressive flaggers** (>90% sens, <30% spec): claude-haiku, claude-sonnet-4-5 (failed §7.3 parse-rate floor), claude-sonnet-4-6, llama-3.3-70b, grok-4.20, grok-4-fast, gpt-5.4-mini, kimi-k2.6, mimo-v2-pro, glm-4.7
   - **Calibrated discriminators**: z-ai/glm-5 (only model with spec > 50%), gemini-3-flash-preview, gemini-3.1-pro-preview, gemini-3.1-flash-lite-preview, sonnet-4-6, gpt-5-mini, qwen-max, gemini-2.5-flash
2. **One model fails §7.3 floor**: `claude-sonnet-4-5` (82% parse rate due to 18% non-binary outputs). Per-spec, moves to supplementary tables.
3. **PUBG cells specifically have a chain-rendering artifact** (synchronized boarding at t=0.6s) that aggressive-flaggers misclassify as violations. Worth flagging in discussion as a data-presentation finding.
4. **v5 → v5.1 calibration mismatch is real but compound**. Direct-comparison calibration (haiku, v5 prompt wording, 50 chains/cell) showed 0% v5-style detection vs v5 phase D's 60%. Likely from cumulative changes (chain rendering A4-A8, question-text update). Cannot make a clean v5 vs v5.1 numerical claim.

## Your tasks (in order)

1. **Read** `OSF_PREREG.md` §1, §7, §8, §9 in full before writing any code.
2. **Read** `results/v5_1/phase3_consolidated/manifest.json` and `per_model_summary.csv` to verify dataset shape (expect 22 models if DeepSeek finished, otherwise 20). Check for any §7.3 floor failures.
3. **Verify Phase 2 completion cleanliness**: cross-check the source dirs in `manifest.json` against `results/v5_1/full_*/cost_ledger.json` — flag any models with kill-switch trips for §9.1 sensitivity.
4. **Propose a step-by-step plan and wait for my approval** before writing analysis code.
5. Once approved, implement the analysis as `scripts/run_analysis.R` (R via `lme4::glmer` + `emmeans`) plus a Python wrapper `scripts/run_analysis.py` that loads `consolidated.jsonl` → tidy R-readable CSV and shells out to R.
6. **Apply the §8.1 random-effects fallback hierarchy** if `glmer` fails to converge.
7. Run **S1 (specificity null)** and **S2 (ITT-style sensitivity)** per §8.4.
8. Implement the **§8.8 OR-exclusion regime sensitivity**.
9. Apply **BH-FDR correction** across H1, H2, H3 per §8.3.
10. Apply **§9.1 3-imputation sensitivity** for any model that tripped the kill-switch (likely `z-ai/glm-4.7` and possibly `gemini-3.1-pro-preview` from full_20260502_064050; check ledger).
11. Write a results summary to `RESULTS/v5_1/PHASE3_RESULTS.md` containing:
    - Headline H1/H2/H3/S1/S2 outcomes (pass/fail vs MDE, point estimate, 95% CI, FDR-adjusted p)
    - **Per-model sensitivity / specificity table** with the calibration-cluster classification
    - Forest plot of per-model intervention effects
    - **Discussion section** addressing the cross-model calibration finding and the PUBG rendering artifact (label as exploratory)
    - Plain-English interpretation
    - Any deviations from §8 with justification
12. Save the fitted model as RDS for later inspection.

## Constraints
- **This is a registered analysis.** No exploratory analyses unless flagged as exploratory in a clearly labelled subsection.
- **All randomness must be seeded** (set.seed in R, np.random.seed in Python).
- **Don't push to GitHub until I approve the results summary.**
- **The §6.3.2 reasoning-control results should be consistent with whatever H1 outcome you find** — if they're not, flag the inconsistency.
- **Do not get the YES/NO interpretation wrong.** The semantic note above is binding.

## Start by

Confirming the working directory exists, then reading `OSF_PREREG.md` §8.

---

**End of prompt — paste everything above (including the closing `---`) into the fresh session.**
