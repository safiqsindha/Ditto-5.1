# v5 Risk Register + Mitigations

**Date:** 2026-04-27 (Phase A complete, pre-Phase B)  
**Format:** Risk → severity → likelihood → concrete mitigation → owner → priority

Mitigations are tagged **"now"** (do before Phase B), **"phase-B"** (do during T design), or **"phase-C"** (do during real-data acquisition).

---

## Methodological risks

### M1 — v3's Phase 2 hasn't been run; effect size is anticipated, not observed
**Severity:** 🔴 Critical · **Likelihood:** Certain  
**Impact:** v5 was sized to detect v3-like effect (gap ≥ 0.06). If actual v3 effect turns out smaller (e.g., 0.03), v5 also misses real signal at n=1,200.

**Mitigation:**
- **NOW**: Pre-register an interim analysis after the first 600 chains/cell. If observed `|c−b|/n` is < 0.04 in any cell, pause that cell and reconsider sample size before continuing.
- **NOW**: Add an `interim_check.py` script that runs McNemar on the first half of chains and reports MDE for the remaining half.
- **PHASE-D**: After each cell completes, log effect-size estimate with CI; if narrow CI excludes effect ≥ 0.04, declare null with confidence rather than claiming underpowered.
- **PROGRAM-LEVEL**: Coordinate with the v3 team to surface their Phase 2 results before v5 runs (if timeline allows).

**Owner:** Lead author  
**Priority:** High

---

### M2 — Mock T 100% retention masks Gate 2 behavior under realistic distributions
**Severity:** 🟠 High · **Likelihood:** Certain  
**Impact:** Real T may extract chains where 30–60% of events are non-actionable; Gate 2 floor of 50% may filter out a large fraction we haven't budgeted for.

**Mitigation:**
- **PHASE-B**: As soon as T is designed, write a "noisy mock T" that intentionally produces some sub-50% chains so we exercise Gate 2 with realistic input.
- **PHASE-B**: Run the noisy-mock pilot before any real-data run. If retention drops below 80%, scale upstream sample target to compensate.
- **PHASE-D**: First-run real-data pilot with 50 streams/cell BEFORE full acquisition. Gate 2 retention at this small scale tells us how much to scale upstream.

**Owner:** Me (pipeline work) + lead author (review)  
**Priority:** High

---

### M3 — Bonferroni divisor=5 may be too conservative
**Severity:** 🟠 High · **Likelihood:** Medium  
**Impact:** Per-cell α=0.01 means a real but modest effect (e.g., 0.05) could fail to clear statistical bar in any single cell, leading to false null conclusion.

**Mitigation:**
- **NOW**: Document the per-cell secondary aggregate test in SPEC. Even if no individual cell hits α=0.01, a pooled test across cells can show generalization with α=0.05.
- **PHASE-D**: Report ALL three: (a) per-cell at α=0.01, (b) pooled at α=0.05, (c) Cohen's h with CI per cell. Don't let Bonferroni mask interesting effect sizes.
- **POST-V5**: ME-1 (variable-length) and ME-3 (NBA play-level) micro-experiments could swap in if results are marginal.

**Owner:** Lead author  
**Priority:** Medium

---

## Technical risks

### T1 — Real-data fetch+parse paths are untested in CI
**Severity:** 🟠 High · **Likelihood:** Medium  
**Impact:** Format drift in any of 5 sources between now and Phase C could cause silent data loss or parser crashes.

**Mitigation:**
- **NOW**: HTTP-mock tests for each pipeline's `fetch()` (using `unittest.mock.patch` on `requests.Session.get`). Confirms our request/response handling works against expected schemas.
- **NOW**: Subprocess-mock tests for parsers that shell out (carball, rrrocket, dotnet, npx).
- **PHASE-C**: Run a 10-stream-per-cell smoke test BEFORE full acquisition. If any cell fails parsing, fix before scaling up.
- **MAINTENANCE**: Add a quarterly "format-drift check" — pull 5 streams per cell from current week, run extractors, verify event distribution matches expectations.

**Owner:** Me  
**Priority:** High → **In progress (coverage push)**

---

### T2 — carball is unmaintained; RL replay format may have drifted
**Severity:** 🟠 High · **Likelihood:** Medium  
**Impact:** carball could fail to parse 2024 RLCS replays.

**Mitigation:**
- **NOW**: Pin `carball<2.0` in requirements. Lock to last-known-working version.
- **PHASE-C**: First step of RL acquisition — try parsing 5 known-good RLCS 2024 replays. If carball fails, immediately fall back to rrrocket subprocess (already a fallback in our pipeline).
- **NOW**: Document the fallback explicitly in `REAL_DATA_GUIDE.md`. Already done.
- **CONTINGENCY**: If both fail, swap RL cell for a different domain (Apex Legends? Valorant?). Would require a new sign-off round.

**Owner:** Me  
**Priority:** High

---

### T3 — NBA Stats API column names are undocumented and may change
**Severity:** 🟡 Medium · **Likelihood:** Low  
**Impact:** Our `col_idx[...]` lookups assume specific header strings. NBA changes these without warning.

**Mitigation:**
- **NOW**: Add a defensive fallback in `_parse_row_raw` — if a column is missing, log a warning and use a default. Already partially implemented (`col_idx.get(..., -1)`).
- **NOW**: Test with current `nba_api` version's sample data; add as fixture.
- **MAINTENANCE**: Re-run sanity test before Phase C.

**Owner:** Me  
**Priority:** Medium

---

### T4 — Hearthstone log format changes per game patch
**Severity:** 🟡 Medium · **Likelihood:** High over time  
**Impact:** hslog versioning must match game-data version of replays we collect.

**Mitigation:**
- **NOW**: Pin `hearthstone>=2.2` in requirements.
- **PHASE-C**: All 300 HS replays should be from the same expansion era (2024 standard format) to minimize version drift within the dataset.
- **DOCUMENT**: Note in REAL_DATA_GUIDE that hslog version drift is the most likely failure mode; have a fallback to direct XML parsing if hslog breaks.

**Owner:** Me  
**Priority:** Medium

---

### T5 — Fortnite Epic CDN auth is fragile; account-flagging risk
**Severity:** 🟠 High · **Likelihood:** Medium  
**Impact:** Burner account gets flagged → cannot pull replays → cell blocked.

**Mitigation:**
- **PHASE-C**: Use the user-provided Node.js async queue (per D-15) with rate-limiting + proxy.
- **PHASE-C**: Pre-warm the burner account by pulling smaller sample (50 replays) over 2–3 days before the full pull.
- **PHASE-C**: If account flagged, FortniteTracker (TRN) API as last-resort fallback (loses spatial telemetry — would need to mark cell as "limited"; needs sign-off).
- **PROGRAM**: Flag this risk to your Mew Fortnite work too — burner account practice generalizes.

**Owner:** Lead author + me  
**Priority:** High

---

## Data quality risks

### D1 — Per-cell chain length still null in harness.yaml
**Severity:** 🔴 Critical (will break Phase D) · **Likelihood:** Certain to fail loudly when wired up  
**Impact:** Without locked N values, ChainBuilder raises `ValueError` on use. So this fails fast — not a silent bug, but blocks Phase D until resolved.

**Mitigation:**
- **PHASE-B**: Lock all 5 N values during the joint session (Q F-3, N-3, C-3, R-3, H-3 in PHASE_B_PREP.md).
- **NOW**: ChainBuilder's `get_chain_length` raises with a clear error message pointing to the fix. Already implemented.

**Owner:** Both authors  
**Priority:** Resolved on Phase B

---

### D2 — Mock data event-type distribution is too clean
**Severity:** 🟡 Medium · **Likelihood:** Certain  
**Impact:** Pilot validation against mock doesn't tell us anything useful about real-data filter behavior.

**Mitigation:**
- See M2.

**Owner:** Me  
**Priority:** Medium → Phase B

---

## Operational risks

### O1 — Cost estimate ($1.95) is a guess
**Severity:** 🟡 Medium · **Likelihood:** Medium (could be 2-3× off)  
**Impact:** Budget surprise.

**Mitigation:**
- **PHASE-B**: After T is designed, run cost_estimator with actual prompt token counts (use `tiktoken` or Anthropic's tokenizer to count). Update the estimate.
- **PHASE-D**: Run a 10-chain-per-cell calibration BEFORE full eval — measures actual per-call cost.
- **GUARDRAIL**: Set spend cap on Anthropic API key for v5 ($25 ceiling).

**Owner:** Me + lead author  
**Priority:** Medium

---

### O2 — Real-data acquisition could take 6–12 hours
**Severity:** 🟡 Medium · **Likelihood:** High  
**Impact:** Wall-clock cost.

**Mitigation:**
- **PHASE-C**: Parallelize across cells (each cell runs in its own process; 5× speedup).
- **PHASE-C**: Use checkpointing — `pipeline.fetch()` already skips files that exist on disk. Resume-on-interrupt works.
- **DOCUMENT**: Add "expected wall-clock" to REAL_DATA_GUIDE per cell.

**Owner:** Me  
**Priority:** Medium

---

## Architectural risks

### A1 — PromptBuilder constraint-context is placeholder
**Severity:** 🟡 Medium · **Likelihood:** Certain  
**Impact:** Real prompts won't match what's tested.

**Mitigation:**
- See PHASE_B_PREP §*-5 questions.
- **PHASE-B**: Lock per-cell language during joint session.
- **NOW**: Add a test that catches the placeholder and warns when used in a real eval call.

**Owner:** Both authors  
**Priority:** Resolved on Phase B

---

### A2 — T interface assumes per-game translation; cross-game state unclear
**Severity:** ⚪ Low · **Likelihood:** Low  
**Impact:** If T design needs cross-game state (player history, opponent context), interface needs reworking.

**Mitigation:**
- **PHASE-B**: Surface this as an explicit question during T design — does T need anything beyond `EventStream`?
- **PROGRAM**: If yes, refactor `TranslationFunction.translate()` to optionally accept a "context" dict.

**Owner:** Both authors  
**Priority:** Low

---

## Coverage risk

### C1 — 24% of code uncovered, mostly real-fetch + CLI mains
**Severity:** 🟡 Medium · **Likelihood:** Certain  
**Impact:** The riskiest paths (network I/O, parser invocations) are the least-tested.

**Mitigation:**
- **NOW**: HTTP-mock + subprocess-mock tests for fetch+parse paths. **In progress (this round).**
- **NOW**: Test all `main()` functions with argparse.
- **NOW**: Test print_summary methods with `capsys`.
- **TARGET**: 90%+ coverage by end of session.

**Owner:** Me  
**Priority:** Resolved by end of this session

---

## Summary table

| ID | Risk | Severity | Status | Mitigated by |
|----|------|----------|--------|--------------|
| M1 | v3 Phase 2 not run | 🔴 | Pending | Interim analysis script |
| M2 | Mock T 100% retention | 🟠 | Pending | Noisy-mock T in Phase B |
| M3 | Bonferroni too conservative | 🟠 | Documented | Triple-report (per-cell, pooled, h) |
| T1 | Real-fetch untested | 🟠 | **In progress** | HTTP/subprocess mocks |
| T2 | Carball unmaintained | 🟠 | Pinned + fallback | rrrocket subprocess |
| T3 | NBA API drift | 🟡 | Defensive coding | col_idx.get fallback |
| T4 | HS log format | 🟡 | Pinned + same-era | Single expansion era |
| T5 | Fortnite Epic CDN | 🟠 | Burner + queue | D-15 mitigation |
| D1 | Chain length null | 🔴 | Phase B | Lock during T design |
| D2 | Mock too clean | 🟡 | See M2 | Noisy-mock T |
| O1 | Cost guess | 🟡 | Calibration | tiktoken count + spend cap |
| O2 | Long acquisition | 🟡 | Parallelize | Per-cell processes |
| A1 | Prompt placeholder | 🟡 | Phase B | Lock during T design |
| A2 | T cross-game state | ⚪ | Surface in B | Optional context dict |
| C1 | Coverage gaps | 🟡 | **In progress** | This round |

---

## What's actively being mitigated right now

- **C1 (coverage)** — adding HTTP/subprocess mocks + CLI tests + output tests in this session
- **T1 (real-fetch testing)** — same effort as C1; the mocks ARE the tests

What's blocked on Phase B:
- M2, D1, D2, A1 (all need T design first)

What's blocked on Phase C:
- T2, T5 (real data acquisition needed to discover format drift)

What needs explicit author decisions:
- M1 (interim analysis policy)
- M3 (reporting scheme)
- T5 (burner account operational details)
