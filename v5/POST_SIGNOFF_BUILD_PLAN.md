# v5 Post-Sign-Off Build Plan

**SPEC status:** ✅ Signed 2026-04-27 by lead author + Myriam  
**This plan:** what gets built now that the SPEC is locked  
**Replaces:** the original `BUILD_PLAN.md` was the infrastructure-first plan; this is the post-sign-off continuation

---

## What sign-off unlocked

| Was blocked | Now unblocked |
|-------------|---------------|
| ChainBuilder implementation (Q4) | ✅ Build `FixedPerCellChainBuilder` |
| Extractor granularity (Q5–Q8) | ✅ Update NBA → possession-level, RL → boost-enriched |
| Evaluation prompt structure (Q1) | ✅ Build prompt template + response parser |
| Bonferroni divisor (Q2) | ✅ Locked to 5; cost estimator can run |
| Sample sizing (Q3) | ✅ Locked to 1,200/cell; acquisition targets confirmed |
| T implementations | 🟡 Design unblocked; **implementation still gates on joint authoring session** (per v6-discipline) |

---

## Phase A — Autonomous remote work (no credentials, no joint session needed)

These are buildable now without you or Myriam in the room. Sequenced by dependency.

### A.1 — Extractor updates per Q5–Q8 sign-off

| Cell | Change | Why |
|------|--------|-----|
| NBA | Switch from play-level to possession-level (D-21) | Q6 sign-off |
| Rocket League | Promote boost events to first-class merged stream (D-22) | Q7 sign-off |
| CS:GO | No change (already round-level) | Q5 sign-off matched default |
| Hearthstone | No change (already per-action) | Q8 sign-off matched default |
| Fortnite | No change (T design will define granularity) | Not affected by Q5–Q8 |

Each cell's mock data regenerates after extractor change. Pilot must re-pass.

### A.2 — `FixedPerCellChainBuilder` (Q4)

Replace the `DefaultChainBuilder` stub in `src/interfaces/chain_builder.py` with a working implementation that:
- Accepts a per-cell `chain_length` mapping
- Slides non-overlapping windows over each EventStream
- Emits ChainCandidates of fixed length per cell
- Per-cell N values stay parameterized (locked in T-design joint session — see Phase B)

### A.3 — Evaluation prompt template + response parser (Q1)

Build `src/harness/prompts.py`:
- Baseline prompt: chain events serialized → "what does the model predict?" (no constraint context)
- Intervention prompt: same chain + constraint-context segment prepended
- Response parser: extract structured prediction from model text → maps to ground-truth space for scoring

Output: a `PromptBuilder` class that `cell_runner.py` calls; one builder per cell with cell-specific phrasing.

**Gate:** This builds the *infrastructure* for evaluation. The actual prompt content per cell — the constraint-context language, the prediction question — requires T-design context. So this lands as a templated stub with cell-specific phrasing TBD until T design.

### A.4 — Cost estimator (Q1, Q3)

`src/harness/cost_estimator.py`:
- Inputs: n_chains/cell, calls/chain (Q1=B → 2), cells (5)
- Outputs: total API calls, expected Haiku spend (using Haiku pricing), per-cell breakdown
- v5 Phase 1 estimate: 12,000 calls ≈ $4

### A.5 — Synthetic-fixture extractor tests *(item 1 from menu)*

Currently extractors only have happy-path coverage via mock pipelines. Build small synthetic fixture JSON files shaped like real outputs (one per cell):
- `tests/fixtures/nba_pbp_sample.json` — synthetic PlayByPlayV3 response
- `tests/fixtures/csgo_awpy_sample.json` — synthetic awpy DemoParser output
- `tests/fixtures/rl_carball_sample.json` — synthetic carball decompiled-replay structure
- `tests/fixtures/hs_hslog_sample.json` — synthetic hslog entity-tree dump
- `tests/fixtures/fortnite_decomp_sample.json` — synthetic FortniteReplayDecompressor output

Then unit-test each extractor's parse path against the fixture. Catches regressions when real data formats drift.

### A.6 — Config sanity check CLI *(item 2 from menu)*

`python -m v5.check_config` reports per cell:
- Display name, data source, sample target
- Time range, stratification
- Required env vars and which are set/missing
- Whether mock fallback would activate
- Pre-flight green/yellow/red light per cell before real acquisition

### A.7 — Coverage report + CI integration *(item 3 from menu)*

- Add `pytest-cov` to requirements
- Update CI workflow to run with `--cov=v5/src --cov-report=xml --cov-report=html`
- Upload coverage artifact in CI
- Surface untested code paths
- Optional follow-up: add a Codecov badge to README if you sign up for Codecov

### A.8 — Linting (ruff) + pre-commit *(item 4 from menu)*

- Add `ruff.toml` with sensible defaults (line-length 100, all default rule sets except docstring style)
- Run `ruff check --fix` on the codebase (catches dead imports, unused vars, etc.)
- Add `.pre-commit-config.yaml` running ruff + pytest on commit
- Update CI to run ruff before pytest

### A.9 — CHANGELOG + CI badge *(item 5 from menu)*

- Create `v5/CHANGELOG.md` listing each round (round-1: infrastructure, round-2: extras, round-3: post-sign-off)
- Add CI status badge to top of `v5/README.md`

### A.10 — Notebook templates *(item 6 from menu)*

Skeleton notebooks in `v5/notebooks/`:
- `01_pilot_diagnostics.ipynb` — load `RESULTS/pilot_report*.json`, plot retention rate per cell, chain length distributions, event type histograms
- `02_cell_comparison.ipynb` — placeholder for per-cell McNemar results once real eval runs
- `03_aggregate_report.ipynb` — placeholder for cross-cell aggregate

These are templates only — no analysis decisions baked in beyond what's already in SPEC.

### A.11 — Performance benchmarking *(item 7 from menu)*

`v5/scripts/benchmark_pilot.py`:
- Time the full pilot per cell (mock data)
- Measure peak memory
- Report results in `RESULTS/benchmark.md`
- Useful as baseline for when real data acquisition timing is measured

---

## Phase B — Joint authoring session (you + Myriam in the room)

T design is the single biggest remaining design decision. Per v6-discipline, T implementations require both-author review. **Cannot be done remotely.**

### B.1 — T design session

For each of five cells, decide:
1. **Chain content selection rule** — which events form a chain? (e.g., "all events within one possession" for NBA, "all events within one round" for CS:GO)
2. **Constraint extraction rule** — what constraint(s) does T attach to each chain? (e.g., "this player has X boost", "this team has economy advantage")
3. **Prediction target** — what is the model asked to predict? (e.g., "next event type", "winning side at end of round", "score margin")
4. **Per-cell chain length N** — final value to lock in `chain_length.per_cell` (currently null in `harness.yaml`)

### B.2 — T implementation per cell

Five implementations, ~50–150 lines each. Replace the stubs in `src/interfaces/translation.py` (`FortniteT`, `NBAT`, `CSGOT`, `RocketLeagueT`, `HearthstoneT`).

### B.3 — Lock final per-cell chain lengths

Update `config/harness.yaml` with concrete N values; remove `null` placeholders.

---

## Phase C — Real data acquisition (gated on credentials)

Cannot be done remotely from this session — requires API credentials and outbound network.

### C.1 — Provision credentials

Per `docs/REAL_DATA_GUIDE.md`:
- Burner Epic account (https://www.epicgames.com/id/register) + dev portal app (https://dev.epicgames.com/portal/) → `EPIC_ACCOUNT_ID`, `EPIC_ACCESS_TOKEN`
- BallChasing free signup (https://ballchasing.com/login) → `BALLCHASING_TOKEN`
- HSReplay API key (https://hsreplay.net/account/api/) → `HSREPLAY_API_KEY`
- (NBA, CS:GO need no keys)

### C.2 — Add Fortnite Node.js queue

Drop the user's burner-account async queue snippet into `v5/src/cells/fortnite/fetch_queue.js`. Wire `pipeline.fetch()` to invoke it via subprocess.

### C.3 — Run real acquisition

```bash
python -m v5.check_config              # Phase A.6 sanity check
python -m v5.run_pilot --output v5/RESULTS/pilot_real.json   # real-data pilot
python -m v5.src.pilot.render_report v5/RESULTS/pilot_real.json
```

### C.4 — Verify Gate 2 retention

Per Q3 contingency: if any cell falls below 50% retention, scale up upstream acquisition for that cell.

---

## Phase D — Evaluation runs (gated on T + credentials)

### D.1 — Cost projection lock

Run `cost_estimator.py` against final per-cell n_chains. Confirm budget. Estimated total ≈ $4 for v5 Phase 1.

### D.2 — Baseline + intervention runs

Two API calls per chain × 1,200 chains × 5 cells = 12,000 Haiku calls.

### D.3 — Scoring + McNemar

Run cell_runner per cell, aggregate per Q2 (divisor = 5).

### D.4 — Results

Render to markdown + commit to `v5/RESULTS/v5_phase1_results.md`.

---

## Tracker — items from menu, mapped to phases

| Menu item | Phase | Status |
|-----------|-------|--------|
| 1. Synthetic-fixture extractor tests | A.5 | Phase A — remote |
| 2. Config sanity check CLI | A.6 | Phase A — remote |
| 3. Coverage report + CI integration | A.7 | Phase A — remote |
| 4. Linting (ruff) + pre-commit | A.8 | Phase A — remote |
| 5. CHANGELOG + CI badge | A.9 | Phase A — remote |
| 6. Notebook templates | A.10 | Phase A — remote |
| 7. Performance benchmarking | A.11 | Phase A — remote |

**All seven menu items are slotted into Phase A.** None are forgotten.

---

## Gate sequence

```
[SPEC sign-off] ✅
       ↓
   Phase A (remote, in progress)
       ↓
   Phase B (joint authoring, requires you + Myriam)
       ↓
   Phase C (real-data acquisition, requires credentials)
       ↓
   Phase D (evaluation runs, ~$4 spend)
       ↓
   Results report
       ↓
   (optional) Micro-experiments per MICRO_EXPERIMENTS.md if results justify
```

---

## What I'm doing right now

In this session, executing Phase A in order: A.1 (NBA + RL extractor updates), A.2 (ChainBuilder), A.3 (prompt template), A.4 (cost estimator), then A.5–A.11 in sequence. All seven menu items will be done before I stop. Will commit/push at logical breakpoints so you can pull progress at any time.
