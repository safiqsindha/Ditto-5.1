# v5: Five-Cell Parallel Detection Experiment

[![v5 tests](https://github.com/safiqsindha/Ditto-V5/actions/workflows/v5-tests.yml/badge.svg)](https://github.com/safiqsindha/Ditto-V5/actions/workflows/v5-tests.yml)

Five-cell parallel replication of v3's constraint-chain detection methodology across new game domains. Subject model: Claude Haiku 4.5.

**Status:** ✅ **CLOSED 2026-04-30** — Phase D complete at n=1,200 chains/cell, results locked, methodology paused for v5.1 cross-model replication (scoped separately). Tag: [`v5.0`](https://github.com/safiqsindha/Ditto-V5/releases/tag/v5.0).

**Headline finding:** 4-tier representational hierarchy of LLM constraint reasoning. 4 of 5 cells significant past Bonferroni at α/5 by many orders of magnitude. See [`STATUS.md`](STATUS.md) and [`MEMO.md`](MEMO.md).

| Cell | Δ Det@Int | p_Bonferroni | Tier |
|------|----------:|-------------:|------|
| pubg          | +24.1% | 1.1e-63   | 1 (aligned) |
| nba           | +42.6% | 5.2e-112  | 1 (aligned) |
| csgo          | +32.9% | 9.2e-87   | 2 (partial-observability) |
| rocket_league | +5.9%  | 4.9e-16   | 3 (misaligned) |
| poker         | -0.1%  | 1.000     | 0 (saturated/ceiling) |

**Reference experiments:** v3 (Chess/Checkers), v4 (single-cell methodology characterization). v5.1 = cross-model replication via OpenRouter, scoped separately.

---

## Domains (one cell each)

| Cell | Domain | Data Source | Sample Target |
|------|---------|-------------|---------------|
| `pubg` | PUBG sample matches (squad-fpp / solo-fpp) | PUBG Developer API (telemetry) | 25 matches (smoke), scaling to 80+ for full corpus |
| `nba` | 2023-24 season | NBA Stats API (PlayByPlayV3) | 300 games |
| `csgo` | 2024 S-tier (CS2) | HLTV demo archive + awpy | 150 maps |
| `rocket_league` | RLCS 2024 | BallChasing.com + rrrocket | 250 replays |
| `poker` | NLHE — HandHQ + WSOP 2023 (all-human) | PHH Dataset v3 (tomli) | 3,500 hands |

**SPEC v1.1 amendments** ([SPEC.md](SPEC.md)):
- **A1** (pre-current session): Hearthstone → Poker — HSReplay friction; PHH Dataset v3 is open
- **A2** (2026-04-28): Fortnite → PUBG — Epic CDN locked down public chunk access; PUBG offers a documented public API ([D-35](DECISION_LOG.md), [D-36](DECISION_LOG.md))
- **A3** (2026-04-28): Poker corpus Pluribus → HandHQ — Pluribus hands include Facebook's superhuman bot in one of 6 seats, contaminating ~17% of actions; switched to HandHQ (anonymized human cash games) + WSOP 2023 ([D-37](DECISION_LOG.md))

---

## Quickstart

```bash
# Install dependencies (Python 3.11+)
pip install -r requirements.txt

# Copy and fill in your API keys
cp .env.example .env   # then edit .env

# Run the test suite
python -m pytest tests/

# Dry-run evaluation (mock data, no API or LLM calls)
python run_eval.py --dry-run --output RESULTS/eval_dry_run.json

# Run the pilot validator (mock data, no API calls)
python run_pilot.py

# Run pilot for one cell only
python run_pilot.py --cells nba

# Save pilot report to JSON
python run_pilot.py --output RESULTS/pilot_report.json
```

---

## Read these first

| File | Purpose |
|------|---------|
| `STATUS.md` | End-state status (Phase D closed 2026-04-30, results locked) |
| `MEMO.md` | Internal closeout memo + bridge document for the eventual V1–V5.1 arXiv preprint |
| `DECISION_LOG.md` | D-0 → D-45, every methodology decision with default/alternative/reversibility |
| `SPEC.md` | Pre-registered specification (signed 2026-04-27, 7 amendments adopted) |
| `BUILD_PLAN.md` | Sequenced build plan with dependency graph |
| `docs/REAL_DATA_GUIDE.md` | How to set up credentials and run real-data acquisition |
| `docs/STATUS_BUILD_DAY_2026-04-27.md` | Original end-of-build-day status (preserved for history) |

---

## Repository Layout

```
./
├── BUILD_PLAN.md, SPEC.md, DECISION_LOG.md, STATUS.md   # Documentation
├── README.md                                             # This file
├── run_pilot.py                                          # Pilot validation entry point
├── run_eval.py                                           # Phase D evaluation entry point
├── requirements.txt
├── config/
│   ├── cells.yaml          # Per-cell sample targets, stratification, env vars
│   └── harness.yaml        # Statistical harness parameters (Bonferroni, alpha, etc.)
├── src/
│   ├── common/             # GameEvent, EventStream, ChainCandidate; config loader
│   ├── harness/            # McNemar, scoring, variance, ACTIONABLE_TYPES, cell runner
│   ├── interfaces/         # TranslationFunction and ChainBuilder ABCs
│   ├── cells/              # One subdirectory per domain: pipeline.py + extractor.py
│   │   ├── pubg/           # active battle-royale cell (replaces fortnite per A2)
│   │   ├── fortnite/       # legacy — kept for tests; not in active configs
│   │   ├── nba/
│   │   ├── csgo/
│   │   ├── rocket_league/
│   │   └── poker/
│   └── pilot/              # MockT + PilotValidator + render_report
├── data/
│   ├── raw/                # Per-cell raw downloads (gitignored)
│   ├── processed/          # Per-cell parsed records (gitignored)
│   └── events/             # Per-cell normalized GameEvent streams (gitignored)
├── RESULTS/                # Pilot reports, evaluation outputs (gitignored)
├── notebooks/              # Analysis notebooks
└── tests/                  # pytest suite (401 tests)
```

---

## What's Implemented

- All five data acquisition pipelines (real-fetch + mock fallback) — pubg, nba, csgo, rocket_league, poker
- All five domain event extractors with derived-state markers (per D-43)
- All five Translation Functions T (`src/interfaces/translation.py`)
- Per-cell PromptBuilder + `_MarkerSurfacing` for chain rendering (per A4–A6)
- Per-cell violation injectors (`src/harness/violation_injector.py`) for the violation-detection diagnostic (D-42)
- Statistical harness: McNemar (continuity correction or exact binomial when n_disc<25), Bonferroni, bootstrap CI
- Anthropic Batches API caller with positional custom_id integrity (`src/harness/model_evaluator.py`)
- Phase D entry point (`run_diagnostic_violations.py`) with strict-grounding (D-44 Layer 1)
- Layer-2 CoT FP diagnostic (`run_phase_d_cot.py`)
- Phase D synthesis pipeline (`synthesize_phase_d.py`)
- Raw batch archival (`archive_phase_d_batches.py` → `RESULTS/phase_d_raw_batches/`, 23,998 records committed)
- Test suite: 415 tests passing (9 pre-existing failures in Fortnite + CSGO mock tests, tracked in [#5](https://github.com/safiqsindha/Ditto-V5/issues/5) — unrelated to v5 results)

## Reproducing the Headline Numbers

```bash
.venv/bin/python synthesize_phase_d.py
# -> RESULTS/phase_d_final.json + console table

# Layer-2 CoT diagnostic on residual NBA + CSGO FPs:
.venv/bin/python run_phase_d_cot.py --cells nba csgo
```

If 60-day Anthropic batch retention has expired, the raw responses are archived locally at `RESULTS/phase_d_raw_batches/*.jsonl` — modify `fetch_batch` in `retrieve_phase_d_partial.py` to read those files instead of hitting the API.

---

## What's Deferred to v5.1+ / v6

- **v5.1 cross-model replication via OpenRouter** — frozen Phase D prompts replayed across Anthropic / OpenAI / Google / open-weights models with a derived-state-marker ablation as a second axis. Pre-registered design in `MEMO.md` §6.
- **v5.2 CSGO awpy fix** — adds bomb-site observability via parsed CS2 demos. Run *after* cross-model so capability vs. observability can be cleanly separated.
- **v5.2 Rocket League per-event extraction** — carball / boxcars-py replay parsing. Same deferral logic as CSGO.
- **v6 chain-length sweep, reasoning-mode toggles** — open candidates.

---

## License

MIT.
