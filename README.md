# v5: Five-Cell Parallel Detection Experiment

[![v5 tests](https://github.com/safiqsindha/Ditto-V5/actions/workflows/v5-tests.yml/badge.svg)](https://github.com/safiqsindha/Ditto-V5/actions/workflows/v5-tests.yml)

Five-cell parallel replication of v3's constraint-chain detection methodology across new game domains. Subject model: Claude Haiku.

**Status:** SPEC v1.0 signed (both authors, 2026-04-27). Phase A + B complete (infrastructure, poker cell, harness, CF-3=A leakage analysis). Phase C/D (T implementation, evaluation runs) pending SPEC sign-off.
**Reference experiments:** v3 (Chess/Checkers), v4 (single-cell methodology characterization).

---

## Domains (one cell each)

| Cell | Domain | Data Source | Sample Target |
|------|---------|-------------|---------------|
| `fortnite` | FNCS / Cash Cup 2024 | Epic CDN replays + FortniteReplayDecompressor | 200 matches |
| `nba` | 2023-24 season | NBA Stats API (PlayByPlayV3) | 300 games |
| `csgo` | 2024 S-tier (CS2) | HLTV demo archive + awpy | 150 maps |
| `rocket_league` | RLCS 2024 | BallChasing.com + rrrocket | 250 replays |
| `poker` | NLHE — Pluribus + WSOP 2023 | PHH Dataset v3 (pokerkit) | 300 games |

---

## Quickstart

```bash
# Install dependencies (Python 3.9+)
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
| `BUILD_PLAN.md` | Sequenced build plan with dependency graph (written before any code) |
| `SPEC.md` | Draft specification — **8 [REQUIRES SIGN-OFF] questions** block T implementation |
| `DECISION_LOG.md` | Every non-pre-specified decision logged with default/alternative/reversibility |
| `STATUS.md` | End-of-build summary |
| `docs/REAL_DATA_GUIDE.md` | How to set up credentials and run real-data acquisition |

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
│   │   ├── fortnite/
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

## What's Implemented vs Stubbed

### Implemented
- All five data acquisition pipelines (real-fetch + mock fallback)
- All five domain event extractors
- Statistical harness: McNemar (continuity correction, Bonferroni, bootstrap CI), scoring, variance, cell runner
- ACTIONABLE_TYPES whitelist (v1.1: ResourceBudget added, phase_ prefix stripped)
- ChainBuilder: fixed per-cell chain lengths, non-overlapping windows, CF-3=A shuffle controls
- CF-3=A leakage diagnosis: quantified leakage ratio with direction check
- MDE and post-hoc power surfaced in all evaluation reports
- Pilot validation harness with MockT + NoisyMockT
- Phase D evaluation entry point (`run_eval.py`) with dry-run mode
- Test suite: 401 tests covering harness, pilot, cells, and CLI integration

### Stubbed (Out of Scope Until SPEC Sign-Off)
- All five Translation Functions T (raise `NotImplementedError`)
- All evaluation runs against the real Haiku API

---

## Pre-Registration Sign-Off Blockers

Before T can be implemented or evaluation runs can begin, both authors must answer the eight questions in `SPEC.md`:

1. Baseline/intervention prompt structure
2. Bonferroni divisor (5 vs 1 vs mixed)
3. Sample size adequacy (n=1,200/cell)
4. Chain length consistency (fixed vs variable)
5. CS:GO granularity (round-level vs tick-sampled vs clutch-level)
6. NBA granularity (possession vs play-level)
7. Rocket League state handling (hit-level vs possession vs boost-enriched)
8. Poker granularity (per-action vs per-hand vs street-level)

---

## License

MIT.
