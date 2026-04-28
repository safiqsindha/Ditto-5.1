# v5 Changelog

All notable changes to the v5 infrastructure.

## [Unreleased]

## [1.2.0] — 2026-04-27 — Phase A complete

### Added
- **Phase A.1**: NBA possession-level extraction (Q6-A); RL boost-enriched extraction (Q7-C)
- **Phase A.2**: `FixedPerCellChainBuilder` per Q4-B (was a stub raising `NotImplementedError`)
- **Phase A.3**: Evaluation `PromptBuilder` per Q1-B with baseline+intervention pair generation
- **Phase A.3**: Response parser handling abstain patterns + JSON-wrapped answers
- **Phase A.4**: `cost_estimator.py` — projects $1.95 total Phase 1 spend
- **Phase A.5**: 5 synthetic fixture JSON files + 31 extractor tests + tests for new modules
- **Phase A.6**: `python -m v5.check_config` CLI with green/yellow/red light per cell
- **Phase A.7**: pytest-cov integration; CI uploads coverage HTML/XML artifacts
- **Phase A.8**: ruff config + pre-commit hooks; ruff auto-fixed 302 issues
- **Phase A.9**: This CHANGELOG; CI status badge in README

### Fixed
- **cell_runner.py**: was referencing non-existent `self.config.cells` attribute. Changed
  to iterate over the union of registered cells and cells in the streams dict.
  Without this fix `CellRunner.run()` was unrunnable.
- **variance.py**: removed unused `p` assignment in `mcnemar_power`. Cosmetic.

### Changed
- **NBA extractor**: now produces possession-level events (~85/game) instead of
  play-level (~220/game). Legacy `_parse_row` preserved for ME-3 micro-experiment.
- **RL extractor**: derives `resource_budget` events from boost-history transitions
  crossing `LOW_BOOST_THRESHOLD=25` (per v1.1).
- **chain_builder.py**: stub replaced with `FixedPerCellChainBuilder`. Raises if a
  cell's chain length is not set in `per_cell_chain_length` map (no silent defaults).

### Stats
- Tests: 29 → 76 → 166 (this round)
- Coverage: 60% → 73%
- ruff: 0 errors

---

## [1.1.0] — 2026-04-27 — SPEC sign-off

### Added
- All 8 SPEC questions locked (Q1–Q8)
- `MICRO_EXPERIMENTS.md` tracking flagged variants (Q4-C, Q5-C, Q6-B)
- `POST_SIGNOFF_BUILD_PLAN.md` sequencing post-sign-off work
- DECISION_LOG entries D-16 through D-23 covering each sign-off
- `config/harness.yaml`: bonferroni_divisor=5 locked; chain_length.mode=fixed_per_cell

### Sign-off summary
| Q | Locked choice |
|---|---------------|
| Q1 | (B) two separate API calls per chain |
| Q2 | (A) Bonferroni divisor = 5 |
| Q3 | n = 1,200 chains/cell (matches v3 design) |
| Q4 | (B) fixed length per cell |
| Q5 | (A) CS:GO round-level |
| Q6 | (A) NBA possession-level |
| Q7 | (C) RL boost-enriched |
| Q8 | (A) HS per-action |

---

## [1.0.0] — 2026-04-27 — Round-2 additions

### Added
- `v5/README.md` navigation hub
- `v5/docs/REAL_DATA_GUIDE.md` with token-request URLs per cell
- `v5/src/pilot/render_report.py` JSON → markdown report renderer
- `.github/workflows/v5-tests.yml` CI on push/PR
- Tests for: schema, config, pipelines mock paths, render_report

### Fixed
- `BasePipeline.run()` accepts `force_mock=True` parameter to override
  the credentials-satisfied check (caught in audit when NBA/CS:GO with
  empty `env_vars` lists triggered real-fetch attempts during pilot)
- Tests had broken relative imports past package root; switched to absolute

---

## [0.1.0] — 2026-04-27 — Initial infrastructure

### Added
- v5/ scaffold (BUILD_PLAN, SPEC draft, DECISION_LOG, STATUS)
- All 5 data acquisition pipelines + event extractors (mock fallbacks)
- Statistical harness: McNemar, scoring, variance, ACTIONABLE_TYPES (v1.1)
- TranslationFunction (T) + ChainBuilder interface stubs
- MockT + PilotValidator
- 1,200 mock event streams persisted to `data/events/{cell}/`
