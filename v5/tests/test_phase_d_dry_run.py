"""
Phase D end-to-end dry run — exercises the full evaluation pipeline with
mocked Haiku API responses (no real API spend).

Pipeline under test:
  mock EventStreams → real T → Gate 2 → PromptBuilder → ModelEvaluator (dry_run)
  → CellRunner (McNemar) → RunReport

Catches integration bugs before real spend. Tests are ordered to follow the
Phase D sequence: chains → prompts → evaluation → statistical analysis.
"""

from __future__ import annotations

import pytest
from v5.src.common.config import HarnessConfig
from v5.src.common.schema import EventStream, GameEvent
from v5.src.harness.cell_runner import CellRunner, RunReport
from v5.src.harness.model_evaluator import EvaluationResult, ModelEvaluator
from v5.src.harness.prompts import PER_CELL_PROMPT_BUILDERS, PromptPair
from v5.src.interfaces.translation import DOMAIN_T_STUBS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nba_stream(n_events: int = 60, game_id: str = "dry_g1") -> EventStream:
    """NBA stream with period stamps so NBAT groups correctly."""
    stream = EventStream(game_id=game_id, cell="nba")
    actionable = [
        "engage_decision", "position_commit", "rotation_commit",
        "team_coordinate", "timing_commit", "resource_gain",
    ]
    for i in range(n_events):
        stream.append(GameEvent(
            timestamp=float(i),
            event_type=actionable[i % len(actionable)],
            actor=f"player_{i % 5}",
            location_context={"period": (i // (n_events // 4)) + 1},
            raw_data_blob={},
            cell="nba",
            game_id=game_id,
            sequence_idx=i,
        ))
    return stream


def _csgo_stream(n_events: int = 50, game_id: str = "dry_cg1") -> EventStream:
    """CSGO stream with round stamps so CSGOT groups correctly."""
    stream = EventStream(game_id=game_id, cell="csgo")
    actionable = [
        "engage_decision", "position_commit", "rotation_commit",
        "objective_capture", "zone_enter", "resource_budget",
    ]
    for i in range(n_events):
        stream.append(GameEvent(
            timestamp=float(i),
            event_type=actionable[i % len(actionable)],
            actor=f"ct_{i % 5}",
            location_context={"round": i // 5},
            raw_data_blob={},
            cell="csgo",
            game_id=game_id,
            sequence_idx=i,
        ))
    return stream


def _harness_cfg(**kw) -> HarnessConfig:
    base = dict(
        alpha=0.05, bonferroni_divisor=5, continuity_correction=True,
        bootstrap_iterations=50, bootstrap_seed=7,
        gate2_retention_floor=0.50, min_discordant_pairs=1,
    )
    base.update(kw)
    return HarnessConfig(**base)


# ---------------------------------------------------------------------------
# 1. Chain generation with real T
# ---------------------------------------------------------------------------

class TestPhaseDChainGeneration:
    def test_nba_real_t_produces_chains(self):
        t = DOMAIN_T_STUBS["nba"]
        stream = _nba_stream(n_events=60)
        chains = t.translate(stream)
        assert len(chains) > 0, "NBAT should produce at least one chain"

    def test_csgo_real_t_produces_chains(self):
        t = DOMAIN_T_STUBS["csgo"]
        stream = _csgo_stream(n_events=50)
        chains = t.translate(stream)
        assert len(chains) > 0, "CSGOT should produce at least one round chain"

    def test_all_five_t_run_without_error(self):
        cell_streams = {
            "fortnite": EventStream(game_id="fn1", cell="fortnite"),
            "nba": _nba_stream(),
            "csgo": _csgo_stream(),
            "rocket_league": EventStream(game_id="rl1", cell="rocket_league"),
            "poker": EventStream(game_id="pk1", cell="poker"),
        }
        # Add events to fortnite / rl streams so T has something to work with
        for ev_type, cell in [
            ("zone_enter", "fortnite"), ("objective_capture", "rocket_league"),
        ]:
            s = cell_streams[cell]
            for i in range(20):
                s.append(GameEvent(
                    timestamp=float(i), event_type=ev_type,
                    actor="p1", location_context={}, raw_data_blob={},
                    cell=cell, game_id=s.game_id, sequence_idx=i,
                ))
        # Poker: 6 actors × 5 events each so PokerT produces chains
        pk = cell_streams["poker"]
        for actor_idx in range(6):
            for j in range(5):
                pk.append(GameEvent(
                    timestamp=float(actor_idx * 10 + j),
                    event_type="engage_decision",
                    actor=f"actor_{actor_idx}", location_context={}, raw_data_blob={},
                    cell="poker", game_id=pk.game_id,
                    sequence_idx=actor_idx * 10 + j,
                ))

        for cell, stream in cell_streams.items():
            t = DOMAIN_T_STUBS[cell]
            result = t.translate(stream)
            assert isinstance(result, list), f"{cell} T must return list"


# ---------------------------------------------------------------------------
# 2. Prompt construction
# ---------------------------------------------------------------------------

class TestPhaseDPromptConstruction:
    def test_prompt_pair_built_for_nba_chain(self):
        t = DOMAIN_T_STUBS["nba"]
        chains = t.translate(_nba_stream(n_events=60))
        assert chains, "Need chains to test prompt builder"

        builder = PER_CELL_PROMPT_BUILDERS["nba"]()
        pair = builder.build(chains[0])
        assert isinstance(pair, PromptPair)
        assert "YES or NO" in pair.intervention_prompt
        assert "Constraint Context" in pair.intervention_prompt
        assert "Constraint Context" not in pair.baseline_prompt

    def test_all_five_prompt_builders_run(self):
        """Each cell's PromptBuilder should produce a PromptPair without error."""
        for cell, BuilderClass in PER_CELL_PROMPT_BUILDERS.items():
            t = DOMAIN_T_STUBS[cell]
            stream = _nba_stream(game_id=f"test_{cell}")
            stream.cell = cell  # re-tag for non-nba cells
            for ev in stream.events:
                ev.cell = cell
            chains = t.translate(stream)
            if not chains:
                continue  # some T need domain-specific triggers
            builder = BuilderClass()
            pair = builder.build(chains[0])
            assert pair.chain_id == chains[0].chain_id
            assert pair.cell == cell


# ---------------------------------------------------------------------------
# 3. ModelEvaluator dry-run
# ---------------------------------------------------------------------------

class TestModelEvaluatorDryRun:
    def test_dry_run_returns_responses_for_all_pairs(self):
        t = DOMAIN_T_STUBS["nba"]
        chains = t.translate(_nba_stream(n_events=60))[:5]
        builder = PER_CELL_PROMPT_BUILDERS["nba"]()
        pairs = [builder.build(c) for c in chains]

        evaluator = ModelEvaluator(dry_run=True)
        results, baseline, intervention = evaluator.evaluate_pairs(pairs)

        assert len(results) == len(pairs)
        assert len(baseline) == len(pairs)
        assert len(intervention) == len(pairs)
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_dry_run_responses_are_yes_or_no(self):
        t = DOMAIN_T_STUBS["nba"]
        chains = t.translate(_nba_stream(n_events=60))[:10]
        builder = PER_CELL_PROMPT_BUILDERS["nba"]()
        pairs = [builder.build(c) for c in chains]

        evaluator = ModelEvaluator(dry_run=True)
        _, baseline, intervention = evaluator.evaluate_pairs(pairs)
        for r in baseline + intervention:
            assert r in ("yes", "no", ""), f"Unexpected parsed response: {r!r}"

    def test_dry_run_intervention_bias_toward_yes(self):
        """Mock is designed so intervention > baseline YES-rate."""
        t = DOMAIN_T_STUBS["nba"]
        chains = t.translate(_nba_stream(n_events=200))[:50]
        builder = PER_CELL_PROMPT_BUILDERS["nba"]()
        pairs = [builder.build(c) for c in chains]

        evaluator = ModelEvaluator(dry_run=True)
        _, baseline, intervention = evaluator.evaluate_pairs(pairs)
        b_yes = sum(1 for r in baseline if r == "yes")
        i_yes = sum(1 for r in intervention if r == "yes")
        assert i_yes > b_yes, "Intervention should have more YES than baseline in dry-run"

    def test_dry_run_deterministic(self):
        t = DOMAIN_T_STUBS["nba"]
        chains = t.translate(_nba_stream(n_events=60))[:5]
        builder = PER_CELL_PROMPT_BUILDERS["nba"]()
        pairs = [builder.build(c) for c in chains]
        evaluator = ModelEvaluator(dry_run=True)
        _, b1, i1 = evaluator.evaluate_pairs(pairs)
        _, b2, i2 = evaluator.evaluate_pairs(pairs)
        assert b1 == b2
        assert i1 == i2


# ---------------------------------------------------------------------------
# 4. Full CellRunner pipeline (end-to-end dry run)
# ---------------------------------------------------------------------------

class TestPhaseDCellRunnerEndToEnd:
    def _build_responses(self, cell: str, n_streams: int = 10) -> tuple:
        """Generate chains and dry-run model responses for a cell."""
        t = DOMAIN_T_STUBS[cell]
        streams = [_nba_stream(n_events=60, game_id=f"e2e_{cell}_{i}")
                   for i in range(n_streams)]
        # Re-tag streams for non-NBA cells
        if cell != "nba":
            for s in streams:
                s.cell = cell
                for ev in s.events:
                    ev.cell = cell

        all_chains = []
        for s in streams:
            all_chains.extend(t.translate(s))

        if not all_chains:
            return streams, [], [], []

        builder_cls = PER_CELL_PROMPT_BUILDERS.get(cell)
        if builder_cls is None:
            return streams, [], [], []
        builder = builder_cls()
        pairs = [builder.build(c) for c in all_chains]

        evaluator = ModelEvaluator(dry_run=True)
        _, baseline, intervention = evaluator.evaluate_pairs(pairs)
        ground_truth = ["yes"] * len(all_chains)
        return streams, baseline, intervention, ground_truth

    def test_nba_full_pipeline(self):
        cell = "nba"
        streams, baseline, intervention, gt = self._build_responses(cell, n_streams=10)
        if not baseline:
            pytest.skip("No chains produced for nba in dry-run")

        runner = CellRunner(config=_harness_cfg())
        runner.register_cell(cell, DOMAIN_T_STUBS[cell])
        report = runner.run(
            {cell: streams},
            baseline_responses={cell: baseline},
            intervention_responses={cell: intervention},
            ground_truths={cell: gt},
        )
        assert isinstance(report, RunReport)
        cell_result = next(r for r in report.cells if r.cell == cell)
        assert cell_result.n_chains_pre_gate2 > 0
        assert cell_result.mcnemar is not None
        assert cell_result.mcnemar.cell == cell

    def test_report_has_aggregate_stats(self):
        cell = "nba"
        streams, baseline, intervention, gt = self._build_responses(cell, n_streams=8)
        if not baseline:
            pytest.skip("No chains produced")

        runner = CellRunner(config=_harness_cfg())
        runner.register_cell(cell, DOMAIN_T_STUBS[cell])
        report = runner.run(
            {cell: streams},
            baseline_responses={cell: baseline},
            intervention_responses={cell: intervention},
            ground_truths={cell: gt},
        )
        assert "n_cells" in report.aggregate

    def test_report_is_serialisable(self, tmp_path):
        cell = "nba"
        streams, baseline, intervention, gt = self._build_responses(cell, n_streams=5)
        runner = CellRunner(config=_harness_cfg())
        runner.register_cell(cell, DOMAIN_T_STUBS[cell])
        report = runner.run(
            {cell: streams},
            baseline_responses={cell: baseline} if baseline else None,
            intervention_responses={cell: intervention} if baseline else None,
            ground_truths={cell: gt} if baseline else None,
        )
        path = tmp_path / "phase_d_dry_run.json"
        report.save(path)
        assert path.exists()
        import json
        with open(path) as f:
            data = json.load(f)
        assert "run_id" in data
        assert "cells" in data

    def test_no_real_api_calls_in_dry_run(self, monkeypatch):
        """Confirm dry_run=True never touches anthropic client."""
        import sys

        class _FakeAnthropic:
            class Anthropic:
                def __init__(self):
                    raise RuntimeError("SHOULD NOT BE CALLED IN DRY RUN")

        monkeypatch.setitem(sys.modules, "anthropic", _FakeAnthropic)
        evaluator = ModelEvaluator(dry_run=True)
        from v5.src.harness.prompts import PromptPair
        pair = PromptPair(
            chain_id="test_chain",
            cell="nba",
            baseline_prompt="baseline",
            intervention_prompt="intervention",
            metadata={},
        )
        results, baseline, intervention = evaluator.evaluate_pairs([pair])
        assert len(results) == 1   # no exception = no real call made
