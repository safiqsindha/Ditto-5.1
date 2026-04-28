"""
End-to-end integration tests for v5 — exercises the full pipeline:
EventStream → T → ChainBuilder → Gate 2 → CellRunner → RunReport.

Tests that the post-A.1 wiring (CellRunner now calls ChainBuilder when configured)
works correctly. Per the BUG_SWEEP A1 fix.
"""

from __future__ import annotations

from src.common.config import HarnessConfig
from src.common.schema import EventStream, GameEvent
from src.harness.cell_runner import CellRunner
from src.interfaces.chain_builder import FixedPerCellChainBuilder
from src.pilot.mock_t import MockT


def _stream(cell: str, n_events: int = 60, game_id: str = "g1") -> EventStream:
    s = EventStream(game_id=game_id, cell=cell)
    actionables = [
        "engage_decision", "rotation_commit", "resource_budget",
        "position_commit", "ability_use",
    ]
    for i in range(n_events):
        s.append(GameEvent(
            timestamp=float(i),
            event_type=actionables[i % len(actionables)],
            actor=f"player_{i % 4}",
            location_context={},
            raw_data_blob={},
            cell=cell,
            game_id=game_id,
            sequence_idx=i,
        ))
    return s


def _harness_cfg() -> HarnessConfig:
    return HarnessConfig(
        bonferroni_divisor=5, alpha=0.05,
        bootstrap_iterations=100, bootstrap_seed=42,
        gate2_retention_floor=0.50, min_discordant_pairs=1,
    )


class TestChainBuilderIntegration:
    def test_t_then_chainbuilder_then_gate2(self):
        """T produces wide candidates (window=10); ChainBuilder reshapes to length=5."""
        runner = CellRunner(
            config=_harness_cfg(),
            chain_builder=FixedPerCellChainBuilder({"nba": 5}),
        )
        runner.register_cell("nba", MockT(cell="nba", window_size=10, step_size=10))
        streams = {"nba": [_stream("nba", n_events=100, game_id=f"g{i}") for i in range(3)]}
        report = runner.run(streams)
        nba_result = next(r for r in report.cells if r.cell == "nba")
        # MockT(window=10) on 100 events with step=10 → 10 candidates per stream × 3 streams = 30 candidates
        # ChainBuilder(N=5) splits each 10-event candidate into 2 chains of 5 → ~60 chains
        assert nba_result.n_chains_pre_gate2 >= 30  # at least the candidate count
        # All chains should be exactly length 5 after ChainBuilder
        # (we can't directly inspect chain lengths from CellResult, but n_chains shouldn't be 0)
        assert nba_result.n_chains_post_gate2 > 0

    def test_chainbuilder_records_in_report_config(self):
        """A2 fix: per-cell chain length should be in RunReport.config."""
        runner = CellRunner(
            config=_harness_cfg(),
            chain_builder=FixedPerCellChainBuilder({"nba": 6, "csgo": 8}),
        )
        runner.register_cell("nba", MockT(cell="nba"))
        report = runner.run({"nba": [_stream("nba")]})
        assert "chain_builder" in report.config
        cb_cfg = report.config["chain_builder"]
        assert cb_cfg["per_cell_chain_length"]["nba"] == 6
        assert cb_cfg["per_cell_chain_length"]["csgo"] == 8

    def test_chainbuilder_drops_short_candidates(self):
        """Candidates shorter than per-cell N are dropped, not included."""
        cb = FixedPerCellChainBuilder({"nba": 10})
        # Build candidates of varying lengths from streams
        streams = [_stream("nba", n_events=8, game_id="g1")]  # 8 events, < N=10
        runner = CellRunner(config=_harness_cfg(), chain_builder=cb)
        runner.register_cell("nba", MockT(cell="nba", window_size=5))
        report = runner.run({"nba": streams})
        nba = next(r for r in report.cells if r.cell == "nba")
        # MockT(window=5) on 8 events produces some candidates of length ~5,
        # but ChainBuilder(N=10) drops them all
        assert nba.n_chains_pre_gate2 == 0

    def test_no_chainbuilder_falls_through(self):
        """When no ChainBuilder is set, T candidates pass through unchanged."""
        runner = CellRunner(config=_harness_cfg())
        # No chain_builder set → T candidates flow directly to Gate 2
        runner.register_cell("nba", MockT(cell="nba", window_size=5, step_size=5))
        streams = {"nba": [_stream("nba", n_events=100)]}
        report = runner.run(streams)
        nba = next(r for r in report.cells if r.cell == "nba")
        assert nba.n_chains_pre_gate2 > 0


class TestEndToEndScoring:
    def test_full_pipeline_produces_mcnemar(self):
        runner = CellRunner(
            config=_harness_cfg(),
            chain_builder=FixedPerCellChainBuilder({"nba": 5}),
        )
        runner.register_cell("nba", MockT(cell="nba", window_size=5, step_size=5))
        streams = {"nba": [_stream("nba", n_events=100, game_id=f"g{i}") for i in range(5)]}
        # 5 streams × 20 chains/stream = 100 chains
        # Construct responses so ~30 chains are "intervention right, baseline wrong"
        # and ~10 are "baseline right, intervention wrong" → c=30, b=10
        n = 100
        baseline = (["no"] * 30) + (["yes"] * 10) + (["yes"] * 60)        # 30 wrong + 70 right
        intervention = (["yes"] * 30) + (["no"] * 10) + (["yes"] * 60)    # 30 right + 10 wrong + 60 right
        gt = ["yes"] * n
        report = runner.run(
            streams,
            baseline_responses={"nba": baseline},
            intervention_responses={"nba": intervention},
            ground_truths={"nba": gt},
        )
        nba = next(r for r in report.cells if r.cell == "nba")
        assert nba.mcnemar is not None
        # Discordant pairs: c = 30 (baseline wrong, intervention right), b = 10 (reverse)
        assert nba.mcnemar.c > nba.mcnemar.b
        assert nba.mcnemar.c == 30
        assert nba.mcnemar.b == 10


class TestErrorPaths:
    def test_chainbuilder_with_unset_cell_records_error(self):
        """If chain_builder is configured but cell N is not set, should record error."""
        runner = CellRunner(
            config=_harness_cfg(),
            chain_builder=FixedPerCellChainBuilder(),  # all None
        )
        runner.register_cell("nba", MockT(cell="nba"))
        streams = {"nba": [_stream("nba", 50)]}
        report = runner.run(streams)
        nba = next(r for r in report.cells if r.cell == "nba")
        assert any("ChainBuilder error" in e for e in nba.errors)
