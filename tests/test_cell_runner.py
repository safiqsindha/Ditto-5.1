"""
Tests for CellRunner — five-cell evaluation orchestrator.
"""


from src.common.config import HarnessConfig
from src.common.schema import EventStream, GameEvent
from src.harness.cell_runner import CellResult, CellRunner, RunReport
from src.pilot.mock_t import MockT


def _stream(cell: str, n_events: int = 30, game_id: str = "g1") -> EventStream:
    s = EventStream(game_id=game_id, cell=cell)
    actionables = ["engage_decision", "rotation_commit", "resource_budget",
                   "position_commit", "ability_use"]
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


def _harness_cfg(**overrides) -> HarnessConfig:
    base = dict(
        alpha=0.05, bonferroni_divisor=5, continuity_correction=True,
        bootstrap_iterations=100, bootstrap_seed=42,
        gate2_retention_floor=0.50, min_discordant_pairs=10,
    )
    base.update(overrides)
    return HarnessConfig(**base)


class TestCellRunnerNoModelResponses:
    """Without baseline/intervention responses, the runner reports infrastructure only."""

    def test_runs_with_registered_t(self):
        runner = CellRunner(config=_harness_cfg())
        runner.register_cell("nba", MockT(cell="nba"))
        streams = {"nba": [_stream("nba", n_events=30, game_id=f"g{i}") for i in range(3)]}
        report = runner.run(streams)
        assert isinstance(report, RunReport)
        cell_result = next(r for r in report.cells if r.cell == "nba")
        assert cell_result.n_chains_pre_gate2 > 0
        assert cell_result.mcnemar is None  # no model responses

    def test_unregistered_cell_records_error(self):
        runner = CellRunner(config=_harness_cfg())
        # Register only nba, but config expects 5 cells
        runner.register_cell("nba", MockT(cell="nba"))
        streams = {"nba": [_stream("nba", 20)], "csgo": [_stream("csgo", 20)]}
        report = runner.run(streams)
        csgo_result = next(r for r in report.cells if r.cell == "csgo")
        assert any("No TranslationFunction" in e for e in csgo_result.errors)

    def test_nba_t_runs_and_produces_chains(self):
        from src.interfaces.translation import NBAT
        runner = CellRunner(config=_harness_cfg())
        runner.register_cell("nba", NBAT())
        streams = {"nba": [_stream("nba", 20)]}
        report = runner.run(streams)
        nba_result = next(r for r in report.cells if r.cell == "nba")
        # Real T runs without "not implemented" errors
        assert not any("not implemented" in e.lower() for e in nba_result.errors)

    def test_run_id_and_timestamp_set(self):
        runner = CellRunner(config=_harness_cfg())
        runner.register_cell("nba", MockT(cell="nba"))
        report = runner.run({"nba": []})
        assert report.run_id.startswith("v5_run_")
        assert report.timestamp.endswith("Z")


class TestCellRunnerWithMockResponses:
    """When responses are provided, the runner runs McNemar."""

    def _run_with_responses(self):
        runner = CellRunner(config=_harness_cfg(min_discordant_pairs=1))
        runner.register_cell("nba", MockT(cell="nba", window_size=5, step_size=3))
        streams = {"nba": [_stream("nba", n_events=30, game_id=f"g{i}") for i in range(5)]}
        # Pre-flight: see how many post-gate2 chains we'd get
        # Just provide many responses; runner truncates to actual chain count
        n_max = 200
        baseline = ["yes"] * (n_max // 2) + ["no"] * (n_max // 2)
        intervention = ["yes"] * int(n_max * 0.7) + ["no"] * (n_max - int(n_max * 0.7))
        gt = ["yes"] * n_max
        return runner.run(
            streams,
            baseline_responses={"nba": baseline},
            intervention_responses={"nba": intervention},
            ground_truths={"nba": gt},
        )

    def test_mcnemar_populated_when_responses_provided(self):
        report = self._run_with_responses()
        nba_result = next(r for r in report.cells if r.cell == "nba")
        assert nba_result.mcnemar is not None
        assert nba_result.mcnemar.cell == "nba"
        assert nba_result.variance_baseline is not None
        assert nba_result.variance_intervention is not None
        assert nba_result.power is not None

    def test_aggregate_populated(self):
        report = self._run_with_responses()
        assert "n_cells" in report.aggregate
        assert report.aggregate["n_cells"] >= 1


class TestCellResult:
    def test_to_dict(self):
        result = CellResult(
            cell="nba", n_events_total=100, n_chains_pre_gate2=10,
            n_chains_post_gate2=8, retention_rate=0.8, gate2_pass=True,
            mcnemar=None, variance_baseline=None, variance_intervention=None,
            power=None, mde=0.05, errors=[],
        )
        d = result.to_dict()
        assert d["cell"] == "nba"
        assert d["retention_rate"] == 0.8


class TestRunReportSave:
    def test_save_to_disk(self, tmp_path):
        runner = CellRunner(config=_harness_cfg())
        runner.register_cell("nba", MockT(cell="nba"))
        report = runner.run({"nba": [_stream("nba", 20)]})
        path = tmp_path / "report.json"
        report.save(path)
        assert path.exists()
        import json
        with open(path) as f:
            data = json.load(f)
        assert "run_id" in data
        assert "cells" in data
