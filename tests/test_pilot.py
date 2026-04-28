"""
Tests for the v5 pilot validation harness and MockT.
"""


from src.common.schema import EventStream, GameEvent
from src.pilot.mock_t import MockT
from src.pilot.validator import PilotValidator


def _make_stream(cell: str, n_events: int = 50, seed: int = 0) -> EventStream:
    import random
    rng = random.Random(seed)
    actionable_types = [
        "engage_decision", "rotation_commit", "resource_budget",
        "position_commit", "ability_use", "team_coordinate",
    ]
    non_actionable = ["player_idle", "camera_move", "unknown_event"]
    stream = EventStream(game_id=f"test_{cell}_{seed}", cell=cell)
    for i in range(n_events):
        etype = rng.choice(actionable_types if rng.random() > 0.3 else non_actionable)
        stream.append(GameEvent(
            timestamp=float(i),
            event_type=etype,
            actor=f"player_{rng.randint(0,5)}",
            location_context={"x": rng.random() * 100},
            raw_data_blob={},
            cell=cell,
            game_id=stream.game_id,
            sequence_idx=i,
        ))
    return stream


class TestMockT:
    def test_returns_chains(self):
        stream = _make_stream("nba", n_events=50)
        t = MockT(cell="nba", window_size=5, step_size=3)
        chains = t.translate(stream)
        assert len(chains) > 0

    def test_chain_ids_unique(self):
        stream = _make_stream("csgo", n_events=60)
        t = MockT(cell="csgo")
        chains = t.translate(stream)
        ids = [c.chain_id for c in chains]
        assert len(ids) == len(set(ids)), "Duplicate chain IDs"

    def test_empty_stream(self):
        stream = EventStream(game_id="empty", cell="nba")
        t = MockT(cell="nba")
        chains = t.translate(stream)
        assert chains == []

    def test_cell_attribute(self):
        t = MockT(cell="fortnite")
        assert t.cell == "fortnite"

    def test_chain_events_within_stream(self):
        stream = _make_stream("poker", n_events=30)
        t = MockT(cell="poker", window_size=5)
        chains = t.translate(stream)
        for chain in chains:
            assert all(e in stream.events for e in chain.events)


class TestPilotValidator:
    def _make_streams(self, cell: str, n_games: int = 20) -> list:
        return [_make_stream(cell, n_events=60, seed=i) for i in range(n_games)]

    def test_single_cell_pass(self):
        validator = PilotValidator(gate2_floor=0.50)
        validator.register_cell("nba", MockT(cell="nba"))
        streams = {"nba": self._make_streams("nba")}
        report = validator.run(streams)
        assert len(report.cells) == 1
        cell_r = report.cells[0]
        assert cell_r.cell == "nba"
        assert cell_r.n_chains_raw > 0
        assert cell_r.retention_rate >= 0.0

    def test_five_cells(self):
        cells = ["fortnite", "nba", "csgo", "rocket_league", "poker"]
        validator = PilotValidator(gate2_floor=0.50)
        streams = {}
        for c in cells:
            validator.register_cell(c, MockT(cell=c))
            streams[c] = self._make_streams(c, n_games=10)
        report = validator.run(streams)
        assert len(report.cells) == 5
        cell_names = {r.cell for r in report.cells}
        assert cell_names == set(cells)

    def test_default_mock_t_used(self):
        validator = PilotValidator()
        validator.register_cell("csgo")  # No T provided → uses MockT
        streams = {"csgo": self._make_streams("csgo")}
        report = validator.run(streams)
        assert report.cells[0].n_chains_raw > 0

    def test_report_structure(self):
        validator = PilotValidator()
        validator.register_cell("fortnite", MockT(cell="fortnite"))
        streams = {"fortnite": self._make_streams("fortnite", n_games=5)}
        report = validator.run(streams)
        r = report.cells[0]
        assert hasattr(r, "chain_length_mean")
        assert hasattr(r, "event_type_distribution")
        assert hasattr(r, "sample_chain_summaries")
        assert isinstance(r.event_type_distribution, dict)

    def test_fortnite_t_runs_without_errors(self):
        from src.interfaces.translation import FortniteT
        validator = PilotValidator()
        validator.register_cell("fortnite", FortniteT())
        streams = {"fortnite": self._make_streams("fortnite", n_games=3)}
        report = validator.run(streams)
        r = report.cells[0]
        assert r.cell == "fortnite"
        # Real T runs without "not implemented" errors
        assert not any("not implemented" in e.lower() for e in r.errors)
