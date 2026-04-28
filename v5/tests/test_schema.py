"""
Tests for v5 common schema: GameEvent, EventStream, ChainCandidate.
"""

import json

import pytest
from v5.src.common.schema import (
    VALID_CELLS,
    ChainCandidate,
    EventStream,
    GameEvent,
)


def _ev(**overrides) -> GameEvent:
    base = dict(
        timestamp=0.0, event_type="engage_decision", actor="player_1",
        location_context={}, raw_data_blob={}, cell="nba",
        game_id="g1", sequence_idx=0,
    )
    base.update(overrides)
    return GameEvent(**base)


class TestGameEvent:
    def test_valid_construction(self):
        ev = _ev()
        assert ev.event_type == "engage_decision"
        assert ev.cell == "nba"

    def test_phase_prefix_stripped(self):
        ev = _ev(event_type="phase_engage_decision")
        assert ev.event_type == "engage_decision"

    def test_phase_prefix_only_at_start(self):
        ev = _ev(event_type="some_phase_thing")
        assert ev.event_type == "some_phase_thing"

    def test_invalid_cell_rejected(self):
        with pytest.raises(ValueError, match="Unknown cell"):
            _ev(cell="not_a_cell")

    def test_negative_sequence_idx_rejected(self):
        with pytest.raises(ValueError, match="sequence_idx"):
            _ev(sequence_idx=-1)

    def test_to_dict_roundtrip(self):
        ev = _ev(timestamp=12.5, location_context={"x": 1, "y": 2})
        d = ev.to_dict()
        ev2 = GameEvent.from_dict(d)
        assert ev2.timestamp == 12.5
        assert ev2.location_context == {"x": 1, "y": 2}

    def test_to_json_roundtrip(self):
        ev = _ev(timestamp=42.0)
        s = ev.to_json()
        # Parseable JSON
        parsed = json.loads(s)
        assert parsed["timestamp"] == 42.0
        ev2 = GameEvent.from_json(s)
        assert ev2.timestamp == 42.0

    def test_valid_cells_set(self):
        assert VALID_CELLS == frozenset(
            ["fortnite", "nba", "csgo", "rocket_league", "poker"]
        )


class TestEventStream:
    def test_empty_stream(self):
        s = EventStream(game_id="g1", cell="nba")
        assert len(s) == 0
        assert list(s) == []

    def test_append(self):
        s = EventStream(game_id="g1", cell="nba")
        s.append(_ev(sequence_idx=0))
        s.append(_ev(sequence_idx=1, timestamp=1.0))
        assert len(s) == 2

    def test_iteration(self):
        s = EventStream(game_id="g1", cell="csgo")
        for i in range(5):
            s.append(_ev(cell="csgo", sequence_idx=i, timestamp=float(i)))
        timestamps = [e.timestamp for e in s]
        assert timestamps == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_jsonl_roundtrip(self, tmp_path):
        original = EventStream(
            game_id="test_game", cell="poker",
            metadata={"subset": "pluribus", "n_players": 6},
        )
        for i in range(10):
            original.append(_ev(
                cell="poker",
                game_id="test_game",
                sequence_idx=i,
                timestamp=float(i),
                event_type="engage_decision",
                location_context={"street": "preflop"},
            ))

        path = tmp_path / "stream.jsonl"
        original.to_jsonl(path)
        assert path.exists()

        loaded = EventStream.from_jsonl(path)
        assert loaded.game_id == "test_game"
        assert loaded.cell == "poker"
        assert loaded.metadata == {"subset": "pluribus", "n_players": 6}
        assert len(loaded) == 10
        assert all(loaded.events[i].sequence_idx == i for i in range(10))
        assert loaded.events[5].location_context == {"street": "preflop"}


class TestChainCandidate:
    def test_construction(self):
        events = [_ev(sequence_idx=i) for i in range(5)]
        chain = ChainCandidate(
            chain_id="c1", game_id="g1", cell="nba", events=events,
        )
        assert len(chain) == 5
        assert chain.is_actionable is None  # not yet evaluated
        assert chain.scored_correct is None
