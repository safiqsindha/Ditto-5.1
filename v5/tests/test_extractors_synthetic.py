"""
Synthetic-fixture extractor tests.

Tests the actual parsing logic in each cell's extractor against synthetic
fixtures shaped like real upstream-tool outputs. Catches regressions when
real data formats drift.
"""

import json
from pathlib import Path

from v5.src.cells.csgo.extractor import CSGOExtractor
from v5.src.cells.fortnite.extractor import FortniteExtractor
from v5.src.cells.poker.extractor import PokerExtractor
from v5.src.cells.nba.extractor import NBAExtractor
from v5.src.cells.rocket_league.extractor import RocketLeagueExtractor
from v5.src.common.schema import EventStream

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    with open(FIXTURES / name) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fortnite
# ---------------------------------------------------------------------------

class TestFortniteExtractor:
    def test_parses_synthetic_fixture(self):
        record = _load("fortnite_decomp_sample.json")
        extractor = FortniteExtractor()
        stream = extractor.extract(record)
        assert isinstance(stream, EventStream)
        assert stream.cell == "fortnite"
        assert stream.game_id.startswith("fnc_")
        assert len(stream.events) > 0

    def test_storm_events_become_zone_enter(self):
        record = _load("fortnite_decomp_sample.json")
        stream = FortniteExtractor().extract(record)
        zone_events = [e for e in stream.events if e.event_type == "zone_enter"]
        assert len(zone_events) == 3  # 3 storm phases in fixture

    def test_eliminations_become_engage_decision(self):
        record = _load("fortnite_decomp_sample.json")
        stream = FortniteExtractor().extract(record)
        engage_events = [e for e in stream.events if e.event_type == "engage_decision"]
        assert len(engage_events) >= 2  # 2 eliminations in fixture

    def test_non_actionable_events_filtered(self):
        record = _load("fortnite_decomp_sample.json")
        stream = FortniteExtractor().extract(record)
        # PlayerJump should map to None and be excluded
        for e in stream.events:
            assert e.event_type != "PlayerJump"

    def test_events_sorted_by_timestamp(self):
        record = _load("fortnite_decomp_sample.json")
        stream = FortniteExtractor().extract(record)
        timestamps = [e.timestamp for e in stream.events]
        assert timestamps == sorted(timestamps)

    def test_sequence_idx_monotonic(self):
        record = _load("fortnite_decomp_sample.json")
        stream = FortniteExtractor().extract(record)
        for i, e in enumerate(stream.events):
            assert e.sequence_idx == i

    def test_empty_record_returns_empty_stream(self):
        stream = FortniteExtractor().extract({})
        assert len(stream) == 0


# ---------------------------------------------------------------------------
# NBA — possession-level (per Q6-A)
# ---------------------------------------------------------------------------

class TestNBAExtractor:
    def test_parses_synthetic_fixture(self):
        record = _load("nba_pbp_sample.json")
        stream = NBAExtractor().extract(record)
        assert isinstance(stream, EventStream)
        assert stream.cell == "nba"
        assert stream.game_id.startswith("nba_")
        assert len(stream.events) > 0

    def test_possession_level_grouping(self):
        """Per Q6-A: each possession produces one summary event, not per-play."""
        record = _load("nba_pbp_sample.json")
        stream = NBAExtractor().extract(record)
        # Fixture has multiple possession-ending events:
        # - Made shot (msgtype=1) at row 2
        # - Defensive rebound transition at row 5
        # - Made layup (msgtype=1) at row 6
        # - Turnover (msgtype=5) at row 7
        # - End period (msgtype=13) at row 9
        # Should produce at most ~5 possession events (much fewer than 10 plays)
        assert len(stream.events) < 10  # Less than per-play count
        assert len(stream.events) >= 3

    def test_possession_event_has_terminal_msgtype(self):
        record = _load("nba_pbp_sample.json")
        stream = NBAExtractor().extract(record)
        for ev in stream.events:
            ctx = ev.location_context
            assert "terminal_msgtype" in ctx
            assert "n_plays" in ctx

    def test_no_resultsets_returns_empty(self):
        stream = NBAExtractor().extract({"parameters": {"GameID": "test"}})
        assert len(stream) == 0

    def test_legacy_play_level_parser_still_works(self):
        """_parse_row reserved for ME-3 micro-experiment."""
        record = _load("nba_pbp_sample.json")
        ext = NBAExtractor()
        rs = record["resultSets"][0]
        col_idx = {col: i for i, col in enumerate(rs["headers"])}
        row = rs["rowSet"][2]  # Player B 2-pt jump shot
        ev = ext._parse_row(row, col_idx, "test_game", 0)
        assert ev is not None
        assert ev.event_type == "engage_decision"  # made shot


# ---------------------------------------------------------------------------
# CS:GO
# ---------------------------------------------------------------------------

class TestCSGOExtractor:
    def test_parses_synthetic_fixture(self):
        record = _load("csgo_awpy_sample.json")
        stream = CSGOExtractor().extract(record)
        assert isinstance(stream, EventStream)
        assert stream.cell == "csgo"
        assert "csgo" in stream.game_id

    def test_kills_become_engage_decision(self):
        record = _load("csgo_awpy_sample.json")
        stream = CSGOExtractor().extract(record)
        engage = [e for e in stream.events if e.event_type == "engage_decision"]
        assert len(engage) == 3  # 3 kills in fixture

    def test_grenades_become_ability_use(self):
        record = _load("csgo_awpy_sample.json")
        stream = CSGOExtractor().extract(record)
        ability = [e for e in stream.events if e.event_type == "ability_use"]
        assert len(ability) >= 1  # 1 flashbang

    def test_buy_phase_becomes_resource_budget(self):
        """v1.1 amendment — buy phase is resource_budget."""
        record = _load("csgo_awpy_sample.json")
        stream = CSGOExtractor().extract(record)
        budget = [e for e in stream.events if e.event_type == "resource_budget"]
        assert len(budget) == 2  # 2 rounds in fixture

    def test_bomb_events_normalized(self):
        record = _load("csgo_awpy_sample.json")
        stream = CSGOExtractor().extract(record)
        bomb_events = [e for e in stream.events if "objective" in e.event_type]
        assert len(bomb_events) >= 1  # 1 plant_begin in fixture

    def test_round_phase_set(self):
        record = _load("csgo_awpy_sample.json")
        stream = CSGOExtractor().extract(record)
        round_phases = {e.phase for e in stream.events if e.phase}
        assert any("round_" in p for p in round_phases)

    def test_events_sorted_chronologically(self):
        record = _load("csgo_awpy_sample.json")
        stream = CSGOExtractor().extract(record)
        timestamps = [e.timestamp for e in stream.events]
        assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# Rocket League — boost-enriched (per Q7-C)
# ---------------------------------------------------------------------------

class TestRLExtractor:
    def test_parses_synthetic_fixture(self):
        record = _load("rl_carball_sample.json")
        stream = RocketLeagueExtractor().extract(record)
        assert isinstance(stream, EventStream)
        assert stream.cell == "rocket_league"

    def test_hits_extracted(self):
        record = _load("rl_carball_sample.json")
        stream = RocketLeagueExtractor().extract(record)
        # 3 hits in fixture: shot, save, goal
        engage = [e for e in stream.events if e.event_type == "engage_decision"]
        save = [e for e in stream.events if e.event_type == "disengage_decision"]
        goal = [e for e in stream.events if e.event_type == "objective_capture"]
        assert len(engage) >= 1
        assert len(save) >= 1
        assert len(goal) >= 1

    def test_boost_events_extracted(self):
        record = _load("rl_carball_sample.json")
        stream = RocketLeagueExtractor().extract(record)
        boost_gain = [e for e in stream.events if e.event_type == "resource_gain"]
        boost_spend = [e for e in stream.events if e.event_type == "resource_spend"]
        assert len(boost_gain) >= 2  # 2 pickups in fixture
        assert len(boost_spend) >= 1  # 1 use in fixture

    def test_low_boost_transitions_become_resource_budget(self):
        """Q7-C boost-enriched: low-boost transition → resource_budget."""
        record = _load("rl_carball_sample.json")
        stream = RocketLeagueExtractor().extract(record)
        budget = [e for e in stream.events if e.event_type == "resource_budget"]
        # Both players cross the LOW_BOOST_THRESHOLD=25 in their boost_history.
        # Player blue_0: 100 → 80 → 30 → 20 (crosses 25 going from 30 to 20)
        # Player orange_0: 50 → 62 → 35 → 15 (crosses going from 35 to 15)
        assert len(budget) >= 1

    def test_streams_chronologically_sorted(self):
        record = _load("rl_carball_sample.json")
        stream = RocketLeagueExtractor().extract(record)
        timestamps = [e.timestamp for e in stream.events]
        assert timestamps == sorted(timestamps)

    def test_unknown_format_warns_but_returns_empty(self):
        stream = RocketLeagueExtractor().extract({"unknown_format": True})
        assert len(stream) == 0


# ---------------------------------------------------------------------------
# Poker — synthetic PHH-style hand dict
# ---------------------------------------------------------------------------

def _mock_poker_record(actions: list[str] | None = None, game_id: str = "pk_test_syn") -> dict:
    if actions is None:
        actions = [
            "d dh p0 ??", "d dh p1 ??", "d dh p2 ??",
            "p2 f",
            "p0 cbr 300",
            "p1 cc",
            "d db 2s3h4d",   # flop
            "p0 cbr 600",
            "p1 f",
        ]
    return {
        "game_id": game_id,
        "players": ["actor_0", "actor_1", "actor_2"],
        "n_players": 3,
        "starting_stacks": [10000, 10000, 10000],
        "blinds": [50, 100, 0],
        "big_blind": 100,
        "actions": actions,
        "subset": "pluribus",
    }


class TestPokerExtractorSynthetic:
    def test_parses_simple_hand(self):
        stream = PokerExtractor().extract(_mock_poker_record())
        assert isinstance(stream, EventStream)
        assert stream.cell == "poker"

    def test_player_actions_produce_events(self):
        stream = PokerExtractor().extract(_mock_poker_record())
        # p2 f, p0 cbr 300, p1 cc, (flop), p0 cbr 600, p1 f → 5 events
        assert len(stream.events) == 5

    def test_fold_is_disengage_decision(self):
        stream = PokerExtractor().extract(_mock_poker_record())
        folds = [e for e in stream.events if e.location_context.get("action") == "f"]
        assert all(e.event_type == "disengage_decision" for e in folds)

    def test_raise_is_engage_decision(self):
        stream = PokerExtractor().extract(_mock_poker_record())
        raises = [e for e in stream.events if e.location_context.get("action") == "cbr"]
        assert all(e.event_type == "engage_decision" for e in raises)

    def test_street_changes_to_flop_after_deal_board(self):
        stream = PokerExtractor().extract(_mock_poker_record())
        flop_events = [e for e in stream.events if e.phase == "flop"]
        assert len(flop_events) >= 1

    def test_actors_match_anonymised_names(self):
        stream = PokerExtractor().extract(_mock_poker_record())
        for e in stream.events:
            assert e.actor in {"actor_0", "actor_1", "actor_2"}

    def test_empty_actions_returns_empty_stream(self):
        record = _mock_poker_record(actions=[])
        stream = PokerExtractor().extract(record)
        assert len(stream) == 0
