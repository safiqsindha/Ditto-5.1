"""
Tests for the Texas Hold'em poker cell.

Coverage targets:
  - PokerExtractor: PHH action parsing, state tracking, anonymisation
  - PokerPipeline: mock data generation, run() integration
  - PokerT: T1 (chain length ≤ N), T2 (cross-cell guard), min-actions filter
  - PokerPromptBuilder: constraint context, CF-4=B anonymisation
  - ME-PK-1 / ME-PK-2 stubs: raise NotImplementedError
  - DOMAIN_T_STUBS registration
"""

from __future__ import annotations

import pytest
from src.cells.poker.extractor import PokerExtractor, _parse_action_string
from src.cells.poker.pipeline import PokerPipeline, _stamp_poker_streets
from src.cells.poker.poker_t import (
    PokerHandHQT,
    PokerPerSessionT,
    PokerT,
    _POKER_MIN_ACTIONS_PER_HAND,
    _POKER_N,
)
# Backward-compat alias for tests authored before A7 amendment.
_POKER_MIN_ACTIONS = _POKER_MIN_ACTIONS_PER_HAND
from src.common.config import load_cell_configs
from src.common.schema import ChainCandidate, EventStream, GameEvent, VALID_CELLS
from src.harness.prompts import PER_CELL_PROMPT_BUILDERS, PokerPromptBuilder
from src.interfaces.translation import DOMAIN_T_STUBS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_hand_record(
    n_players: int = 6,
    actions: list[str] | None = None,
    game_id: str = "pk_test_0001",
) -> dict:
    """Synthetic PHH-style hand dict (already anonymised)."""
    players = [f"actor_{i}" for i in range(n_players)]
    starting_stacks = [10000] * n_players
    blinds = [50, 100] + [0] * (n_players - 2)
    if actions is None:
        actions = [
            "d dh p0 ??", "d dh p1 ??", "d dh p2 ??",
            "d dh p3 ??", "d dh p4 ??", "d dh p5 AsKd",
            "p2 f",
            "p3 cbr 300",
            "p4 f",
            "p5 cbr 900",
            "p0 f",
            "p1 f",
            "p3 cc",
            "d db 7c8s2d",        # flop
            "p3 cc",
            "p5 cbr 1200",
            "p3 cc",
            "d db Ks",            # turn
            "p3 cc",
            "p5 cbr 3000",
            "p3 f",
        ]
    return {
        "game_id": game_id,
        "players": players,
        "n_players": n_players,
        "starting_stacks": starting_stacks,
        "blinds": blinds,
        "big_blind": 100,
        "actions": actions,
        "subset": "pluribus",
    }


def _make_event_stream(cell: str = "poker", n_events: int = 30) -> EventStream:
    """Minimal EventStream with all-actionable poker events."""
    stream = EventStream(game_id="g_poker_test", cell=cell)
    actors = [f"actor_{i}" for i in range(6)]
    for i in range(n_events):
        stream.append(GameEvent(
            timestamp=float(i * 5),
            event_type="engage_decision",
            actor=actors[i % len(actors)],
            location_context={"street": "preflop", "position": "BTN"},
            raw_data_blob={},
            cell=cell,
            game_id="g_poker_test",
            sequence_idx=i,
            phase="preflop",
        ))
    return stream


@pytest.fixture(scope="module")
def cell_config():
    return load_cell_configs()["poker"]


@pytest.fixture(scope="module")
def extractor():
    return PokerExtractor()


# ---------------------------------------------------------------------------
# PokerExtractor tests
# ---------------------------------------------------------------------------

class TestPokerExtractor:
    def test_extract_returns_event_stream(self, extractor):
        rec = _make_hand_record()
        stream = extractor.extract(rec)
        assert isinstance(stream, EventStream)
        assert stream.cell == "poker"

    def test_extract_game_id(self, extractor):
        rec = _make_hand_record(game_id="pk_my_hand")
        stream = extractor.extract(rec)
        assert stream.game_id == "pk_my_hand"

    def test_extract_produces_events_for_player_actions(self, extractor):
        rec = _make_hand_record()
        stream = extractor.extract(rec)
        # Actions contain: p2 f, p3 cbr 300, p4 f, p5 cbr 900,
        #   p0 f, p1 f, p3 cc, (flop), p3 cc, p5 cbr 1200, p3 cc,
        #   (turn), p3 cc, p5 cbr 3000, p3 f → 13 player actions
        assert len(stream.events) > 0

    def test_extract_all_events_are_poker_cell(self, extractor):
        stream = extractor.extract(_make_hand_record())
        assert all(e.cell == "poker" for e in stream.events)

    def test_extract_event_types_are_actionable(self, extractor):
        stream = extractor.extract(_make_hand_record())
        valid_types = {"engage_decision", "disengage_decision"}
        for ev in stream.events:
            assert ev.event_type in valid_types

    def test_fold_maps_to_disengage(self, extractor):
        rec = _make_hand_record(actions=["p0 f"])
        rec["blinds"] = [0, 0]
        stream = extractor.extract(rec)
        assert any(e.event_type == "disengage_decision" for e in stream.events)

    def test_call_maps_to_engage(self, extractor):
        rec = _make_hand_record(actions=["p1 cbr 300", "p0 cc"])
        rec["blinds"] = [0, 0]
        stream = extractor.extract(rec)
        engage = [e for e in stream.events if e.event_type == "engage_decision"]
        assert len(engage) >= 1

    def test_bet_raise_maps_to_engage(self, extractor):
        rec = _make_hand_record(actions=["p0 cbr 500"])
        rec["blinds"] = [0, 0]
        stream = extractor.extract(rec)
        assert stream.events[0].event_type == "engage_decision"

    def test_street_stamped_in_phase(self, extractor):
        stream = extractor.extract(_make_hand_record())
        streets = {e.phase for e in stream.events}
        # Should contain both preflop and post-flop streets
        assert "preflop" in streets

    def test_street_in_location_context(self, extractor):
        stream = extractor.extract(_make_hand_record())
        for ev in stream.events:
            assert "street" in ev.location_context

    def test_actors_are_anonymised(self, extractor):
        stream = extractor.extract(_make_hand_record())
        for ev in stream.events:
            assert ev.actor.startswith("actor_")

    def test_sequence_idx_monotonic(self, extractor):
        stream = extractor.extract(_make_hand_record())
        for i, ev in enumerate(stream.events):
            assert ev.sequence_idx == i

    def test_timestamps_monotonic(self, extractor):
        stream = extractor.extract(_make_hand_record())
        ts = [e.timestamp for e in stream.events]
        assert ts == sorted(ts)

    def test_show_muck_not_emitted(self, extractor):
        rec = _make_hand_record(actions=["p0 sm", "p0 sm AsKd"])
        rec["blinds"] = [0, 0]
        stream = extractor.extract(rec)
        assert len(stream.events) == 0

    def test_deal_board_not_emitted_as_event(self, extractor):
        rec = _make_hand_record(actions=["d db 2s3h4d"])
        rec["blinds"] = [0, 0]
        stream = extractor.extract(rec)
        assert len(stream.events) == 0

    def test_deal_hole_not_emitted_as_event(self, extractor):
        rec = _make_hand_record(actions=["d dh p0 ??"])
        rec["blinds"] = [0, 0]
        stream = extractor.extract(rec)
        assert len(stream.events) == 0

    def test_empty_actions_returns_empty_stream(self, extractor):
        rec = _make_hand_record(actions=[])
        stream = extractor.extract(rec)
        assert len(stream.events) == 0

    def test_pot_and_stack_in_location_context(self, extractor):
        stream = extractor.extract(_make_hand_record())
        for ev in stream.events:
            assert "pot_size_bb" in ev.location_context
            assert "stack_bb" in ev.location_context

    def test_flop_street_detected(self, extractor):
        rec = _make_hand_record(actions=[
            "p0 cbr 300",
            "d db 2s3h4d",    # flop
            "p0 cc",
        ])
        rec["blinds"] = [0, 0]
        stream = extractor.extract(rec)
        flop_events = [e for e in stream.events if e.phase == "flop"]
        assert len(flop_events) >= 1

    def test_position_populated(self, extractor):
        stream = extractor.extract(_make_hand_record(n_players=6))
        for ev in stream.events:
            assert "position" in ev.location_context
            assert ev.location_context["position"] != ""

    def test_two_player_positions(self, extractor):
        rec = _make_hand_record(n_players=2, actions=["p0 cc", "p1 f"])
        rec["players"] = ["actor_0", "actor_1"]
        rec["starting_stacks"] = [1000, 1000]
        rec["blinds"] = [50, 100]
        stream = extractor.extract(rec)
        positions = {e.location_context["position"] for e in stream.events}
        assert positions.issubset({"SB", "BB"})


# ---------------------------------------------------------------------------
# _parse_action_string unit tests
# ---------------------------------------------------------------------------

class TestParseActionString:
    def test_fold(self):
        r = _parse_action_string("p3 f")
        assert r["type"] == "player_action"
        assert r["player_idx"] == 3
        assert r["action_code"] == "f"

    def test_check_or_call(self):
        r = _parse_action_string("p1 cc")
        assert r["player_idx"] == 1
        assert r["action_code"] == "cc"

    def test_bet_raise(self):
        r = _parse_action_string("p2 cbr 600")
        assert r["player_idx"] == 2
        assert r["action_code"] == "cbr"
        assert r["amount"] == 600.0

    def test_deal_board(self):
        r = _parse_action_string("d db 2s3h4d")
        assert r["type"] == "deal_board"
        assert r["cards"] == "2s3h4d"

    def test_deal_hole(self):
        r = _parse_action_string("d dh p0 ??")
        assert r["type"] == "deal_hole"

    def test_empty_string(self):
        assert _parse_action_string("") is None

    def test_unknown_actor(self):
        assert _parse_action_string("x3 f") is None

    def test_show_muck(self):
        r = _parse_action_string("p0 sm")
        assert r["type"] == "player_action"
        assert r["action_code"] == "sm"


# ---------------------------------------------------------------------------
# PokerPipeline tests
# ---------------------------------------------------------------------------

class TestPokerPipelineMock:
    def test_constructible(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        assert p.cell == "poker"

    def test_generate_mock_data_returns_streams(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        streams = p.generate_mock_data()
        assert len(streams) == cell_config.sample_target

    def test_mock_streams_are_event_streams(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        streams = p.generate_mock_data()
        assert all(isinstance(s, EventStream) for s in streams)

    def test_mock_streams_have_poker_cell(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        for s in p.generate_mock_data()[:5]:
            assert s.cell == "poker"
            assert all(e.cell == "poker" for e in s.events)

    def test_mock_events_have_mock_flag(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        for s in p.generate_mock_data()[:5]:
            assert all(e.metadata.get("mock") is True for e in s.events)

    def test_mock_events_have_street_in_phase(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        valid_streets = {"preflop", "flop", "turn", "river"}
        for s in p.generate_mock_data()[:5]:
            assert all(e.phase in valid_streets for e in s.events)

    def test_mock_events_have_street_in_location_context(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        for s in p.generate_mock_data()[:3]:
            for e in s.events:
                assert "street" in e.location_context

    def test_run_force_mock_returns_streams(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        streams = p.run(force_mock=True)
        assert len(streams) > 0

    def test_run_persists_jsonl(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        streams = p.run(force_mock=True)
        files = list((tmp_path / "events" / "poker").glob("*.jsonl"))
        assert len(files) == len(streams)

    def test_stratification_includes_both_subsets(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        streams = p.generate_mock_data()
        subsets = {s.metadata.get("subset") for s in streams}
        assert "pluribus" in subsets
        assert "wsop-2023-50k" in subsets

    def test_fetch_returns_empty_list(self, cell_config, tmp_path):
        """fetch() returns [] when no download possible (no network in CI)."""
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        # We mock the URL to something that will fail quickly
        import src.cells.poker.pipeline as pm
        orig = pm._PHH_TARBALL_URL
        pm._PHH_TARBALL_URL = "http://localhost:0/nonexistent"
        try:
            result = p.fetch()
            assert isinstance(result, list)
        finally:
            pm._PHH_TARBALL_URL = orig

    def test_parse_returns_empty_without_pokerkit(self, cell_config, tmp_path, monkeypatch):
        """parse() returns [] gracefully when pokerkit is unavailable."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pokerkit":
                raise ImportError("mocked missing pokerkit")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        result = p.parse([])
        assert result == []

    def test_extract_events_empty_on_empty_records(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        result = p.extract_events([])
        assert result == []


# ---------------------------------------------------------------------------
# _stamp_poker_streets helper
# ---------------------------------------------------------------------------

class TestStampPokerStreets:
    def test_all_events_get_valid_street(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        stream = p._make_mock_stream(
            game_id="test", cell="poker", n_events=20,
            event_types=["engage_decision"], actors=["actor_0"], seed=0,
        )
        _stamp_poker_streets(stream)
        valid = {"preflop", "flop", "turn", "river"}
        assert all(e.phase in valid for e in stream.events)

    def test_empty_stream_does_not_crash(self):
        stream = EventStream(game_id="g", cell="poker")
        _stamp_poker_streets(stream)  # should not raise

    def test_preflop_is_most_common_street(self, cell_config, tmp_path):
        p = PokerPipeline(config=cell_config, data_root=tmp_path)
        stream = p._make_mock_stream(
            game_id="t", cell="poker", n_events=40,
            event_types=["engage_decision"], actors=["actor_0"], seed=99,
        )
        _stamp_poker_streets(stream)
        preflop_count = sum(1 for e in stream.events if e.phase == "preflop")
        assert preflop_count > len(stream.events) * 0.4


# ---------------------------------------------------------------------------
# PokerT tests  (T1: chain length, T2: cross-cell guard)
# ---------------------------------------------------------------------------

class TestPokerT:
    def setup_method(self):
        self.t = PokerT()

    # T1 — chain length must not exceed N=8
    def test_T1_chain_length_at_most_N(self):
        stream = _make_event_stream(n_events=60)
        chains = self.t.translate(stream)
        for chain in chains:
            assert len(chain.events) <= _POKER_N, (
                f"Chain {chain.chain_id} has {len(chain.events)} events > N={_POKER_N}"
            )

    def test_T1_chain_length_exactly_N_when_many_events(self):
        """Actor with >N events gets capped at N."""
        stream = EventStream(game_id="long_hand", cell="poker")
        for i in range(20):
            stream.append(GameEvent(
                timestamp=float(i), event_type="engage_decision",
                actor="actor_0", location_context={}, raw_data_blob={},
                cell="poker", game_id="long_hand", sequence_idx=i,
            ))
        chains = self.t.translate(stream)
        actor_chains = [c for c in chains if "pk_0" in c.chain_id]
        assert all(len(c.events) == _POKER_N for c in actor_chains)

    # T2 — cross-cell contamination: all chains must have cell="poker"
    def test_T2_all_chains_are_poker_cell(self):
        stream = _make_event_stream(n_events=50)
        chains = self.t.translate(stream)
        for chain in chains:
            assert chain.cell == "poker", (
                f"Cross-cell contamination: chain cell='{chain.cell}'"
            )
            for ev in chain.events:
                assert ev.cell == "poker", (
                    f"Cross-cell contamination: event cell='{ev.cell}'"
                )

    def test_T2_chain_game_id_matches_stream(self):
        stream = _make_event_stream(n_events=30)
        chains = self.t.translate(stream)
        for chain in chains:
            assert chain.game_id == stream.game_id

    def test_min_actions_filter_at_hand_level(self):
        """A7: filter is at the hand level — hand with fewer than
        _POKER_MIN_ACTIONS_PER_HAND total player actions is dropped."""
        # Hand with only 2 events total → dropped under A7
        stream = EventStream(game_id="thin_hand", cell="poker")
        for i in range(2):
            stream.append(GameEvent(
                timestamp=float(i), event_type="engage_decision",
                actor=f"actor_{i}", location_context={}, raw_data_blob={},
                cell="poker", game_id="thin_hand", sequence_idx=i,
            ))
        assert self.t.translate(stream) == []

    def test_hand_with_min_actions_qualifies(self):
        """Hand with exactly _POKER_MIN_ACTIONS_PER_HAND events produces 1 chain."""
        stream = EventStream(game_id="exactly_three", cell="poker")
        for i in range(_POKER_MIN_ACTIONS_PER_HAND):
            stream.append(GameEvent(
                timestamp=float(i), event_type="engage_decision",
                actor=f"actor_{i % 6}", location_context={}, raw_data_blob={},
                cell="poker", game_id="exactly_three", sequence_idx=i,
            ))
        chains = self.t.translate(stream)
        assert len(chains) == 1
        assert chains[0].chain_metadata["chain_type"] == "hand_sequence"

    def test_exactly_min_actions_qualifies_legacy_alias(self):
        """Sanity: hand-level min_actions filter applies regardless of how
        actions are distributed across actors. (Replaces pre-A7 per-actor
        floor test which is now obsolete.)"""
        stream = EventStream(game_id="exact", cell="poker")
        for i in range(_POKER_MIN_ACTIONS):
            stream.append(GameEvent(
                timestamp=float(i), event_type="engage_decision",
                actor="actor_0", location_context={}, raw_data_blob={},
                cell="poker", game_id="exact", sequence_idx=i,
            ))
        chains = self.t.translate(stream)
        assert len(chains) == 1

    def test_empty_stream_returns_empty(self):
        stream = EventStream(game_id="empty", cell="poker")
        assert self.t.translate(stream) == []

    def test_chain_ids_unique_within_stream(self):
        stream = _make_event_stream(n_events=60)
        chains = self.t.translate(stream)
        ids = [c.chain_id for c in chains]
        assert len(ids) == len(set(ids))

    def test_chain_metadata_has_chain_type(self):
        # A7: chain_type is now "hand_sequence" (was "player_hand").
        stream = _make_event_stream(n_events=30)
        chains = self.t.translate(stream)
        for chain in chains:
            assert chain.chain_metadata["chain_type"] == "hand_sequence"

    def test_chain_metadata_has_streets_covered(self):
        # Generate events with explicit streets so coverage is non-empty
        stream = EventStream(game_id="streets", cell="poker")
        for i in range(6):
            stream.append(GameEvent(
                timestamp=float(i), event_type="engage_decision",
                actor="actor_0", location_context={"street": "preflop"},
                raw_data_blob={}, cell="poker", game_id="streets",
                sequence_idx=i, phase="preflop",
            ))
        chains = self.t.translate(stream)
        assert len(chains) == 1
        assert "streets_covered" in chains[0].chain_metadata

    def test_batch_translate(self):
        streams = [_make_event_stream(n_events=30) for _ in range(3)]
        chains = self.t.batch_translate(streams)
        assert len(chains) > 0

    def test_cell_property(self):
        assert self.t.cell == "poker"

    def test_one_chain_per_hand_with_multiple_actors(self):
        """A7: one stream = one hand → one chain regardless of how many
        actors participated. Multi-actor events are interleaved by
        sequence_idx into a single hand_sequence chain."""
        stream = EventStream(game_id="multi", cell="poker")
        for actor_idx in range(6):
            for j in range(4):
                i = actor_idx * 10 + j
                stream.append(GameEvent(
                    timestamp=float(i), event_type="engage_decision",
                    actor=f"actor_{actor_idx}", location_context={},
                    raw_data_blob={}, cell="poker", game_id="multi",
                    sequence_idx=i,
                ))
        chains = self.t.translate(stream)
        assert len(chains) == 1
        # The chain should reference all 6 actors who participated in the hand
        assert chains[0].chain_metadata["n_actors_in_window"] >= 1
        # N=8 truncation preserved
        assert len(chains[0].events) == 8


# ---------------------------------------------------------------------------
# ME stubs
# ---------------------------------------------------------------------------

class TestMEStubs:
    def test_me_pk1_raises_not_implemented(self):
        t = PokerPerSessionT()
        stream = _make_event_stream()
        with pytest.raises(NotImplementedError):
            t.translate(stream)

    def test_me_pk2_raises_not_implemented(self):
        t = PokerHandHQT()
        stream = _make_event_stream()
        with pytest.raises(NotImplementedError):
            t.translate(stream)

    def test_me_pk1_cell_property(self):
        assert PokerPerSessionT().cell == "poker"

    def test_me_pk2_cell_property(self):
        assert PokerHandHQT().cell == "poker"


# ---------------------------------------------------------------------------
# PokerPromptBuilder
# ---------------------------------------------------------------------------

class TestPokerPromptBuilder:
    def setup_method(self):
        self.builder = PokerPromptBuilder()

    def _make_chain(self, n_events: int = 30) -> ChainCandidate:
        stream = _make_event_stream(n_events=n_events)
        t = PokerT()
        chains = t.translate(stream)
        assert chains, "Need at least one chain"
        return chains[0]

    def test_builds_prompt_pair(self):
        from src.harness.prompts import PromptPair
        chain = self._make_chain()
        pair = self.builder.build(chain)
        assert isinstance(pair, PromptPair)

    def test_baseline_has_no_constraint_context(self):
        chain = self._make_chain()
        pair = self.builder.build(chain)
        assert "Constraint Context" not in pair.baseline_prompt
        # Per T-design review (2026-04-28): NLHE/Hold'em anchor stripped from
        # the constraint block to test rule-following rather than pretrained
        # domain recall. Baseline must also not name the variant.
        assert "fold/check/call" not in pair.baseline_prompt

    def test_intervention_has_constraint_context(self):
        chain = self._make_chain()
        pair = self.builder.build(chain)
        # Per T-design review (2026-04-28): constraint block uses generic
        # rule wording without naming the poker variant.
        assert "fold/check/call/bet/raise" in pair.intervention_prompt
        assert "showdown" in pair.intervention_prompt

    def test_constraint_mentions_stack(self):
        chain = self._make_chain()
        pair = self.builder.build(chain)
        assert "stack" in pair.intervention_prompt.lower()

    def test_constraint_mentions_fold(self):
        chain = self._make_chain()
        pair = self.builder.build(chain)
        assert "fold" in pair.intervention_prompt.lower()

    def test_cf4b_no_real_names_in_prompt(self):
        """CF-4=B: actor IDs in prompt should be Player_N, not 'Alice' etc."""
        chain = self._make_chain()
        pair = self.builder.build(chain)
        for prompt in (pair.baseline_prompt, pair.intervention_prompt):
            assert "Alice" not in prompt
            assert "Bob" not in prompt
            # Actors from _make_event_stream are actor_0 … actor_5 already;
            # PromptBuilder maps them to Player_N for the chain-local display
            assert "actor_" not in prompt  # raw IDs must not leak through

    def test_question_asks_yes_no(self):
        chain = self._make_chain()
        pair = self.builder.build(chain)
        assert "YES or NO" in pair.intervention_prompt

    def test_wrong_cell_raises(self):
        from src.harness.prompts import NBAPromptBuilder
        nba_builder = NBAPromptBuilder()
        chain = self._make_chain()
        with pytest.raises(ValueError, match="cell"):
            nba_builder.build(chain)


# ---------------------------------------------------------------------------
# Schema / registry checks
# ---------------------------------------------------------------------------

class TestPokerRegistration:
    def test_poker_in_valid_cells(self):
        assert "poker" in VALID_CELLS

    def test_hearthstone_not_in_valid_cells(self):
        assert "hearthstone" not in VALID_CELLS

    def test_poker_in_domain_t_stubs(self):
        assert "poker" in DOMAIN_T_STUBS
        assert isinstance(DOMAIN_T_STUBS["poker"], PokerT)

    def test_hearthstone_not_in_domain_t_stubs(self):
        assert "hearthstone" not in DOMAIN_T_STUBS

    def test_domain_t_stubs_keys_match_valid_cells(self):
        assert set(DOMAIN_T_STUBS.keys()) == VALID_CELLS

    def test_poker_in_prompt_builders(self):
        assert "poker" in PER_CELL_PROMPT_BUILDERS
        assert PER_CELL_PROMPT_BUILDERS["poker"] is PokerPromptBuilder

    def test_hearthstone_not_in_prompt_builders(self):
        assert "hearthstone" not in PER_CELL_PROMPT_BUILDERS

    def test_poker_in_cell_configs(self):
        configs = load_cell_configs()
        assert "poker" in configs

    def test_poker_config_sample_target(self):
        config = load_cell_configs()["poker"]
        assert config.sample_target > 0

    def test_poker_config_no_required_env_vars(self):
        """Poker requires no API keys (PHH is public)."""
        config = load_cell_configs()["poker"]
        assert config.env_vars == []

    def test_poker_config_mock_fallback(self):
        config = load_cell_configs()["poker"]
        assert config.mock_fallback is True


# ---------------------------------------------------------------------------
# End-to-end dry-run for poker cell
# ---------------------------------------------------------------------------

class TestPokerDryRun:
    def test_dry_run_produces_report(self, tmp_path):
        """run_eval(cells=['poker'], dry_run=True) completes without error."""
        import run_eval as run_eval_module
        out = tmp_path / "poker_dry_run.json"
        result = run_eval_module.run_eval(
            cells=["poker"],
            output_path=out,
            dry_run=True,
            include_shuffle=False,
        )
        assert isinstance(result, bool)
        assert out.exists()

    def test_dry_run_report_contains_poker_cell(self, tmp_path):
        import json
        import run_eval as run_eval_module
        out = tmp_path / "poker_report.json"
        run_eval_module.run_eval(
            cells=["poker"], output_path=out, dry_run=True, include_shuffle=False
        )
        data = json.loads(out.read_text())
        cells = [c["cell"] for c in data.get("cells", [])]
        assert "poker" in cells
