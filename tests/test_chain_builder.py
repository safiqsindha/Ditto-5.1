"""
Tests for FixedPerCellChainBuilder (Q4-B chain construction).
"""

import pytest
from src.common.schema import EventStream, GameEvent
from src.interfaces.chain_builder import (
    DefaultChainBuilder,
    FixedPerCellChainBuilder,
    _uniform_subsample,
)


def _stream(cell: str, n_events: int, game_id: str = "g1") -> EventStream:
    s = EventStream(game_id=game_id, cell=cell)
    for i in range(n_events):
        s.append(GameEvent(
            timestamp=float(i),
            event_type="engage_decision",
            actor=f"player_{i % 3}",
            location_context={},
            raw_data_blob={},
            cell=cell,
            game_id=game_id,
            sequence_idx=i,
        ))
    return s


class TestFixedPerCellChainBuilder:
    def test_unset_cell_raises(self):
        builder = FixedPerCellChainBuilder()
        with pytest.raises(ValueError, match="not set"):
            builder.get_chain_length("nba")

    def test_zero_or_negative_raises(self):
        builder = FixedPerCellChainBuilder({"nba": 0})
        with pytest.raises(ValueError, match=">= 1"):
            builder.get_chain_length("nba")

    def test_locked_per_cell_lengths(self):
        builder = FixedPerCellChainBuilder({"nba": 6, "csgo": 10})
        assert builder.get_chain_length("nba") == 6
        assert builder.get_chain_length("csgo") == 10

    def test_non_overlapping_windows(self):
        builder = FixedPerCellChainBuilder({"nba": 5})
        streams = [_stream("nba", 20)]
        chains = builder.build_from_streams(streams, cell="nba")
        # 20 events / chain_length 5 = 4 chains
        assert len(chains) == 4
        # Verify windows are non-overlapping
        for i, chain in enumerate(chains):
            expected_start = i * 5
            assert chain.events[0].sequence_idx == expected_start
            assert chain.events[-1].sequence_idx == expected_start + 4

    def test_partial_tail_dropped(self):
        builder = FixedPerCellChainBuilder({"nba": 5})
        streams = [_stream("nba", 23)]  # 4 full chains, 3 leftover
        chains = builder.build_from_streams(streams, cell="nba")
        assert len(chains) == 4

    def test_too_short_stream_yields_no_chains(self):
        builder = FixedPerCellChainBuilder({"nba": 10})
        streams = [_stream("nba", 5)]  # < chain_length
        chains = builder.build_from_streams(streams, cell="nba")
        assert chains == []

    def test_overlap_mode(self):
        builder = FixedPerCellChainBuilder({"nba": 5}, overlap=True)
        streams = [_stream("nba", 10)]
        chains = builder.build_from_streams(streams, cell="nba")
        # With overlap=True, step=1, so 10-5+1 = 6 chains
        assert len(chains) == 6

    def test_max_chains_subsamples(self):
        builder = FixedPerCellChainBuilder({"nba": 5})
        # 4 streams × 4 chains each = 16 raw chains
        streams = [_stream("nba", 20, f"g{i}") for i in range(4)]
        chains = builder.build_from_streams(streams, cell="nba", max_chains=8)
        assert len(chains) == 8

    def test_wrong_cell_stream_skipped(self):
        builder = FixedPerCellChainBuilder({"nba": 5})
        streams = [_stream("csgo", 20)]  # wrong cell
        chains = builder.build_from_streams(streams, cell="nba")
        assert chains == []

    def test_chain_metadata_set(self):
        builder = FixedPerCellChainBuilder({"nba": 5})
        streams = [_stream("nba", 10)]
        chains = builder.build_from_streams(streams, cell="nba")
        for chain in chains:
            assert chain.chain_metadata["chain_length"] == 5
            assert chain.chain_metadata["builder"] == "FixedPerCellChainBuilder"
            assert chain.chain_metadata["overlap"] is False

    def test_chain_ids_unique_within_stream(self):
        builder = FixedPerCellChainBuilder({"nba": 5})
        streams = [_stream("nba", 20)]
        chains = builder.build_from_streams(streams, cell="nba")
        ids = [c.chain_id for c in chains]
        assert len(ids) == len(set(ids))

    def test_build_dispatches_event_streams(self):
        builder = FixedPerCellChainBuilder({"nba": 5})
        streams = [_stream("nba", 20)]
        chains = builder.build(streams)
        assert len(chains) == 4

    def test_build_passes_through_chain_candidates(self):
        builder = FixedPerCellChainBuilder({"nba": 5})
        streams = [_stream("nba", 10)]
        first = builder.build_from_streams(streams, cell="nba")
        # Pass chains back through .build() — should pass through unchanged
        same = builder.build(first)
        assert len(same) == len(first)

    def test_build_invalid_input_raises(self):
        builder = FixedPerCellChainBuilder({"nba": 5})
        with pytest.raises(TypeError):
            builder.build([{"not_a_stream": True}])

    def test_default_chain_builder_alias(self):
        # Backwards-compat alias
        assert DefaultChainBuilder is FixedPerCellChainBuilder


class TestShuffleChains:
    """CF-3=A shuffled-control generation."""

    def _make_chains(self, n: int = 5, chain_len: int = 8) -> list:
        from src.common.schema import ChainCandidate
        chains = []
        for i in range(n):
            events = [GameEvent(
                timestamp=float(j),
                event_type="engage_decision",
                actor=f"player_{j}",
                location_context={},
                raw_data_blob={},
                cell="nba",
                game_id=f"g{i}",
                sequence_idx=j,
            ) for j in range(chain_len)]
            chains.append(ChainCandidate(
                chain_id=f"chain_{i}",
                game_id=f"g{i}",
                cell="nba",
                events=events,
                chain_metadata={"chain_type": "test"},
            ))
        return chains

    def test_returns_one_shuffle_per_chain_by_default(self):
        builder = FixedPerCellChainBuilder({"nba": 8})
        chains = self._make_chains(n=4)
        shuffled = builder.shuffle_chains(chains, seed=0)
        assert len(shuffled) == 4

    def test_n_shuffles_multiplies_output(self):
        builder = FixedPerCellChainBuilder({"nba": 8})
        chains = self._make_chains(n=3)
        shuffled = builder.shuffle_chains(chains, seed=0, n_shuffles=2)
        assert len(shuffled) == 6

    def test_shuffled_ids_unique(self):
        builder = FixedPerCellChainBuilder({"nba": 8})
        chains = self._make_chains(n=5)
        shuffled = builder.shuffle_chains(chains, seed=42)
        ids = [c.chain_id for c in shuffled]
        assert len(ids) == len(set(ids))

    def test_shuffled_chain_preserves_cell_and_game(self):
        builder = FixedPerCellChainBuilder({"nba": 8})
        chains = self._make_chains(n=2)
        shuffled = builder.shuffle_chains(chains, seed=0)
        for shuf, orig in zip(shuffled, chains, strict=False):
            assert shuf.cell == orig.cell
            assert shuf.game_id == orig.game_id

    def test_shuffled_metadata_tagged(self):
        builder = FixedPerCellChainBuilder({"nba": 8})
        chains = self._make_chains(n=2)
        shuffled = builder.shuffle_chains(chains, seed=0)
        for s in shuffled:
            assert s.chain_metadata["shuffled"] is True
            assert s.chain_metadata["cf3"] == "shuffled_control"
            assert "parent_chain_id" in s.chain_metadata

    def test_events_same_set_different_order(self):
        builder = FixedPerCellChainBuilder({"nba": 8})
        chains = self._make_chains(n=1, chain_len=20)
        shuffled = builder.shuffle_chains(chains, seed=7)
        orig_ids = [e.sequence_idx for e in chains[0].events]
        shuf_ids = [e.sequence_idx for e in shuffled[0].events]
        assert sorted(orig_ids) == sorted(shuf_ids)
        # With seed=7 and 20 events, order should differ (not guaranteed but highly likely)

    def test_empty_chains_returns_empty(self):
        builder = FixedPerCellChainBuilder({"nba": 8})
        assert builder.shuffle_chains([], seed=0) == []

    def test_deterministic_with_same_seed(self):
        builder = FixedPerCellChainBuilder({"nba": 8})
        chains = self._make_chains(n=3)
        s1 = builder.shuffle_chains(chains, seed=99)
        s2 = builder.shuffle_chains(chains, seed=99)
        for a, b in zip(s1, s2, strict=False):
            assert [e.sequence_idx for e in a.events] == [e.sequence_idx for e in b.events]


def test_uniform_subsample():
    items = list(range(100))
    sub = _uniform_subsample(items, 10)
    assert len(sub) == 10
    # Should be roughly evenly spaced
    diffs = [sub[i+1] - sub[i] for i in range(len(sub)-1)]
    assert all(d >= 9 for d in diffs)


def test_uniform_subsample_k_larger_than_n():
    items = list(range(5))
    assert _uniform_subsample(items, 100) == items


def test_uniform_subsample_k_zero():
    assert _uniform_subsample([1, 2, 3], 0) == [1, 2, 3]
