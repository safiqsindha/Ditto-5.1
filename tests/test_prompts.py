"""
Tests for the prompt builder + response parser.
"""

import pytest
from src.common.schema import ChainCandidate, GameEvent
from src.harness.prompts import (
    PER_CELL_PROMPT_BUILDERS,
    NBAPromptBuilder,
    PromptPair,
    parse_model_response,
)


def _chain(cell: str = "nba", n: int = 5) -> ChainCandidate:
    events = [
        GameEvent(
            timestamp=float(i),
            event_type="engage_decision",
            actor=f"player_{i}",
            location_context={"x": i},
            raw_data_blob={},
            cell=cell,
            game_id="g1",
            sequence_idx=i,
        )
        for i in range(n)
    ]
    return ChainCandidate(chain_id="c1", game_id="g1", cell=cell, events=events)


class TestPromptBuilder:
    def test_build_returns_prompt_pair(self):
        chain = _chain("nba")
        builder = NBAPromptBuilder()
        pair = builder.build(chain)
        assert isinstance(pair, PromptPair)
        assert pair.cell == "nba"
        assert pair.chain_id == "c1"

    def test_baseline_lacks_constraint_block(self):
        chain = _chain("nba")
        pair = NBAPromptBuilder().build(chain)
        assert "Constraint Context" not in pair.baseline_prompt
        assert "Event Chain" in pair.baseline_prompt
        assert "Question" in pair.baseline_prompt

    def test_intervention_includes_constraint_block(self):
        chain = _chain("nba")
        pair = NBAPromptBuilder().build(chain)
        assert "Constraint Context" in pair.intervention_prompt
        assert "Event Chain" in pair.intervention_prompt
        assert "Question" in pair.intervention_prompt

    def test_metadata_populated(self):
        chain = _chain("nba", n=7)
        pair = NBAPromptBuilder().build(chain)
        assert pair.metadata["n_events"] == 7
        assert pair.metadata["chain_length"] == 7

    def test_wrong_cell_raises(self):
        chain = _chain("csgo")
        with pytest.raises(ValueError, match="cell"):
            NBAPromptBuilder().build(chain)

    def test_event_format_includes_type_and_anonymised_actor(self):
        chain = _chain("nba")
        builder = NBAPromptBuilder()
        from src.harness.prompts import _build_actor_map
        actor_map = _build_actor_map(chain.events)
        line = builder.format_event(chain.events[0], 0, actor_map=actor_map)
        assert "engage_decision" in line
        # CF-4=B: real actor name must NOT appear; anonymised slot must
        assert "player_0" not in line
        assert "Player_0" in line

    def test_format_chain_anonymises_actors(self):
        chain = _chain("nba", n=5)
        builder = NBAPromptBuilder()
        rendered = builder.format_chain(chain)
        # Real actor names must not appear in the rendered chain
        for i in range(5):
            assert f"player_{i}" not in rendered
        # Anonymised slots must appear
        assert "Player_0" in rendered


class TestPerCellBuilders:
    def test_all_five_builders_registered(self):
        # Per A2 (D-35): 5 active builders (pubg replaces fortnite) + fortnite
        # kept as a legacy fixture in the registry.
        expected = {"pubg", "nba", "csgo", "rocket_league", "poker", "fortnite"}
        assert set(PER_CELL_PROMPT_BUILDERS.keys()) == expected

    @pytest.mark.parametrize(
        "cell", ["pubg", "nba", "csgo", "rocket_league", "poker", "fortnite"]
    )
    def test_builder_has_correct_cell(self, cell):
        builder = PER_CELL_PROMPT_BUILDERS[cell]()
        assert builder.cell == cell


class TestResponseParser:
    def test_simple_response(self):
        assert parse_model_response("yes") == "yes"

    def test_strips_whitespace(self):
        assert parse_model_response("  yes  \n") == "yes"

    def test_takes_first_line(self):
        assert parse_model_response("yes\nadditional info") == "yes"

    def test_strips_trailing_punctuation(self):
        assert parse_model_response("yes.") == "yes"
        assert parse_model_response("no!") == "no"

    def test_lowercases(self):
        assert parse_model_response("YES") == "yes"

    def test_empty_returns_empty(self):
        assert parse_model_response("") == ""
        assert parse_model_response("   ") == ""

    def test_abstain_patterns(self):
        for s in ["I don't know", "abstain", "n/a", "unknown",
                  "cannot determine", "no answer", "skip"]:
            assert parse_model_response(s) == "", f"Failed for: {s}"

    def test_json_response_extracts_answer(self):
        assert parse_model_response('{"answer": "yes"}') == "yes"
        assert parse_model_response('{"prediction": "no"}') == "no"
        assert parse_model_response('{"value": "Maybe"}') == "maybe"

    def test_malformed_json_falls_through(self):
        # Not parseable JSON → use first-line parsing
        assert parse_model_response("{not json: yes}") == "{not json: yes}"

    def test_allowed_predictions_exact_match(self):
        assert parse_model_response("yes", allowed_predictions=["yes", "no"]) == "yes"
        assert parse_model_response("no", allowed_predictions=["yes", "no"]) == "no"

    def test_allowed_predictions_substring_no_longer_matches(self):
        # Substring matching was removed in the Phase-D-prep code review (C1)
        # because it could invert answers — e.g. "There's no doubt — yes, this
        # is consistent" contains both "no" and "yes" with "no" appearing
        # first in scan order. Non-conforming responses now abstain (return "")
        # and are excluded from McNemar pairs.
        assert parse_model_response("the answer is yes",
                                     allowed_predictions=["yes", "no"]) == ""

    def test_allowed_predictions_inverted_substring_returns_empty(self):
        # Specifically the inversion failure mode: "no" appears before "yes"
        # in the response text, but the actual answer is yes.
        assert parse_model_response("there's no doubt — yes",
                                     allowed_predictions=["yes", "no"]) == ""

    def test_allowed_predictions_no_match_returns_empty(self):
        assert parse_model_response("maybe",
                                     allowed_predictions=["yes", "no"]) == ""
