"""
Tests for the v5 statistical harness.
"""


from v5.src.common.schema import ChainCandidate, GameEvent
from v5.src.harness.actionables import (
    ACTIONABLE_TYPES,
    compute_retention_rate,
    gate2_check,
    is_actionable,
)
from v5.src.harness.mcnemar import aggregate_results, run_mcnemar
from v5.src.harness.scoring import ChainScore, extract_binary_vectors, score_chain
from v5.src.harness.variance import bootstrap_proportion_ci, mcnemar_power


def make_event(event_type: str, cell: str = "nba", seq: int = 0) -> GameEvent:
    return GameEvent(
        timestamp=float(seq),
        event_type=event_type,
        actor="player_1",
        location_context={},
        raw_data_blob={},
        cell=cell,
        game_id="test_game_001",
        sequence_idx=seq,
    )


def make_chain(events, cell="nba", chain_id="test_chain") -> ChainCandidate:
    return ChainCandidate(
        chain_id=chain_id,
        game_id="test_game_001",
        cell=cell,
        events=events,
    )


class TestActionables:
    def test_known_actionable_type(self):
        ev = make_event("engage_decision")
        assert is_actionable(ev)

    def test_non_actionable_type(self):
        ev = make_event("player_idle")
        assert not is_actionable(ev)

    def test_phase_prefix_stripped(self):
        # GameEvent strips phase_ in __post_init__
        ev = make_event("phase_engage_decision")
        assert ev.event_type == "engage_decision"
        assert is_actionable(ev)

    def test_resource_budget_v1_1(self):
        ev = make_event("resource_budget")
        assert "resource_budget" in ACTIONABLE_TYPES
        assert is_actionable(ev)

    def test_gate2_check_pass(self):
        events = [make_event("engage_decision", seq=i) for i in range(5)]
        chain = make_chain(events)
        assert gate2_check(chain, floor=0.50)
        assert chain.is_actionable is True

    def test_gate2_check_fail(self):
        actionable = [make_event("engage_decision", seq=0)]
        non_actionable = [make_event("player_idle", seq=i+1) for i in range(9)]
        chain = make_chain(actionable + non_actionable)
        assert not gate2_check(chain, floor=0.50)
        assert chain.is_actionable is False

    def test_compute_retention_rate(self):
        chains = []
        for i in range(10):
            events = [make_event("engage_decision", seq=j) for j in range(5)]
            chains.append(make_chain(events, chain_id=f"chain_{i}"))
        result = compute_retention_rate(chains, floor=0.50)
        assert result["n_total"] == 10
        assert result["retention_rate"] == 1.0
        assert result["gate2_pass"]


class TestMcNemar:
    def test_basic_mcnemar_significant(self):
        # c >> b: intervention clearly better
        n = 100
        baseline = [True] * 40 + [False] * 60
        intervention = [True] * 70 + [False] * 30
        result = run_mcnemar(baseline, intervention, cell="nba",
                             alpha=0.05, bonferroni_divisor=1,
                             bootstrap_iterations=100)
        assert result.n_chains == n
        assert result.c > result.b  # intervention wins on discordants

    def test_mcnemar_no_discordants(self):
        # Completely concordant — should not be significant
        baseline = [True] * 50 + [False] * 50
        intervention = [True] * 50 + [False] * 50
        result = run_mcnemar(baseline, intervention, cell="fortnite",
                             alpha=0.05, bonferroni_divisor=5,
                             bootstrap_iterations=100)
        assert result.n_discordant == 0
        assert result.p_value == 1.0

    def test_aggregate_results_empty(self):
        agg = aggregate_results([])
        assert agg["n_cells"] == 0
        assert agg["total_chains"] == 0
        assert agg["total_discordant"] == 0
        assert agg["cell_summaries"] == []

    def test_bonferroni_correction_applied(self):
        baseline = [True] * 40 + [False] * 60
        intervention = [True] * 70 + [False] * 30
        r1 = run_mcnemar(baseline, intervention, cell="nba",
                         bonferroni_divisor=1, bootstrap_iterations=100)
        r5 = run_mcnemar(baseline, intervention, cell="nba",
                         bonferroni_divisor=5, bootstrap_iterations=100)
        assert r5.p_value_corrected >= r1.p_value_corrected


class TestScoring:
    def _dummy_chain(self, i=0) -> ChainCandidate:
        return ChainCandidate(
            chain_id=f"chain_{i}",
            game_id="game_001",
            cell="nba",
            events=[make_event("engage_decision")],
        )

    def test_correct(self):
        score = score_chain(self._dummy_chain(), "yes", "yes")
        assert score.correct is True
        assert score.score_label == 1

    def test_incorrect(self):
        score = score_chain(self._dummy_chain(), "yes", "no")
        assert score.correct is False
        assert score.score_label == 0

    def test_abstain(self):
        score = score_chain(self._dummy_chain(), "yes", "abstain")
        assert score.correct is None
        assert score.score_label == -1

    def test_empty_response_abstain(self):
        score = score_chain(self._dummy_chain(), "yes", "")
        assert score.score_label == -1

    def test_extract_binary_vectors_excludes_abstain(self):
        b = [ChainScore("c0", "nba", True, "yes", "yes", 1),
             ChainScore("c1", "nba", None, "abstain", "yes", -1),
             ChainScore("c2", "nba", False, "no", "yes", 0)]
        i = [ChainScore("c0", "nba", True, "yes", "yes", 1),
             ChainScore("c1", "nba", None, "abstain", "yes", -1),
             ChainScore("c2", "nba", True, "yes", "yes", 1)]
        bv, iv = extract_binary_vectors(b, i, exclude_abstain=True)
        assert len(bv) == 2  # c1 excluded
        assert bv == [True, False]
        assert iv == [True, True]


class TestVariance:
    def test_bootstrap_ci_50pct(self):
        correct = [True] * 50 + [False] * 50
        lo, hi = bootstrap_proportion_ci(correct, iterations=1000, seed=42)
        assert 0.38 <= lo <= 0.50
        assert 0.50 <= hi <= 0.62

    def test_bootstrap_ci_empty(self):
        lo, hi = bootstrap_proportion_ci([])
        assert lo == 0.0 and hi == 0.0

    def test_mcnemar_power_positive(self):
        power = mcnemar_power(b=10, c=40, alpha=0.05, bonferroni_divisor=5)
        assert 0.0 < power <= 1.0
