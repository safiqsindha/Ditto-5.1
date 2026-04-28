"""
Tests for the cost estimator.
"""

import pytest
from v5.src.harness.cost_estimator import (
    HAIKU_PRICE_PER_M_INPUT,
    HAIKU_PRICE_PER_M_OUTPUT,
    CostEstimate,
    estimate_cost,
)


class TestEstimateCost:
    def test_default_locked_spec(self):
        """v5 defaults: 1200 chains × 5 cells × 2 calls = 12,000 calls."""
        est = estimate_cost()
        assert est.total_calls == 12_000
        assert est.n_chains_per_cell == 1200
        assert est.n_cells == 5
        assert est.calls_per_chain == 2

    def test_total_cost_matches_token_math(self):
        est = estimate_cost(n_chains_per_cell=100, n_cells=1, calls_per_chain=1,
                            input_tokens_per_call=1000, output_tokens_per_call=100)
        # 100 calls × 1000 input = 100,000 tokens = 0.1M
        # cost = 0.1 × HAIKU_PRICE_PER_M_INPUT
        expected_input = 0.1 * HAIKU_PRICE_PER_M_INPUT
        expected_output = 0.01 * HAIKU_PRICE_PER_M_OUTPUT
        assert est.input_cost_usd == pytest.approx(expected_input)
        assert est.output_cost_usd == pytest.approx(expected_output)
        assert est.total_cost_usd == pytest.approx(expected_input + expected_output)

    def test_per_cell_breakdown(self):
        est = estimate_cost(n_chains_per_cell=1000, n_cells=4, calls_per_chain=2)
        assert est.per_cell_calls == 2000
        assert est.per_cell_cost_usd == pytest.approx(est.total_cost_usd / 4)

    def test_zero_cells(self):
        est = estimate_cost(n_chains_per_cell=1000, n_cells=0)
        assert est.total_calls == 0
        assert est.total_cost_usd == 0.0

    def test_negative_input_raises(self):
        with pytest.raises(ValueError):
            estimate_cost(n_chains_per_cell=-1)

    def test_summary_string_contains_cost(self):
        est = estimate_cost()
        summary = est.summary()
        assert "Total cost:" in summary
        assert "1,200" in summary  # chains per cell
        assert "12,000" in summary  # total calls

    def test_to_dict(self):
        est = estimate_cost()
        d = est.to_dict()
        assert d["total_calls"] == 12_000
        assert "total_cost_usd" in d

    def test_locked_spec_estimate_under_5_dollars(self):
        """Sanity: v5 Phase 1 should cost well under $5."""
        est = estimate_cost()
        assert est.total_cost_usd < 5.0
        assert est.total_cost_usd > 0.5  # but not zero either


class TestCostEstimate:
    def test_dataclass_fields(self):
        est = CostEstimate(
            n_chains_per_cell=1, n_cells=1, calls_per_chain=1,
            input_tokens_per_call=100, output_tokens_per_call=10,
            total_calls=1, total_input_tokens=100, total_output_tokens=10,
            input_cost_usd=0.1, output_cost_usd=0.01, total_cost_usd=0.11,
            per_cell_calls=1, per_cell_cost_usd=0.11,
        )
        assert est.total_calls == 1
        assert est.total_cost_usd == 0.11
