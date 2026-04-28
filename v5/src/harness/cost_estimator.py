"""
API cost estimator for v5 evaluation runs.

Per Q1 sign-off (D-16): two API calls per chain (baseline + intervention).
Per Q3 sign-off (D-18): n = 1,200 chains per cell, 5 cells.

Pricing (Claude Haiku, as of 2026-04, claude-haiku-4-5-20251001):
  Input:  $0.25 per 1M tokens (= $0.00000025/token)
  Output: $1.25 per 1M tokens (= $0.00000125/token)

A typical chain prompt is ~500 input tokens (chain block + question + constraint),
and outputs are short (~30 tokens for a single-prediction reply).

Use:
    cost = estimate_cost(n_chains_per_cell=1200, n_cells=5, calls_per_chain=2)
    print(cost.summary())
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# Pricing per million tokens, claude-haiku-4-5-20251001
HAIKU_PRICE_PER_M_INPUT = 0.25
HAIKU_PRICE_PER_M_OUTPUT = 1.25

# Empirical defaults from infrastructure-build prompt sizing
DEFAULT_INPUT_TOKENS_PER_PROMPT = 500   # chain block + constraint + question
DEFAULT_OUTPUT_TOKENS_PER_RESPONSE = 30  # single-prediction reply


@dataclass
class CostEstimate:
    n_chains_per_cell: int
    n_cells: int
    calls_per_chain: int
    input_tokens_per_call: int
    output_tokens_per_call: int

    total_calls: int
    total_input_tokens: int
    total_output_tokens: int

    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float

    # Per-cell breakdown
    per_cell_calls: int
    per_cell_cost_usd: float

    def summary(self) -> str:
        return (
            f"v5 evaluation cost estimate\n"
            f"---------------------------\n"
            f"Cells: {self.n_cells}\n"
            f"Chains per cell: {self.n_chains_per_cell:,}\n"
            f"Calls per chain: {self.calls_per_chain} (Q1=B: separate baseline/intervention)\n"
            f"Per-call tokens: in={self.input_tokens_per_call}, out={self.output_tokens_per_call}\n"
            f"\n"
            f"Total API calls:    {self.total_calls:>10,}\n"
            f"Total input tokens: {self.total_input_tokens:>10,}\n"
            f"Total output tokens:{self.total_output_tokens:>10,}\n"
            f"\n"
            f"Input cost:  ${self.input_cost_usd:>8.4f}\n"
            f"Output cost: ${self.output_cost_usd:>8.4f}\n"
            f"Total cost:  ${self.total_cost_usd:>8.4f}\n"
            f"\n"
            f"Per-cell:    {self.per_cell_calls:,} calls @ ${self.per_cell_cost_usd:.4f}\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_cost(
    n_chains_per_cell: int = 1200,
    n_cells: int = 5,
    calls_per_chain: int = 2,
    input_tokens_per_call: int = DEFAULT_INPUT_TOKENS_PER_PROMPT,
    output_tokens_per_call: int = DEFAULT_OUTPUT_TOKENS_PER_RESPONSE,
    input_price_per_m: float = HAIKU_PRICE_PER_M_INPUT,
    output_price_per_m: float = HAIKU_PRICE_PER_M_OUTPUT,
) -> CostEstimate:
    """
    Estimate total cost for a v5 evaluation run.

    Defaults match the locked SPEC: 1,200 chains/cell, 5 cells, 2 calls/chain.
    """
    if n_chains_per_cell < 0 or n_cells < 0 or calls_per_chain < 0:
        raise ValueError("Counts must be non-negative")

    total_calls = n_chains_per_cell * n_cells * calls_per_chain
    total_input_tokens = total_calls * input_tokens_per_call
    total_output_tokens = total_calls * output_tokens_per_call

    input_cost = (total_input_tokens / 1_000_000.0) * input_price_per_m
    output_cost = (total_output_tokens / 1_000_000.0) * output_price_per_m
    total_cost = input_cost + output_cost

    per_cell_calls = n_chains_per_cell * calls_per_chain
    per_cell_cost = total_cost / max(n_cells, 1)

    return CostEstimate(
        n_chains_per_cell=n_chains_per_cell,
        n_cells=n_cells,
        calls_per_chain=calls_per_chain,
        input_tokens_per_call=input_tokens_per_call,
        output_tokens_per_call=output_tokens_per_call,
        total_calls=total_calls,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        total_cost_usd=total_cost,
        per_cell_calls=per_cell_calls,
        per_cell_cost_usd=per_cell_cost,
    )


def main():
    """CLI: python -m v5.src.harness.cost_estimator [args]"""
    import argparse
    parser = argparse.ArgumentParser(description="v5 cost estimator")
    parser.add_argument("--chains", type=int, default=1200,
                        help="Chains per cell (default: 1200, locked Q3)")
    parser.add_argument("--cells", type=int, default=5,
                        help="Number of cells (default: 5)")
    parser.add_argument("--calls", type=int, default=2,
                        help="API calls per chain (default: 2, locked Q1=B)")
    parser.add_argument("--input-tokens", type=int, default=DEFAULT_INPUT_TOKENS_PER_PROMPT)
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS_PER_RESPONSE)
    args = parser.parse_args()

    est = estimate_cost(
        n_chains_per_cell=args.chains,
        n_cells=args.cells,
        calls_per_chain=args.calls,
        input_tokens_per_call=args.input_tokens,
        output_tokens_per_call=args.output_tokens,
    )
    print(est.summary())


if __name__ == "__main__":
    main()
