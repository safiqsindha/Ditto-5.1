"""
Chain-level scoring for v5.

Scoring converts a model response string to a binary correct/incorrect label
per chain. The model is given a chain and asked to predict/classify some
property; the scoring function compares the response to the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..common.schema import ChainCandidate
from .prompts import parse_model_response


@dataclass
class ChainScore:
    chain_id: str
    cell: str
    correct: bool | None     # None = abstain / unparseable
    model_response: str
    ground_truth: str
    score_label: int            # 1=correct, 0=incorrect, -1=abstain


def score_chain(
    chain: ChainCandidate,
    ground_truth: str,
    model_response: str,
) -> ChainScore:
    """
    Score a single chain.

    The model_response is the raw string returned by the subject model (Haiku).
    ground_truth is the pre-computed correct answer for this chain.

    Correctness is determined by exact match after normalization (lower, strip).
    A response of "abstain", empty string, or unparseable → abstain (-1).
    """
    # F8 fix: route through parse_model_response for richer abstain detection
    # (handles "I don't know", JSON-wrapped answers, etc.)
    response_norm = parse_model_response(model_response)
    truth_norm = ground_truth.strip().lower()

    if not response_norm:
        label = -1
        correct = None
    elif response_norm == truth_norm:
        label = 1
        correct = True
    else:
        label = 0
        correct = False

    chain.scored_correct = correct
    return ChainScore(
        chain_id=chain.chain_id,
        cell=chain.cell,
        correct=correct,
        model_response=model_response,
        ground_truth=ground_truth,
        score_label=label,
    )


def score_batch(
    chains: list[ChainCandidate],
    ground_truths: list[str],
    model_responses: list[str],
) -> list[ChainScore]:
    if not (len(chains) == len(ground_truths) == len(model_responses)):
        raise ValueError("chains, ground_truths, and model_responses must be same length")
    return [
        score_chain(chain, gt, resp)
        for chain, gt, resp in zip(chains, ground_truths, model_responses)
    ]


def extract_binary_vectors(
    baseline_scores: list[ChainScore],
    intervention_scores: list[ChainScore],
    exclude_abstain: bool = True,
) -> tuple[list[bool], list[bool]]:
    """
    Extract paired binary correct/incorrect vectors for McNemar input.
    Optionally excludes pairs where either response is abstain.
    """
    if len(baseline_scores) != len(intervention_scores):
        raise ValueError("baseline and intervention score lists must be same length")

    baseline_vec = []
    intervention_vec = []
    for b, i in zip(baseline_scores, intervention_scores):
        if exclude_abstain and (b.correct is None or i.correct is None):
            continue
        baseline_vec.append(bool(b.correct))
        intervention_vec.append(bool(i.correct))

    return baseline_vec, intervention_vec
