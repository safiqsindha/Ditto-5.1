"""
ModelEvaluator for v5 Phase D — calls Claude Haiku on baseline + intervention prompts.

Per Q1 sign-off (D-16): two separate API calls per chain.
Model: claude-haiku-4-5-20251001.
dry_run=True returns deterministic mock responses without any network calls.

Usage
-----
evaluator = ModelEvaluator(dry_run=True)          # tests / dry run
evaluator = ModelEvaluator()                       # real evaluation

baseline, intervention = evaluator.evaluate_pairs(prompt_pairs)
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

from .prompts import PromptPair, parse_model_response

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
MAX_OUTPUT_TOKENS = 32          # YES/NO answer is always ≤ 5 tokens
RATE_LIMIT_SLEEP_S = 0.05       # courtesy delay between calls


@dataclass
class EvaluationResult:
    """Raw + parsed responses for one chain's baseline + intervention pair."""
    chain_id: str
    cell: str
    baseline_raw: str
    intervention_raw: str
    baseline_parsed: str
    intervention_parsed: str


class ModelEvaluator:
    """
    Calls Claude Haiku for baseline and intervention prompts.

    Parameters
    ----------
    model      : Anthropic model ID (default: HAIKU_MODEL)
    dry_run    : If True, skip real API calls and return deterministic mock responses.
                 Useful for integration tests and Phase D dry runs without spend.
    allowed_predictions : Passed to parse_model_response; typically ["yes", "no"].
    rate_limit_sleep    : Seconds to sleep between API calls (real mode only).
    """

    def __init__(
        self,
        model: str = HAIKU_MODEL,
        dry_run: bool = False,
        allowed_predictions: list[str] | None = None,
        rate_limit_sleep: float = RATE_LIMIT_SLEEP_S,
    ):
        self.model = model
        self.dry_run = dry_run
        self.allowed_predictions = allowed_predictions or ["yes", "no"]
        self.rate_limit_sleep = rate_limit_sleep
        self._client = None  # lazy-init to avoid import cost when dry_run=True

    # --- Public API ---------------------------------------------------------

    def evaluate_pairs(
        self,
        prompt_pairs: list[PromptPair],
    ) -> tuple[list[EvaluationResult], list[str], list[str]]:
        """
        Call the model for each PromptPair.

        Returns
        -------
        results           : list of EvaluationResult (raw + parsed, per chain)
        baseline_parsed   : list[str] — normalized prediction strings for baseline
        intervention_parsed: list[str] — normalized prediction strings for intervention
        """
        results: list[EvaluationResult] = []
        for i, pair in enumerate(prompt_pairs):
            if self.dry_run:
                b_raw = self._mock_response(pair, "baseline")
                i_raw = self._mock_response(pair, "intervention")
            else:
                b_raw = self._call_api(pair.baseline_prompt)
                time.sleep(self.rate_limit_sleep)
                i_raw = self._call_api(pair.intervention_prompt)
                time.sleep(self.rate_limit_sleep)

            b_parsed = parse_model_response(b_raw, self.allowed_predictions)
            i_parsed = parse_model_response(i_raw, self.allowed_predictions)

            results.append(EvaluationResult(
                chain_id=pair.chain_id,
                cell=pair.cell,
                baseline_raw=b_raw,
                intervention_raw=i_raw,
                baseline_parsed=b_parsed,
                intervention_parsed=i_parsed,
            ))

            if (i + 1) % 100 == 0:
                logger.info(f"[{pair.cell}] Evaluated {i+1}/{len(prompt_pairs)} chains")

        baseline_parsed = [r.baseline_parsed for r in results]
        intervention_parsed = [r.intervention_parsed for r in results]
        return results, baseline_parsed, intervention_parsed

    # --- Private helpers ----------------------------------------------------

    def _call_api(self, prompt: str) -> str:
        """Make a single Anthropic API call. Raises on network or API error."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError as e:
                raise ImportError(
                    "anthropic package required for real evaluation. "
                    "Install: pip install anthropic"
                ) from e

        message = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    @staticmethod
    def _mock_response(pair: PromptPair, variant: str) -> str:
        """
        Deterministic mock response for dry-run mode.

        Baseline: tends toward "NO" (model without constraint context is uncertain).
        Intervention: tends toward "YES" (model with constraint context is confident).

        Uses a stable hashlib digest (NOT Python's built-in hash(), which is
        randomized per-process via PYTHONHASHSEED) so dry-run results are
        reproducible across processes, machines, and CI runs. Without this
        fix, two dry-runs of the same input yield different McNemar stats —
        breaking research reproducibility for any cached/recorded baseline.
        """
        digest = hashlib.sha256(f"{pair.chain_id}_{variant}".encode("utf-8")).digest()
        # Take a single byte and reduce mod 10 → uniform-ish 0..9
        h = digest[0] % 10
        if variant == "baseline":
            return "YES" if h < 4 else "NO"   # 40% YES baseline
        else:
            return "YES" if h < 8 else "NO"   # 80% YES intervention
