"""
ModelEvaluator for v5 Phase D — calls Claude Haiku on baseline + intervention prompts.

Per Q1 sign-off (D-16): two separate API calls per chain.
Model: claude-haiku-4-5-20251001.

Three execution modes:
  dry_run=True       deterministic mock responses, no network calls
  use_batch=False    sequential messages.create calls (default)
  use_batch=True     Anthropic Batches API — 50% input/output discount,
                     24h SLA, single submission for all calls

Usage
-----
evaluator = ModelEvaluator(dry_run=True)              # tests / dry run
evaluator = ModelEvaluator()                          # real, sequential
evaluator = ModelEvaluator(use_batch=True)            # real, batched (50% off)

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
RATE_LIMIT_SLEEP_S = 0.05       # courtesy delay between sequential calls
BATCH_POLL_INTERVAL_S = 30      # poll batch status every 30s
# Anthropic Batch API supports up to 100K requests per batch. We chunk for
# pragmatic memory/observability reasons even though most cells fit in one.
MAX_REQUESTS_PER_BATCH = 50_000


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
        use_batch: bool = False,
        batch_poll_interval_s: float = BATCH_POLL_INTERVAL_S,
    ):
        self.model = model
        self.dry_run = dry_run
        self.allowed_predictions = allowed_predictions or ["yes", "no"]
        self.rate_limit_sleep = rate_limit_sleep
        self.use_batch = use_batch
        self.batch_poll_interval_s = batch_poll_interval_s
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
        if self.dry_run:
            return self._evaluate_dry_run(prompt_pairs)
        if self.use_batch:
            return self._evaluate_batch(prompt_pairs)
        return self._evaluate_sequential(prompt_pairs)

    # --- Sequential path (default real-mode) -------------------------------

    def _evaluate_sequential(
        self, prompt_pairs: list[PromptPair]
    ) -> tuple[list[EvaluationResult], list[str], list[str]]:
        results: list[EvaluationResult] = []
        for i, pair in enumerate(prompt_pairs):
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

    # --- Dry-run path -------------------------------------------------------

    def _evaluate_dry_run(
        self, prompt_pairs: list[PromptPair]
    ) -> tuple[list[EvaluationResult], list[str], list[str]]:
        results: list[EvaluationResult] = []
        for i, pair in enumerate(prompt_pairs):
            b_raw = self._mock_response(pair, "baseline")
            i_raw = self._mock_response(pair, "intervention")
            b_parsed = parse_model_response(b_raw, self.allowed_predictions)
            i_parsed = parse_model_response(i_raw, self.allowed_predictions)
            results.append(EvaluationResult(
                chain_id=pair.chain_id, cell=pair.cell,
                baseline_raw=b_raw, intervention_raw=i_raw,
                baseline_parsed=b_parsed, intervention_parsed=i_parsed,
            ))
            if (i + 1) % 100 == 0:
                logger.info(f"[{pair.cell}] Evaluated {i+1}/{len(prompt_pairs)} chains")
        return (
            results,
            [r.baseline_parsed for r in results],
            [r.intervention_parsed for r in results],
        )

    # --- Batch path (Anthropic Batches API, 50% off) -----------------------

    def _evaluate_batch(
        self, prompt_pairs: list[PromptPair]
    ) -> tuple[list[EvaluationResult], list[str], list[str]]:
        """
        Submit all calls as a single batch (or a small number of chunks if
        the volume exceeds MAX_REQUESTS_PER_BATCH). Each chain produces two
        requests (baseline + intervention), tagged by custom_id so we can
        re-pair after polling.

        Cost: 50% discount on input + output tokens vs sequential.
        Latency: 24h SLA but typically completes in minutes for our volumes.
        """
        self._ensure_client()
        cell = prompt_pairs[0].cell if prompt_pairs else "?"

        # Build requests: 2 per chain, custom_id encodes positional index +
        # chain_id + variant.
        #
        # The positional index is critical: at n=1200 some cells produce
        # PromptPair lists with duplicate chain_ids (e.g., poker hands sharing
        # a chain_id namespace after deduplication shifts; or any future bug
        # that lets two pairs share a chain_id). The Anthropic Batches API
        # rejects the entire batch with a 400 if ANY two custom_ids collide.
        # Prefixing with the zero-padded position guarantees uniqueness within
        # the request list while still being parseable on the way back.
        requests = []
        for idx, pair in enumerate(prompt_pairs):
            requests.append(self._make_batch_request(
                custom_id=f"{idx:06d}__{pair.chain_id}__baseline",
                prompt=pair.baseline_prompt,
            ))
            requests.append(self._make_batch_request(
                custom_id=f"{idx:06d}__{pair.chain_id}__intervention",
                prompt=pair.intervention_prompt,
            ))

        n_unique_chain_ids = len({p.chain_id for p in prompt_pairs})
        logger.info(
            f"[{cell}] Batch mode: {len(prompt_pairs)} chains "
            f"({n_unique_chain_ids} unique chain_ids) -> {len(requests)} requests"
        )
        if n_unique_chain_ids < len(prompt_pairs):
            logger.warning(
                f"[{cell}] {len(prompt_pairs) - n_unique_chain_ids} duplicate "
                f"chain_ids detected; positional prefix will be used to keep "
                f"custom_ids unique"
            )

        # Submit and collect results across chunks.
        results_by_custom_id: dict[str, str] = {}
        for chunk_start in range(0, len(requests), MAX_REQUESTS_PER_BATCH):
            chunk = requests[chunk_start:chunk_start + MAX_REQUESTS_PER_BATCH]
            chunk_results = self._submit_and_wait_batch(chunk, cell=cell)
            results_by_custom_id.update(chunk_results)

        # Re-pair by (positional index, chain_id) and parse.
        results: list[EvaluationResult] = []
        missing: list[str] = []
        for idx, pair in enumerate(prompt_pairs):
            b_raw = results_by_custom_id.get(
                f"{idx:06d}__{pair.chain_id}__baseline", ""
            )
            i_raw = results_by_custom_id.get(
                f"{idx:06d}__{pair.chain_id}__intervention", ""
            )
            if not b_raw and not i_raw:
                missing.append(pair.chain_id)
            results.append(EvaluationResult(
                chain_id=pair.chain_id, cell=pair.cell,
                baseline_raw=b_raw, intervention_raw=i_raw,
                baseline_parsed=parse_model_response(b_raw, self.allowed_predictions),
                intervention_parsed=parse_model_response(i_raw, self.allowed_predictions),
            ))

        if missing:
            logger.warning(
                f"[{cell}] {len(missing)} chains had no batch result for either "
                f"variant; will be scored as abstain. First few: {missing[:3]}"
            )

        return (
            results,
            [r.baseline_parsed for r in results],
            [r.intervention_parsed for r in results],
        )

    def _make_batch_request(self, custom_id: str, prompt: str) -> dict:
        """Build one request dict for the Batches API."""
        return {
            "custom_id": custom_id,
            "params": {
                "model": self.model,
                "max_tokens": MAX_OUTPUT_TOKENS,
                "messages": [{"role": "user", "content": prompt}],
            },
        }

    def _submit_and_wait_batch(
        self, requests: list[dict], cell: str
    ) -> dict[str, str]:
        """Submit one chunk and poll until it ends. Returns custom_id → text."""
        batch = self._client.messages.batches.create(requests=requests)
        batch_id = batch.id
        logger.info(
            f"[{cell}] Submitted batch {batch_id} ({len(requests)} requests); "
            f"polling every {self.batch_poll_interval_s}s"
        )

        while True:
            # Anthropic create→retrieve has eventual consistency. The first
            # retrieve immediately after create occasionally 404s — retry a
            # few times with backoff before giving up.
            batch = None
            last_err = None
            for attempt in range(5):
                try:
                    batch = self._client.messages.batches.retrieve(batch_id)
                    break
                except Exception as e:
                    last_err = e
                    # 404 NotFoundError from anthropic SDK is the typical case.
                    if "not_found" in str(e).lower() or "404" in str(e):
                        time.sleep(2 ** attempt)  # 1, 2, 4, 8, 16
                        continue
                    raise
            if batch is None:
                raise RuntimeError(
                    f"[{cell}] Could not retrieve batch {batch_id} after retries: {last_err}"
                )
            counts = batch.request_counts
            logger.info(
                f"[{cell}] batch {batch_id[:12]} status={batch.processing_status} "
                f"succeeded={counts.succeeded} errored={counts.errored} "
                f"processing={counts.processing} canceled={counts.canceled} "
                f"expired={counts.expired}"
            )
            if batch.processing_status == "ended":
                break
            time.sleep(self.batch_poll_interval_s)

        out: dict[str, str] = {}
        for result in self._client.messages.batches.results(batch_id):
            text = self._extract_batch_text(result)
            if text is not None:
                out[result.custom_id] = text
            else:
                logger.warning(
                    f"[{cell}] batch result {result.custom_id} type={result.result.type}"
                )
        return out

    @staticmethod
    def _extract_batch_text(result) -> str | None:
        """Extract the text content from a MessageBatchIndividualResponse."""
        r = result.result
        if r.type != "succeeded":
            return None
        message = r.message
        if not message.content:
            return ""
        # First content block, defensive on type
        block = message.content[0]
        return getattr(block, "text", "") or ""

    def _ensure_client(self) -> None:
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError as e:
                raise ImportError(
                    "anthropic package required for real evaluation. "
                    "Install: pip install anthropic"
                ) from e

    # --- Private helpers ----------------------------------------------------

    def _call_api(self, prompt: str) -> str:
        """Make a single Anthropic API call. Raises on network or API error."""
        self._ensure_client()
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
