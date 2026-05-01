"""
v5.1 OpenRouter runner — provider-pinned, synchronous, per-call usage logging.

Key contracts (per SPEC §4.3 and BUILD_PLAN Task 2):
  - provider.order = [pinned_provider] on every request
  - allow_fallbacks = False  — fail loud, never silently re-route
  - usage = {include: True}  — OR returns ground-truth cost in response
  - resolved_provider logged per call; mismatch triggers ProviderDriftError

Concurrency model (updated for full-run scale):
  - The orchestrator spawns one thread per OR model (unchanged).
  - Within each model, evaluate() runs up to `max_concurrent` chains in
    parallel using a bounded semaphore — each chain is still sequential
    (baseline → intervention), preserving causal pairing.
  - All threads sharing the same upstream provider share one TokenBucket
    (passed in by the orchestrator) for cross-model rate control.

Usage records:
  - cost_usd: taken from OR's `usage.cost` field (authoritative)
  - input_tokens / output_tokens: from OR's usage.prompt_tokens / completion_tokens
  - cache_*: not available from OR (OR doesn't expose per-model caching stats)
"""

from __future__ import annotations

import json
import logging
import threading
import time
import random
from pathlib import Path
from typing import TYPE_CHECKING

from .runner_native import (
    V51Result,
    PathologicalModelError,
    _compute_cost,
    _log_usage,
    _parse_response,
    _parse_response_lenient,
    MAX_TOKENS_STANDARD,
    MAX_TOKENS_STANDARD_RETRY,
    MAX_TOKENS_REASONING,
    RATE_LIMIT_SLEEP_S,
    STANDARD_STOP_SEQUENCES,
    PATHOLOGICAL_WINDOW,
    PATHOLOGICAL_THRESHOLD,
    PATHOLOGICAL_MIN_SAMPLES,
)
from collections import deque
from .rate_limiter import TokenBucket

if TYPE_CHECKING:
    from .prompts import PromptPair

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ProviderDriftError(RuntimeError):
    """Raised when the resolved model org doesn't match the requested org."""


class OpenRouterRunner:
    """
    Evaluates prompt pairs via OpenRouter with hard provider pinning.

    Parameters
    ----------
    model_id         : OpenRouter model ID (e.g. "x-ai/grok-4-fast")
    pinned_provider  : Provider name from provider_pinning.json (e.g. "xAI")
    rates            : Fallback pricing per token (used if OR cost is absent)
    is_reasoning     : Whether to use 1024-token cap vs 64-token cap
    drift_action     : "error" raises ProviderDriftError; "warn" logs and continues
    limiter          : Shared TokenBucket for the upstream provider (optional).
                       When supplied, every _call() acquires a token before
                       sending the request, providing cross-model rate control.
    """

    def __init__(
        self,
        model_id: str,
        pinned_provider: str,
        rates: dict[str, float] | None = None,
        is_reasoning: bool = False,
        drift_action: str = "error",
        limiter: TokenBucket | None = None,
    ):
        self.model_id        = model_id
        self.pinned_provider = pinned_provider
        self.rates           = rates or {}
        self.is_reasoning    = is_reasoning
        self.drift_action    = drift_action
        self._limiter        = limiter
        self._client         = None
        self._ledger: dict   = {}
        self._ledger_lock    = threading.Lock()
        self._log_lock       = threading.Lock()
        # Pathological-model tracking: rolling failure-rate guardrail.
        # See runner_native.PATHOLOGICAL_* constants for the design rationale —
        # streak-based counting was statistically broken under chain
        # parallelism (artificial coupling of independent Bernoulli trials).
        # We now keep a sliding deque of the last PATHOLOGICAL_WINDOW outcomes
        # and trigger only when the empty fraction stays above threshold
        # across the full window, after a min-sample warmup.
        self._call_history       = deque(maxlen=PATHOLOGICAL_WINDOW)
        self._empty_streak_lock  = threading.Lock()

    def _ensure_client(self):
        if self._client is None:
            import os
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url=_OPENROUTER_BASE_URL,
            )

    def _max_tokens(self) -> int:
        return MAX_TOKENS_REASONING if self.is_reasoning else MAX_TOKENS_STANDARD

    def _raw_call(self, prompt: str, max_tokens: int,
                  stop: list[str] | None = None) -> dict:
        """
        Single API call with provider-limiter + 429 backoff. No retry-on-empty
        logic — that lives in _call() which orchestrates the two-stage pattern.
        """
        from openai import RateLimitError

        if self._limiter:
            self._limiter.acquire()

        extra_body = {
            "provider": {
                "order": [self.pinned_provider],
                "allow_fallbacks": False,
            },
            "usage": {"include": True},
        }
        # OR-standardised parameter to disable secret-reasoning on models that
        # do CoT by default. Verified 2026-05-01:
        #   qwen-plus default: 19s/1017 tokens → with disabled: 1.1s/1 token
        #   glm-4.7, mimo×2: empty → "YES"
        # Reasoning models opt out (they need the reasoning).
        if not self.is_reasoning:
            extra_body["reasoning"] = {"enabled": False}

        kwargs: dict = {
            "model":      self.model_id,
            "messages":   [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "extra_body": extra_body,
        }
        if stop:
            kwargs["stop"] = stop

        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except RateLimitError:
                if self._limiter:
                    self._limiter.record_429()
                if attempt == max_retries - 1:
                    raise
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    f"[{self.model_id}] 429 rate-limit, "
                    f"retry {attempt+1}/{max_retries} in {wait:.1f}s"
                )
                time.sleep(wait)
                if self._limiter:
                    self._limiter.acquire()

        if self._limiter:
            self._limiter.record_success()

        resolved = getattr(response, "model", "") or ""
        self._check_provider_drift(resolved)

        u = response.usage
        or_cost = getattr(u, "cost", None) if u else None

        choice = response.choices[0] if response.choices else None
        text = (choice.message.content or "") if choice else ""
        finish_reason = getattr(choice, "finish_reason", "") if choice else ""
        input_tokens  = u.prompt_tokens     if u else 0
        output_tokens = u.completion_tokens if u else 0

        if or_cost is not None:
            cost_usd = float(or_cost)
        else:
            cost_usd = _compute_cost(input_tokens, output_tokens, 0, 0,
                                     self.rates, is_batch=False)

        return {
            "text":                  text,
            "finish_reason":         finish_reason,
            "input_tokens":          input_tokens,
            "output_tokens":         output_tokens,
            "cache_read_tokens":     0,
            "cache_creation_tokens": 0,
            "cost_usd":              cost_usd,
            "resolved_provider":     resolved,
        }

    def _call(self, prompt: str) -> dict:
        """
        Two-stage retry policy (per second-AI consultation 2026-05-01):

          Stage 1: cheap pass with stop=["\n"] and tight token cap.
                   ~95% of compliant models finish here.

          Stage 2: if Stage 1 returned empty OR finish_reason=length,
                   retry once with the larger STANDARD_RETRY cap (no stop).
                   Catches "secretly reasoning" models like glm-4.7 that
                   need ~500-1000 tokens to produce the answer.

          Pathological detection: track consecutive empties post-Stage-2.
                   If >= PATHOLOGICAL_EMPTY_STREAK, raise to kill the
                   model thread — bounds runaway cost on broken models.

        Reasoning models (is_reasoning=True) skip Stage 1 entirely and go
        straight to MAX_TOKENS_REASONING — they're designed to think.
        """
        if self.is_reasoning:
            # Reasoning models: single shot at REASONING cap, no stop sequences.
            resp = self._raw_call(prompt, max_tokens=MAX_TOKENS_REASONING)
            resp["needed_retry"] = False
            self._track_empty(resp["text"])
            return resp

        # Stage 1: cheap, tight, with stop sequences
        resp = self._raw_call(
            prompt,
            max_tokens=MAX_TOKENS_STANDARD,
            stop=STANDARD_STOP_SEQUENCES,
        )
        if resp["text"] and resp["finish_reason"] != "length":
            # Stage-1 success — reset streak, return
            resp["needed_retry"] = False
            self._track_empty(resp["text"])
            return resp

        # Stage 2: recovery pass — larger budget, no stop
        logger.info(
            f"[{self.model_id}] Stage-1 empty (finish={resp['finish_reason']}, "
            f"tokens={resp['output_tokens']}); retrying at {MAX_TOKENS_STANDARD_RETRY}"
        )
        resp2 = self._raw_call(prompt, max_tokens=MAX_TOKENS_STANDARD_RETRY)
        # Aggregate cost: charge for both attempts
        resp2["cost_usd"] += resp["cost_usd"]
        resp2["input_tokens"] += resp["input_tokens"]
        resp2["output_tokens"] += resp["output_tokens"]
        resp2["needed_retry"] = True
        self._track_empty(resp2["text"])
        return resp2

    def _track_empty(self, text: str) -> None:
        """
        Rolling failure-rate guardrail. See runner_native.PATHOLOGICAL_* docs
        for why this replaced consecutive-streak counting.
        """
        with self._empty_streak_lock:
            self._call_history.append(not bool(text))
            if len(self._call_history) >= PATHOLOGICAL_MIN_SAMPLES:
                empty_count = sum(self._call_history)
                empty_rate  = empty_count / len(self._call_history)
                if empty_rate >= PATHOLOGICAL_THRESHOLD:
                    raise PathologicalModelError(self.model_id, empty_count)

    def _check_provider_drift(self, resolved: str) -> None:
        if not resolved or "/" not in resolved:
            return
        requested_org = self.model_id.split("/")[0].lower()
        resolved_org  = resolved.split("/")[0].lower()
        if requested_org != resolved_org:
            msg = (
                f"Unexpected model for {self.model_id!r}: "
                f"expected org '{requested_org}', resolved='{resolved}'"
            )
            if self.drift_action == "error":
                raise ProviderDriftError(msg)
            logger.warning(msg)

    def evaluate(
        self,
        prompt_pairs: list["PromptPair"],
        condition: str = "",
        usage_log_path: Path | None = None,
        max_concurrent: int = 8,
    ) -> list[V51Result]:
        """
        Evaluate all pairs, running up to `max_concurrent` chains in parallel.

        Within each chain the calls remain sequential (baseline → intervention)
        to preserve causal pairing. The shared provider limiter controls
        cross-chain and cross-model request rates.
        """
        self._ensure_client()
        out: list[V51Result] = []
        out_lock = threading.Lock()

        def run_pair(pair: "PromptPair") -> None:
            b  = self._call(pair.baseline_prompt)
            time.sleep(RATE_LIMIT_SLEEP_S)
            iv = self._call(pair.intervention_prompt)
            time.sleep(RATE_LIMIT_SLEEP_S)

            b_cost  = b["cost_usd"]
            iv_cost = iv["cost_usd"]

            with self._log_lock:
                _log_usage(
                    usage_log_path, self.model_id, pair.chain_id, pair.cell,
                    condition, "baseline",
                    b["input_tokens"], b["output_tokens"], 0, 0, b_cost, False,
                    extra={"resolved_provider": b["resolved_provider"],
                           "or_model": b["resolved_provider"]},
                )
                _log_usage(
                    usage_log_path, self.model_id, pair.chain_id, pair.cell,
                    condition, "intervention",
                    iv["input_tokens"], iv["output_tokens"], 0, 0, iv_cost, False,
                    extra={"resolved_provider": iv["resolved_provider"],
                           "or_model": iv["resolved_provider"]},
                )

            result = V51Result(
                chain_id=pair.chain_id, cell=pair.cell, condition=condition,
                model_id=self.model_id,
                resolved_provider=b.get("resolved_provider", ""),
                baseline_raw=b["text"], intervention_raw=iv["text"],
                baseline_parsed=_parse_response(b["text"]),
                intervention_parsed=_parse_response(iv["text"]),
                baseline_input_tokens=b["input_tokens"],
                baseline_output_tokens=b["output_tokens"],
                baseline_cache_read_tokens=0,
                baseline_cache_creation_tokens=0,
                baseline_cost_usd=b_cost,
                intervention_input_tokens=iv["input_tokens"],
                intervention_output_tokens=iv["output_tokens"],
                intervention_cache_read_tokens=0,
                intervention_cache_creation_tokens=0,
                intervention_cost_usd=iv_cost,
                baseline_parsed_lenient=_parse_response_lenient(b["text"]),
                intervention_parsed_lenient=_parse_response_lenient(iv["text"]),
                baseline_needed_retry=b.get("needed_retry", False),
                intervention_needed_retry=iv.get("needed_retry", False),
                route="openrouter",
            )

            with self._ledger_lock:
                self._update_ledger(b, b_cost)
                self._update_ledger(iv, iv_cost)

            with out_lock:
                out.append(result)

        sem = threading.Semaphore(max_concurrent)

        def run_pair_bounded(pair: "PromptPair") -> None:
            with sem:
                run_pair(pair)

        threads = [
            threading.Thread(target=run_pair_bounded, args=(pair,), daemon=True)
            for pair in prompt_pairs
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return out

    def _update_ledger(self, usage: dict, cost_usd: float) -> None:
        # Caller must hold self._ledger_lock
        entry = self._ledger.setdefault(self.model_id, {
            "n_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
            "total_cache_read_tokens": 0, "total_cache_creation_tokens": 0,
            "total_cost_usd": 0.0, "max_single_call_cost_usd": 0.0,
            "resolved_providers": [],
        })
        entry["n_calls"]             += 1
        entry["total_input_tokens"]  += usage.get("input_tokens", 0)
        entry["total_output_tokens"] += usage.get("output_tokens", 0)
        entry["total_cost_usd"]       = round(entry["total_cost_usd"] + cost_usd, 8)
        entry["max_single_call_cost_usd"] = max(
            entry["max_single_call_cost_usd"], cost_usd)
        resolved = usage.get("resolved_provider", "")
        if resolved and resolved not in entry["resolved_providers"]:
            entry["resolved_providers"].append(resolved)

    def flush_ledger(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._ledger, indent=2))
