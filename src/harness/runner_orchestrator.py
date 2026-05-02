"""
v5.1 runner orchestrator — coordinates all provider runners across conditions.

Responsibilities:
  1. Dispatch each model to the correct runner (Anthropic / OpenAI / Google /
     DeepSeek / OpenRouter)
  2. Iterate over all 8 conditions × all cells × all chain pairs
  3. Update the cumulative cost ledger after every batch of calls
  4. Enforce the 110% kill-switch per model
  5. Detect provider drift (OpenRouter only) and halt the run
  6. Handle DeepSeek off-peak scheduling (park until window opens)

The orchestrator does NOT build prompts — it accepts pre-built PromptPair
lists indexed by (condition, cell) and dispatches them to runners.

Usage
-----
    config = OrchestratorConfig.from_files(
        provider_pinning_path=Path("provider_pinning.json"),
        env_path=Path(".env"),
    )
    orch = RunnerOrchestrator(config)
    results = orch.run(
        prompt_pairs_by_condition_cell=pairs,   # dict[(condition, cell), list[PromptPair]]
        models=["claude-haiku-4-5-20251001", "x-ai/grok-4-fast", ...],
        output_dir=Path("results/v5_1/"),
        is_smoke=True,          # False for full run
    )
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .runner_native import (
    AnthropicRunner,
    DeepSeekPeakHoursError,
    DeepSeekRunner,
    GoogleRunner,
    MoonshotRunner,
    OpenAIRunner,
    PathologicalModelError,
    V51Result,
    is_deepseek_off_peak,
    seconds_until_off_peak,
)
from .runner_openrouter import OpenRouterRunner, ProviderDriftError
from .runner_native import atomic_write_text
from .rate_limiter import TokenBucket

if TYPE_CHECKING:
    from .prompts import PromptPair

logger = logging.getLogger(__name__)

# Per-model cost kill-switch: 110% of mean allocation.
# Mean allocation = $189 / 22 = $8.59 per model → kill at $9.45.
# Updated 2026-05-01: 32 → 22 models. Drops:
#   - opus, gpt-5.4, gpt-5.5, grok-4         (cost)
#   - minimax-m2.5, gemini-2.5-pro           (mandatory-reasoning, can't disable)
#   - deepseek-v3.2, v4, r1                  (deprecated; API only serves v4-pro/v4-flash)
#   - llama-4-maverick                       (verbose-preface, no format toggle)
#   - moonshotai/kimi-k2-thinking            (Novita ignores max_tokens; 18K out)
# Net additions: deepseek-v4-flash.
_KILL_SWITCH_RATIO = 1.10
_TOTAL_BUDGET_USD  = 189.0
_N_MODELS          = 22
_MEAN_ALLOCATION   = _TOTAL_BUDGET_USD / _N_MODELS


# ---------------------------------------------------------------------------
# Model registry — maps model string → (runner class, constructor kwargs)
# ---------------------------------------------------------------------------

# Anthropic pricing (USD / token)
_ANTHROPIC_RATES = {
    "claude-haiku-4-5-20251001": {
        "input": 0.80e-6, "output": 4.00e-6,
        "cache_write": 1.00e-6, "cache_read": 0.08e-6,
        "batch_input": 0.40e-6, "batch_output": 2.00e-6,
    },
    "claude-sonnet-4-5": {
        "input": 3.00e-6, "output": 15.00e-6,
        "cache_write": 3.75e-6, "cache_read": 0.30e-6,
        "batch_input": 1.50e-6, "batch_output": 7.50e-6,
    },
    "claude-sonnet-4-6": {
        "input": 3.00e-6, "output": 15.00e-6,
        "cache_write": 3.75e-6, "cache_read": 0.30e-6,
        "batch_input": 1.50e-6, "batch_output": 7.50e-6,
    },
    "claude-opus-4-7": {
        "input": 15.00e-6, "output": 75.00e-6,
        "cache_write": 18.75e-6, "cache_read": 1.50e-6,
        "batch_input": 7.50e-6, "batch_output": 37.50e-6,
    },
}

# OpenAI pricing (USD / token)
_OPENAI_RATES = {
    # Placeholder rates — update with actual GPT-5 pricing when published
    "gpt-5-mini":   {"input": 0.40e-6, "output": 1.60e-6, "batch_input": 0.20e-6, "batch_output": 0.80e-6},
    "gpt-5":        {"input": 2.50e-6, "output": 10.00e-6, "batch_input": 1.25e-6, "batch_output": 5.00e-6},
    "gpt-5.4-mini": {"input": 1.10e-6, "output": 4.40e-6, "batch_input": 0.55e-6, "batch_output": 2.20e-6},
    "gpt-5.4":      {"input": 5.00e-6, "output": 15.00e-6, "batch_input": 2.50e-6, "batch_output": 7.50e-6},
    "gpt-5.5":      {"input": 10.00e-6, "output": 30.00e-6, "batch_input": 5.00e-6, "batch_output": 15.00e-6},
}
_OPENAI_REASONING = {"gpt-5.4-mini"}  # gpt-5.4, gpt-5.5 dropped to fit budget

# Google pricing (USD / token)
_GOOGLE_RATES = {
    # Placeholder rates — update with actual pricing
    "gemini-3.1-flash-lite-preview": {"input": 0.075e-6, "output": 0.30e-6},
    "gemini-2.5-flash":              {"input": 0.075e-6, "output": 0.30e-6},
    "gemini-2.5-pro":                {"input": 1.25e-6,  "output": 5.00e-6},
    "gemini-3-flash-preview":        {"input": 0.15e-6,  "output": 0.60e-6},
    "gemini-3.1-pro-preview":        {"input": 2.50e-6,  "output": 10.00e-6},
}

# DeepSeek pricing (USD / token, off-peak rates).
# 2026-05-01: API consolidated to v4-pro + v4-flash. Old IDs deprecated.
_DEEPSEEK_RATES = {
    "deepseek-v4-pro":   {"input": 0.07e-6, "output": 0.28e-6},
    "deepseek-v4-flash": {"input": 0.04e-6, "output": 0.16e-6},
}
_DEEPSEEK_REASONING = set()  # No explicit reasoning model anymore (r1 deprecated)
_DEEPSEEK_THINKING_OFF = {"deepseek-v4-pro", "deepseek-v4-flash"}  # thinking off

# OpenRouter models from provider_pinning.json (x-ai/grok-4 dropped to fit budget;
# moonshotai/kimi-k2.6 moved to direct Moonshot — see _MOONSHOT_RATES below)
_OR_MODEL_IDS = {
    "x-ai/grok-4-fast", "x-ai/grok-4.20",
    "meta-llama/llama-3.3-70b-instruct",
    # meta-llama/llama-4-maverick dropped (verbose preface, no format toggle)
    "xiaomi/mimo-v2-pro", "xiaomi/mimo-v2.5-pro",
    "qwen/qwen3.6-plus", "qwen/qwen3.6-max-preview",
    "z-ai/glm-4.7", "z-ai/glm-5",
    # moonshotai/kimi-k2-thinking dropped — Novita doesn't honor max_tokens,
    # 18K-token outputs bust the per-model budget. See ALL_MODELS comment.
    # minimax/minimax-m2.5 dropped — mandatory reasoning incompatible with
    # our prompt regime. See ALL_MODELS comment.
}
_OR_REASONING = {"x-ai/grok-4.20"}

# Direct Moonshot AI models (Kimi). Routed away from OpenRouter so we can
# pass the `thinking` extra_body parameter — kimi-k2.6 returns empty content
# without `{"type": "disabled"}`. See MoonshotRunner docstring for evidence.
# Pricing per https://platform.kimi.ai/docs/pricing/chat-k26 (sync).
_MOONSHOT_RATES = {
    "kimi-k2.6": {
        "input":      0.95e-6,   # cache-miss
        "output":     4.00e-6,
        "cache_read": 0.16e-6,   # cache-hit input rate
        "batch_input":  0.57e-6, # batch sync = 60% of standard
        "batch_output": 2.40e-6,
        "cache_read_batch": 0.10e-6,
    },
}
_MOONSHOT_MODEL_IDS = set(_MOONSHOT_RATES.keys())

# OR models known to route through DeepInfra (aggressive rate limiter).
# kimi-k2.6 moved to direct Moonshot (MoonshotRunner); minimax-m2.5 dropped
# (mandatory-reasoning model, see run_v5_1_replication.py). llama-4-maverick
# is also DeepInfra but doesn't need throttling — DeepInfra natively allows
# 200 concurrent per model (verified 2026-05-01).
_OR_DEEPINFRA = set()

# Stagger delay between OR thread starts (seconds) — eliminates synchronized bursts
_OR_THREAD_STAGGER_S = 1.5

# Default token-bucket rates per provider class
_PROVIDER_RATE_DEFAULTS = {
    "deepinfra": (2.0,  4.0),   # (rate/s, burst) — DeepInfra throttles aggressively
    "default":   (5.0, 10.0),   # all other providers
}

# Chain-parallel concurrency caps per model category.
# Bumped 2026-05-01 after smoke showed reasoning models are the wall-time
# bottleneck (kimi-k2-thinking on Novita = 30s/call × 12,800 calls).
# 4 → 16 cuts full-run reasoning wall time from ~27 hrs → ~7 hrs.
# Token bucket prevents bursts; if any provider 429s, adaptive backoff trims.
_MAX_CONCURRENT_REASONING = 16  # was 4
_MAX_CONCURRENT_DEEPINFRA = 2   # DeepInfra models: rate-limited
_MAX_CONCURRENT_STANDARD  = 16  # was 8 — chain parallelism + provider limiter is sufficient


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class KillSwitchTriggered(RuntimeError):
    model_id: str
    actual_cost: float
    threshold: float


class RunnerOrchestrator:
    """
    Coordinates v5.1 evaluation across all 32 models.

    Parameters
    ----------
    provider_pinning  : Loaded provider_pinning.json dict
    output_dir        : Base directory for results + ledger files
    per_model_budget  : Kill-switch threshold per model (USD). Default: 110%
                        of mean allocation ($5.91 → $6.50).
    """

    def __init__(
        self,
        provider_pinning: dict,
        output_dir: Path,
        per_model_budget: float | None = None,
        dry_run: bool = False,
    ):
        self.provider_pinning = provider_pinning
        self.output_dir       = output_dir
        self.dry_run          = dry_run
        self.per_model_budget = (
            per_model_budget
            if per_model_budget is not None
            else _MEAN_ALLOCATION * _KILL_SWITCH_RATIO
        )
        self._ledger: dict  = {}       # model_id → cumulative cost/token totals
        self._ledger_lock   = threading.Lock()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "usage_logs").mkdir(parents=True, exist_ok=True)
        (output_dir / "ledgers").mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(
        self,
        prompt_pairs_by_condition_cell: dict[tuple[str, str], list["PromptPair"]],
        models: list[str],
        is_smoke: bool = True,
    ) -> dict[str, list[V51Result]]:
        """
        Evaluate all (model, condition, cell) combinations.

        Returns
        -------
        results_by_model : dict mapping model_id → flat list of V51Result
                           (across all conditions and cells)
        """
        results_by_model: dict[str, list[V51Result]] = {}

        # Split models by routing
        anthropic_models = [m for m in models if m.startswith("claude-")]
        openai_models    = [m for m in models if m in _OPENAI_RATES]
        google_models    = [m for m in models if m in _GOOGLE_RATES]
        deepseek_models  = [m for m in models if m in _DEEPSEEK_RATES]
        moonshot_models  = [m for m in models if m in _MOONSHOT_MODEL_IDS]
        or_models        = [m for m in models if m in _OR_MODEL_IDS]

        # Phase A: Native batch APIs (or sync for smoke) — run in threads
        native_threads = []
        native_results: dict[str, list[V51Result]] = {}

        for model_id in anthropic_models:
            t = threading.Thread(
                target=self._run_anthropic,
                args=(model_id, prompt_pairs_by_condition_cell,
                      is_smoke, native_results),
                daemon=True,
            )
            native_threads.append(t)

        for model_id in openai_models:
            t = threading.Thread(
                target=self._run_openai,
                args=(model_id, prompt_pairs_by_condition_cell,
                      is_smoke, native_results),
                daemon=True,
            )
            native_threads.append(t)

        for model_id in google_models:
            t = threading.Thread(
                target=self._run_google,
                args=(model_id, prompt_pairs_by_condition_cell,
                      is_smoke, native_results),
                daemon=True,
            )
            native_threads.append(t)

        # Phase A.5: Moonshot direct (Kimi). Sync chain-parallel — same shape
        # as OpenRouter runners, but uses MoonshotRunner for the `thinking`
        # extra_body parameter that OR strips.
        moonshot_results: dict[str, list[V51Result]] = {}
        for model_id in moonshot_models:
            t = threading.Thread(
                target=self._run_moonshot,
                args=(model_id, prompt_pairs_by_condition_cell, moonshot_results),
                daemon=True,
            )
            native_threads.append(t)

        for t in native_threads:
            t.start()

        # Phase B: DeepSeek — each thread handles its own off-peak wait internally.
        # In smoke mode, skip DeepSeek if currently in peak hours rather than waiting
        # 8+ hours for the window; full run always waits.
        deepseek_results: dict[str, list[V51Result]] = {}
        deepseek_threads = []
        for model_id in deepseek_models:
            t = threading.Thread(
                target=self._run_deepseek,
                args=(model_id, prompt_pairs_by_condition_cell,
                      deepseek_results, is_smoke),
                daemon=True,
            )
            deepseek_threads.append(t)

        # Phase C: OpenRouter — one thread per model, staggered starts.
        # OR enforces rate limits per API key (not per upstream provider),
        # so we build one shared TokenBucket per `key_env`. Models on the
        # same key share a bucket; models on different keys (e.g. via the
        # 5/4 split across OPENROUTER_API_KEY and OPENROUTER_API_KEY_2)
        # get independent buckets — doubling aggregate OR throughput.
        # The DeepInfra-tier rate still applies if any model on a key
        # routes through DeepInfra.
        or_results: dict[str, list[V51Result]] = {}
        key_limiters: dict[str, TokenBucket] = {}
        or_thread_args: list[tuple] = []

        for model_id in or_models:
            pin = self.provider_pinning.get(model_id, {})
            provider = pin.get("pinned_provider", "unknown")
            key_env = pin.get("key_env", "OPENROUTER_API_KEY")

            if key_env not in key_limiters:
                # If any model on this key routes through DeepInfra, use the
                # tighter DeepInfra rate; otherwise default. Computed once
                # per key on first encounter.
                key_models = [
                    m for m in or_models
                    if self.provider_pinning.get(m, {}).get(
                        "key_env", "OPENROUTER_API_KEY"
                    ) == key_env
                ]
                key_has_deepinfra = any(
                    self.provider_pinning.get(m, {})
                        .get("pinned_provider", "").lower() == "deepinfra"
                    for m in key_models
                )
                if key_has_deepinfra:
                    rate, burst = _PROVIDER_RATE_DEFAULTS["deepinfra"]
                else:
                    rate, burst = _PROVIDER_RATE_DEFAULTS["default"]
                key_limiters[key_env] = TokenBucket(rate=rate, burst=burst)

            limiter = key_limiters[key_env]

            if model_id in _OR_DEEPINFRA:
                max_concurrent = _MAX_CONCURRENT_DEEPINFRA
            elif model_id in _OR_REASONING:
                max_concurrent = _MAX_CONCURRENT_REASONING
            else:
                max_concurrent = _MAX_CONCURRENT_STANDARD

            or_thread_args.append((model_id, limiter, max_concurrent))

        or_threads = []
        for model_id, limiter, max_concurrent in or_thread_args:
            t = threading.Thread(
                target=self._run_openrouter,
                args=(model_id, prompt_pairs_by_condition_cell,
                      or_results, limiter, max_concurrent),
                daemon=True,
            )
            or_threads.append(t)

        # Start B (DeepSeek) first, then stagger OR thread launches to
        # prevent a synchronized burst of requests at t=0.
        for t in deepseek_threads:
            t.start()

        for i, t in enumerate(or_threads):
            t.start()
            if i < len(or_threads) - 1:
                time.sleep(_OR_THREAD_STAGGER_S)

        # Wait for all threads
        for t in native_threads + deepseek_threads + or_threads:
            t.join()

        results_by_model.update(native_results)
        results_by_model.update(moonshot_results)
        results_by_model.update(deepseek_results)
        results_by_model.update(or_results)

        self._flush_ledger()
        return results_by_model

    # -----------------------------------------------------------------------
    # Per-provider run helpers (called in threads)
    # -----------------------------------------------------------------------

    def _parallel_batches(self, model_id, pairs_dict, runner_factory, log_path):
        """
        Submit one batch per (condition, cell) IN PARALLEL using fresh runner
        instances per thread, then merge results + ledgers.

        Wall-time win: turns 64 sequential batch submissions per model
        (~12 hrs for Anthropic) into 64 concurrent submissions (~12 min total).

        Each thread creates its own runner instance to avoid races on the
        shared `runner._ledger` dict — much simpler than locking that struct.
        Kill-switch is checked once after all threads finish (acceptable for
        batch pricing where per-call cost is bounded and predictable).
        """
        results_by_cond: dict[tuple, list[V51Result]] = {}
        runners_by_cond: dict[tuple, object] = {}
        threads: list[threading.Thread] = []

        def _run_one(condition, cell, pairs):
            runner = runner_factory()
            try:
                res = runner.evaluate(pairs, condition=condition,
                                      usage_log_path=log_path)
                results_by_cond[(condition, cell)] = res
            except Exception as e:
                logger.error(
                    f"[{model_id}] {condition}/{cell} batch failed: {e}",
                    exc_info=True,
                )
                results_by_cond[(condition, cell)] = []
            runners_by_cond[(condition, cell)] = runner

        for (condition, cell), pairs in pairs_dict.items():
            t = threading.Thread(
                target=_run_one, args=(condition, cell, pairs), daemon=True
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Merge results in original (condition, cell) iteration order
        all_results = []
        for key in pairs_dict.keys():
            all_results.extend(results_by_cond.get(key, []))

        # Merge per-thread ledger fragments into one canonical model ledger
        merged: dict = {}
        for runner in runners_by_cond.values():
            for mid, entry in runner._ledger.items():
                dest = merged.setdefault(mid, {
                    "n_calls": 0, "total_input_tokens": 0,
                    "total_output_tokens": 0, "total_cache_read_tokens": 0,
                    "total_cache_creation_tokens": 0, "total_cost_usd": 0.0,
                    "max_single_call_cost_usd": 0.0,
                    "n_yes": 0, "n_no": 0, "n_abstain": 0, "n_stage2_retries": 0,
                })
                dest["n_calls"]                  += entry.get("n_calls", 0)
                dest["total_input_tokens"]       += entry.get("total_input_tokens", 0)
                dest["total_output_tokens"]      += entry.get("total_output_tokens", 0)
                dest["total_cache_read_tokens"]  += entry.get("total_cache_read_tokens", 0)
                dest["total_cache_creation_tokens"] += entry.get("total_cache_creation_tokens", 0)
                dest["total_cost_usd"] = round(
                    dest["total_cost_usd"] + entry.get("total_cost_usd", 0.0), 8)
                dest["max_single_call_cost_usd"] = max(
                    dest["max_single_call_cost_usd"],
                    entry.get("max_single_call_cost_usd", 0.0))
                dest["n_yes"]            += entry.get("n_yes", 0)
                dest["n_no"]             += entry.get("n_no", 0)
                dest["n_abstain"]        += entry.get("n_abstain", 0)
                dest["n_stage2_retries"] += entry.get("n_stage2_retries", 0)

        # Post-hoc kill-switch (informational; batch can't be cancelled mid-flight)
        try:
            self._check_kill_switch(model_id, merged)
        except KillSwitchTriggered as e:
            logger.error(
                f"[{model_id}] POST-HOC kill-switch triggered "
                f"(actual=${e.actual_cost:.2f} > threshold=${e.threshold:.2f}); "
                f"results retained, model excluded from future runs."
            )

        # Flush the merged ledger as one canonical file
        ledger_path = self._ledger_path(model_id)
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(ledger_path, json.dumps(merged, indent=2))
        with self._ledger_lock:
            self._merge_ledger(merged)

        return all_results

    def _run_anthropic(self, model_id, pairs_dict, is_smoke, out_dict):
        # Mega-batch: all (condition, cell) submitted in parallel.
        # 64 sequential batches → 64 concurrent batches per model.
        log_path = self._usage_log_path(model_id)
        out_dict[model_id] = self._parallel_batches(
            model_id=model_id,
            pairs_dict=pairs_dict,
            runner_factory=lambda: AnthropicRunner(
                model_id=model_id, use_batch=True, dry_run=self.dry_run,
            ),
            log_path=log_path,
        )

    def _run_openai(self, model_id, pairs_dict, is_smoke, out_dict):
        rates = _OPENAI_RATES.get(model_id, {"input": 5e-6, "output": 15e-6})
        log_path = self._usage_log_path(model_id)
        # Smoke uses sync (fast iteration, no batch long-tail problem).
        # Full run uses batch (50% cost savings; 24h SLA acceptable).
        # Verified 2026-05-01: smoke #4 had a single gpt-5 batch stuck in
        # in_progress for 37+ min, blocking the entire run. With sync +
        # reasoning_effort=minimal, gpt-5 takes ~2s/call → 160 calls finish
        # in ~20s vs batch's variable 5-40 min tail.
        use_batch = not is_smoke
        out_dict[model_id] = self._parallel_batches(
            model_id=model_id,
            pairs_dict=pairs_dict,
            runner_factory=lambda: OpenAIRunner(
                model_id=model_id, rates=rates, use_batch=use_batch,
                is_reasoning=(model_id in _OPENAI_REASONING),
            ),
            log_path=log_path,
        )

    def _run_google(self, model_id, pairs_dict, is_smoke, out_dict):
        rates = _GOOGLE_RATES.get(model_id, {"input": 1e-6, "output": 4e-6})
        log_path = self._usage_log_path(model_id)
        out_dict[model_id] = self._parallel_batches(
            model_id=model_id,
            pairs_dict=pairs_dict,
            runner_factory=lambda: GoogleRunner(
                model_id=model_id, rates=rates, use_batch=True,
            ),
            log_path=log_path,
        )

    def _run_deepseek(self, model_id, pairs_dict, out_dict, is_smoke=False):
        import os as _os
        # DEEPSEEK_FORCE_PEAK=1 lets us run DeepSeek smoke tests outside the
        # 16:30–00:30 UTC discount window. Used for pre-launch validation;
        # never set in production runs (paying full price defeats the point).
        force_peak = _os.environ.get("DEEPSEEK_FORCE_PEAK") == "1"
        if is_smoke and not is_deepseek_off_peak() and not force_peak:
            wait_min = seconds_until_off_peak() / 60
            logger.warning(
                f"[{model_id}] Skipping DeepSeek in smoke mode — peak hours "
                f"({wait_min:.0f} min until off-peak). Re-run during UTC 16:30–00:30 "
                f"or set DEEPSEEK_FORCE_PEAK=1 to bypass (charges full price)."
            )
            out_dict[model_id] = []
            return
        if force_peak:
            logger.warning(
                f"[{model_id}] DEEPSEEK_FORCE_PEAK=1 set — running outside "
                f"discount window at full price. Smoke/validation only."
            )

        rates = _DEEPSEEK_RATES.get(model_id, {"input": 0.14e-6, "output": 0.55e-6})
        runner = DeepSeekRunner(
            model_id=model_id,
            rates=rates,
            is_reasoning=(model_id in _DEEPSEEK_REASONING),
            thinking_mode=(model_id not in _DEEPSEEK_THINKING_OFF),
            enforce_off_peak=not force_peak,
        )
        log_path = self._usage_log_path(model_id)
        all_results = []
        for (condition, cell), pairs in pairs_dict.items():
            try:
                res = runner.evaluate(pairs, condition=condition,
                                      usage_log_path=log_path)
                self._check_kill_switch(model_id, runner._ledger)
                all_results.extend(res)
            except KillSwitchTriggered:
                logger.error(f"[{model_id}] Kill-switch triggered; halting model.")
                break
            except DeepSeekPeakHoursError as e:
                logger.warning(f"[{model_id}] {e}  Waiting for window...")
                wait_s = seconds_until_off_peak()
                logger.info(f"[{model_id}] Sleeping {wait_s/60:.1f} min for off-peak window.")
                time.sleep(wait_s + 10)   # +10s buffer
                # Retry this condition after sleep
                try:
                    res = runner.evaluate(pairs, condition=condition,
                                          usage_log_path=log_path)
                    self._check_kill_switch(model_id, runner._ledger)
                    all_results.extend(res)
                except Exception as e2:
                    logger.error(f"[{model_id}] Retry after off-peak wait failed: {e2}")
                    break
        runner.flush_ledger(self._ledger_path(model_id))
        with self._ledger_lock:
            self._merge_ledger(runner._ledger)
        out_dict[model_id] = all_results

    def _run_moonshot(self, model_id: str, pairs_dict: dict, out_dict: dict) -> None:
        """Direct Moonshot AI runner — chain-parallel sync with two-stage retry.

        kimi-k2.6 specifically requires `extra_body={"thinking":{"type":"disabled"}}`
        which OR strips. MoonshotRunner sets this by default (thinking_disabled=True).
        """
        rates = _MOONSHOT_RATES.get(model_id, {"input": 1e-6, "output": 4e-6})
        runner = MoonshotRunner(
            model_id=model_id,
            rates=rates,
            is_reasoning=False,
            thinking_disabled=True,
        )
        log_path = self._usage_log_path(model_id)
        all_results = []
        try:
            for (condition, cell), pairs in pairs_dict.items():
                try:
                    res = runner.evaluate(
                        pairs, condition=condition,
                        usage_log_path=log_path,
                        max_concurrent=_MAX_CONCURRENT_STANDARD,
                    )
                    self._check_kill_switch(model_id, runner._ledger)
                    all_results.extend(res)
                except KillSwitchTriggered:
                    logger.error(f"[{model_id}] Kill-switch triggered; halting model.")
                    break
                except PathologicalModelError as e:
                    logger.error(
                        f"[{model_id}] PATHOLOGICAL MODEL — {e.streak} consecutive "
                        f"empties post-retry; terminating model thread."
                    )
                    break
        except Exception as e:
            logger.error(
                f"[{model_id}] Unhandled error in Moonshot thread: {e}", exc_info=True
            )
        finally:
            try:
                runner.flush_ledger(self._ledger_path(model_id))
            except Exception as flush_err:
                logger.warning(f"[{model_id}] Ledger flush failed: {flush_err}")
            with self._ledger_lock:
                self._merge_ledger(runner._ledger)
            out_dict[model_id] = all_results

    def _run_openrouter(
        self,
        model_id: str,
        pairs_dict: dict,
        out_dict: dict,
        limiter: "TokenBucket | None" = None,
        max_concurrent: int = 8,
    ) -> None:
        pin = self.provider_pinning.get(model_id, {})
        pinned_provider = pin.get("pinned_provider", "")
        key_env = pin.get("key_env", "OPENROUTER_API_KEY")
        in_price  = pin.get("input_price_per_m", 1.0)
        out_price = pin.get("output_price_per_m", 4.0)
        rates = {"input": in_price / 1_000_000, "output": out_price / 1_000_000}

        runner = OpenRouterRunner(
            model_id=model_id,
            pinned_provider=pinned_provider,
            rates=rates,
            is_reasoning=(model_id in _OR_REASONING),
            drift_action="error",
            limiter=limiter,
            key_env=key_env,
        )
        log_path = self._usage_log_path(model_id)
        all_results = []
        try:
            for (condition, cell), pairs in pairs_dict.items():
                try:
                    res = runner.evaluate(
                        pairs, condition=condition,
                        usage_log_path=log_path,
                        max_concurrent=max_concurrent,
                    )
                    self._check_kill_switch(model_id, runner._ledger)
                    all_results.extend(res)
                except KillSwitchTriggered:
                    logger.error(f"[{model_id}] Kill-switch triggered; halting model.")
                    break
                except ProviderDriftError as e:
                    logger.error(f"[{model_id}] PROVIDER DRIFT HALT: {e}")
                    break
                except PathologicalModelError as e:
                    logger.error(
                        f"[{model_id}] PATHOLOGICAL MODEL — {e.streak} consecutive "
                        f"empties post-retry; terminating model thread."
                    )
                    break
        except Exception as e:
            logger.error(
                f"[{model_id}] Unhandled error in OR thread: {e}", exc_info=True
            )
        finally:
            try:
                runner.flush_ledger(self._ledger_path(model_id))
            except Exception as flush_err:
                logger.warning(f"[{model_id}] Ledger flush failed: {flush_err}")
            with self._ledger_lock:
                self._merge_ledger(runner._ledger)
            out_dict[model_id] = all_results

    # -----------------------------------------------------------------------
    # Kill-switch
    # -----------------------------------------------------------------------

    def _check_kill_switch(self, model_id: str, runner_ledger: dict) -> None:
        entry = runner_ledger.get(model_id, {})
        cumulative = entry.get("total_cost_usd", 0.0)
        if cumulative > self.per_model_budget:
            raise KillSwitchTriggered(
                model_id=model_id,
                actual_cost=cumulative,
                threshold=self.per_model_budget,
            )

    # -----------------------------------------------------------------------
    # Ledger helpers
    # -----------------------------------------------------------------------

    def _merge_ledger(self, runner_ledger: dict) -> None:
        for model_id, entry in runner_ledger.items():
            dest = self._ledger.setdefault(model_id, {
                "n_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
                "total_cache_read_tokens": 0, "total_cache_creation_tokens": 0,
                "total_cost_usd": 0.0, "max_single_call_cost_usd": 0.0,
                "n_yes": 0, "n_no": 0, "n_abstain": 0, "n_stage2_retries": 0,
            })
            # If dest was created by an earlier flush before counters existed
            # (or by a runner that doesn't track them), backfill the keys so
            # the += ops below don't KeyError.
            for k in ("n_yes", "n_no", "n_abstain", "n_stage2_retries"):
                dest.setdefault(k, 0)
            dest["n_calls"]              += entry.get("n_calls", 0)
            dest["total_input_tokens"]   += entry.get("total_input_tokens", 0)
            dest["total_output_tokens"]  += entry.get("total_output_tokens", 0)
            dest["total_cache_read_tokens"]  += entry.get("total_cache_read_tokens", 0)
            dest["total_cache_creation_tokens"] += entry.get("total_cache_creation_tokens", 0)
            dest["total_cost_usd"] = round(
                dest["total_cost_usd"] + entry.get("total_cost_usd", 0.0), 8)
            dest["max_single_call_cost_usd"] = max(
                dest["max_single_call_cost_usd"],
                entry.get("max_single_call_cost_usd", 0.0))
            dest["n_yes"]             += entry.get("n_yes", 0)
            dest["n_no"]              += entry.get("n_no", 0)
            dest["n_abstain"]         += entry.get("n_abstain", 0)
            dest["n_stage2_retries"]  += entry.get("n_stage2_retries", 0)

    def _flush_ledger(self) -> None:
        path = self.output_dir / "cost_ledger.json"
        with self._ledger_lock:
            ledger_snapshot = dict(self._ledger)
        atomic_write_text(path, json.dumps(ledger_snapshot, indent=2))
        total = sum(e.get("total_cost_usd", 0) for e in ledger_snapshot.values())
        logger.info(f"Cost ledger flushed to {path} (total=${total:.4f})")

    def _usage_log_path(self, model_id: str) -> Path:
        safe = model_id.replace("/", "_").replace(":", "_")
        return self.output_dir / "usage_logs" / f"{safe}_usage.jsonl"

    def _ledger_path(self, model_id: str) -> Path:
        safe = model_id.replace("/", "_").replace(":", "_")
        return self.output_dir / "ledgers" / f"{safe}_ledger.json"

    def _wait_for_deepseek_window(self) -> None:
        if not is_deepseek_off_peak():
            wait_s = seconds_until_off_peak()
            logger.info(
                f"[DeepSeek] Peak hours. Waiting {wait_s/60:.1f} minutes "
                f"for off-peak window (16:30 UTC)."
            )
            time.sleep(wait_s + 10)
