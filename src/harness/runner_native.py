"""
v5.1 native API runners — Anthropic, OpenAI, Google, DeepSeek.

Each runner exposes:
    evaluate(prompt_pairs, condition, model_id, usage_log_path, is_batch) -> list[V51Result]
    flush_ledger(path) -> None

The AnthropicRunner wraps the existing ModelEvaluator; the rest implement
their provider's API directly.

DeepSeek: off-peak-only scheduling enforced here (UTC 16:30 – 00:30).
Calls issued outside that window raise DeepSeekPeakHoursError so the
orchestrator can reschedule rather than fail the run.

Usage record format matches model_evaluator.py's _log_usage so the same
audit scripts work across all providers.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .prompts import PromptPair

logger = logging.getLogger(__name__)

# Output-token caps per SPEC §4.5
# Two-stage retry: Stage 1 = STANDARD (cheap, biased toward YES/NO direct answers).
# If empty / finish_reason=length, retry once with STANDARD_RETRY (recovery budget).
# Models that still produce empty after Stage 2 are marked pathological and skipped.
MAX_TOKENS_STANDARD       = 96    # Stage 1 cap for non-reasoning models
MAX_TOKENS_STANDARD_RETRY = 1024  # Stage 2 cap on retry
MAX_TOKENS_REASONING      = 384   # explicit reasoning models use this cap (no retry)

# Stop sequences enforced for non-reasoning Stage 1 calls.
# Newline forces early termination after YES/NO; period covers "Yes."/"No.".
STANDARD_STOP_SEQUENCES = ["\n"]

# Pathological-model detection (revised 2026-05-01 per second-AI feedback).
# Old "5 consecutive empties" approach failed: with chain parallelism (16
# concurrent), independent ~20% empty rate produces frequent artificial
# streaks — coupling independent stochastic events via a global counter.
# That's measuring Bernoulli variance, not model pathology.
#
# New approach: rolling failure-rate guardrail. Sliding window of N=50 most
# recent _call() outcomes; trigger PathologicalModelError if empty rate
# stays above 50% across the full window. Won't fire on streaks or normal
# Bernoulli noise — only on truly broken models that systematically fail.
PATHOLOGICAL_WINDOW            = 50    # sliding window size
PATHOLOGICAL_THRESHOLD         = 0.50  # empty fraction trigger
PATHOLOGICAL_MIN_SAMPLES       = 20    # don't fire before this many samples
# Legacy alias kept for any callers reading PATHOLOGICAL_EMPTY_STREAK
PATHOLOGICAL_EMPTY_STREAK      = PATHOLOGICAL_WINDOW

# DeepSeek off-peak window: UTC 16:30 – 00:30 (8.0 hours, 75% discount)
_DEEPSEEK_WINDOW_START = 990   # 16*60 + 30
_DEEPSEEK_WINDOW_END   = 30    # 00*60 + 30  (wraps midnight)

RATE_LIMIT_SLEEP_S = 0.05


# ---------------------------------------------------------------------------
# Atomic ledger writes (added 2026-05-02 per pre-launch hardening review)
# ---------------------------------------------------------------------------

def atomic_write_text(path: Path, content: str) -> None:
    """
    Write `content` to `path` atomically: write to a sibling tempfile, then
    os.replace() onto the target. On POSIX (Darwin, Linux), os.replace is
    atomic — readers see either the old file or the new one, never a
    half-written file.

    Why this exists: the orchestrator flushes per-model and global
    cost_ledger.json files repeatedly during a 2-4 hour run. A naïve
    `path.write_text(json.dumps(...))` truncates first, then writes — if
    the process is killed (or OOMs, or the laptop hibernates) between
    truncate and write completion, the ledger is silently corrupted.
    Post-hoc analysis then sees an invalid JSON file and the canonical
    cost record is lost. atomic_write_text() prevents that class of bug.
    """
    import os as _os
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    _os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class V51Result:
    """One evaluated PromptPair with full v5.1 metadata.

    Per second-AI consultation 2026-05-01, also tracks:
      - parsed_lenient: alternative parse using whole-line YES/NO scan
                        (separates 'wrong answer' from 'wrong format')
      - needed_retry:   True if the call hit Stage-2 of the retry policy
                        (asymmetric second chance — needs to be reportable)
      - route:          which dispatch path served the call ("native_batch",
                        "openrouter", "moonshot_direct", "deepseek_direct").
                        Used for routing-invariance analysis.
    """
    chain_id: str
    cell: str
    condition: str              # CONDITION_LABELS entry
    model_id: str
    resolved_provider: str      # set by OpenRouterRunner; empty for natives
    baseline_raw: str
    intervention_raw: str
    baseline_parsed: str
    intervention_parsed: str
    baseline_input_tokens: int = 0
    baseline_output_tokens: int = 0
    baseline_cache_read_tokens: int = 0
    baseline_cache_creation_tokens: int = 0
    baseline_cost_usd: float = 0.0
    intervention_input_tokens: int = 0
    intervention_output_tokens: int = 0
    intervention_cache_read_tokens: int = 0
    intervention_cache_creation_tokens: int = 0
    intervention_cost_usd: float = 0.0
    # New fields for methodological transparency (default to safe values
    # so existing call sites keep working).
    baseline_parsed_lenient: str = ""
    intervention_parsed_lenient: str = ""
    baseline_needed_retry: bool = False
    intervention_needed_retry: bool = False
    route: str = ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class PathologicalModelError(RuntimeError):
    """Raised when a model produces ≥PATHOLOGICAL_EMPTY_STREAK consecutive
    empty responses (after both Stage-1 and Stage-2 of the retry policy).
    Signals the model is non-compliant on this prompt class — orchestrator
    should kill the model thread to bound cost."""

    def __init__(self, model_id: str, streak: int):
        super().__init__(
            f"Model {model_id!r} returned {streak} consecutive empty responses "
            f"after retry — terminating thread to bound cost."
        )
        self.model_id = model_id
        self.streak = streak


def _parse_response(text: str) -> str:
    """
    STRICT parser — normalize model output to 'yes', 'no', or 'abstain'.

    First-token-anchored AFTER markdown-wrapper stripping. We strip leading
    markdown wrappers (e.g. ```YES```, **YES**, `YES`) before tokenisation
    since some models comply with "one token" but wrap it in formatting.
    Then we lowercase and strip trailing punctuation/markdown closers from
    the first token. Responses that start with prose (e.g. "Looking at the
    events, yes") are still rejected as 'abstain' — they signal the model
    ignored the "exactly one token" directive.

    Use _parse_response_lenient() for a "did the model land on a classification
    anywhere in the first line" view; reporting both lets us separate "model
    reasoned correctly but ignored format" from "model genuinely abstained".

    This implementation is the OSF-pre-registered authoritative parser
    (OSF_PREREG.md §7.1). Any change here invalidates the pre-registration.
    """
    if not text:
        return "abstain"
    # Exhaustively strip leading markdown wrappers — handles nested cases
    # like ```**YES**``` where a single-pass loop would leave residue.
    stripped = text.strip()
    while True:
        start_len = len(stripped)
        for wrapper in ("```", "``", "`", "**", "__", "*", "_"):
            if stripped.startswith(wrapper):
                stripped = stripped[len(wrapper):].lstrip()
        if len(stripped) == start_len:
            break
    parts = stripped.split()
    if not parts:
        return "abstain"
    first = parts[0].lower().rstrip(".,!?:;\"'`*_")
    if first in ("yes", "y"):
        return "yes"
    if first in ("no", "n"):
        return "no"
    return "abstain"


def _parse_response_lenient(text: str) -> str:
    """
    LENIENT parser — scan the first line of output for a YES/NO classification.

    Catches verbose-prefaced responses like "Looking at the events, yes" or
    "## Step 1\nThe answer is YES" that the strict parser rejects. Used for
    side-by-side reporting (strict vs lenient parse rates) so we can separate:
      - format-compliance failure (strict=abstain, lenient=yes/no)
      - genuine abstain or empty (both abstain)
    """
    if not text:
        return "abstain"
    import re
    # Take only first line of stripped text — multi-line responses still
    # require the answer to be near the start, not buried after paragraphs.
    first_line = text.strip().split("\n", 1)[0]
    # Whole-word match (avoids matching "yesterday" or "no" inside "now")
    has_yes = bool(re.search(r"\byes\b", first_line, re.IGNORECASE))
    has_no  = bool(re.search(r"\bno\b",  first_line, re.IGNORECASE))
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return "abstain"


def _compute_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    rates: dict[str, float],
    is_batch: bool,
) -> float:
    in_rate  = rates.get("batch_input",  rates["input"])  if is_batch else rates["input"]
    out_rate = rates.get("batch_output", rates["output"]) if is_batch else rates["output"]
    standard_in = max(input_tokens - cache_read_tokens - cache_creation_tokens, 0)
    return (
        standard_in        * in_rate
        + cache_read_tokens     * rates.get("cache_read", 0.0)
        + cache_creation_tokens * rates.get("cache_write", 0.0)
        + output_tokens         * out_rate
    )


def _log_usage(
    log_path: Path | None,
    model_id: str,
    chain_id: str,
    cell: str,
    condition: str,
    variant: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
    cost_usd: float,
    is_batch: bool,
    extra: dict | None = None,
) -> None:
    if log_path is None:
        return
    record: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "chain_id": chain_id,
        "cell": cell,
        "condition": condition,
        "variant": variant,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "cost_usd": round(cost_usd, 8),
        "is_batch": is_batch,
    }
    if extra:
        record.update(extra)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# AnthropicRunner
# ---------------------------------------------------------------------------

class AnthropicRunner:
    """
    Thin wrapper around the existing ModelEvaluator.

    Translates V5's EvaluationResult into V51Result so the orchestrator
    sees a uniform interface across all providers.
    """

    def __init__(self, model_id: str, use_batch: bool = False, dry_run: bool = False):
        self.model_id  = model_id
        self.use_batch = use_batch
        self.dry_run   = dry_run
        self._ledger: dict = {}

    def evaluate(
        self,
        prompt_pairs: list["PromptPair"],
        condition: str = "",
        usage_log_path: Path | None = None,
    ) -> list[V51Result]:
        from .model_evaluator import ModelEvaluator

        ev = ModelEvaluator(
            model=self.model_id,
            use_batch=self.use_batch,
            usage_log_path=usage_log_path,
            dry_run=self.dry_run,
        )
        results_v5, _, _ = ev.evaluate_pairs(prompt_pairs)

        # Merge per-model ledger totals
        for m, entry in ev._ledger.items():
            dest = self._ledger.setdefault(m, {
                "n_calls": 0, "total_input_tokens": 0,
                "total_output_tokens": 0, "total_cache_read_tokens": 0,
                "total_cache_creation_tokens": 0, "total_cost_usd": 0.0,
                "max_single_call_cost_usd": 0.0,
                "n_yes": 0, "n_no": 0, "n_abstain": 0, "n_stage2_retries": 0,
            })
            dest["n_calls"]                  += entry["n_calls"]
            dest["total_input_tokens"]       += entry["total_input_tokens"]
            dest["total_output_tokens"]      += entry["total_output_tokens"]
            dest["total_cache_read_tokens"]  += entry["total_cache_read_tokens"]
            dest["total_cache_creation_tokens"] += entry["total_cache_creation_tokens"]
            dest["total_cost_usd"]           = round(
                dest["total_cost_usd"] + entry["total_cost_usd"], 8)
            dest["max_single_call_cost_usd"] = max(
                dest["max_single_call_cost_usd"], entry["max_single_call_cost_usd"])

        # Accumulate label-class counters from results (Anthropic batch has no
        # two-stage retry concept, so n_stage2_retries stays 0 for this route).
        # Note: ModelEvaluator's ledger doesn't carry these counter fields,
        # and v5 EvaluationResult doesn't carry model_id either — but
        # AnthropicRunner is per-model, so all rows belong to self.model_id.
        # We backfill the counter keys explicitly because setdefault is a
        # no-op when an entry already exists from the merge loop above.
        for model_key in list(self._ledger.keys()) + [self.model_id]:
            entry = self._ledger.setdefault(model_key, {
                "n_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
                "total_cache_read_tokens": 0, "total_cache_creation_tokens": 0,
                "total_cost_usd": 0.0, "max_single_call_cost_usd": 0.0,
                "n_yes": 0, "n_no": 0, "n_abstain": 0, "n_stage2_retries": 0,
            })
            for k in ("n_yes", "n_no", "n_abstain", "n_stage2_retries"):
                entry.setdefault(k, 0)
        target_entry = self._ledger[self.model_id]
        for r in results_v5:
            for parsed in (r.baseline_parsed, r.intervention_parsed):
                if parsed in ("yes", "no", "abstain"):
                    target_entry[f"n_{parsed}"] += 1

        out = []
        route_label = "anthropic_batch" if self.use_batch else "anthropic_sync"
        for pair, r in zip(prompt_pairs, results_v5):
            out.append(V51Result(
                chain_id=r.chain_id, cell=r.cell, condition=condition,
                model_id=self.model_id, resolved_provider="",
                baseline_raw=r.baseline_raw, intervention_raw=r.intervention_raw,
                baseline_parsed=r.baseline_parsed,
                intervention_parsed=r.intervention_parsed,
                baseline_input_tokens=r.baseline_input_tokens,
                baseline_output_tokens=r.baseline_output_tokens,
                baseline_cache_read_tokens=r.baseline_cache_read_tokens,
                baseline_cache_creation_tokens=r.baseline_cache_creation_tokens,
                baseline_cost_usd=r.baseline_cost_usd,
                intervention_input_tokens=r.intervention_input_tokens,
                intervention_output_tokens=r.intervention_output_tokens,
                intervention_cache_read_tokens=r.intervention_cache_read_tokens,
                intervention_cache_creation_tokens=r.intervention_cache_creation_tokens,
                intervention_cost_usd=r.intervention_cost_usd,
                baseline_parsed_lenient=_parse_response_lenient(r.baseline_raw),
                intervention_parsed_lenient=_parse_response_lenient(r.intervention_raw),
                route=route_label,
            ))
        return out

    def flush_ledger(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(self._ledger, indent=2))


# ---------------------------------------------------------------------------
# OpenAIRunner
# ---------------------------------------------------------------------------

class OpenAIRunner:
    """
    OpenAI Batches API (full run) + synchronous (smoke test).

    Pricing rates are passed in at construction so the same runner class
    handles GPT-5 Mini / GPT-5 / GPT-5.4+ at their respective rates.

    Usage fields:
      input_tokens             → prompt_tokens
      output_tokens            → completion_tokens
      cache_read_tokens        → prompt_tokens_details.cached_tokens
      cache_creation_tokens    → always 0 (OpenAI caches automatically)
    """

    def __init__(
        self,
        model_id: str,
        rates: dict[str, float],
        use_batch: bool = False,
        is_reasoning: bool = False,
        batch_poll_interval_s: float = 60.0,
    ):
        self.model_id = model_id
        self.rates    = rates
        self.use_batch = use_batch
        self.is_reasoning = is_reasoning
        self.batch_poll_interval_s = batch_poll_interval_s
        self._client = None
        self._ledger: dict = {}

    def _ensure_client(self):
        if self._client is None:
            import os
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def evaluate(
        self,
        prompt_pairs: list["PromptPair"],
        condition: str = "",
        usage_log_path: Path | None = None,
    ) -> list[V51Result]:
        self._ensure_client()
        if self.use_batch:
            return self._evaluate_batch(prompt_pairs, condition, usage_log_path)
        return self._evaluate_sequential(prompt_pairs, condition, usage_log_path)

    def _max_tokens(self) -> int:
        return MAX_TOKENS_REASONING if self.is_reasoning else MAX_TOKENS_STANDARD

    def _call(self, prompt: str) -> dict:
        # GPT-5.x and o-series require max_completion_tokens; use it universally.
        # reasoning_effort="minimal" empirically discovered 2026-05-01: gpt-5
        # and gpt-5-mini do internal CoT by default and burn 970-1024 output
        # tokens before producing YES/NO (50% empty-content rate at cap=1024).
        # With minimal: 10 tokens, ~2s, correct answer. Reasoning models keep
        # default effort.
        kwargs: dict = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self._max_tokens(),
        }
        if not self.is_reasoning:
            kwargs["reasoning_effort"] = "minimal"
        response = self._client.chat.completions.create(**kwargs)
        u = response.usage
        cached = 0
        if hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
            cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
        return {
            "text": response.choices[0].message.content or "",
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "cache_read_tokens": cached,
            "cache_creation_tokens": 0,
        }

    def _evaluate_sequential(self, pairs, condition, log_path):
        out = []
        for pair in pairs:
            b  = self._call(pair.baseline_prompt)
            time.sleep(RATE_LIMIT_SLEEP_S)
            iv = self._call(pair.intervention_prompt)
            time.sleep(RATE_LIMIT_SLEEP_S)

            b_cost  = _compute_cost(b["input_tokens"], b["output_tokens"],
                                    b["cache_read_tokens"], b["cache_creation_tokens"],
                                    self.rates, is_batch=False)
            iv_cost = _compute_cost(iv["input_tokens"], iv["output_tokens"],
                                    iv["cache_read_tokens"], iv["cache_creation_tokens"],
                                    self.rates, is_batch=False)

            _log_usage(log_path, self.model_id, pair.chain_id, pair.cell,
                       condition, "baseline",
                       b["input_tokens"], b["output_tokens"],
                       b["cache_read_tokens"], b["cache_creation_tokens"], b_cost, False)
            _log_usage(log_path, self.model_id, pair.chain_id, pair.cell,
                       condition, "intervention",
                       iv["input_tokens"], iv["output_tokens"],
                       iv["cache_read_tokens"], iv["cache_creation_tokens"], iv_cost, False)
            b_parsed  = _parse_response(b["text"])
            iv_parsed = _parse_response(iv["text"])
            self._update_ledger(b, b_cost,
                                parsed=b_parsed,
                                needed_retry=b.get("needed_retry", False))
            self._update_ledger(iv, iv_cost,
                                parsed=iv_parsed,
                                needed_retry=iv.get("needed_retry", False))

            out.append(V51Result(
                chain_id=pair.chain_id, cell=pair.cell, condition=condition,
                model_id=self.model_id, resolved_provider="",
                baseline_raw=b["text"], intervention_raw=iv["text"],
                baseline_parsed=b_parsed,
                intervention_parsed=iv_parsed,
                baseline_input_tokens=b["input_tokens"],
                baseline_output_tokens=b["output_tokens"],
                baseline_cache_read_tokens=b["cache_read_tokens"],
                baseline_cache_creation_tokens=b["cache_creation_tokens"],
                baseline_cost_usd=b_cost,
                intervention_input_tokens=iv["input_tokens"],
                intervention_output_tokens=iv["output_tokens"],
                intervention_cache_read_tokens=iv["cache_read_tokens"],
                intervention_cache_creation_tokens=iv["cache_creation_tokens"],
                intervention_cost_usd=iv_cost,
                baseline_parsed_lenient=_parse_response_lenient(b["text"]),
                intervention_parsed_lenient=_parse_response_lenient(iv["text"]),
                route="openai_sync",
            ))
        return out

    def _evaluate_batch(self, pairs, condition, log_path):
        import io
        # Build JSONL input file
        lines = []
        for idx, pair in enumerate(pairs):
            for variant, prompt in [("baseline", pair.baseline_prompt),
                                     ("intervention", pair.intervention_prompt)]:
                custom_id = f"{idx:06d}__{pair.chain_id}__{variant}"
                # Batch can't do two-stage retry mid-batch, so allocate the
                # Stage-2 budget directly (1024 for standard, REASONING for
                # reasoning models). 50% batch discount keeps cost bounded.
                batch_cap = (
                    MAX_TOKENS_REASONING if self.is_reasoning
                    else MAX_TOKENS_STANDARD_RETRY
                )
                body: dict = {
                    "model": self.model_id,
                    "max_completion_tokens": batch_cap,
                    "messages": [{"role": "user", "content": prompt}],
                }
                # See _call() docstring: gpt-5/gpt-5-mini secretly reason and
                # exhaust the cap producing empty content unless we set
                # reasoning_effort=minimal. Reasoning models keep default.
                if not self.is_reasoning:
                    body["reasoning_effort"] = "minimal"
                lines.append(json.dumps({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }))
        jsonl_bytes = ("\n".join(lines)).encode()

        # Upload file
        batch_file = self._client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
            purpose="batch",
        )
        logger.info(f"[{self.model_id}] Uploaded batch file {batch_file.id} "
                    f"({len(pairs)*2} requests)")

        # Create batch
        batch = self._client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        logger.info(f"[{self.model_id}] Batch {batch.id} submitted")

        # Poll
        while batch.status not in ("completed", "failed", "cancelled", "expired"):
            time.sleep(self.batch_poll_interval_s)
            batch = self._client.batches.retrieve(batch.id)
            logger.info(f"[{self.model_id}] Batch {batch.id[:12]} status={batch.status}")

        if batch.status != "completed":
            raise RuntimeError(f"OpenAI batch ended with status={batch.status}")

        # Download results
        raw = self._client.files.content(batch.output_file_id).text
        results_by_id: dict[str, dict] = {}
        for line in raw.strip().split("\n"):
            if not line:
                continue
            rec = json.loads(line)
            cid = rec["custom_id"]
            body = rec.get("response", {}).get("body", {})
            choices = body.get("choices", [])
            usage   = body.get("usage", {})
            text = choices[0]["message"]["content"] if choices else ""
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            results_by_id[cid] = {
                "text": text or "",
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_read_tokens": cached,
                "cache_creation_tokens": 0,
            }

        out = []
        for idx, pair in enumerate(pairs):
            b  = results_by_id.get(f"{idx:06d}__{pair.chain_id}__baseline", {})
            iv = results_by_id.get(f"{idx:06d}__{pair.chain_id}__intervention", {})
            b_text  = b.get("text", ""); iv_text = iv.get("text", "")
            b_cost  = _compute_cost(b.get("input_tokens", 0), b.get("output_tokens", 0),
                                    b.get("cache_read_tokens", 0), 0, self.rates, is_batch=True)
            iv_cost = _compute_cost(iv.get("input_tokens", 0), iv.get("output_tokens", 0),
                                    iv.get("cache_read_tokens", 0), 0, self.rates, is_batch=True)
            if b:
                _log_usage(log_path, self.model_id, pair.chain_id, pair.cell,
                           condition, "baseline",
                           b["input_tokens"], b["output_tokens"],
                           b["cache_read_tokens"], 0, b_cost, True)
                self._update_ledger(b, b_cost,
                                    parsed=_parse_response(b_text),
                                    needed_retry=False)
            if iv:
                _log_usage(log_path, self.model_id, pair.chain_id, pair.cell,
                           condition, "intervention",
                           iv["input_tokens"], iv["output_tokens"],
                           iv["cache_read_tokens"], 0, iv_cost, True)
                self._update_ledger(iv, iv_cost,
                                    parsed=_parse_response(iv_text),
                                    needed_retry=False)
            out.append(V51Result(
                chain_id=pair.chain_id, cell=pair.cell, condition=condition,
                model_id=self.model_id, resolved_provider="",
                baseline_raw=b_text, intervention_raw=iv_text,
                baseline_parsed=_parse_response(b_text),
                intervention_parsed=_parse_response(iv_text),
                baseline_input_tokens=b.get("input_tokens", 0),
                baseline_output_tokens=b.get("output_tokens", 0),
                baseline_cache_read_tokens=b.get("cache_read_tokens", 0),
                baseline_cache_creation_tokens=0,
                baseline_cost_usd=b_cost,
                intervention_input_tokens=iv.get("input_tokens", 0),
                intervention_output_tokens=iv.get("output_tokens", 0),
                intervention_cache_read_tokens=iv.get("cache_read_tokens", 0),
                intervention_cache_creation_tokens=0,
                intervention_cost_usd=iv_cost,
                baseline_parsed_lenient=_parse_response_lenient(b_text),
                intervention_parsed_lenient=_parse_response_lenient(iv_text),
                route="openai_batch",
            ))
        return out

    def _update_ledger(self, usage: dict, cost_usd: float,
                       parsed: str = "", needed_retry: bool = False) -> None:
        entry = self._ledger.setdefault(self.model_id, {
            "n_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
            "total_cache_read_tokens": 0, "total_cache_creation_tokens": 0,
            "total_cost_usd": 0.0, "max_single_call_cost_usd": 0.0,
            "n_yes": 0, "n_no": 0, "n_abstain": 0, "n_stage2_retries": 0,
        })
        entry["n_calls"]              += 1
        entry["total_input_tokens"]   += usage.get("input_tokens", 0)
        entry["total_output_tokens"]  += usage.get("output_tokens", 0)
        entry["total_cache_read_tokens"] += usage.get("cache_read_tokens", 0)
        entry["total_cost_usd"]        = round(entry["total_cost_usd"] + cost_usd, 8)
        entry["max_single_call_cost_usd"] = max(
            entry["max_single_call_cost_usd"], cost_usd)
        if parsed in ("yes", "no", "abstain"):
            entry[f"n_{parsed}"] += 1
        if needed_retry:
            entry["n_stage2_retries"] += 1

    def flush_ledger(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(self._ledger, indent=2))


# ---------------------------------------------------------------------------
# MoonshotRunner — direct Kimi/Moonshot API
# ---------------------------------------------------------------------------

class MoonshotRunner:
    """
    Direct Moonshot AI (Kimi) API runner — OpenAI SDK compatible.

    Routes Kimi models away from OpenRouter to gain access to:
      - The `thinking` extra_body parameter (kimi-k2.6 needs
        `{"type": "disabled"}` or its message.content comes back empty)
      - Native Moonshot rate limits / billing tier

    Empirically validated 2026-05-01:
      - kimi-k2.6 via OR (any provider): 100% empty content (thinking on)
      - kimi-k2.6 via direct Moonshot, default: 100% empty (thinking on)
      - kimi-k2.6 via direct Moonshot, thinking={"type":"disabled"}: ~7s,
        4 output tokens, finish=stop, content="YES" — works perfectly

    Two-stage retry + pathological detection (same as OpenRouterRunner):
      Stage 1: max_tokens=96, stop=["\n"]  — cheap, biases YES/NO direct
      Stage 2: max_tokens=1024, no stop    — recovery for unusual responses
      ≥5 consecutive empties → PathologicalModelError → kill model thread
    """

    _BASE_URL = "https://api.moonshot.ai/v1"

    def __init__(
        self,
        model_id: str,
        rates: dict[str, float],
        is_reasoning: bool = False,
        thinking_disabled: bool = True,
    ):
        self.model_id          = model_id
        self.rates             = rates
        self.is_reasoning      = is_reasoning
        self.thinking_disabled = thinking_disabled
        self._client           = None
        self._ledger: dict     = {}
        self._ledger_lock      = threading.Lock()
        self._log_lock         = threading.Lock()
        # Pathological-model tracking — rolling failure-rate guardrail
        # (see runner_native.PATHOLOGICAL_* docs for design rationale).
        from collections import deque as _deque
        self._call_history       = _deque(maxlen=PATHOLOGICAL_WINDOW)
        self._empty_streak_lock  = threading.Lock()

    def _ensure_client(self):
        if self._client is None:
            import os
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ["MOONSHOT_API_KEY"],
                base_url=self._BASE_URL,
            )

    def _max_tokens(self) -> int:
        return MAX_TOKENS_REASONING if self.is_reasoning else MAX_TOKENS_STANDARD

    def _extra_body(self) -> dict:
        body: dict = {}
        if self.thinking_disabled:
            body["thinking"] = {"type": "disabled"}
        return body

    def _raw_call(self, prompt: str, max_tokens: int,
                  stop: list[str] | None = None) -> dict:
        kwargs: dict = {
            "model":      self.model_id,
            "messages":   [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if stop:
            kwargs["stop"] = stop
        eb = self._extra_body()
        if eb:
            kwargs["extra_body"] = eb

        response = self._client.chat.completions.create(**kwargs)
        u = response.usage
        choice = response.choices[0] if response.choices else None
        text = (choice.message.content or "") if choice else ""
        finish_reason = getattr(choice, "finish_reason", "") if choice else ""
        input_tokens  = u.prompt_tokens     if u else 0
        output_tokens = u.completion_tokens if u else 0
        # Moonshot exposes cached prompt tokens via prompt_tokens_details
        cached = 0
        if u and hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
            cached = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
        cost_usd = _compute_cost(
            input_tokens, output_tokens, cached, 0, self.rates, is_batch=False,
        )
        return {
            "text":                  text,
            "finish_reason":         finish_reason,
            "input_tokens":          input_tokens,
            "output_tokens":         output_tokens,
            "cache_read_tokens":     cached,
            "cache_creation_tokens": 0,
            "cost_usd":              cost_usd,
        }

    def _call(self, prompt: str) -> dict:
        """Two-stage retry — same shape as OpenRouterRunner._call."""
        if self.is_reasoning:
            resp = self._raw_call(prompt, max_tokens=MAX_TOKENS_REASONING)
            resp["needed_retry"] = False
            self._track_empty(resp["text"])
            return resp

        resp = self._raw_call(
            prompt,
            max_tokens=MAX_TOKENS_STANDARD,
            stop=STANDARD_STOP_SEQUENCES,
        )
        if resp["text"] and resp["finish_reason"] != "length":
            resp["needed_retry"] = False
            self._track_empty(resp["text"])
            return resp

        logger.info(
            f"[{self.model_id}] Stage-1 empty (finish={resp['finish_reason']}, "
            f"tokens={resp['output_tokens']}); retrying at {MAX_TOKENS_STANDARD_RETRY}"
        )
        resp2 = self._raw_call(prompt, max_tokens=MAX_TOKENS_STANDARD_RETRY)
        resp2["cost_usd"]      += resp["cost_usd"]
        resp2["input_tokens"]  += resp["input_tokens"]
        resp2["output_tokens"] += resp["output_tokens"]
        resp2["needed_retry"] = True
        self._track_empty(resp2["text"])
        return resp2

    def _track_empty(self, text: str) -> None:
        """Rolling failure-rate guardrail — see runner_native.PATHOLOGICAL_*."""
        with self._empty_streak_lock:
            self._call_history.append(not bool(text))
            if len(self._call_history) >= PATHOLOGICAL_MIN_SAMPLES:
                empty_count = sum(self._call_history)
                empty_rate  = empty_count / len(self._call_history)
                if empty_rate >= PATHOLOGICAL_THRESHOLD:
                    raise PathologicalModelError(self.model_id, empty_count)

    def evaluate(
        self,
        prompt_pairs: list["PromptPair"],
        condition: str = "",
        usage_log_path: Path | None = None,
        max_concurrent: int = 8,
    ) -> list[V51Result]:
        """Chain-parallel sync evaluation; one fresh thread per pair, bounded
        by `max_concurrent` semaphore. Same pattern as OpenRouterRunner.evaluate."""
        self._ensure_client()
        out: list[V51Result] = []
        out_lock = threading.Lock()

        def run_pair(pair: "PromptPair") -> None:
            b  = self._call(pair.baseline_prompt)
            time.sleep(RATE_LIMIT_SLEEP_S)
            iv = self._call(pair.intervention_prompt)
            time.sleep(RATE_LIMIT_SLEEP_S)

            with self._log_lock:
                _log_usage(
                    usage_log_path, self.model_id, pair.chain_id, pair.cell,
                    condition, "baseline",
                    b["input_tokens"], b["output_tokens"],
                    b["cache_read_tokens"], 0, b["cost_usd"], False,
                )
                _log_usage(
                    usage_log_path, self.model_id, pair.chain_id, pair.cell,
                    condition, "intervention",
                    iv["input_tokens"], iv["output_tokens"],
                    iv["cache_read_tokens"], 0, iv["cost_usd"], False,
                )

            result = V51Result(
                chain_id=pair.chain_id, cell=pair.cell, condition=condition,
                model_id=self.model_id, resolved_provider="moonshot-direct",
                baseline_raw=b["text"], intervention_raw=iv["text"],
                baseline_parsed=_parse_response(b["text"]),
                intervention_parsed=_parse_response(iv["text"]),
                baseline_input_tokens=b["input_tokens"],
                baseline_output_tokens=b["output_tokens"],
                baseline_cache_read_tokens=b["cache_read_tokens"],
                baseline_cache_creation_tokens=0,
                baseline_cost_usd=b["cost_usd"],
                intervention_input_tokens=iv["input_tokens"],
                intervention_output_tokens=iv["output_tokens"],
                intervention_cache_read_tokens=iv["cache_read_tokens"],
                intervention_cache_creation_tokens=0,
                intervention_cost_usd=iv["cost_usd"],
                baseline_parsed_lenient=_parse_response_lenient(b["text"]),
                intervention_parsed_lenient=_parse_response_lenient(iv["text"]),
                baseline_needed_retry=b.get("needed_retry", False),
                intervention_needed_retry=iv.get("needed_retry", False),
                route="moonshot_direct",
            )

            with self._ledger_lock:
                self._update_ledger(b,
                                    parsed=_parse_response(b["text"]),
                                    needed_retry=b.get("needed_retry", False))
                self._update_ledger(iv,
                                    parsed=_parse_response(iv["text"]),
                                    needed_retry=iv.get("needed_retry", False))
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

    def _update_ledger(self, usage: dict,
                       parsed: str = "", needed_retry: bool = False) -> None:
        # Caller must hold self._ledger_lock
        entry = self._ledger.setdefault(self.model_id, {
            "n_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
            "total_cache_read_tokens": 0, "total_cache_creation_tokens": 0,
            "total_cost_usd": 0.0, "max_single_call_cost_usd": 0.0,
            "n_yes": 0, "n_no": 0, "n_abstain": 0, "n_stage2_retries": 0,
        })
        cost_usd = usage["cost_usd"]
        entry["n_calls"]                 += 1
        entry["total_input_tokens"]      += usage.get("input_tokens", 0)
        entry["total_output_tokens"]     += usage.get("output_tokens", 0)
        entry["total_cache_read_tokens"] += usage.get("cache_read_tokens", 0)
        entry["total_cost_usd"] = round(entry["total_cost_usd"] + cost_usd, 8)
        entry["max_single_call_cost_usd"] = max(
            entry["max_single_call_cost_usd"], cost_usd)
        if parsed in ("yes", "no", "abstain"):
            entry[f"n_{parsed}"] += 1
        if needed_retry:
            entry["n_stage2_retries"] += 1

    def flush_ledger(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(self._ledger, indent=2))


# ---------------------------------------------------------------------------
# GoogleRunner
# ---------------------------------------------------------------------------

class GoogleRunner:
    """
    Google Gemini API runner (google.genai SDK).

    Usage fields:
      input_tokens             → prompt_token_count
      output_tokens            → candidates_token_count
      cache_read_tokens        → cached_content_token_count
      cache_creation_tokens    → always 0 (caching not used in this runner)
    """

    def __init__(
        self,
        model_id: str,
        rates: dict[str, float],
        use_batch: bool = False,
        is_reasoning: bool = False,
    ):
        self.model_id     = model_id
        self.rates        = rates
        self.use_batch    = use_batch
        self.is_reasoning = is_reasoning
        self._client      = None
        self._ledger: dict = {}

    def _ensure_client(self):
        if self._client is None:
            import os
            from google import genai
            self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def evaluate(
        self,
        prompt_pairs: list["PromptPair"],
        condition: str = "",
        usage_log_path: Path | None = None,
    ) -> list[V51Result]:
        self._ensure_client()
        if self.use_batch:
            return self._evaluate_batch(prompt_pairs, condition, usage_log_path)
        return self._evaluate_sequential(prompt_pairs, condition, usage_log_path)

    def _max_tokens(self) -> int:
        return MAX_TOKENS_REASONING if self.is_reasoning else MAX_TOKENS_STANDARD

    def _call(self, prompt: str) -> dict:
        import re as _re
        from google.genai import types
        from google.genai.errors import ClientError
        max_retries = 6
        for attempt in range(max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=self._max_tokens(),
                    ),
                )
                break
            except ClientError as e:
                if "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e):
                    raise
                if attempt == max_retries - 1:
                    raise
                # Parse "Please retry in Xs" from the error message if present
                m = _re.search(r"retry in ([0-9.]+)s", str(e))
                wait = float(m.group(1)) if m else (2 ** attempt + 1)
                logger.warning(f"[{self.model_id}] 429 quota, retry {attempt+1}/{max_retries} in {wait:.1f}s")
                time.sleep(wait)
        text = resp.text or ""
        um = resp.usage_metadata
        return {
            "text": text,
            "input_tokens":        (um.prompt_token_count or 0) if um else 0,
            "output_tokens":       (um.candidates_token_count or 0) if um else 0,
            "cache_read_tokens":   (getattr(um, "cached_content_token_count", None) or 0) if um else 0,
            "cache_creation_tokens": 0,
        }

    def _evaluate_sequential(self, pairs, condition, log_path):
        out = []
        for pair in pairs:
            b  = self._call(pair.baseline_prompt)
            time.sleep(RATE_LIMIT_SLEEP_S)
            iv = self._call(pair.intervention_prompt)
            time.sleep(RATE_LIMIT_SLEEP_S)

            b_cost  = _compute_cost(b["input_tokens"], b["output_tokens"],
                                    b["cache_read_tokens"], 0, self.rates, False)
            iv_cost = _compute_cost(iv["input_tokens"], iv["output_tokens"],
                                    iv["cache_read_tokens"], 0, self.rates, False)

            _log_usage(log_path, self.model_id, pair.chain_id, pair.cell,
                       condition, "baseline",
                       b["input_tokens"], b["output_tokens"],
                       b["cache_read_tokens"], 0, b_cost, False)
            _log_usage(log_path, self.model_id, pair.chain_id, pair.cell,
                       condition, "intervention",
                       iv["input_tokens"], iv["output_tokens"],
                       iv["cache_read_tokens"], 0, iv_cost, False)
            b_parsed  = _parse_response(b["text"])
            iv_parsed = _parse_response(iv["text"])
            self._update_ledger(b, b_cost,
                                parsed=b_parsed,
                                needed_retry=b.get("needed_retry", False))
            self._update_ledger(iv, iv_cost,
                                parsed=iv_parsed,
                                needed_retry=iv.get("needed_retry", False))

            out.append(V51Result(
                chain_id=pair.chain_id, cell=pair.cell, condition=condition,
                model_id=self.model_id, resolved_provider="",
                baseline_raw=b["text"], intervention_raw=iv["text"],
                baseline_parsed=b_parsed,
                intervention_parsed=iv_parsed,
                baseline_input_tokens=b["input_tokens"],
                baseline_output_tokens=b["output_tokens"],
                baseline_cache_read_tokens=b["cache_read_tokens"],
                baseline_cache_creation_tokens=0,
                baseline_cost_usd=b_cost,
                intervention_input_tokens=iv["input_tokens"],
                intervention_output_tokens=iv["output_tokens"],
                intervention_cache_read_tokens=iv["cache_read_tokens"],
                intervention_cache_creation_tokens=0,
                intervention_cost_usd=iv_cost,
                baseline_parsed_lenient=_parse_response_lenient(b["text"]),
                intervention_parsed_lenient=_parse_response_lenient(iv["text"]),
                route="google_sync",
            ))
        return out

    def _evaluate_batch(self, pairs, condition, log_path):
        """
        Submit pairs as a Gemini Batch API job using inline requests
        (google.genai SDK, modern path).

        Per https://ai.google.dev/gemini-api/docs/batch-api:
          - 50% pricing vs sync API
          - 24-hour target turnaround
          - Inline responses returned in the same order they were submitted
            (we rely on positional matching, since the new SDK doesn't
             surface per-request keys on inlined responses)
        """
        import time as time_mod
        from google.genai import types
        from google.genai.errors import ClientError

        # Batch can't do two-stage retry mid-batch — allocate Stage-2 budget
        # directly. 50% batch discount keeps cost bounded.
        max_tokens = (
            MAX_TOKENS_REASONING if self.is_reasoning
            else MAX_TOKENS_STANDARD_RETRY
        )

        # Build inline requests; track (pair, variant) by submission position.
        # The InlinedRequest schema accepts: model, contents, metadata, config.
        # generation params live on `config` as a GenerateContentConfig.
        gen_cfg = types.GenerateContentConfig(max_output_tokens=max_tokens)
        inline_requests = []
        metadata: list[tuple] = []
        for pair in pairs:
            for variant, prompt in [("baseline", pair.baseline_prompt),
                                     ("intervention", pair.intervention_prompt)]:
                inline_requests.append({
                    "contents": [{"parts": [{"text": prompt}], "role": "user"}],
                    "config": gen_cfg,
                })
                metadata.append((pair, variant))

        # Submit
        batch_job = self._client.batches.create(
            model=self.model_id,
            src=inline_requests,
            config={"display_name": f"v51_{condition}_{int(time_mod.time())}"},
        )
        job_name = batch_job.name
        logger.info(
            f"[{self.model_id}] Batch {job_name} submitted "
            f"({len(inline_requests)} requests)"
        )

        # Poll until terminal state
        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
        while batch_job.state.name not in terminal:
            time_mod.sleep(30)
            try:
                batch_job = self._client.batches.get(name=job_name)
            except ClientError as e:
                logger.warning(f"[{self.model_id}] Batch poll error (continuing): {e}")
                continue
            logger.info(
                f"[{self.model_id}] Batch {job_name} state={batch_job.state.name}"
            )

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(
                f"[{self.model_id}] Batch ended in state={batch_job.state.name}"
            )

        # Extract inline responses (order matches submission order)
        inlined = (batch_job.dest.inlined_responses
                   if batch_job.dest else None) or []
        if len(inlined) != len(metadata):
            raise RuntimeError(
                f"[{self.model_id}] Batch returned {len(inlined)} responses, "
                f"expected {len(metadata)}"
            )

        # Map (chain_id, variant) → usage dict
        extracted: dict[tuple, dict] = {}
        for inline_resp, (pair, variant) in zip(inlined, metadata):
            if getattr(inline_resp, "error", None):
                logger.warning(
                    f"[{self.model_id}] Batch entry error for "
                    f"{pair.chain_id}/{variant}: {inline_resp.error}"
                )
                extracted[(pair.chain_id, variant)] = {
                    "text": "", "input_tokens": 0, "output_tokens": 0,
                    "cache_read_tokens": 0, "cache_creation_tokens": 0,
                }
                continue
            resp = inline_resp.response
            # Gemini batch responses can include non-text parts (e.g.
            # `thought_signature` for Pro models), which causes the `.text`
            # shortcut to return None even when the model produced a valid
            # answer. Iterate candidates[0].content.parts and concatenate
            # any .text fields directly. Verified 2026-05-01: gemini-2.5-pro
            # was 71/80 empty via batch using just `.text`; this should
            # restore those responses.
            text = ""
            try:
                if resp:
                    text = resp.text or ""
            except Exception:
                text = ""
            if not text and resp and getattr(resp, "candidates", None):
                try:
                    parts = resp.candidates[0].content.parts or []
                    text = "".join(
                        (getattr(p, "text", "") or "") for p in parts
                    )
                except Exception:
                    pass
            um = resp.usage_metadata if resp else None
            extracted[(pair.chain_id, variant)] = {
                "text":               text,
                "input_tokens":       (um.prompt_token_count or 0) if um else 0,
                "output_tokens":      (um.candidates_token_count or 0) if um else 0,
                "cache_read_tokens":  (getattr(um, "cached_content_token_count", None) or 0) if um else 0,
                "cache_creation_tokens": 0,
            }

        # Build V51Results in pair order
        out = []
        for pair in pairs:
            b  = extracted.get((pair.chain_id, "baseline"), {})
            iv = extracted.get((pair.chain_id, "intervention"), {})
            b_text  = b.get("text", "")
            iv_text = iv.get("text", "")

            b_cost  = _compute_cost(b.get("input_tokens", 0), b.get("output_tokens", 0),
                                    b.get("cache_read_tokens", 0), 0, self.rates, True)
            iv_cost = _compute_cost(iv.get("input_tokens", 0), iv.get("output_tokens", 0),
                                    iv.get("cache_read_tokens", 0), 0, self.rates, True)

            _log_usage(log_path, self.model_id, pair.chain_id, pair.cell,
                       condition, "baseline",
                       b.get("input_tokens", 0), b.get("output_tokens", 0),
                       b.get("cache_read_tokens", 0), 0, b_cost, True)
            _log_usage(log_path, self.model_id, pair.chain_id, pair.cell,
                       condition, "intervention",
                       iv.get("input_tokens", 0), iv.get("output_tokens", 0),
                       iv.get("cache_read_tokens", 0), 0, iv_cost, True)
            b_parsed  = _parse_response(b_text)
            iv_parsed = _parse_response(iv_text)
            self._update_ledger(b, b_cost,
                                parsed=b_parsed,
                                needed_retry=False)
            self._update_ledger(iv, iv_cost,
                                parsed=iv_parsed,
                                needed_retry=False)

            out.append(V51Result(
                chain_id=pair.chain_id, cell=pair.cell, condition=condition,
                model_id=self.model_id, resolved_provider="",
                baseline_raw=b_text, intervention_raw=iv_text,
                baseline_parsed=b_parsed,
                intervention_parsed=iv_parsed,
                baseline_input_tokens=b.get("input_tokens", 0),
                baseline_output_tokens=b.get("output_tokens", 0),
                baseline_cache_read_tokens=b.get("cache_read_tokens", 0),
                baseline_cache_creation_tokens=0,
                baseline_cost_usd=b_cost,
                intervention_input_tokens=iv.get("input_tokens", 0),
                intervention_output_tokens=iv.get("output_tokens", 0),
                intervention_cache_read_tokens=iv.get("cache_read_tokens", 0),
                intervention_cache_creation_tokens=0,
                intervention_cost_usd=iv_cost,
                baseline_parsed_lenient=_parse_response_lenient(b_text),
                intervention_parsed_lenient=_parse_response_lenient(iv_text),
                route="google_batch",
            ))
        return out

    def _update_ledger(self, usage: dict, cost_usd: float,
                       parsed: str = "", needed_retry: bool = False) -> None:
        entry = self._ledger.setdefault(self.model_id, {
            "n_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
            "total_cache_read_tokens": 0, "total_cache_creation_tokens": 0,
            "total_cost_usd": 0.0, "max_single_call_cost_usd": 0.0,
            "n_yes": 0, "n_no": 0, "n_abstain": 0, "n_stage2_retries": 0,
        })
        entry["n_calls"]             += 1
        entry["total_input_tokens"]  += usage.get("input_tokens", 0)
        entry["total_output_tokens"] += usage.get("output_tokens", 0)
        entry["total_cache_read_tokens"] += usage.get("cache_read_tokens", 0)
        entry["total_cost_usd"]       = round(entry["total_cost_usd"] + cost_usd, 8)
        entry["max_single_call_cost_usd"] = max(
            entry["max_single_call_cost_usd"], cost_usd)
        if parsed in ("yes", "no", "abstain"):
            entry[f"n_{parsed}"] += 1
        if needed_retry:
            entry["n_stage2_retries"] += 1

    def flush_ledger(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(self._ledger, indent=2))


# ---------------------------------------------------------------------------
# DeepSeekRunner
# ---------------------------------------------------------------------------

class DeepSeekPeakHoursError(RuntimeError):
    """Raised when a DeepSeek call is attempted outside the off-peak window."""


def is_deepseek_off_peak() -> bool:
    """
    True when UTC clock is inside the 75%-discount window.
    Window: 16:30 – 00:30 UTC (next day).
    """
    utc = datetime.now(timezone.utc)
    total_minutes = utc.hour * 60 + utc.minute
    return total_minutes >= _DEEPSEEK_WINDOW_START or total_minutes < _DEEPSEEK_WINDOW_END


def seconds_until_off_peak() -> float:
    """Seconds from now until the off-peak window opens (16:30 UTC)."""
    utc = datetime.now(timezone.utc)
    total_minutes = utc.hour * 60 + utc.minute
    if is_deepseek_off_peak():
        return 0.0
    # Window opens at 16:30 UTC = 990 minutes
    wait = _DEEPSEEK_WINDOW_START - total_minutes
    return wait * 60.0


class DeepSeekRunner:
    """
    DeepSeek synchronous runner (DeepSeek has no batch API).

    Hard constraint: calls are only issued during the 75%-discount off-peak
    window (UTC 16:30 – 00:30). Calls outside this window raise
    DeepSeekPeakHoursError. The orchestrator must reschedule, not retry.

    Usage fields:
      input_tokens             → prompt_tokens (includes both cache hit + miss)
      output_tokens            → completion_tokens
      cache_read_tokens        → prompt_cache_hit_tokens
      cache_creation_tokens    → 0 (DeepSeek caches automatically)
    """

    _DEEPSEEK_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        model_id: str,
        rates: dict[str, float],
        is_reasoning: bool = False,
        thinking_mode: bool = False,
        enforce_off_peak: bool = True,
    ):
        self.model_id        = model_id
        self.rates           = rates
        self.is_reasoning    = is_reasoning
        self.thinking_mode   = thinking_mode
        self.enforce_off_peak = enforce_off_peak
        self._client         = None
        self._ledger: dict   = {}

    def _ensure_client(self):
        if self._client is None:
            import os
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=self._DEEPSEEK_BASE_URL,
            )

    def evaluate(
        self,
        prompt_pairs: list["PromptPair"],
        condition: str = "",
        usage_log_path: Path | None = None,
    ) -> list[V51Result]:
        if self.enforce_off_peak and not is_deepseek_off_peak():
            wait_s = seconds_until_off_peak()
            raise DeepSeekPeakHoursError(
                f"DeepSeek off-peak window opens in {wait_s/60:.1f} minutes "
                f"(16:30 UTC). Reschedule this run."
            )
        self._ensure_client()
        return self._evaluate_sequential(prompt_pairs, condition, usage_log_path)

    def _max_tokens(self) -> int:
        return MAX_TOKENS_REASONING if self.is_reasoning else MAX_TOKENS_STANDARD

    def _call(self, prompt: str) -> dict:
        kwargs: dict = dict(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens(),
        )
        # DeepSeek thinking mode (V4 Flash / V4 Pro when thinking=off)
        if self.thinking_mode is False and not self.is_reasoning:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        response = self._client.chat.completions.create(**kwargs)
        u = response.usage
        cached = getattr(u, "prompt_cache_hit_tokens", 0) or 0
        return {
            "text": response.choices[0].message.content or "",
            "input_tokens":          u.prompt_tokens,
            "output_tokens":         u.completion_tokens,
            "cache_read_tokens":     cached,
            "cache_creation_tokens": 0,
        }

    def _evaluate_sequential(self, pairs, condition, log_path):
        out = []
        for pair in pairs:
            b  = self._call(pair.baseline_prompt)
            time.sleep(RATE_LIMIT_SLEEP_S)
            iv = self._call(pair.intervention_prompt)
            time.sleep(RATE_LIMIT_SLEEP_S)

            b_cost  = _compute_cost(b["input_tokens"], b["output_tokens"],
                                    b["cache_read_tokens"], 0, self.rates, False)
            iv_cost = _compute_cost(iv["input_tokens"], iv["output_tokens"],
                                    iv["cache_read_tokens"], 0, self.rates, False)

            _log_usage(log_path, self.model_id, pair.chain_id, pair.cell,
                       condition, "baseline",
                       b["input_tokens"], b["output_tokens"],
                       b["cache_read_tokens"], 0, b_cost, False)
            _log_usage(log_path, self.model_id, pair.chain_id, pair.cell,
                       condition, "intervention",
                       iv["input_tokens"], iv["output_tokens"],
                       iv["cache_read_tokens"], 0, iv_cost, False)
            b_parsed  = _parse_response(b["text"])
            iv_parsed = _parse_response(iv["text"])
            self._update_ledger(b, b_cost,
                                parsed=b_parsed,
                                needed_retry=b.get("needed_retry", False))
            self._update_ledger(iv, iv_cost,
                                parsed=iv_parsed,
                                needed_retry=iv.get("needed_retry", False))

            out.append(V51Result(
                chain_id=pair.chain_id, cell=pair.cell, condition=condition,
                model_id=self.model_id, resolved_provider="",
                baseline_raw=b["text"], intervention_raw=iv["text"],
                baseline_parsed=b_parsed,
                intervention_parsed=iv_parsed,
                baseline_input_tokens=b["input_tokens"],
                baseline_output_tokens=b["output_tokens"],
                baseline_cache_read_tokens=b["cache_read_tokens"],
                baseline_cache_creation_tokens=0,
                baseline_cost_usd=b_cost,
                intervention_input_tokens=iv["input_tokens"],
                intervention_output_tokens=iv["output_tokens"],
                intervention_cache_read_tokens=iv["cache_read_tokens"],
                intervention_cache_creation_tokens=0,
                intervention_cost_usd=iv_cost,
                baseline_parsed_lenient=_parse_response_lenient(b["text"]),
                intervention_parsed_lenient=_parse_response_lenient(iv["text"]),
                route="deepseek_sync",
            ))
        return out

    def _update_ledger(self, usage: dict, cost_usd: float,
                       parsed: str = "", needed_retry: bool = False) -> None:
        entry = self._ledger.setdefault(self.model_id, {
            "n_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
            "total_cache_read_tokens": 0, "total_cache_creation_tokens": 0,
            "total_cost_usd": 0.0, "max_single_call_cost_usd": 0.0,
            "n_yes": 0, "n_no": 0, "n_abstain": 0, "n_stage2_retries": 0,
        })
        entry["n_calls"]             += 1
        entry["total_input_tokens"]  += usage.get("input_tokens", 0)
        entry["total_output_tokens"] += usage.get("output_tokens", 0)
        entry["total_cache_read_tokens"] += usage.get("cache_read_tokens", 0)
        entry["total_cost_usd"]       = round(entry["total_cost_usd"] + cost_usd, 8)
        entry["max_single_call_cost_usd"] = max(
            entry["max_single_call_cost_usd"], cost_usd)
        if parsed in ("yes", "no", "abstain"):
            entry[f"n_{parsed}"] += 1
        if needed_retry:
            entry["n_stage2_retries"] += 1

    def flush_ledger(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(self._ledger, indent=2))
