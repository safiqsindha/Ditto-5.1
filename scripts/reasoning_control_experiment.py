#!/usr/bin/env python3
"""
Reasoning ON-vs-OFF control experiment (per second-AI consultation 2026-05-01).

Question: does extra_body={"reasoning":{"enabled":False}} change the
classification answer for the same prompt, or is it a pure cost/latency
optimization?

Procedure:
  - 4 OR models (with reasoning toggle):
      qwen-plus, glm-4.7, mimo-v2-pro, llama-3.3-70b
  - For each model, sample N chains × 2 conditions (ON, OFF) × 2 calls
    (baseline, intervention). All ON calls run concurrently across chains;
    all OFF calls run concurrently. Per-call timeout enforced.
  - Same prompts in both conditions.
  - Compute per-model agreement rate between ON and OFF answers.

Decision rule (per second-AI):
  - agreement >= 95% → flag is a defensible cost optimization
  - agreement <  95% → flag is an experimental condition; must be disclosed

Output: JSON report at results/v5_1/reasoning_control_<timestamp>.json

Usage:
  python3 -u scripts/reasoning_control_experiment.py [n_chains]

Notes (lessons from first attempt that hung 25+ min):
  - Use `python3 -u` or set PYTHONUNBUFFERED=1 to see progress live
  - Per-call timeout (default 60s) prevents hung requests from blocking the run
  - ThreadPoolExecutor parallelizes across (chain × condition) tuples
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Force unbuffered stdout so progress is visible when running in background
sys.stdout.reconfigure(line_buffering=True)  # py 3.7+

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai import OpenAI

from src.harness.runner_native import (
    _parse_response,
    _parse_response_lenient,
)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if k and v and not os.environ.get(k, "").strip():
            os.environ[k] = v


load_env(ROOT / ".env")

# Per-call timeout — important to prevent hung requests from blocking forever.
# 90s is generous for reasoning-ON which can produce 1000+ tokens at slow tps.
PER_CALL_TIMEOUT_S = 90.0

# Concurrent workers per model. With reasoning ON taking 10-20s per call and
# OFF taking 1-2s, 8 concurrent gives a good balance and stays well within
# OR rate limits for a one-off diagnostic.
WORKERS_PER_MODEL = 8

OR_CLIENT = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    timeout=PER_CALL_TIMEOUT_S,
)
OPENAI_CLIENT = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    timeout=PER_CALL_TIMEOUT_S,
)
MOONSHOT_CLIENT = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.ai/v1",
    timeout=PER_CALL_TIMEOUT_S,
)

# All 13 flag-bearing panel models. Anthropic (3), Google (4), DeepSeek (2)
# are excluded by design — they expose no reasoning-disable parameter on
# their public APIs, so ON/OFF testing is undefined for them. The §8.4
# regression in the prereg restricts to flag_models_only for this reason.
#
# Each entry: (model_id, route, route_param)
#   route ∈ {"openrouter", "openai_direct", "moonshot_direct"}
#   route_param: provider name for OR; ignored for direct routes
MODELS = [
    # OpenRouter (9 models with reasoning:enabled toggle)
    ("qwen/qwen3.6-plus",                  "openrouter", "Alibaba"),
    ("qwen/qwen3.6-max-preview",           "openrouter", "Alibaba"),
    ("z-ai/glm-4.7",                       "openrouter", "Cerebras"),
    ("z-ai/glm-5",                         "openrouter", "Baidu"),
    ("xiaomi/mimo-v2-pro",                 "openrouter", "Xiaomi"),
    ("xiaomi/mimo-v2.5-pro",               "openrouter", "Xiaomi"),
    ("meta-llama/llama-3.3-70b-instruct",  "openrouter", "WandB"),
    ("x-ai/grok-4-fast",                   "openrouter", "xAI"),
    ("x-ai/grok-4.20",                     "openrouter", "xAI"),  # reasoning model — ON is default

    # OpenAI direct (3 models with reasoning_effort=minimal)
    ("gpt-5",                              "openai_direct",   None),
    ("gpt-5-mini",                         "openai_direct",   None),
    ("gpt-5.4-mini",                       "openai_direct",   None),  # reasoning model — minimal vs default

    # Moonshot direct (1 model with thinking:type)
    ("kimi-k2.6",                          "moonshot_direct", None),
]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """## Event Chain
{chain}

## Question
Is the sequence of events above consistent with the rules of Counter-Strike?
Answer with exactly one token:
YES
NO
If you output anything else, the answer is invalid."""

CHAIN_VALID = """  0. t=    0.0s | round_start             | actor=Player_0     | action=spawn, team=team_A, round=1, map=de_dust2
  1. t=   12.4s | weapon_pickup           | actor=Player_0     | action=ak47, team=team_A, round=1, map=de_dust2
  2. t=   45.2s | bomb_plant              | actor=Player_3     | action=plant_a, team=team_A, round=1, map=de_dust2
  3. t=   78.9s | kill                    | actor=Player_0     | action=ak47, team=team_A, round=1, map=de_dust2
  4. t=   95.1s | round_end               | actor=Player_3     | action=t_win, team=team_A, round=1, map=de_dust2"""

CHAIN_BROKEN = """  0. t=    0.0s | round_start             | actor=Player_0     | action=spawn, team=team_A, round=1, map=de_dust2
  1. t=    1.0s | bomb_plant              | actor=Player_0     | action=plant_at_spawn, team=team_A, round=1, map=de_dust2
  2. t=    2.0s | bomb_plant              | actor=Player_0     | action=plant_third_site_C, team=team_A, round=1, map=de_dust2
  3. t=    3.0s | kill                    | actor=Player_DEAD  | action=knife, team=team_A, round=1, map=de_dust2
  4. t=    4.0s | round_end               | actor=Player_0     | action=t_win, team=team_A, round=1, map=de_dust2"""


def build_prompt_pairs(n: int) -> list[tuple[str, str]]:
    out = []
    for i in range(n):
        if i % 2 == 0:
            label, chain = "valid", CHAIN_VALID
        else:
            label, chain = "broken", CHAIN_BROKEN
        out.append((label, PROMPT_TEMPLATE.format(chain=chain)))
    return out


# ---------------------------------------------------------------------------
# Single call (with explicit timeout + error capture)
# ---------------------------------------------------------------------------

def call_with_reasoning(
    model_id: str,
    route: str,
    route_param: str | None,
    prompt: str,
    reasoning_enabled: bool,
) -> dict:
    """Single call with reasoning ON or OFF, dispatched per route.

    For each route the OFF condition uses the route's natural disable
    parameter; the ON condition omits the parameter (provider default).
    """
    t0 = time.monotonic()
    try:
        if route == "openrouter":
            extra_body = {
                "provider": {"order": [route_param], "allow_fallbacks": False},
                "usage": {"include": True},
            }
            if not reasoning_enabled:
                extra_body["reasoning"] = {"enabled": False}
            r = OR_CLIENT.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                extra_body=extra_body,
                timeout=PER_CALL_TIMEOUT_S,
            )
        elif route == "openai_direct":
            kwargs = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 1024,
                "timeout": PER_CALL_TIMEOUT_S,
            }
            if not reasoning_enabled:
                kwargs["reasoning_effort"] = "minimal"
            r = OPENAI_CLIENT.chat.completions.create(**kwargs)
        elif route == "moonshot_direct":
            kwargs = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "timeout": PER_CALL_TIMEOUT_S,
            }
            if not reasoning_enabled:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            r = MOONSHOT_CLIENT.chat.completions.create(**kwargs)
        else:
            raise ValueError(f"Unknown route: {route}")
        elapsed = time.monotonic() - t0
        c = r.choices[0]
        text = (c.message.content or "")
        return {
            "text":           text,
            "parsed_strict":  _parse_response(text),
            "parsed_lenient": _parse_response_lenient(text),
            "finish_reason":  c.finish_reason,
            "tokens_in":      r.usage.prompt_tokens,
            "tokens_out":     r.usage.completion_tokens,
            "elapsed_s":      round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.monotonic() - t0
        return {
            "error":          f"{type(e).__name__}: {str(e)[:160]}",
            "parsed_strict":  "abstain",
            "parsed_lenient": "abstain",
            "elapsed_s":      round(elapsed, 2),
        }


# ---------------------------------------------------------------------------
# Per-model parallel run
# ---------------------------------------------------------------------------

def run_one_chain(model_id: str, route: str, route_param: str | None,
                  idx: int, label: str, prompt: str) -> dict:
    """Two calls (ON, OFF) for a single chain."""
    on  = call_with_reasoning(model_id, route, route_param, prompt, reasoning_enabled=True)
    off = call_with_reasoning(model_id, route, route_param, prompt, reasoning_enabled=False)
    return {"idx": idx, "label": label, "on": on, "off": off}


def run_model(model_id: str, route: str, route_param: str | None, n_chains: int) -> dict:
    prompts = build_prompt_pairs(n_chains)
    print(f"\n[{model_id}] {n_chains} chains × 2 conditions = {2*n_chains} calls "
          f"(parallel workers={WORKERS_PER_MODEL})", flush=True)

    samples: list[dict] = []
    completed = 0
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=WORKERS_PER_MODEL) as ex:
        futures = {
            ex.submit(run_one_chain, model_id, route, route_param, i, label, prompt): i
            for i, (label, prompt) in enumerate(prompts)
        }
        for fut in as_completed(futures):
            try:
                s = fut.result()
                samples.append(s)
            except Exception as e:
                print(f"  chain {futures[fut]} crashed: {e}", flush=True)
            completed += 1
            if completed % 5 == 0 or completed == n_chains:
                elapsed = time.monotonic() - t0
                print(f"  {completed}/{n_chains} done ({elapsed:.0f}s)", flush=True)

    samples.sort(key=lambda s: s["idx"])

    # Metrics
    n = len(samples)
    strict_agree = sum(1 for s in samples if s["on"]["parsed_strict"] == s["off"]["parsed_strict"])
    lenient_agree = sum(1 for s in samples if s["on"]["parsed_lenient"] == s["off"]["parsed_lenient"])
    on_strict_dist  = Counter(s["on"]["parsed_strict"] for s in samples)
    off_strict_dist = Counter(s["off"]["parsed_strict"] for s in samples)

    on_lat  = [s["on"]["elapsed_s"]  for s in samples if "elapsed_s" in s["on"]]
    off_lat = [s["off"]["elapsed_s"] for s in samples if "elapsed_s" in s["off"]]
    on_tok  = [s["on"]["tokens_out"]  for s in samples if "tokens_out" in s["on"]]
    off_tok = [s["off"]["tokens_out"] for s in samples if "tokens_out" in s["off"]]

    n_on_err  = sum(1 for s in samples if "error" in s["on"])
    n_off_err = sum(1 for s in samples if "error" in s["off"])

    def med(xs):
        return round(sorted(xs)[len(xs)//2], 2) if xs else None

    return {
        "model_id":              model_id,
        "route":                 route,
        "route_param":           route_param,
        "n":                     n,
        "n_errors_on":           n_on_err,
        "n_errors_off":          n_off_err,
        "strict_agreement":      round(strict_agree  / n, 4) if n else 0,
        "lenient_agreement":     round(lenient_agree / n, 4) if n else 0,
        "on_distribution":       dict(on_strict_dist),
        "off_distribution":      dict(off_strict_dist),
        "median_latency_on_s":   med(on_lat),
        "median_latency_off_s":  med(off_lat),
        "median_tokens_on":      med(on_tok),
        "median_tokens_off":     med(off_tok),
        "samples":               samples,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_chains: int = 25) -> None:
    print(f"Reasoning ON-vs-OFF control: {len(MODELS)} models × {n_chains} chains × 2 conditions",
          flush=True)
    print(f"Total calls: {len(MODELS) * n_chains * 2}  |  per-call timeout: {PER_CALL_TIMEOUT_S}s",
          flush=True)

    results = []
    t_start = time.monotonic()
    for model_id, route, route_param in MODELS:
        try:
            r = run_model(model_id, route, route_param, n_chains)
            results.append(r)
            print(f"  → strict_agreement={r['strict_agreement']:.0%}  "
                  f"latency ON/OFF={r['median_latency_on_s']}s/{r['median_latency_off_s']}s  "
                  f"tokens ON/OFF={r['median_tokens_on']}/{r['median_tokens_off']}  "
                  f"errors ON/OFF={r['n_errors_on']}/{r['n_errors_off']}", flush=True)
        except Exception as e:
            print(f"  [{model_id}] FAILED: {e}", flush=True)
            results.append({"model_id": model_id, "error": str(e)})

    elapsed = time.monotonic() - t_start
    out_dir = ROOT / "results" / "v5_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"reasoning_control_{ts}.json"
    out_path.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_chains_per_model": n_chains,
        "elapsed_s": round(elapsed, 1),
        "models": results,
    }, indent=2))

    # Decision summary
    print()
    print("=" * 72, flush=True)
    print("DECISION SUMMARY (per second-AI: ≥95% strict agreement = optimization)",
          flush=True)
    print("=" * 72, flush=True)
    for r in results:
        if "error" in r:
            print(f"  {r['model_id']:42s}  ERROR: {r['error'][:60]}", flush=True)
            continue
        agree = r["strict_agreement"]
        flag = "OPTIMIZATION ✓" if agree >= 0.95 else "EXPERIMENTAL CONDITION ⚠"
        print(f"  {r['model_id']:42s}  strict={agree:.0%}  →  {flag}", flush=True)
    print(flush=True)
    print(f"Report → {out_path}", flush=True)
    print(f"Total wall time: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    main(n_chains=n)
