"""
v5.1 Cross-Model Replication — main entry point.

Runs 32 models × 8 conditions × 5 cells across all providers.

Modes
-----
  smoke  : 32 models × 8 conditions × 1 cell (CSGO) × 10 chains = 5,120 calls
  full   : 32 models × 8 conditions × 5 cells × full chain set = 320,000 calls

Usage
-----
    # Smoke test (sync, CSGO only, first 10 chain pairs per cell):
    python run_v5_1_replication.py smoke

    # Full run (batch APIs where available):
    python run_v5_1_replication.py full

    # Single model debug:
    python run_v5_1_replication.py smoke --model claude-haiku-4-5-20251001 --dry-run

Options
-------
    --model MODEL_ID    Run only this model (repeatable)
    --cell CELL         Run only this cell (repeatable)
    --condition COND    Run only this condition (repeatable)
    --n-chains N        Override number of chain pairs per cell (default: all)
    --dry-run           Use ModelEvaluator dry-run mode for Anthropic; mock for others
    --output-dir DIR    Override output directory (default: results/v5_1/<date>/)
    --no-kill-switch    Disable per-model cost kill-switch (smoke only)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT    = Path(__file__).parent
_V51_ROOT     = _REPO_ROOT.parent / "Ditto V5.1"
_ENV_PATH     = _REPO_ROOT / ".env"
_PINNING_PATH = _V51_ROOT / "provider_pinning.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v5_1_replication")


# ---------------------------------------------------------------------------
# Model panel (SPEC §4.1)
# ---------------------------------------------------------------------------

# NOTE 2026-05-01: dropped 4 most-expensive models to fit $189 budget.
# Removed: claude-opus-4-7, gpt-5.4, gpt-5.5, x-ai/grok-4
# Estimated full-run cost reduction: $479 → $149.
# Within-family std-vs-reasoning pairs preserved via:
#   OpenAI (gpt-5 vs gpt-5.4-mini), MoonshotAI (kimi-k2.6 vs kimi-k2-thinking),
#   DeepSeek (v3.2 vs r1).
# Lost: xAI within-family reasoning comparison; Anthropic Opus tier.
ALL_MODELS: dict[str, str] = {
    # Anthropic
    "claude-haiku-4-5-20251001": "anthropic",
    "claude-sonnet-4-5":         "anthropic",
    "claude-sonnet-4-6":         "anthropic",
    # OpenAI
    "gpt-5-mini":   "openai",
    "gpt-5":        "openai",
    "gpt-5.4-mini": "openai",
    # Google
    "gemini-3.1-flash-lite-preview": "google",
    "gemini-2.5-flash":              "google",
    # gemini-2.5-pro DROPPED 2026-05-01: Google docs explicitly state
    # "Cannot disable thinking" for 2.5 Pro. Empirically 96% empty content
    # over 160 calls — model exhausts thinking budget without producing
    # visible output. Same exclusion criterion as minimax/minimax-m2.5
    # (mandatory reasoning incompatible with binary-classification regime).
    # gemini-3.1-pro-preview retained — its thinking IS controllable.
    "gemini-3-flash-preview":        "google",
    "gemini-3.1-pro-preview":        "google",
    # DeepSeek (off-peak only) — updated 2026-05-01: API now serves only
    # `deepseek-v4-pro` and `deepseek-v4-flash`. Old IDs (v3.2, v4, r1)
    # return 400 "invalid_request_error". Lost: explicit reasoning slot
    # (deepseek-r1). v4-pro / v4-flash both support thinking_mode toggle.
    "deepseek-v4-pro":   "deepseek",
    "deepseek-v4-flash": "deepseek",
    # OpenRouter
    "x-ai/grok-4-fast":                    "openrouter",
    "x-ai/grok-4.20":                      "openrouter",
    "meta-llama/llama-3.3-70b-instruct":   "openrouter",
    # llama-4-maverick DROPPED 2026-05-01: model produces verbose prose
    # preface ("To determine if the sequence...") instead of YES/NO and
    # has no documented format-control parameter. 32% parse rate (strict
    # AND lenient — answer never appears in first line of output).
    "xiaomi/mimo-v2-pro":                  "openrouter",
    "xiaomi/mimo-v2.5-pro":               "openrouter",
    "qwen/qwen3.6-plus":                   "openrouter",
    "qwen/qwen3.6-max-preview":            "openrouter",
    "z-ai/glm-4.7":                        "openrouter",
    "z-ai/glm-5":                          "openrouter",
    # minimax/minimax-m2.5 DROPPED 2026-05-01: provider explicitly rejects
    # `reasoning: {enabled: false}` — "Reasoning is mandatory for this endpoint
    # and cannot be disabled." Mandatory reasoning produces empty content under
    # our prompt class and any cap below ~3K tokens. Untenable for budget.
    # moonshotai/kimi-k2-thinking DROPPED 2026-05-01: model genuinely needs
    # 1.7K-18K output tokens per call — Novita doesn't honor max_tokens=384.
    # Per-model cost extrapolates to ~$55, well over the $7.88 per-model budget
    # (would trip $9.04 kill-switch mid-run). Not exposed via direct Moonshot.
    # Same exclusion rationale as gemini-2.5-pro / minimax (mandatory reasoning
    # incompatible with budget envelope). MoonshotAI represented by kimi-k2.6.
    # Kimi K2.6 routed direct to Moonshot AI — needs the `thinking:disabled`
    # extra_body parameter that OpenRouter strips. Without it, kimi-k2.6
    # produces empty content (verified 2026-05-01 across Novita, Moonshot AI,
    # and DeepInfra via OR — all produce empty).
    "kimi-k2.6":                           "moonshot",
}

ALL_CELLS    = ["poker", "pubg", "nba", "csgo", "rocket_league"]
SMOKE_CELL   = "csgo"
SMOKE_N_CHAINS = 10   # 10 × 8 conditions × 22 models × 2 calls = 3,520 total

# Per-cell chain lengths (used by FixedPerCellChainBuilder)
CHAIN_LENGTHS = {
    "pubg": 8, "nba": 5, "csgo": 10, "rocket_league": 12, "poker": 8,
}


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------

def load_env(env_path: Path) -> None:
    """Load .env key=value pairs into os.environ (no overwrite of real values)."""
    import os
    if not env_path.exists():
        logger.warning(f".env not found at {env_path}; relying on shell environment")
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip(); v = v.strip()
        if k and v and not os.environ.get(k, "").strip():
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Chain pair generation — runs v5's pipeline + violation injector
# ---------------------------------------------------------------------------

def build_chain_pairs_for_cell(
    cell: str,
    n_chains: int | None,
    force_mock: bool = False,
) -> list[tuple]:
    """
    Run v5's data pipeline + T + violation injector for one cell.

    Returns a list of (clean_chain, adv_chain) ChainCandidate pairs,
    filtered to chains where injection succeeded.
    """
    from src.cells.csgo.pipeline import CSGOPipeline
    from src.cells.nba.pipeline import NBAPipeline
    from src.cells.poker.pipeline import PokerPipeline
    from src.cells.pubg.pipeline import PUBGPipeline
    from src.cells.rocket_league.pipeline import RocketLeaguePipeline
    from src.common.config import load_cell_configs, load_harness_config
    from src.harness.actionables import compute_retention_rate
    from src.harness.violation_injector import INJECTORS
    from src.interfaces.chain_builder import FixedPerCellChainBuilder
    from src.interfaces.translation import DOMAIN_T_STUBS

    pipeline_map = {
        "pubg": PUBGPipeline,
        "nba": NBAPipeline,
        "csgo": CSGOPipeline,
        "rocket_league": RocketLeaguePipeline,
        "poker": PokerPipeline,
    }

    cell_configs    = load_cell_configs()
    harness_config  = load_harness_config()
    config = cell_configs.get(cell)
    if config is None:
        raise RuntimeError(f"No config for cell '{cell}'")

    pipeline_cls = pipeline_map.get(cell)
    if pipeline_cls is None:
        raise RuntimeError(f"Unknown cell '{cell}'")

    pipeline = pipeline_cls(config=config)
    streams = pipeline.run(force_mock=force_mock)
    logger.info(f"[{cell}] {len(streams)} streams fetched")

    t_fn = DOMAIN_T_STUBS.get(cell)
    if t_fn is None:
        raise RuntimeError(f"No T function for cell '{cell}'")

    chain_builder = FixedPerCellChainBuilder(
        per_cell_chain_length={cell: CHAIN_LENGTHS.get(cell, 8)}
    )
    candidates = []
    for stream in streams:
        candidates.extend(t_fn.translate(stream))

    chains = chain_builder.build_from_candidates(candidates, cell=cell)
    # compute_retention_rate is a side-effectful call that sets chain.is_actionable
    compute_retention_rate(chains, floor=harness_config.gate2_retention_floor)
    chains_passed = [c for c in chains if c.is_actionable]
    logger.info(f"[{cell}] {len(chains_passed)} chains passed Gate 2")

    if n_chains is not None:
        chains_passed = chains_passed[:n_chains]

    injector = INJECTORS.get(cell)
    if injector is None:
        raise RuntimeError(f"No violation injector for cell '{cell}'")

    pairs = []
    n_failed = 0
    for clean_chain in chains_passed:
        result = injector(clean_chain)
        if result is None or result.chain is None:
            n_failed += 1
            continue
        pairs.append((clean_chain, result.chain))

    if n_failed:
        logger.warning(f"[{cell}] {n_failed} chains had injection failures (skipped)")
    logger.info(f"[{cell}] {len(pairs)} valid (clean, adv) chain pairs")
    return pairs


def build_prompt_pairs_by_condition(
    chain_pairs: list[tuple],
    cell: str,
) -> dict[tuple[str, str], list]:
    """
    Apply v5.1's 8-condition variant generator to all (clean, adv) chain pairs.

    Returns dict keyed by (condition_label, cell) → list[PromptPair].
    Each PromptPair's:
      baseline_prompt    = clean chain under the condition's context level
      intervention_prompt = adv chain under the condition's context level
    """
    from src.harness.prompt_variants import CONDITION_LABELS, build_eight_variants
    from src.harness.prompts import PER_CELL_PROMPT_BUILDERS

    builder_cls = PER_CELL_PROMPT_BUILDERS.get(cell)
    if builder_cls is None:
        raise RuntimeError(f"No PromptBuilder for cell '{cell}'")
    prompt_builder = builder_cls()

    out: dict[tuple[str, str], list] = {
        (cond, cell): [] for cond in CONDITION_LABELS
    }

    for clean_chain, adv_chain in chain_pairs:
        try:
            variants = build_eight_variants(clean_chain, adv_chain, prompt_builder)
        except Exception as e:
            logger.warning(
                f"[{cell}] build_eight_variants failed for {clean_chain.chain_id}: {e}"
            )
            continue
        for cond, pair in variants.items():
            out[(cond, cell)].append(pair)

    total = sum(len(v) for v in out.values())
    logger.info(f"[{cell}] Built {total} condition × chain prompt pairs "
                f"({len(chain_pairs)} chain pairs × 8 conditions)")
    return out


# ---------------------------------------------------------------------------
# Results writer
# ---------------------------------------------------------------------------

def write_results(results_by_model: dict, output_dir: Path) -> None:
    import dataclasses
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for model_id, results in results_by_model.items():
        safe = model_id.replace("/", "_").replace(":", "_")
        out_path = raw_dir / f"{safe}_results.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(dataclasses.asdict(r)) + "\n")
        logger.info(f"Wrote {len(results)} results for {model_id} → {out_path}")


def write_smoke_report(results_by_model: dict, output_dir: Path) -> None:
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_models": len(results_by_model),
        "models": {},
    }
    for model_id, results in results_by_model.items():
        total = len(results)
        if total == 0:
            report["models"][model_id] = {"n_results": 0}
            continue
        by_condition: dict[str, dict] = {}
        for r in results:
            cond = r.condition
            entry = by_condition.setdefault(cond, {
                "total": 0,
                "b_yes": 0, "b_no": 0, "b_abstain": 0,
                "iv_yes": 0, "iv_no": 0, "iv_abstain": 0,
                "b_lenient_yes": 0, "b_lenient_no": 0, "b_lenient_abstain": 0,
                "iv_lenient_yes": 0, "iv_lenient_no": 0, "iv_lenient_abstain": 0,
                "needed_retry_count": 0,
                "total_cost_usd": 0.0,
            })
            entry["total"] += 1
            # Defensive: if parsed value isn't one of yes/no/abstain
            # (e.g., empty string from a runner that pre-dates the strict
            # parser), bucket as abstain rather than KeyError.
            b_strict  = r.baseline_parsed if r.baseline_parsed in ("yes", "no", "abstain") else "abstain"
            iv_strict = r.intervention_parsed if r.intervention_parsed in ("yes", "no", "abstain") else "abstain"
            entry[f"b_{b_strict}"]   += 1
            entry[f"iv_{iv_strict}"] += 1
            # Lenient parse (added 2026-05-01 per second-AI consultation):
            # separates "wrong format" from "actually abstained/refused".
            b_len_raw = getattr(r, "baseline_parsed_lenient", "") or "abstain"
            iv_len_raw = getattr(r, "intervention_parsed_lenient", "") or "abstain"
            b_len  = b_len_raw if b_len_raw in ("yes", "no", "abstain") else "abstain"
            iv_len = iv_len_raw if iv_len_raw in ("yes", "no", "abstain") else "abstain"
            entry[f"b_lenient_{b_len}"]   += 1
            entry[f"iv_lenient_{iv_len}"] += 1
            # Stage-2 retry tracking — exposes the asymmetric second chance.
            if getattr(r, "baseline_needed_retry", False):
                entry["needed_retry_count"] += 1
            if getattr(r, "intervention_needed_retry", False):
                entry["needed_retry_count"] += 1
            entry["total_cost_usd"] += r.baseline_cost_usd + r.intervention_cost_usd

        parse_rates = {}
        for cond, e in by_condition.items():
            n = e["total"]
            parse_rates[cond] = {
                "baseline_parse_rate":         round(1 - e["b_abstain"]  / n, 4) if n else 0,
                "intervention_parse_rate":     round(1 - e["iv_abstain"] / n, 4) if n else 0,
                "baseline_parse_rate_lenient":     round(1 - e["b_lenient_abstain"]  / n, 4) if n else 0,
                "intervention_parse_rate_lenient": round(1 - e["iv_lenient_abstain"] / n, 4) if n else 0,
                "needed_retry_rate":           round(e["needed_retry_count"] / (2 * n), 4) if n else 0,
            }
        total_cost = sum(r.baseline_cost_usd + r.intervention_cost_usd for r in results)
        resolved_providers = list({r.resolved_provider for r in results if r.resolved_provider})
        # Aggregate route distribution for routing-invariance analysis
        routes = {}
        for r in results:
            route = getattr(r, "route", "") or "unknown"
            routes[route] = routes.get(route, 0) + 1
        report["models"][model_id] = {
            "n_results": total,
            "total_cost_usd": round(total_cost, 6),
            "parse_rates_by_condition": parse_rates,
            "resolved_providers": resolved_providers,
            "route_distribution": routes,
        }

    smoke_dir = output_dir / "smoke_results"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = smoke_dir / f"v5_1_smoke_{date_str}.json"
    out_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Smoke report → {out_path}")
    total_cost = sum(m.get("total_cost_usd", 0) for m in report["models"].values())
    logger.info(f"Smoke total cost: ${total_cost:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="v5.1 Cross-Model Replication")
    parser.add_argument("mode", choices=["smoke", "full"])
    parser.add_argument("--model",      action="append", dest="models")
    parser.add_argument("--cell",       action="append", dest="cells")
    parser.add_argument("--condition",  action="append", dest="conditions")
    parser.add_argument("--n-chains",   type=int, default=None)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--force-mock", action="store_true",
                        help="Use mock data instead of real API data for pipelines")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-kill-switch", action="store_true")
    args = parser.parse_args()

    load_env(_ENV_PATH)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        date_str   = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_dir = _REPO_ROOT / "results" / "v5_1" / f"{args.mode}_{date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_dir}")

    # Models
    if args.models:
        models  = [m for m in args.models if m in ALL_MODELS]
        unknown = [m for m in args.models if m not in ALL_MODELS]
        if unknown:
            logger.warning(f"Unknown model IDs (skipped): {unknown}")
    else:
        models = list(ALL_MODELS.keys())

    # Cells + n_chains
    if args.mode == "smoke":
        cells    = args.cells or [SMOKE_CELL]
        n_chains = args.n_chains if args.n_chains is not None else SMOKE_N_CHAINS
        is_smoke = True
    else:
        cells    = args.cells or ALL_CELLS
        n_chains = args.n_chains
        is_smoke = False

    logger.info(f"Mode={args.mode}  models={len(models)}  cells={cells}  "
                f"n_chains={n_chains or 'all'}")

    # Load provider pinning
    if not _PINNING_PATH.exists():
        logger.error(f"provider_pinning.json not found at {_PINNING_PATH}")
        sys.exit(1)
    provider_pinning = json.loads(_PINNING_PATH.read_text())

    # Build chain pairs + 8-condition prompt pairs
    prompt_pairs_by_condition_cell: dict[tuple[str, str], list] = {}

    for cell in cells:
        logger.info(f"[{cell}] Building chain pairs via v5 pipeline + injector...")
        try:
            chain_pairs = build_chain_pairs_for_cell(
                cell=cell,
                n_chains=n_chains,
                force_mock=args.force_mock or args.dry_run,
            )
        except Exception as e:
            logger.error(f"[{cell}] Chain pair build failed: {e}")
            sys.exit(1)

        condition_pairs = build_prompt_pairs_by_condition(chain_pairs, cell)

        if args.conditions:
            condition_pairs = {k: v for k, v in condition_pairs.items()
                               if k[0] in args.conditions}

        prompt_pairs_by_condition_cell.update(condition_pairs)

    n_total = sum(len(v) for v in prompt_pairs_by_condition_cell.values())
    logger.info(f"Total prompt pairs: {n_total}  "
                f"(×2 API calls each = {n_total*2} calls per model)")

    if n_total == 0:
        logger.error("No prompt pairs generated — check pipeline + injection logs above")
        sys.exit(1)

    # Orchestrate
    from src.harness.runner_orchestrator import RunnerOrchestrator

    orch = RunnerOrchestrator(
        provider_pinning=provider_pinning,
        output_dir=output_dir,
        per_model_budget=None,  # use default 110% of mean allocation
        dry_run=args.dry_run,
    )

    logger.info("Starting evaluation run...")
    results_by_model = orch.run(
        prompt_pairs_by_condition_cell=prompt_pairs_by_condition_cell,
        models=models,
        is_smoke=is_smoke,
    )

    write_results(results_by_model, output_dir)
    if is_smoke:
        write_smoke_report(results_by_model, output_dir)

    total_results = sum(len(v) for v in results_by_model.values())
    logger.info(f"Done. {total_results} results across {len(results_by_model)} models.")


if __name__ == "__main__":
    main()
