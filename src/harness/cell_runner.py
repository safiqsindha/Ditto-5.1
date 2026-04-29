"""
Cell runner for v5 — orchestrates parallel cell evaluation.

Accepts a TranslationFunction (T) for each cell and runs the full
McNemar pipeline: event stream → chains (via T) → Gate 2 filter →
scoring → McNemar test → cell-level + aggregate report.

T is injected as a pluggable dependency. Until T implementations
are built and pre-registered, use the MockT from src/pilot/mock_t.py.

NOTE: No real Haiku API calls are made in this module. The `evaluate_chains`
method is a stub that requires a real ModelEvaluator to be injected.
"""

from __future__ import annotations

import datetime
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..common.config import HarnessConfig, load_harness_config
from ..common.schema import ChainCandidate, EventStream
from ..interfaces.chain_builder import ChainBuilder
from ..interfaces.translation import TranslationFunction
from .actionables import compute_retention_rate
from .mcnemar import McnemarResult, aggregate_results, run_mcnemar
from .scoring import extract_binary_vectors, score_batch
from .variance import mcnemar_power, minimum_detectable_effect, variance_summary


@dataclass
class CellResult:
    """Complete result for one cell's evaluation run."""
    cell: str
    n_events_total: int
    n_chains_pre_gate2: int
    n_chains_post_gate2: int
    retention_rate: float
    gate2_pass: bool
    mcnemar: McnemarResult | None
    variance_baseline: dict | None
    variance_intervention: dict | None
    power: float | None
    mde: float | None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.mcnemar:
            d["mcnemar_summary"] = self.mcnemar.summary()
        return d


@dataclass
class RunReport:
    """Aggregate report across all cells."""
    run_id: str
    timestamp: str
    config: dict
    cells: list[CellResult]
    aggregate: dict

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": self.config,
            "cells": [c.to_dict() for c in self.cells],
            "aggregate": self.aggregate,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class CellRunner:
    """
    Orchestrates evaluation across all five v5 cells.

    Usage
    -----
    runner = CellRunner(config=harness_config)
    runner.register_cell("fortnite", translation_fn=MyFortniteT())
    runner.register_cell("nba", translation_fn=MockT(cell="nba"))
    ...
    report = runner.run(event_streams_by_cell)
    """

    def __init__(
        self,
        config: HarnessConfig | None = None,
        chain_builder: ChainBuilder | None = None,
    ):
        self.config = config or load_harness_config()
        self.chain_builder = chain_builder
        self._cell_translations: dict[str, TranslationFunction] = {}

    def register_cell(self, cell: str, translation_fn: TranslationFunction) -> None:
        self._cell_translations[cell] = translation_fn

    def set_chain_builder(self, chain_builder: ChainBuilder) -> None:
        """Wire a ChainBuilder so T-output candidates flow through chain construction."""
        self.chain_builder = chain_builder

    def run(
        self,
        event_streams: dict[str, list[EventStream]],
        baseline_responses: dict[str, list[str]] | None = None,
        intervention_responses: dict[str, list[str]] | None = None,
        ground_truths: dict[str, list[str]] | None = None,
        n_per_cell: int | None = None,
    ) -> RunReport:
        """
        Run the full pipeline for all registered cells.

        Parameters
        ----------
        event_streams         : {cell: [EventStream, ...]}
        baseline_responses    : {cell: [model_response_str, ...]}  (per chain)
        intervention_responses: {cell: [model_response_str, ...]}
        ground_truths         : {cell: [ground_truth_str, ...]}
        n_per_cell            : optional cap on chains per cell after Gate 2.
                                Used by run_eval --n-per-cell for pilots; must
                                match the cap used to generate response lists
                                or the H1 pairing assertion will fire.

        When baseline/intervention/ground_truth are None, scoring is skipped
        (useful for infrastructure validation without real model calls).
        """
        run_id = f"v5_run_{int(time.time())}"
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        cell_results = []
        mcnemar_results = []

        # Iterate over the union of registered cells and cells with streams.
        # This way: registered-but-streamless cells get a CellResult with errors,
        # streams-but-unregistered cells get a CellResult flagging the missing T.
        cells_to_run = sorted(set(self._cell_translations) | set(event_streams))

        for cell in cells_to_run:
            streams = event_streams.get(cell, [])
            t_fn = self._cell_translations.get(cell)

            result = self._run_cell(
                cell=cell,
                streams=streams,
                translation_fn=t_fn,
                baseline_responses=(baseline_responses or {}).get(cell),
                intervention_responses=(intervention_responses or {}).get(cell),
                ground_truths=(ground_truths or {}).get(cell),
                n_per_cell=n_per_cell,
            )
            cell_results.append(result)
            if result.mcnemar:
                mcnemar_results.append(result.mcnemar)

        agg = aggregate_results(mcnemar_results) if mcnemar_results else {}

        # Capture full config including ChainBuilder per-cell lengths (A2 fix)
        full_config = asdict(self.config)
        if self.chain_builder is not None:
            full_config["chain_builder"] = {
                "class": type(self.chain_builder).__name__,
                "per_cell_chain_length": getattr(
                    self.chain_builder, "per_cell_chain_length", {}
                ),
                "overlap": getattr(self.chain_builder, "overlap", None),
            }

        return RunReport(
            run_id=run_id,
            timestamp=ts,
            config=full_config,
            cells=cell_results,
            aggregate=agg,
        )

    def _run_cell(
        self,
        cell: str,
        streams: list[EventStream],
        translation_fn: TranslationFunction | None,
        baseline_responses: list[str] | None,
        intervention_responses: list[str] | None,
        ground_truths: list[str] | None,
        n_per_cell: int | None = None,
    ) -> CellResult:
        errors = []
        n_events = sum(len(s) for s in streams)

        # Step 1: Translate event streams to chain candidates via T
        candidates: list[ChainCandidate] = []
        if translation_fn is None:
            errors.append(f"No TranslationFunction registered for cell '{cell}'")
        else:
            try:
                for stream in streams:
                    candidates.extend(translation_fn.translate(stream))
            except NotImplementedError:
                errors.append(f"T not implemented for cell '{cell}' (stub)")
            except Exception as e:
                errors.append(f"T raised exception for cell '{cell}': {e}")

        # Step 1b: ChainBuilder enforces per-cell chain length (Q4-B) — A1 fix
        if self.chain_builder is not None and candidates:
            try:
                chains: list[ChainCandidate] = self.chain_builder.build_from_candidates(
                    candidates, cell=cell
                )
            except (ValueError, TypeError) as e:
                errors.append(f"ChainBuilder error for cell '{cell}': {e}")
                chains = candidates  # fall through with raw candidates
        else:
            chains = candidates

        n_pre_gate2 = len(chains)

        # Step 2: Gate 2 filter
        retention_info = compute_retention_rate(
            chains, floor=self.config.gate2_retention_floor
        )
        chains_passed = [c for c in chains if c.is_actionable]

        # If a per-cell cap is set (e.g. for the real-Haiku pilot), trim here
        # so the chain count matches what was sent to the model upstream.
        # Trim BEFORE capturing n_post_gate2 so the McNemar result reflects
        # the actual scored count, not the upstream pre-cap count.
        if n_per_cell is not None and len(chains_passed) > n_per_cell:
            chains_passed = chains_passed[:n_per_cell]
        n_post_gate2 = len(chains_passed)

        # Step 3: Scoring (skipped if no model responses provided)
        mcnemar_result = None
        var_baseline = None
        var_intervention = None
        power = None
        mde_val = None

        if baseline_responses and intervention_responses and ground_truths:
            # Pairing contract: baseline_responses[i], intervention_responses[i],
            # and ground_truths[i] correspond to chains_passed[i]. The current
            # ModelEvaluator is sequential so this holds today, but if batching
            # is introduced, response lists could be reordered. Assert the
            # lengths match exactly — silently truncating with min() would mask
            # any mismatch caused by batching/parallelism.
            n = len(chains_passed)
            if not (len(baseline_responses) == len(intervention_responses)
                    == len(ground_truths) == n):
                raise ValueError(
                    f"[{cell}] response/chain length mismatch: "
                    f"chains={n} baseline={len(baseline_responses)} "
                    f"intervention={len(intervention_responses)} "
                    f"gt={len(ground_truths)} — positional pairing broken"
                )
            chains_scored = chains_passed
            b_scores = score_batch(chains_scored, ground_truths, baseline_responses)
            i_scores = score_batch(chains_scored, ground_truths, intervention_responses)

            b_vec, i_vec = extract_binary_vectors(b_scores, i_scores)
            var_baseline = variance_summary([s.correct or False for s in b_scores], cell)
            var_intervention = variance_summary([s.correct or False for s in i_scores], cell)

            mcnemar_result = run_mcnemar(
                baseline_correct=b_vec,
                intervention_correct=i_vec,
                cell=cell,
                alpha=self.config.alpha,
                bonferroni_divisor=self.config.bonferroni_divisor,
                continuity_correction=self.config.continuity_correction,
                bootstrap_iterations=self.config.bootstrap_iterations,
                bootstrap_seed=self.config.bootstrap_seed,
                min_discordant_pairs=self.config.min_discordant_pairs,
            )
            power = mcnemar_power(
                mcnemar_result.b, mcnemar_result.c,
                alpha=self.config.alpha,
                bonferroni_divisor=self.config.bonferroni_divisor,
            )
        else:
            pass  # no responses; power stays None

        # MDE is always computed from n_post_gate2 — useful both pre-run
        # (when no responses yet) and post-hoc (to show detection threshold).
        mde_val = minimum_detectable_effect(
            n_chains=n_post_gate2,
            alpha=self.config.alpha,
            bonferroni_divisor=self.config.bonferroni_divisor,
        )

        return CellResult(
            cell=cell,
            n_events_total=n_events,
            n_chains_pre_gate2=n_pre_gate2,
            n_chains_post_gate2=n_post_gate2,
            retention_rate=retention_info["retention_rate"],
            gate2_pass=retention_info["gate2_pass"],
            mcnemar=mcnemar_result,
            variance_baseline=var_baseline,
            variance_intervention=var_intervention,
            power=power,
            mde=mde_val,
            errors=errors,
        )
