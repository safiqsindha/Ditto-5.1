"""
Pilot validation harness for v5.

Takes 50-100 sample chains per cell and produces:
  1. Retention rate vs Gate 2 floor (50%)
  2. Sample chain inspection output
  3. Basic distributional sanity checks per cell

T is pluggable — accepts any TranslationFunction implementation.
Default: MockT (deterministic heuristic window, no real T).

No Haiku API calls are made here. This harness validates pipeline
and harness correctness only.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from ..common.schema import ChainCandidate, EventStream
from ..harness.actionables import compute_retention_rate
from ..interfaces.translation import TranslationFunction
from .mock_t import MockT

logger = logging.getLogger(__name__)

PILOT_SAMPLE_SIZE = 75  # chains to sample per cell for inspection


@dataclass
class CellPilotReport:
    """Pilot validation results for one cell."""
    cell: str
    n_streams: int
    n_events_total: int
    n_chains_raw: int
    n_chains_post_gate2: int
    retention_rate: float
    gate2_floor: float
    gate2_pass: bool

    # Distributional stats on chain length
    chain_length_mean: float
    chain_length_min: int
    chain_length_max: int
    chain_length_median: float

    # Distributional stats on actionable fraction
    actionable_frac_mean: float
    actionable_frac_min: float
    actionable_frac_max: float

    # Event type distribution (top 10)
    event_type_distribution: dict

    # Sample chains for inspection (up to PILOT_SAMPLE_SIZE)
    sample_chain_summaries: list[dict]

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.gate2_pass and not self.errors

    def print_summary(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        print(f"\n{'='*60}")
        print(f"[{self.cell.upper()}] Pilot Validation — {status}")
        print(f"{'='*60}")
        print(f"  Streams:       {self.n_streams}")
        print(f"  Total events:  {self.n_events_total}")
        print(f"  Raw chains:    {self.n_chains_raw}")
        print(f"  Post-Gate2:    {self.n_chains_post_gate2}")
        print(f"  Retention:     {self.retention_rate:.1%} (floor={self.gate2_floor:.0%})")
        print(f"  Gate 2:        {'PASS' if self.gate2_pass else 'FAIL'}")
        print(f"  Chain length:  mean={self.chain_length_mean:.1f} "
              f"min={self.chain_length_min} max={self.chain_length_max} "
              f"median={self.chain_length_median:.1f}")
        print(f"  Actionable:    mean={self.actionable_frac_mean:.1%} "
              f"min={self.actionable_frac_min:.1%} max={self.actionable_frac_max:.1%}")
        if self.warnings:
            for w in self.warnings:
                print(f"  WARN: {w}")
        if self.errors:
            for e in self.errors:
                print(f"  ERROR: {e}")
        print()
        print("  Top event types:")
        for etype, count in sorted(
            self.event_type_distribution.items(), key=lambda x: -x[1]
        )[:10]:
            print(f"    {etype:<35} {count:>5}")

    def to_dict(self) -> dict:
        return {
            "cell": self.cell,
            "n_streams": self.n_streams,
            "n_events_total": self.n_events_total,
            "n_chains_raw": self.n_chains_raw,
            "n_chains_post_gate2": self.n_chains_post_gate2,
            "retention_rate": self.retention_rate,
            "gate2_floor": self.gate2_floor,
            "gate2_pass": self.gate2_pass,
            "chain_length_mean": self.chain_length_mean,
            "chain_length_min": self.chain_length_min,
            "chain_length_max": self.chain_length_max,
            "chain_length_median": self.chain_length_median,
            "actionable_frac_mean": self.actionable_frac_mean,
            "actionable_frac_min": self.actionable_frac_min,
            "actionable_frac_max": self.actionable_frac_max,
            "event_type_distribution": self.event_type_distribution,
            "n_sample_chains": len(self.sample_chain_summaries),
            "warnings": self.warnings,
            "errors": self.errors,
            "passed": self.passed,
        }


@dataclass
class PilotReport:
    """Aggregate pilot validation results across all cells."""
    cells: list[CellPilotReport]
    all_passed: bool

    def print_summary(self) -> None:
        for cell_report in self.cells:
            cell_report.print_summary()
        print(f"\n{'='*60}")
        print(f"AGGREGATE PILOT RESULT: {'ALL PASS' if self.all_passed else 'SOME FAILURES'}")
        for r in self.cells:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.cell}: retention={r.retention_rate:.1%} "
                  f"chains={r.n_chains_post_gate2}")
        print(f"{'='*60}\n")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "cells": [c.to_dict() for c in self.cells],
                "all_passed": self.all_passed,
            }, f, indent=2)


class PilotValidator:
    """
    Runs pilot validation for one or more cells.

    Usage
    -----
    validator = PilotValidator(gate2_floor=0.50)
    validator.register_cell("fortnite", MockT(cell="fortnite"))
    report = validator.run(streams_by_cell)
    report.print_summary()
    """

    def __init__(self, gate2_floor: float = 0.50):
        self.gate2_floor = gate2_floor
        self._cell_translations: dict[str, TranslationFunction] = {}

    def register_cell(self, cell: str, translation_fn: TranslationFunction | None = None) -> None:
        if translation_fn is None:
            translation_fn = MockT(cell=cell)
        self._cell_translations[cell] = translation_fn

    def run(
        self,
        streams_by_cell: dict[str, list[EventStream]],
        sample_size: int = PILOT_SAMPLE_SIZE,
    ) -> PilotReport:
        cell_reports = []
        for cell, streams in streams_by_cell.items():
            t_fn = self._cell_translations.get(cell, MockT(cell=cell))
            report = self._validate_cell(cell, streams, t_fn, sample_size)
            cell_reports.append(report)

        all_passed = all(r.passed for r in cell_reports)
        return PilotReport(cells=cell_reports, all_passed=all_passed)

    def _validate_cell(
        self,
        cell: str,
        streams: list[EventStream],
        translation_fn: TranslationFunction,
        sample_size: int,
    ) -> CellPilotReport:
        errors = []
        warnings = []

        n_events = sum(len(s) for s in streams)

        # Translate to chains
        chains: list[ChainCandidate] = []
        try:
            for stream in streams:
                chains.extend(translation_fn.translate(stream))
        except NotImplementedError as e:
            errors.append(f"T not implemented: {e}")
        except Exception as e:
            errors.append(f"T error: {e}")

        n_raw = len(chains)

        # Gate 2
        gate2_info = compute_retention_rate(chains, floor=self.gate2_floor)
        chains_passed = [c for c in chains if c.is_actionable]
        n_passed = len(chains_passed)

        # F11 fix: SPEC Q3 locked target is 1,200 chains/cell
        if n_passed < 1200:
            warnings.append(
                f"Only {n_passed} chains post-Gate2; SPEC Q3 target is 1,200/cell. "
                "Per Q3 contingency, scale upstream acquisition to maintain 1,200 post-filter."
            )
        if gate2_info["retention_rate"] < self.gate2_floor:
            warnings.append(
                f"Retention rate {gate2_info['retention_rate']:.1%} < "
                f"floor {self.gate2_floor:.0%}"
            )

        # Chain length distribution
        lengths = [len(c.events) for c in chains_passed] if chains_passed else [0]
        chain_length_mean = statistics.mean(lengths)
        chain_length_min = min(lengths)
        chain_length_max = max(lengths)
        chain_length_median = statistics.median(lengths)

        # Actionable fraction distribution
        fracs = [
            c.chain_metadata.get("gate2_actionable_fraction", 0.0)
            for c in chains_passed
        ]
        if fracs:
            actionable_mean = statistics.mean(fracs)
            actionable_min = min(fracs)
            actionable_max = max(fracs)
        else:
            actionable_mean = actionable_min = actionable_max = 0.0

        # Event type distribution
        event_type_dist = _count_event_types(chains_passed)

        # Sample chain summaries
        sample_chains = chains_passed[:sample_size]
        sample_summaries = [_chain_summary(c) for c in sample_chains]

        # Sanity checks
        if event_type_dist:
            top_type, top_count = max(event_type_dist.items(), key=lambda x: x[1])
            total_ev = sum(event_type_dist.values())
            if total_ev > 0 and top_count / total_ev > 0.80:
                warnings.append(
                    f"Event type '{top_type}' dominates "
                    f"({top_count/total_ev:.0%} of events). "
                    "Check extractor normalization."
                )

        return CellPilotReport(
            cell=cell,
            n_streams=len(streams),
            n_events_total=n_events,
            n_chains_raw=n_raw,
            n_chains_post_gate2=n_passed,
            retention_rate=gate2_info["retention_rate"],
            gate2_floor=self.gate2_floor,
            gate2_pass=gate2_info["gate2_pass"],
            chain_length_mean=chain_length_mean,
            chain_length_min=chain_length_min,
            chain_length_max=chain_length_max,
            chain_length_median=float(chain_length_median),
            actionable_frac_mean=actionable_mean,
            actionable_frac_min=actionable_min,
            actionable_frac_max=actionable_max,
            event_type_distribution=event_type_dist,
            sample_chain_summaries=sample_summaries,
            warnings=warnings,
            errors=errors,
        )


def _count_event_types(chains: list[ChainCandidate]) -> dict:
    counts: dict = {}
    for chain in chains:
        for ev in chain.events:
            counts[ev.event_type] = counts.get(ev.event_type, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _chain_summary(chain: ChainCandidate) -> dict:
    return {
        "chain_id": chain.chain_id,
        "game_id": chain.game_id,
        "cell": chain.cell,
        "n_events": len(chain.events),
        "event_types": [e.event_type for e in chain.events],
        "actors": list({e.actor for e in chain.events}),
        "t_start": chain.events[0].timestamp if chain.events else 0,
        "t_end": chain.events[-1].timestamp if chain.events else 0,
        "actionable_fraction": chain.chain_metadata.get("gate2_actionable_fraction", 0),
        "is_actionable": chain.is_actionable,
        "mock": chain.chain_metadata.get("mock", False),
    }
