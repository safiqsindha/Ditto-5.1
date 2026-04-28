"""
ChainBuilder for v5.

Per Q4 sign-off (D-19): chain length is fixed PER CELL, varying across cells.
Per-cell N values are pre-registered at T-design time (joint authoring session).

Overlap policy: non-overlapping (sliding window with step = chain_length).
Sampling: sequential within game; cells subsample chains uniformly to hit n_chains target.

Variant (C) variable-bounded chain length is flagged for ME-1 micro-experiment.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod

from ..common.schema import ChainCandidate, EventStream

logger = logging.getLogger(__name__)


class ChainBuilder(ABC):
    """Abstract base. See FixedPerCellChainBuilder for the v5 implementation."""

    @abstractmethod
    def build(
        self,
        candidates_or_streams,
        chain_length: int | None = None,
        overlap: bool = False,
    ) -> list[ChainCandidate]:
        ...


class FixedPerCellChainBuilder(ChainBuilder):
    """
    Q4-locked chain builder. Constructs fixed-length non-overlapping chains
    from EventStreams using a per-cell `chain_length` map.

    Per-cell N values are pre-registered (None = not yet locked). The builder
    raises if asked to build for a cell with N still unset, ensuring no
    accidental defaulting.

    Usage
    -----
    builder = FixedPerCellChainBuilder(per_cell_chain_length={
        "fortnite": 8, "nba": 5, "csgo": 10, "rocket_league": 12, "poker": 8,
    })
    chains = builder.build_from_streams(streams, cell="nba")
    """

    def __init__(
        self,
        per_cell_chain_length: dict[str, int | None] | None = None,
        overlap: bool = False,
        sampling: str = "sequential",
    ):
        # Per Q4-locked config: non-overlapping, sequential
        self.per_cell_chain_length: dict[str, int | None] = per_cell_chain_length or {
            "fortnite": None,
            "nba": None,
            "csgo": None,
            "rocket_league": None,
            "poker": None,
        }
        self.overlap = overlap
        self.sampling = sampling

    def get_chain_length(self, cell: str) -> int:
        """Return the locked chain length for the cell, or raise if not set."""
        n = self.per_cell_chain_length.get(cell)
        if n is None:
            raise ValueError(
                f"chain_length for cell '{cell}' is not set. "
                "Per Q4 sign-off, per-cell N values must be locked at T-design time. "
                "Set via FixedPerCellChainBuilder(per_cell_chain_length={...}) or in "
                "config/harness.yaml chain_length.per_cell.{cell}."
            )
        if n < 1:
            raise ValueError(f"chain_length for '{cell}' must be >= 1, got {n}")
        return n

    def build_from_streams(
        self,
        streams: list[EventStream],
        cell: str,
        max_chains: int | None = None,
    ) -> list[ChainCandidate]:
        """
        Build chains from a list of EventStreams for one cell.

        max_chains caps the total number of chains returned across all streams.
        Sampling is sequential (deterministic, in-order); to get an
        evenly-distributed sample, the caller should pass max_chains and the
        builder will subsample uniformly across streams.
        """
        chain_length = self.get_chain_length(cell)
        all_chains: list[ChainCandidate] = []

        for stream in streams:
            if stream.cell != cell:
                logger.warning(
                    f"Stream {stream.game_id} has cell={stream.cell} != requested {cell}; skipping"
                )
                continue
            stream_chains = self._build_one_stream(stream, chain_length)
            all_chains.extend(stream_chains)

        if max_chains is None or len(all_chains) <= max_chains:
            return all_chains

        return _uniform_subsample(all_chains, max_chains)

    def build_from_candidates(
        self,
        candidates: list[ChainCandidate],
        cell: str,
        max_chains: int | None = None,
    ) -> list[ChainCandidate]:
        """
        Build fixed-length chains from T-output ChainCandidates (variable length).

        For each candidate longer than the per-cell N, slide a non-overlapping
        window of size N over its events and emit one chain per window.
        Candidates shorter than N are dropped (with a warning).

        Per Q4-B sign-off — used when T produces variable-length candidates that
        need to be normalized to per-cell N before Gate 2 + scoring.
        """
        chain_length = self.get_chain_length(cell)
        all_chains: list[ChainCandidate] = []

        for cand in candidates:
            if cand.cell != cell:
                logger.warning(
                    f"Candidate {cand.chain_id} cell={cand.cell} != requested {cell}; skipping"
                )
                continue
            if len(cand.events) < chain_length:
                logger.debug(
                    f"Candidate {cand.chain_id} too short ({len(cand.events)} < {chain_length}); dropped"
                )
                continue

            step = chain_length if not self.overlap else 1
            n_subchains = 0
            for i in range(0, len(cand.events) - chain_length + 1, step):
                window = cand.events[i:i + chain_length]
                sub_id = f"{cand.chain_id}_sub{n_subchains:03d}"
                all_chains.append(ChainCandidate(
                    chain_id=sub_id,
                    game_id=cand.game_id,
                    cell=cell,
                    events=window,
                    chain_metadata={
                        **cand.chain_metadata,
                        "parent_chain_id": cand.chain_id,
                        "chain_length": chain_length,
                        "subwindow_start": i,
                        "overlap": self.overlap,
                        "builder": "FixedPerCellChainBuilder",
                    },
                ))
                n_subchains += 1

        if max_chains is None or len(all_chains) <= max_chains:
            return all_chains
        return _uniform_subsample(all_chains, max_chains)

    def _build_one_stream(
        self, stream: EventStream, chain_length: int
    ) -> list[ChainCandidate]:
        """Slide a fixed window over the stream's events. Non-overlapping."""
        chains: list[ChainCandidate] = []
        events = stream.events
        if len(events) < chain_length:
            return chains

        step = chain_length if not self.overlap else 1
        for i in range(0, len(events) - chain_length + 1, step):
            window = events[i:i + chain_length]
            chain_id = f"{stream.game_id}_chain_{i // step:04d}"
            chains.append(ChainCandidate(
                chain_id=chain_id,
                game_id=stream.game_id,
                cell=stream.cell,
                events=window,
                chain_metadata={
                    "chain_length": chain_length,
                    "window_start": i,
                    "overlap": self.overlap,
                    "builder": "FixedPerCellChainBuilder",
                },
            ))
        return chains

    def shuffle_chains(
        self,
        chains: list[ChainCandidate],
        seed: int | None = None,
        n_shuffles: int = 1,
    ) -> list[ChainCandidate]:
        """
        CF-3=A: generate one shuffled-event control per real chain.

        Returns only the shuffled controls; caller is responsible for
        appending them to the real chains for null-distribution comparison.
        Events within each chain are randomly reordered; chain_id, game_id,
        and cell are preserved. Shuffled chains are tagged in chain_metadata
        with cf3="shuffled_control".
        """
        rng = random.Random(seed)
        shuffled: list[ChainCandidate] = []
        for chain in chains:
            for k in range(n_shuffles):
                events = list(chain.events)
                rng.shuffle(events)
                shuffled.append(ChainCandidate(
                    chain_id=f"{chain.chain_id}_shuf{k:02d}",
                    game_id=chain.game_id,
                    cell=chain.cell,
                    events=events,
                    chain_metadata={
                        **chain.chain_metadata,
                        "parent_chain_id": chain.chain_id,
                        "shuffled": True,
                        "shuffle_idx": k,
                        "cf3": "shuffled_control",
                    },
                ))
        return shuffled

    # Legacy interface compatibility — accepts either a list of ChainCandidates
    # or a list of EventStreams. The runner uses build_from_streams() directly;
    # this is here so the abstract method signature matches.
    def build(
        self,
        candidates_or_streams,
        chain_length: int | None = None,
        overlap: bool = False,
    ) -> list[ChainCandidate]:
        if not candidates_or_streams:
            return []
        first = candidates_or_streams[0]
        if isinstance(first, EventStream):
            cell = first.cell
            return self.build_from_streams(candidates_or_streams, cell=cell)
        if isinstance(first, ChainCandidate):
            # Already chains; pass through
            return list(candidates_or_streams)
        raise TypeError(
            f"build() expected list of EventStream or ChainCandidate, got {type(first).__name__}"
        )


def _uniform_subsample(items: list, k: int) -> list:
    """Return k items roughly uniformly from items. Deterministic order-preserving."""
    n = len(items)
    if k >= n or k <= 0:
        return items
    step = n / k
    return [items[int(i * step)] for i in range(k)]


# Backwards-compat alias — the original DefaultChainBuilder name is referenced
# in older docs. The post-sign-off implementation IS this class.
DefaultChainBuilder = FixedPerCellChainBuilder
