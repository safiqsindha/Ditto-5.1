"""
Base class for translation functions. Lives in its own module to avoid
circular imports — `src/cells/poker/poker_t.py` and the other cell T
implementations need TranslationFunction at class-definition time, but
`src/interfaces/translation.py` imports from `poker_t.py` to populate
the DOMAIN_T_STUBS registry. Splitting the ABC here lets both sides
import from a leaf module.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from ..common.schema import ChainCandidate, EventStream


class TranslationFunction(ABC):
    """
    Abstract base class for domain-specific translation functions.

    translate() takes one game's normalized EventStream and returns a list of
    ChainCandidates — variable-length event sequences representing candidate
    constraint chains. ChainBuilder then trims each candidate to the per-cell
    fixed length N.
    """

    @property
    @abstractmethod
    def cell(self) -> str:
        """Domain cell this T serves. Must match one of VALID_CELLS."""
        ...

    @abstractmethod
    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        """
        Convert a normalized event stream to candidate constraint chains.

        Returns list of ChainCandidates; may be empty. Each candidate may be
        longer than the target N — ChainBuilder handles trimming.
        """
        ...

    def batch_translate(self, streams: list[EventStream]) -> list[ChainCandidate]:
        """Translate multiple streams. Default: calls translate() per stream."""
        chains = []
        for stream in streams:
            chains.extend(self.translate(stream))
        return chains

    @staticmethod
    def _chain_id(game_id: str, tag: str) -> str:
        raw = f"{game_id}__{tag}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
