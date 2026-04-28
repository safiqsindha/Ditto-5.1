"""
MockT — deterministic mock TranslationFunction for pilot harness testing.

MockT converts event streams to chain candidates using simple heuristics
(sliding window over actionable events). It is NOT a real T implementation —
it exists solely to let the pilot harness and cell runner run end-to-end
without a real T.

NoisyMockT is a variant that intentionally injects non-actionable events so
that a fraction of output chains fall below the Gate 2 actionable-fraction
floor. Use this to validate that Gate 2 filtering works correctly under
realistic (non-100%) retention rates (RISK_MITIGATIONS M2).

Real T implementations require both-author review and are out of scope.
"""

from __future__ import annotations

import hashlib

from ..common.schema import ChainCandidate, EventStream
from ..harness.actionables import is_actionable
from ..interfaces.translation import TranslationFunction


class MockT(TranslationFunction):
    """
    Deterministic mock translation function.

    Algorithm: sliding window over actionable events in the stream.
    Window size = window_size. Step = step_size.
    Each window becomes one ChainCandidate.

    This is intentionally simplistic — it demonstrates interface compliance
    without making any real constraint-chain decisions.
    """

    def __init__(
        self,
        cell: str,
        window_size: int = 5,
        step_size: int = 3,
        min_actionable_fraction: float = 0.0,
    ):
        self._cell = cell
        self.window_size = window_size
        self.step_size = step_size
        self.min_actionable_fraction = min_actionable_fraction

    @property
    def cell(self) -> str:
        return self._cell

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        """
        Slide a window over the event stream, emitting one chain per window.
        Only includes events that are actionable under v1.1 rules.
        """
        events = stream.events
        if not events:
            return []

        chains = []
        i = 0
        chain_idx = 0
        while i + self.window_size <= len(events):
            window = events[i:i + self.window_size]
            actionable = [e for e in window if is_actionable(e, self.cell)]

            if not actionable:
                i += self.step_size
                continue

            frac = len(actionable) / len(window)
            if frac < self.min_actionable_fraction:
                i += self.step_size
                continue

            chain_id = self._make_chain_id(stream.game_id, i, chain_idx)
            candidate = ChainCandidate(
                chain_id=chain_id,
                game_id=stream.game_id,
                cell=self.cell,
                events=window,
                chain_metadata={
                    "window_start": i,
                    "window_size": len(window),
                    "n_actionable": len(actionable),
                    "actionable_fraction": frac,
                    "mock": True,
                },
            )
            chains.append(candidate)
            chain_idx += 1
            i += self.step_size

        # Handle partial tail window (if remaining events > 0)
        if i < len(events) and len(events) - i >= 2:
            tail = events[i:]
            actionable = [e for e in tail if is_actionable(e, self.cell)]
            if actionable:
                chain_id = self._make_chain_id(stream.game_id, i, chain_idx)
                chains.append(ChainCandidate(
                    chain_id=chain_id,
                    game_id=stream.game_id,
                    cell=self.cell,
                    events=tail,
                    chain_metadata={
                        "window_start": i,
                        "window_size": len(tail),
                        "n_actionable": len(actionable),
                        "actionable_fraction": len(actionable) / len(tail),
                        "mock": True,
                        "is_tail": True,
                    },
                ))

        return chains

    @staticmethod
    def _make_chain_id(game_id: str, window_start: int, chain_idx: int) -> str:
        raw = f"{game_id}_{window_start}_{chain_idx}"
        return "mock_chain_" + hashlib.md5(raw.encode()).hexdigest()[:12]


class NoisyMockT(MockT):
    """
    Noisy variant of MockT for Gate 2 validation (RISK_MITIGATIONS M2).

    Intentionally degrades a fraction of output chains by injecting
    non-actionable filler events until the chain's actionable fraction
    drops below ``noise_frac``. This exercises the Gate 2 filter under
    realistic (non-100%) retention scenarios.

    Parameters
    ----------
    noise_rate : float
        Fraction of chains to degrade (0.0–1.0). Default 0.4 means 40% of
        chains will be made sub-Gate-2.
    target_actionable_frac : float
        Actionable fraction to inject into degraded chains. Defaults to 0.30,
        which is below the Gate 2 floor of 0.50.
    """

    def __init__(
        self,
        cell: str,
        window_size: int = 5,
        step_size: int = 3,
        noise_rate: float = 0.4,
        target_actionable_frac: float = 0.30,
    ):
        super().__init__(cell=cell, window_size=window_size, step_size=step_size)
        self.noise_rate = noise_rate
        self.target_actionable_frac = target_actionable_frac

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        chains = super().translate(stream)
        if not chains:
            return chains

        # Deterministically select chains to degrade using their index
        degraded = []
        for idx, chain in enumerate(chains):
            if (idx % 10) < int(self.noise_rate * 10):
                chain = self._degrade_chain(chain, idx)
            degraded.append(chain)
        return degraded

    def _degrade_chain(self, chain: ChainCandidate, seed: int) -> ChainCandidate:
        """
        Replace the chain's metadata to reflect a low actionable fraction.

        We do not re-filter the actual events here (the events remain the same),
        but we override the actionable_fraction metadata so Gate 2 sees a
        sub-floor value. This is sufficient to test the Gate 2 filter logic
        without needing to construct artificial non-actionable event types.
        """
        n_events = len(chain.events)
        n_actionable = max(1, int(n_events * self.target_actionable_frac))
        updated_meta = {
            **chain.chain_metadata,
            "n_actionable": n_actionable,
            "actionable_fraction": n_actionable / n_events,
            "noisy_mock": True,
            "degraded_seed": seed,
        }
        return ChainCandidate(
            chain_id=chain.chain_id,
            game_id=chain.game_id,
            cell=chain.cell,
            events=chain.events,
            chain_metadata=updated_meta,
        )
