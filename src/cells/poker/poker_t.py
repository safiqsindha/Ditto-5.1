"""
PokerT — Translation function for the Texas Hold'em poker cell.

Phase B decisions (P-1 through P-6), originally locked 2026-04-27:
  P-1: chain unit = one player's decisions within a single hand across
       all streets — **AMENDED by A7 (2026-04-29)**: chain unit is now
       per-hand sequence (all actors interleaved by sequence_idx). The
       pre-Phase-D pilot showed real cash-game data averages ~7.5 player
       actions/hand spread across 6 actors → most actors have 1–2 actions
       and the original P-4 floor (≥3 per actor) discarded 99.9% of pairs.
  P-2: chain triggers on player-decision events only (fold/check/call/bet/raise) — preserved
  P-3: N = 8 — preserved
  P-4: minimum 3 actions — **AMENDED by A7**: filter is now applied at
       the HAND level (≥3 total player actions in the hand) instead of
       the per-actor level.
  P-5: constraint = stack limits, action order, showdown rules — preserved
  P-6: actionable-type additions — preserved

CF-4=B: actors arrive already anonymised as actor_0 … actor_N from
  PokerPipeline.parse(). PokerT preserves these identifiers; no further
  anonymisation is needed here.

Micro-experiment stubs (NOT implemented):
  ME-PK-1: per-session chains (stack evolution across multiple hands)
  ME-PK-2: HandHQ skill-stratified chains (win-rate threshold for "expert")
"""

from __future__ import annotations

# Import the ABC from a leaf module to avoid circular import — translation.py
# imports PokerT from this module to populate DOMAIN_T_STUBS, so we can't go
# back the other way through translation.py.
from ...interfaces.translation_base import TranslationFunction
from ...common.schema import ChainCandidate, EventStream, GameEvent

_POKER_N = 8                     # P-3 preserved
_POKER_MIN_ACTIONS_PER_HAND = 3  # P-4 amended (A7) — filter at hand level


class PokerT(TranslationFunction):
    """
    Extract per-hand chains from a Texas Hold'em event stream.

    Algorithm (per A7 amendment, 2026-04-29):
      1. The stream is one hand (per PokerPipeline.parse).
      2. If the stream contains fewer than _POKER_MIN_ACTIONS_PER_HAND (= 3)
         player-action events, drop the hand (blind-post + immediate-fold).
      3. Take the first _POKER_N (= 8) events in chronological order as
         the chain. Multi-actor — the chain interleaves all players' actions
         in the order they were emitted (sequence_idx).
      4. Yield one ChainCandidate per qualifying hand.

    Chain ID: deterministic hash of (game_id, "pk_hand") — game_id encodes
    the hand identity, so this is stable per hand. CF-4=B preserved: the
    chain_id never embeds a real player name.
    """

    @property
    def cell(self) -> str:
        return "poker"

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        events = stream.events
        if not events:
            return []

        # A7: filter at hand level. PokerExtractor only emits player-decision
        # events (deal/board events update internal state but don't append to
        # the stream), so len(events) is the player-action count for the hand.
        if len(events) < _POKER_MIN_ACTIONS_PER_HAND:
            return []

        # Sort by sequence_idx for safety, then take first N. Real cash-game
        # hands rarely exceed 12 player actions, so the truncation is the
        # exception not the rule.
        ordered = sorted(events, key=lambda e: e.sequence_idx)
        window = ordered[:_POKER_N]

        # Capture the unique actors present in the window for downstream
        # diagnostics. CF-4=B: these are already actor_N labels from parse.
        unique_actors = []
        seen_actors: set[str] = set()
        for ev in window:
            if ev.actor not in seen_actors:
                seen_actors.add(ev.actor)
                unique_actors.append(ev.actor)

        return [
            ChainCandidate(
                chain_id=self._chain_id(stream.game_id, "pk_hand"),
                game_id=stream.game_id,
                cell="poker",
                events=window,
                chain_metadata={
                    "chain_type": "hand_sequence",
                    "n_actions": len(window),
                    "n_actors_in_window": len(unique_actors),
                    "actors_in_window": unique_actors,
                    "streets_covered": _streets_in(window),
                },
            )
        ]


def _streets_in(events: list[GameEvent]) -> list[str]:
    """Return ordered unique streets covered by a chain's events."""
    street_order = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
    seen: set[str] = set()
    ordered: list[str] = []
    for ev in events:
        s = ev.location_context.get("street") or ev.phase or ""
        if s and s not in seen:
            seen.add(s)
            ordered.append(s)
    ordered.sort(key=lambda x: street_order.get(x, 99))
    return ordered


# ---------------------------------------------------------------------------
# ME-PK-1: Per-session chains (stack evolution across hands)
# Pre-registered micro-experiment — NOT YET IMPLEMENTED.
# ---------------------------------------------------------------------------

class PokerPerSessionT(TranslationFunction):
    """
    ME-PK-1: per-session chains where one chain spans multiple hands from
    the same player, using stack evolution as the linking signal.

    Rationale: in a cash-game session, a player's stack size at the start
    of each hand constrains which plays are available (short-stack push/fold
    is a hard mechanical constraint). Tracking stack_bb across consecutive
    hands tests whether the model can reason about cross-hand resource state.

    Chain extraction: group hands by player × session; emit a sliding window
    of N consecutive hands per player. Each event = one hand-level decision
    (represented as the player's most significant action per hand).

    Mirror ME-RL-1 / ME-FN-1 pattern for registration.
    """

    @property
    def cell(self) -> str:
        return "poker"

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        raise NotImplementedError(
            "ME-PK-1 (per-session chains) is not yet implemented. "
            "Requires a multi-hand EventStream format with session metadata."
        )


# ---------------------------------------------------------------------------
# ME-PK-2: HandHQ skill-stratified cell
# Pre-registered micro-experiment — NOT YET IMPLEMENTED.
# ---------------------------------------------------------------------------

class PokerHandHQT(TranslationFunction):
    """
    ME-PK-2: skill-stratified chains from the HandHQ 25NL-1000NL corpus.

    Rationale: 21.6M hands provide a large population sample. Stratifying
    by win-rate (bb/100) allows a skill-gradient study: do constraint
    violations decrease monotonically with skill level?

    Chain extraction: same per-player-hand algorithm as PokerT (P-1 through P-4)
    but applied only to hands where the player's long-run win-rate is above
    a pre-registered threshold (e.g., 5 bb/100 over ≥ 10,000 hands).

    Data: HandHQ 2009 scrapes (excluded from primary cell per SPEC).
    See docs/REAL_DATA_GUIDE.md for credential setup.
    """

    @property
    def cell(self) -> str:
        return "poker"

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        raise NotImplementedError(
            "ME-PK-2 (HandHQ skill-stratified) is not yet implemented. "
            "Requires HandHQ corpus download and win-rate computation."
        )
