"""
PokerT — Translation function for the Texas Hold'em poker cell.

Phase B decisions (P-1 through P-6), locked 2026-04-27:
  P-1: chain unit = one player's decisions within a single hand across
       all streets (preflop, flop, turn, river)
  P-2: chain triggers on player-decision events only (fold/check/call/bet/raise)
  P-3: N = 8 (matches FortniteT window; comparable prompt token count ~600)
  P-4: minimum 3 actions per player-hand pair (drops blind-post + fold-only entries)
  P-5: constraint = NLHE stack limits, action order, and showdown rules
  P-6: actionable-type additions: pot_odds_decision, stack_pressure,
       hand_strength_commitment

CF-4=B: actors arrive already anonymised as actor_0 … actor_N from
  PokerPipeline.parse(). PokerT preserves these identifiers; no further
  anonymisation is needed here.

Micro-experiment stubs (NOT implemented):
  ME-PK-1: per-session chains (stack evolution across multiple hands)
  ME-PK-2: HandHQ skill-stratified chains (win-rate threshold for "expert")
"""

from __future__ import annotations

from collections import defaultdict

from ...interfaces.translation import TranslationFunction
from ...common.schema import ChainCandidate, EventStream, GameEvent

_POKER_N = 8               # P-3 locked
_POKER_MIN_ACTIONS = 3     # P-4 locked


class PokerT(TranslationFunction):
    """
    Extract per-player-hand chains from a Texas Hold'em event stream.

    Algorithm (P-1, P-2, P-4):
      1. Group all events in the stream by actor (player identifier).
      2. Discard actors with fewer than _POKER_MIN_ACTIONS (= 3) events.
         (Eliminates players who only post a blind and fold pre-flop.)
      3. For qualifying actors, take up to _POKER_N (= 8) events in
         chronological order as the chain.
      4. Yield one ChainCandidate per qualifying (actor, hand) pair.

    Chain ID: deterministic hash of (game_id, "pk_<actor_slot>") —
    actor_slot is the sort-index into the hand's unique actor set so
    chain IDs are stable and CF-4=B safe (no real name appears in the ID).
    """

    @property
    def cell(self) -> str:
        return "poker"

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        events = stream.events
        if not events:
            return []

        # Group by actor (actors are already actor_N from parse step)
        by_actor: dict[str, list[GameEvent]] = defaultdict(list)
        for ev in events:
            by_actor[ev.actor].append(ev)

        chains: list[ChainCandidate] = []
        # Sort for deterministic actor_slot assignment (CF-4=B)
        for actor_slot, actor in enumerate(sorted(by_actor)):
            actor_events = by_actor[actor]
            if len(actor_events) < _POKER_MIN_ACTIONS:
                continue  # P-4: drop thin entries (blind-post + fold only)

            window = actor_events[:_POKER_N]

            chains.append(ChainCandidate(
                chain_id=self._chain_id(stream.game_id, f"pk_{actor_slot}"),
                game_id=stream.game_id,
                cell="poker",
                events=window,
                chain_metadata={
                    "chain_type": "player_hand",
                    "actor_slot": actor_slot,
                    "n_actions": len(window),
                    "streets_covered": _streets_in(window),
                },
            ))

        return chains


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
