"""
Violation injectors — diagnostic instrument per the 4-way reviewer synthesis
(Gemini, ChatGPT, Opus #1, Opus #2) recorded in DECISION_LOG D-42.

The pre-Phase-D pilot showed 1 of 5 cells (PUBG) with the predicted McNemar
pattern; the other 4 cells produced 0% YES rates regardless of intervention
or shuffle, indicating a *floor effect* rather than a constraint-reasoning
failure. The model interprets "Is this consistent with the rules of {domain}?"
as "Does this look like a complete real game?" and rejects 8-event chains
because they are too short to be a complete game.

Violation injectors swap the dependent variable from "consistency rating"
to "violation detection." We take a real chain, plant ONE explicit
constraint violation that maps directly to a clause in the locked
constraint context, and ask the model "Does this contain any event that
violates the rules?" Now baseline-vs-intervention measures whether the
constraint context helps the model SPOT planted violations.

This is methodologically defensible because:
1. The diagnostic uses the same locked constraint contexts (no per-cell
   constraint tuning).
2. The injector targets ONE clause per cell, applied uniformly across
   chains.
3. The redesign is in the SCORING layer (detection accuracy), not the
   chain-rendering layer that A4-A7 amended.

Per-cell injectors target:
  - NBA: "A player with 6 fouls is ejected" → bump an actor's foul count
    to 7 in mid-chain and have them act in a subsequent event
  - PUBG: "An eliminated player cannot act" → mark a player as eliminated
    via a kill event then have them act later
  - Poker: "A folded player cannot act again" → make a player fold then
    bet/raise later
  - Rocket League: skipped — best violation candidates need per-hit boost
    state (data ceiling)
  - CS:GO: skipped — synthetic timestamp distribution prevents clean
    intra-round violation construction (data ceiling)

If diagnostic recovers signal on NBA/PUBG/Poker, the framework engagement
is real and a v5.1 follow-up can address RL/CS:GO via demo/replay
extraction (which would also unlock cleaner violation injection).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

from ..common.schema import ChainCandidate, GameEvent

logger = logging.getLogger(__name__)


@dataclass
class InjectionResult:
    """Wraps a chain with metadata describing the violation that was injected."""
    chain: ChainCandidate
    cell: str
    violation_clause: str       # which constraint clause is violated
    violation_description: str  # human-readable summary
    target_actor: str | None    # the actor whose rule was broken
    target_event_idx: int       # index of the event that violates the rule


def inject_nba_foul_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    Plant: an actor has actor_total_fouls=7 (over the 6-foul ejection
    threshold) and then commits a subsequent action.

    Strategy: pick the actor of the LAST event. Walk back to find an event
    where they are NOT the actor (or the second-to-last event), and bump
    actor_total_fouls in THAT event to 7. Then keep them as actor in
    the final event. This creates a sequence where someone with 7 fouls
    (already ejected) still acts.
    """
    if len(chain.events) < 2:
        return None

    events = [_clone(e) for e in chain.events]
    final = events[-1]
    target_actor = final.actor

    # Find an earlier event where actor_total_fouls can be bumped to 7
    # for the target actor. If the target actor doesn't appear earlier,
    # use the second-to-last event and force the bump anyway.
    bump_idx = -1
    for i in range(len(events) - 1):
        if events[i].actor == target_actor:
            bump_idx = i
            break
    if bump_idx == -1:
        bump_idx = len(events) - 2  # second to last; we'll force it

    # Mutate the location_context to show the foul count crossing the threshold.
    bumped = events[bump_idx]
    bumped.location_context = dict(bumped.location_context or {})
    bumped.location_context["actor_foul_count_after"] = 7
    bumped.location_context["terminal_action"] = "Foul"
    # Override actor of the bump event to be the target actor so the count
    # is for THIS player. This is necessary because the chain's terminal
    # action carries that field, and we need the same actor to commit a
    # subsequent action to violate the constraint.
    bumped.actor = target_actor

    # Final event keeps the target actor and shows them acting POST-7-fouls.
    # Cosmetic: the rendered chain will show foul=7 at idx=bump_idx, and the
    # same actor still acting at idx=final_idx. That is the violation.
    if final.location_context is not None:
        final.location_context = dict(final.location_context)
        final.location_context["actor_foul_count_after"] = 7

    new_chain = ChainCandidate(
        chain_id=chain.chain_id + "_adv",
        game_id=chain.game_id,
        cell=chain.cell,
        events=events,
        chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                        "violation_clause": "6_foul_ejection"},
    )
    return InjectionResult(
        chain=new_chain, cell="nba",
        violation_clause="A player with 6 fouls is ejected",
        violation_description=(
            f"Actor {target_actor} reaches 7 fouls at event {bump_idx} but "
            f"continues to act at event {len(events)-1}"
        ),
        target_actor=target_actor,
        target_event_idx=len(events) - 1,
    )


def inject_pubg_elimination_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    Plant: an actor is killed (appears as victim of an engage_decision) and
    then generates a subsequent action.

    Strategy: identify the actor of the FINAL event. Insert (or modify) an
    earlier event so it shows the target actor being eliminated. Specifically,
    we mutate an early event's location_context to say
    `victim=<target_actor> eliminated_by=...` so the rendering shows that
    target_actor was eliminated. Then the final event has them still acting,
    which violates "An eliminated player cannot act."
    """
    if len(chain.events) < 2:
        return None

    events = [_clone(e) for e in chain.events]
    target_actor = events[-1].actor

    # Modify the FIRST event to record the target actor as eliminated.
    first = events[0]
    first.location_context = dict(first.location_context or {})
    first.location_context["eliminated_player"] = target_actor
    first.location_context["elimination_note"] = (
        f"{target_actor} was eliminated"
    )
    # Tag the event_type with a clear violation marker the renderer surfaces.
    if first.event_type != "engage_decision":
        first.event_type = "engage_decision"

    new_chain = ChainCandidate(
        chain_id=chain.chain_id + "_adv",
        game_id=chain.game_id,
        cell=chain.cell,
        events=events,
        chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                        "violation_clause": "eliminated_cannot_act"},
    )
    return InjectionResult(
        chain=new_chain, cell="pubg",
        violation_clause="An eliminated player cannot act",
        violation_description=(
            f"Actor {target_actor} is recorded as eliminated at event 0 but "
            f"continues to act at event {len(events)-1}"
        ),
        target_actor=target_actor,
        target_event_idx=len(events) - 1,
    )


def inject_poker_fold_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    Plant: a player folds (action='f') and then bets/raises later in the
    same hand. Violates "A folded player cannot act again."

    Strategy: pick an actor who appears BOTH early and late in the chain.
    Modify their EARLY event so location_context shows action='f' (fold).
    Leave their later event as-is (showing them taking a non-fold action).
    """
    if len(chain.events) < 3:
        return None

    events = [_clone(e) for e in chain.events]

    # Find an actor with at least 2 actions (one early, one late).
    actor_first_idx: dict[str, int] = {}
    actor_last_idx: dict[str, int] = {}
    for i, ev in enumerate(events):
        if ev.actor not in actor_first_idx:
            actor_first_idx[ev.actor] = i
        actor_last_idx[ev.actor] = i

    candidates = [
        a for a, last in actor_last_idx.items()
        if actor_first_idx[a] != last
    ]
    if not candidates:
        return None

    target_actor = candidates[0]
    fold_idx = actor_first_idx[target_actor]
    later_idx = actor_last_idx[target_actor]

    # Mutate the early event to be a fold.
    fold_event = events[fold_idx]
    fold_event.event_type = "disengage_decision"
    fold_event.location_context = dict(fold_event.location_context or {})
    fold_event.location_context["action"] = "f"
    fold_event.location_context["bet_size_bb"] = 0.0

    # Force the later event to be a non-fold (bet/raise) — make it cbr.
    later_event = events[later_idx]
    later_event.event_type = "engage_decision"
    later_event.location_context = dict(later_event.location_context or {})
    later_event.location_context["action"] = "cbr"
    if "bet_size_bb" not in later_event.location_context:
        later_event.location_context["bet_size_bb"] = 5.0

    new_chain = ChainCandidate(
        chain_id=chain.chain_id + "_adv",
        game_id=chain.game_id,
        cell=chain.cell,
        events=events,
        chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                        "violation_clause": "folded_cannot_act"},
    )
    return InjectionResult(
        chain=new_chain, cell="poker",
        violation_clause="A folded player cannot act again",
        violation_description=(
            f"Actor {target_actor} folds at event {fold_idx} but bets "
            f"again at event {later_idx}"
        ),
        target_actor=target_actor,
        target_event_idx=later_idx,
    )


def inject_csgo_round_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    Plant: a kill event tagged with a round number that exceeds the total
    rounds in the match (e.g., round=99 in a 30-round game). Violates the
    implied "events occur within the round structure of the match."

    Note: this is a weaker violation than NBA/PUBG/Poker because the
    constraint context doesn't explicitly state "events must occur within
    valid round numbers." Included for completeness.
    """
    if not chain.events:
        return None
    events = [_clone(e) for e in chain.events]
    target = events[-1]
    target.location_context = dict(target.location_context or {})
    target.location_context["round"] = 99
    new_chain = ChainCandidate(
        chain_id=chain.chain_id + "_adv",
        game_id=chain.game_id,
        cell=chain.cell,
        events=events,
        chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                        "violation_clause": "round_out_of_bounds"},
    )
    return InjectionResult(
        chain=new_chain, cell="csgo",
        violation_clause="Events occur within match round structure",
        violation_description=(
            f"Event {len(events)-1} tagged with round=99 (impossible)"
        ),
        target_actor=target.actor,
        target_event_idx=len(events) - 1,
    )


def inject_rocket_league_demolished_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    Plant: a player gets demolished and then takes an action immediately
    after (RL respawns are 3 seconds, but adjacent events are typically
    less than that). Approximates "demolished player cannot act for ~3s."
    """
    if len(chain.events) < 2:
        return None
    events = [_clone(e) for e in chain.events]
    final = events[-1]
    target_actor = final.actor

    # Mutate the second-to-last event to record target as demolished.
    bump = events[-2]
    bump.event_type = "risk_accept"
    bump.location_context = dict(bump.location_context or {})
    bump.location_context["action_label"] = "demo_received"
    bump.location_context["demolished_player"] = target_actor

    new_chain = ChainCandidate(
        chain_id=chain.chain_id + "_adv",
        game_id=chain.game_id,
        cell=chain.cell,
        events=events,
        chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                        "violation_clause": "demolished_player_inaction"},
    )
    return InjectionResult(
        chain=new_chain, cell="rocket_league",
        violation_clause="Demolished player cannot act for ~3 seconds",
        violation_description=(
            f"Actor {target_actor} demolished at event {len(events)-2} but "
            f"acts at event {len(events)-1}"
        ),
        target_actor=target_actor,
        target_event_idx=len(events) - 1,
    )


# Per-cell dispatch
INJECTORS = {
    "nba": inject_nba_foul_violation,
    "pubg": inject_pubg_elimination_violation,
    "poker": inject_poker_fold_violation,
    "csgo": inject_csgo_round_violation,
    "rocket_league": inject_rocket_league_demolished_violation,
}


def inject(cell: str, chain: ChainCandidate) -> InjectionResult | None:
    fn = INJECTORS.get(cell)
    if fn is None:
        logger.warning(f"No violation injector for cell {cell}")
        return None
    return fn(chain)


def _clone(e: GameEvent) -> GameEvent:
    """Deep-copy an event so injections don't mutate shared references."""
    return GameEvent(
        timestamp=e.timestamp,
        event_type=e.event_type,
        actor=e.actor,
        location_context=copy.deepcopy(e.location_context),
        raw_data_blob=copy.deepcopy(e.raw_data_blob),
        cell=e.cell,
        game_id=e.game_id,
        sequence_idx=e.sequence_idx,
        actor_team=e.actor_team,
        phase=e.phase,
        metadata=copy.deepcopy(getattr(e, "metadata", {}) or {}),
    )
