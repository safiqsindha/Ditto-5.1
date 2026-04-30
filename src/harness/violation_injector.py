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
    Plant: a player is eliminated and then generates a subsequent action.
    Locally verifiable: every event after the elimination shows the marker
    `eliminated_player=<actor>` directly in the rendered chain, so the
    model can check per-event without cross-event memory.

    Strategy: pick the final event's actor as the target. Mutate ALL events
    AFTER the first to include `eliminated_player=<target>` and
    `eliminated_at_event=0` markers in their location_context. The
    PUBGPromptBuilder.format_event override (added 2026-04-29 alongside
    A4-A7) surfaces this marker explicitly so the 6-key context cap doesn't
    drop it.
    """
    if len(chain.events) < 2:
        return None

    events = [_clone(e) for e in chain.events]
    target_actor = events[-1].actor

    # Mark the FIRST event as the elimination event.
    first = events[0]
    first.location_context = dict(first.location_context or {})
    first.location_context["eliminates_player"] = target_actor
    first.event_type = "engage_decision"

    # Mark every SUBSEQUENT event with the elimination memory so the renderer
    # surfaces it locally. This is the local-violation pattern: the model
    # doesn't have to remember event 0; the marker is on every event.
    for i, ev in enumerate(events[1:], start=1):
        ev.location_context = dict(ev.location_context or {})
        ev.location_context["already_eliminated_player"] = target_actor
        ev.location_context["eliminated_at_event"] = 0

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
            f"Actor {target_actor} eliminated at event 0 but continues to "
            f"act at later events; marker visible per-event"
        ),
        target_actor=target_actor,
        target_event_idx=len(events) - 1,
    )


def inject_poker_folded_acts_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    TIER-1 candidate per the v3 reviewer synthesis (Gemini + Opus + ChatGPT,
    DECISION_LOG D-43). The constraint context says "A folded player cannot
    act again." This injector mirrors the PUBG elimination pattern exactly:

      1. Pick an actor with at least 2 events in the chain.
      2. Mutate their EARLIEST event to be `action=f` (fold).
      3. On every SUBSEQUENT event by that actor, attach a per-event marker
         `NOTE=Player_X_already_folded` so PokerPromptBuilder.format_event
         (override added 2026-04-29) renders the marker on each line.

    The model's check then collapses to a unary per-event predicate:
    "if NOTE says folded, actor must not act." This is structurally
    identical to PUBG's `NOTE=Player_3_already_eliminated` pattern that
    achieved 100% detection.

    The original v1 fold-then-bet injector failed at ~50% because the
    fold event had no persistent marker — the model had to remember the
    earlier fold to detect the later violation. Adding the per-event
    marker eliminates the cross-event memory load.
    """
    if len(chain.events) < 3:
        return None

    events = [_clone(e) for e in chain.events]

    # Find an actor with ≥2 events (so we can fold them early + see them act later).
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

    # Mutate the earliest event to be a fold.
    fold_event = events[fold_idx]
    fold_event.event_type = "disengage_decision"
    fold_event.location_context = dict(fold_event.location_context or {})
    fold_event.location_context["action"] = "f"
    fold_event.location_context["bet_size_bb"] = 0.0

    # Mark every subsequent event from the SAME actor with the persistent
    # folded-status marker. Subsequent events from OTHER actors don't get
    # the marker (they didn't fold).
    for i in range(fold_idx + 1, len(events)):
        if events[i].actor == target_actor:
            events[i].location_context = dict(events[i].location_context or {})
            events[i].location_context["already_folded"] = True
            # Force the actor's later event to be a non-fold (cbr/raise) so
            # the violation is observable: folded player taking action.
            events[i].event_type = "engage_decision"
            if events[i].location_context.get("action") in (None, "f", ""):
                events[i].location_context["action"] = "cbr"
                events[i].location_context.setdefault("bet_size_bb", 5.0)

    new_chain = ChainCandidate(
        chain_id=chain.chain_id + "_adv",
        game_id=chain.game_id,
        cell=chain.cell,
        events=events,
        chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                        "violation_clause": "folded_cannot_act_v3"},
    )
    later_idx = actor_last_idx[target_actor]
    return InjectionResult(
        chain=new_chain, cell="poker",
        violation_clause="A folded player cannot act again",
        violation_description=(
            f"Actor {target_actor} folds at event {fold_idx} but acts at "
            f"event {later_idx}; per-event marker visible on every "
            f"post-fold event for that actor"
        ),
        target_actor=target_actor,
        target_event_idx=later_idx,
    )


def inject_poker_overbet_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    PER-EVENT poker violation anchored to a STATED constraint clause.
    The locked Poker constraint says: "Cannot wager more than held."
    Plant a single event where bet_size_bb > stack_bb — the model can
    detect the violation by reading ONE event line; no cross-event
    memory or aggregation required.

    Strategy: find an event with a non-trivial stack_bb. Set
    bet_size_bb = stack_bb * 2 (clearly larger than held). Both fields
    are rendered per-event in PokerPromptBuilder.format_event, so the
    violation is fully visible on a single line.

    This replaces inject_poker_stack_arithmetic_violation, which v2
    diagnostic showed at 45% detection — the stack-monotonicity rule
    was NOT in the constraint context, so the model had no anchor to
    flag it.
    """
    if len(chain.events) < 2:
        return None

    events = [_clone(e) for e in chain.events]

    # Find an event with a usable stack_bb. Default to mid-chain.
    target_idx = len(events) // 2
    for i, ev in enumerate(events):
        ctx = ev.location_context or {}
        try:
            stack = float(ctx.get("stack_bb", 0))
        except (TypeError, ValueError):
            continue
        if stack > 5:
            target_idx = i
            break

    target = events[target_idx]
    target.location_context = dict(target.location_context or {})
    try:
        stack_val = float(target.location_context.get("stack_bb", 50.0))
    except (TypeError, ValueError):
        stack_val = 50.0
    overbet = round(stack_val * 2.0 + 10.0, 1)
    target.location_context["bet_size_bb"] = overbet
    target.location_context["action"] = "cbr"
    target.event_type = "engage_decision"

    new_chain = ChainCandidate(
        chain_id=chain.chain_id + "_adv",
        game_id=chain.game_id,
        cell=chain.cell,
        events=events,
        chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                        "violation_clause": "wager_exceeds_stack"},
    )
    return InjectionResult(
        chain=new_chain, cell="poker",
        violation_clause="Cannot wager more than held",
        violation_description=(
            f"Event {target_idx}: bet_size_bb={overbet} but stack_bb="
            f"{stack_val} (cannot wager more than held)"
        ),
        target_actor=target.actor,
        target_event_idx=target_idx,
    )


def inject_poker_stack_arithmetic_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    LEGACY — see inject_poker_overbet_violation. Diagnostic v2 showed this
    detected at only 45% because the stack-monotonicity rule isn't in the
    locked constraint context. Kept for reference only.
    """
    if len(chain.events) < 3:
        return None

    events = [_clone(e) for e in chain.events]

    # Find an actor with at least 2 events in the chain.
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
    earlier_idx = actor_first_idx[target_actor]
    later_idx = actor_last_idx[target_actor]

    earlier_stack = (events[earlier_idx].location_context or {}).get("stack_bb", 100.0)
    try:
        earlier_stack_f = float(earlier_stack)
    except (TypeError, ValueError):
        earlier_stack_f = 100.0

    # Force the later event's stack to be HIGHER than the earlier — impossible
    # within a single hand without a hand-end win event in between.
    later = events[later_idx]
    later.location_context = dict(later.location_context or {})
    later.location_context["stack_bb"] = round(earlier_stack_f + 50.0, 1)

    new_chain = ChainCandidate(
        chain_id=chain.chain_id + "_adv",
        game_id=chain.game_id,
        cell=chain.cell,
        events=events,
        chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                        "violation_clause": "stack_arithmetic"},
    )
    return InjectionResult(
        chain=new_chain, cell="poker",
        violation_clause="Stacks can only decrease within a hand",
        violation_description=(
            f"Actor {target_actor} stack at event {earlier_idx} = "
            f"{earlier_stack_f:.1f}bb, at event {later_idx} = "
            f"{earlier_stack_f + 50.0:.1f}bb (impossible mid-hand)"
        ),
        target_actor=target_actor,
        target_event_idx=later_idx,
    )


def inject_poker_fold_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    GLOBAL (cross-event-state-tracking) violation kept for comparison with
    the local stack-arithmetic injector. Diagnostic v1 showed Haiku catches
    only ~50% of these because the model has to remember an earlier fold
    to detect a later bet by the same actor.

    Plant: a player folds (action='f') and then bets/raises later in the
    same hand. Violates "A folded player cannot act again."
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


def inject_csgo_eliminated_acts_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    TIER-1 candidate per v3 reviewer synthesis (DECISION_LOG D-43). The
    constraint context says "Eliminated players don't respawn until next
    round." Mirrors PUBG/Poker pattern:

      1. Pick an actor with ≥2 events.
      2. Mutate their first event to record an elimination marker.
      3. On every SUBSEQUENT event from that actor, attach
         `NOTE=Player_X_eliminated_this_round` — logically inconsistent.

    The model's check is a unary per-event predicate.

    Note: this still has the synthetic-timestamp confound (all events
    stamped to round-start t=115.5s), but if the elimination marker is
    visible per-event, the discrimination signal between adversarial
    and clean chains should still surface even on top of that artifact.
    The diagnostic --ignore-timestamps flag (added 2026-04-29) provides
    an additional instrument correction.
    """
    if len(chain.events) < 3:
        return None

    events = [_clone(e) for e in chain.events]

    # Find an actor with ≥2 events.
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
    elim_idx = actor_first_idx[target_actor]

    elim_event = events[elim_idx]
    elim_event.location_context = dict(elim_event.location_context or {})
    elim_event.location_context["eliminates_player"] = target_actor
    elim_event.location_context["action_label"] = "eliminated"

    for i in range(elim_idx + 1, len(events)):
        if events[i].actor == target_actor:
            events[i].location_context = dict(events[i].location_context or {})
            events[i].location_context["already_eliminated_this_round"] = True

    new_chain = ChainCandidate(
        chain_id=chain.chain_id + "_adv",
        game_id=chain.game_id,
        cell=chain.cell,
        events=events,
        chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                        "violation_clause": "eliminated_no_respawn_v3"},
    )
    later_idx = actor_last_idx[target_actor]
    return InjectionResult(
        chain=new_chain, cell="csgo",
        violation_clause="Eliminated players don't respawn until next round",
        violation_description=(
            f"Actor {target_actor} eliminated at event {elim_idx} but acts "
            f"at event {later_idx}; marker visible per-event"
        ),
        target_actor=target_actor,
        target_event_idx=later_idx,
    )


def inject_csgo_team_flip_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    LOCALLY-VERIFIABLE CS:GO violation within the FACEIT aggregate-stat data
    ceiling. The constraint context implies team integrity ("T wins by ...
    eliminating CTs", "CT wins by ..."). Plant a team-membership flip: have
    the SAME actor appear with two different team labels in the chain.

    Each event renders team=team_A or team=team_B (CF-4=B anonymized via
    md5 hash of the underlying team_id). If the same actor is seen on
    BOTH team_A and team_B within one chain, that violates implicit team
    integrity. Locally checkable: per-event (actor, team) pair tracked.

    Note: this is the cleanest local violation we can construct from
    FACEIT aggregate stats. A v5.1 with awpy demo extraction would unlock
    bomb-site violations, alive/dead state, and other directly-stated
    constraint clauses.
    """
    if len(chain.events) < 3:
        return None
    events = [_clone(e) for e in chain.events]

    # Find an actor that appears at least twice
    actor_indices: dict[str, list[int]] = {}
    for i, ev in enumerate(events):
        actor_indices.setdefault(ev.actor, []).append(i)
    candidates = [a for a, idxs in actor_indices.items() if len(idxs) >= 2]
    if not candidates:
        # Fall back: just plant team_C on a single event
        target = events[-1]
        target.location_context = dict(target.location_context or {})
        target.location_context["team_id"] = "team_INVALID"
        new_chain = ChainCandidate(
            chain_id=chain.chain_id + "_adv",
            game_id=chain.game_id,
            cell=chain.cell,
            events=events,
            chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                            "violation_clause": "invalid_team_id"},
        )
        return InjectionResult(
            chain=new_chain, cell="csgo",
            violation_clause="Players belong to T or CT (not a third team)",
            violation_description=(
                f"Event {len(events)-1} has team='team_INVALID' "
                f"(not T or CT)"
            ),
            target_actor=target.actor,
            target_event_idx=len(events) - 1,
        )

    target_actor = candidates[0]
    indices = actor_indices[target_actor]

    # Get this actor's current team from their first event.
    first_idx = indices[0]
    first_team = (events[first_idx].location_context or {}).get("team_id", "")

    # Flip the LAST occurrence to a different team_id (and the renderer's
    # md5 hash will assign it to the opposite team_A/team_B slot).
    flip_idx = indices[-1]
    flip = events[flip_idx]
    flip.location_context = dict(flip.location_context or {})
    # Use a synthetic team_id that hashes to the OPPOSITE slot. We don't
    # know which slot first_team mapped to, so just append "_FLIPPED" — the
    # renderer's hash will deterministically produce a different slot most
    # of the time.
    flip.location_context["team_id"] = first_team + "_FLIPPED"

    new_chain = ChainCandidate(
        chain_id=chain.chain_id + "_adv",
        game_id=chain.game_id,
        cell=chain.cell,
        events=events,
        chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                        "violation_clause": "team_membership_flip"},
    )
    return InjectionResult(
        chain=new_chain, cell="csgo",
        violation_clause="Player belongs to a single team (T or CT) for the match",
        violation_description=(
            f"Actor {target_actor} appears with mismatched team_ids in "
            f"events {first_idx} and {flip_idx}"
        ),
        target_actor=target_actor,
        target_event_idx=flip_idx,
    )


def inject_csgo_round_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    LEGACY round-out-of-bounds injector. Constraint context doesn't mention
    round bounds, so this is a weak violation. Kept for reference only —
    v2 dispatch uses inject_csgo_team_flip_violation.
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


def inject_rocket_league_post_goal_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    TIER-1 candidate per v3 reviewer synthesis (DECISION_LOG D-43). The
    constraint context says "A goal resets ball and player positions for
    a kickoff." This injector applies the PUBG/Poker derived-state-marker
    pattern uniformly:

      1. Pick an event mid-chain to mark as `goal_scored=True`.
      2. On every SUBSEQUENT event, attach `NOTE=pre_goal_state` — this is
         logically inconsistent because a goal MUST reset positions, so any
         pre-goal state appearing post-goal violates the rule.

    The model's check is a unary per-event predicate: "if NOTE says
    pre_goal_state on a post-goal event, that's a reset violation." No
    cross-event memory required (the marker is on every event).

    The original v2 team-size injector failed at 25% because counting
    distinct actors per team across 12 events is multi-event aggregation
    (Tier 3). This Tier-1 reframe uses ONLY existing data + derived state
    markers — no fabricated boost levels or positions.
    """
    if len(chain.events) < 3:
        return None

    events = [_clone(e) for e in chain.events]

    # Pick a goal point ~1/3 through the chain so we have enough events
    # before AND after to make the violation observable.
    goal_idx = max(1, len(events) // 3)
    goal_event = events[goal_idx]
    goal_event.event_type = "objective_capture"
    goal_event.location_context = dict(goal_event.location_context or {})
    goal_event.location_context["action_label"] = "goal_scored"
    goal_event.location_context["goal_marker"] = True

    # Mark every event AFTER the goal with `pre_goal_state` — this is
    # logically inconsistent because the goal should have reset positions.
    for i in range(goal_idx + 1, len(events)):
        events[i].location_context = dict(events[i].location_context or {})
        events[i].location_context["pre_goal_state_persisting"] = True

    new_chain = ChainCandidate(
        chain_id=chain.chain_id + "_adv",
        game_id=chain.game_id,
        cell=chain.cell,
        events=events,
        chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                        "violation_clause": "goal_reset_violated"},
    )
    return InjectionResult(
        chain=new_chain, cell="rocket_league",
        violation_clause="A goal resets ball and player positions for a kickoff",
        violation_description=(
            f"Goal scored at event {goal_idx}, but events "
            f"{goal_idx+1}..{len(events)-1} carry pre_goal_state markers "
            f"(should have reset)"
        ),
        target_actor=goal_event.actor,
        target_event_idx=goal_idx,
    )


def inject_rocket_league_team_size_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    LOCALLY-VERIFIABLE Rocket League violation. Constraint says "Teams of 3
    score by hitting the ball". Inject a 4th player on the same team_color
    as an existing actor. Every event in the chain renders team={blue|orange}
    so the model can locally count distinct actors per team.

    Strategy: identify existing actors and their teams. Pick the team with
    fewer unique actors. Replace the actor on a chosen event with a fake
    fourth-player ID assigned to that team. Now the chain shows 4 distinct
    actors on the same team, violating "teams of 3".
    """
    if len(chain.events) < 4:
        return None

    events = [_clone(e) for e in chain.events]

    # Catalogue (team_color -> set of unique actors).
    team_actors: dict[str, set] = {"blue": set(), "orange": set()}
    for ev in events:
        team = (ev.location_context or {}).get("team_color") or ev.actor_team
        if team in team_actors:
            team_actors[team].add(ev.actor)
        elif team:
            team_actors.setdefault(team, set()).add(ev.actor)

    # Pick the team with the most unique actors so we just need to add 1.
    target_team = max(team_actors, key=lambda t: len(team_actors[t]))
    if len(team_actors[target_team]) < 1:
        return None

    fake_actor = f"injected_{target_team}_4th"

    # Replace one mid-chain event's actor with the fake 4th-player so
    # the team now has 4 unique actors visible in the rendered chain.
    middle_idx = len(events) // 2
    target_event = events[middle_idx]
    target_event.actor = fake_actor
    target_event.actor_team = target_team
    target_event.location_context = dict(target_event.location_context or {})
    target_event.location_context["team_color"] = target_team

    new_chain = ChainCandidate(
        chain_id=chain.chain_id + "_adv",
        game_id=chain.game_id,
        cell=chain.cell,
        events=events,
        chain_metadata={**(chain.chain_metadata or {}), "adversarial": True,
                        "violation_clause": "team_size_exceeds_3"},
    )
    return InjectionResult(
        chain=new_chain, cell="rocket_league",
        violation_clause="Teams of 3 score by hitting the ball",
        violation_description=(
            f"Team '{target_team}' now shows 4 unique actors in the chain "
            f"(violates teams-of-3); injected actor '{fake_actor}' at "
            f"event {middle_idx}"
        ),
        target_actor=fake_actor,
        target_event_idx=middle_idx,
    )


def inject_rocket_league_demolished_violation(chain: ChainCandidate) -> InjectionResult | None:
    """
    LEGACY: demolished-player-acts violation. Diagnostic v1 didn't reach RL,
    but this rule isn't in the locked constraint context, so the model has
    no anchor to detect it. Kept for reference only — not used in v2 dispatch.
    """
    if len(chain.events) < 2:
        return None
    events = [_clone(e) for e in chain.events]
    final = events[-1]
    target_actor = final.actor

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


# Per-cell dispatch — v2 uses LOCALLY-VERIFIABLE injectors per the local-vs-
# global insight from diagnostic v1 (NBA local works at 95-100%; Poker
# global fails at 45-55%). v2 swaps Poker, RL, CS:GO injectors to local
# variants. NBA stayed on its v1 injector because it was already local.
INJECTORS = {
    "nba": inject_nba_foul_violation,                      # Tier 1 (per-event simple) — 95-100%
    "pubg": inject_pubg_elimination_violation,             # Tier 1 (per-event marker) — 100%
    "poker": inject_poker_folded_acts_violation,           # v4: per-event marker mirroring PUBG — was overbet (Tier 2)
    "csgo": inject_csgo_eliminated_acts_violation,         # v3: per-event marker mirroring PUBG — was team_flip
    "rocket_league": inject_rocket_league_post_goal_violation,  # v3: per-event derived-state marker — was team_size (Tier 3)
}

# Legacy global injectors for comparison studies
LEGACY_GLOBAL_INJECTORS = {
    "poker": inject_poker_fold_violation,
    "rocket_league": inject_rocket_league_demolished_violation,
    "csgo": inject_csgo_round_violation,
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
