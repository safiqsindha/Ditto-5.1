"""
TranslationFunction (T) interface for v5.

T is the domain-specific function that converts a normalized GameEvent stream
into ChainCandidate objects. Each domain cell has its own T implementation.

Phase B decisions locked 2026-04-27 (both authors):
  CF-1=A (natural language constraint), CF-2=D (binary classify),
  CF-3=A (shuffled controls), CF-4=B (domain-only provenance)
  Per-cell N: fortnite=8, nba=5, csgo=10, rocket_league=12, poker=8

To register T with CellRunner:
  runner.register_cell(cell_id, MyT())
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict

from ..common.schema import ChainCandidate, EventStream, GameEvent


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


# ---------------------------------------------------------------------------
# Fortnite T  ⚠ LEGACY — superseded by PUBGT per A2 (D-35).
# Kept as a test fixture and historical reference for the F-1/F-2/F-3
# decisions. Not included in active evaluation runs (PUBG inherits the
# battle-royale slot). Do not use for new work — use PUBGT instead.
# Phase B decisions (preserved for the historical record):
#   F-1 storm-zone boundary + elimination causality,
#   F-2 storm-rotation phase, F-3 N=8
# ---------------------------------------------------------------------------

_FORTNITE_STORM_TRIGGERS = frozenset({"zone_enter", "zone_exit", "position_commit"})
_FORTNITE_BUILD_TRIGGERS = frozenset({"build_decision", "resource_spend", "resource_budget"})
_FORTNITE_N = 8


class FortniteT(TranslationFunction):
    """
    Extract storm-rotation phase windows from a Fortnite event stream.

    Algorithm: for each storm-boundary event (zone_enter / zone_exit /
    position_commit), emit a window of _FORTNITE_N events centered on it.
    Windows are de-duplicated by start index so consecutive triggers don't
    produce overlapping chains.

    Constraint (F-1): storm-zone boundary (player must be inside zone or take
    damage) and elimination causality (eliminated player cannot act).
    Build-cost rule noted for future testing (note: F-1 decision).
    """

    @property
    def cell(self) -> str:
        return "fortnite"

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        events = stream.events
        if not events:
            return []

        half = _FORTNITE_N // 2
        seen_starts: set[int] = set()
        chains: list[ChainCandidate] = []

        for i, ev in enumerate(events):
            if ev.event_type not in _FORTNITE_STORM_TRIGGERS:
                continue
            start = max(0, i - half)
            end = start + _FORTNITE_N
            if end > len(events):
                end = len(events)
                start = max(0, end - _FORTNITE_N)
            if start in seen_starts:
                continue
            seen_starts.add(start)

            window = events[start:end]
            if len(window) < 2:
                continue

            chains.append(ChainCandidate(
                chain_id=self._chain_id(stream.game_id, f"fn_storm_{start}"),
                game_id=stream.game_id,
                cell="fortnite",
                events=window,
                chain_metadata={
                    "chain_type": "storm_rotation",
                    "trigger_type": ev.event_type,
                    "trigger_idx": i,
                    "window_start": start,
                    "window_size": len(window),
                },
            ))

        return chains


# ---------------------------------------------------------------------------
# ME-FN-1: FortniteBuildCostT
# Micro-experiment: centers windows on build_decision / resource_spend /
#   resource_budget instead of storm-boundary events. Tests whether build-cost
#   constraint (can't spend more materials than held) produces a different
#   distributional signal than the primary storm-rotation constraint.
# ---------------------------------------------------------------------------


class FortniteBuildCostT(TranslationFunction):
    """
    ME-FN-1: Extract build-cost decision windows from a Fortnite event stream.

    Same windowing algorithm as FortniteT (centered, de-duplicated, N=8) but
    triggered by build_decision / resource_spend / resource_budget events
    instead of storm-boundary events.

    Constraint (ME-FN-1): build-cost (player cannot place structures that cost
    more materials than they currently hold).
    """

    @property
    def cell(self) -> str:
        return "fortnite"

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        events = stream.events
        if not events:
            return []

        half = _FORTNITE_N // 2
        seen_starts: set[int] = set()
        chains: list[ChainCandidate] = []

        for i, ev in enumerate(events):
            if ev.event_type not in _FORTNITE_BUILD_TRIGGERS:
                continue
            start = max(0, i - half)
            end = start + _FORTNITE_N
            if end > len(events):
                end = len(events)
                start = max(0, end - _FORTNITE_N)
            if start in seen_starts:
                continue
            seen_starts.add(start)

            window = events[start:end]
            if len(window) < 2:
                continue

            chains.append(ChainCandidate(
                chain_id=self._chain_id(stream.game_id, f"fn_build_{start}"),
                game_id=stream.game_id,
                cell="fortnite",
                events=window,
                chain_metadata={
                    "chain_type": "build_cost",
                    "trigger_type": ev.event_type,
                    "trigger_idx": i,
                    "window_start": start,
                    "window_size": len(window),
                    "me": "ME-FN-1",
                },
            ))

        return chains


# ---------------------------------------------------------------------------
# PUBG T
# Per A2 (D-35) — replaces Fortnite as the battle-royale cell. Mirrors
# FortniteT structurally because the constraint type (zone-rotation +
# elimination causality) is preserved across the swap. Bot filter (D-36)
# is enforced upstream in the extractor; PUBGT receives only human-attributed
# events and need not re-filter.
# Decisions:
#   - Triggers: zone_enter / zone_exit / position_commit (mirrors F-1)
#   - N=8 (mirrors F-3, the BR zone-rotation window)
# ---------------------------------------------------------------------------

_PUBG_ZONE_TRIGGERS = frozenset({"zone_enter", "zone_exit", "position_commit"})
_PUBG_N = 8


class PUBGT(TranslationFunction):
    """
    Extract zone-rotation phase windows from a PUBG event stream.

    Algorithm: for each zone-boundary event (zone_enter / zone_exit /
    position_commit), emit a window of _PUBG_N events centered on it.
    Windows are de-duplicated by start index so consecutive triggers don't
    produce overlapping chains. Identical structure to FortniteT —
    the constraint shape is preserved across the A2 cell swap.

    Constraint: blue-zone boundary (player must stay inside the safe zone
    or take damage that scales per phase) + elimination causality
    (eliminated player cannot generate further actions).
    """

    @property
    def cell(self) -> str:
        return "pubg"

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        events = stream.events
        if not events:
            return []

        half = _PUBG_N // 2
        seen_starts: set[int] = set()
        chains: list[ChainCandidate] = []

        for i, ev in enumerate(events):
            if ev.event_type not in _PUBG_ZONE_TRIGGERS:
                continue
            start = max(0, i - half)
            end = start + _PUBG_N
            if end > len(events):
                end = len(events)
                start = max(0, end - _PUBG_N)
            if start in seen_starts:
                continue
            seen_starts.add(start)

            window = events[start:end]
            if len(window) < 2:
                continue

            chains.append(ChainCandidate(
                chain_id=self._chain_id(stream.game_id, f"pubg_zone_{start}"),
                game_id=stream.game_id,
                cell="pubg",
                events=window,
                chain_metadata={
                    "chain_type": "zone_rotation",
                    "trigger_type": ev.event_type,
                    "trigger_idx": i,
                    "window_start": start,
                    "window_size": len(window),
                },
            ))

        return chains


# ---------------------------------------------------------------------------
# NBA T
# Phase B decisions: N-1 shot-clock + foul-out, N-2 consecutive possessions
#   from single quarter, N-3 N=5
# ---------------------------------------------------------------------------

_NBA_N = 5


class NBAT(TranslationFunction):
    """
    Extract consecutive-possession windows from an NBA event stream.

    Events from the NBA extractor are already at possession-level (one event
    per possession, grouped by shot-clock boundary). T groups them by quarter
    (location_context["period"]) and slides non-overlapping windows of N=5
    possessions. Each window becomes one ChainCandidate.

    Constraint (N-1): shot-clock (must shoot within 24s) + foul-out
    (6 fouls = ejection). Possession turnover rule is implicit.
    """

    @property
    def cell(self) -> str:
        return "nba"

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        events = stream.events
        if not events:
            return []

        # Group by period (quarter)
        by_period: dict[int, list[GameEvent]] = defaultdict(list)
        for ev in events:
            period = int(ev.location_context.get("period", 0))
            by_period[period].append(ev)

        chains: list[ChainCandidate] = []
        for period in sorted(by_period):
            p_events = by_period[period]
            # Non-overlapping windows of N=5
            for start in range(0, len(p_events) - _NBA_N + 1, _NBA_N):
                window = p_events[start:start + _NBA_N]
                chains.append(ChainCandidate(
                    chain_id=self._chain_id(
                        stream.game_id, f"nba_q{period}_p{start}"
                    ),
                    game_id=stream.game_id,
                    cell="nba",
                    events=window,
                    chain_metadata={
                        "chain_type": "possession_window",
                        "period": period,
                        "possession_start": start,
                        "n_possessions": len(window),
                    },
                ))

        return chains


# ---------------------------------------------------------------------------
# CS:GO T
# Phase B decisions: C-1 respawn-disabled + bomb-objective, C-2 full round,
#   C-3 N=10
# ---------------------------------------------------------------------------

class CSGOT(TranslationFunction):
    """
    Extract full-round chains from a CS:GO/CS2 event stream.

    Each round is one ChainCandidate containing all events within that round
    (kills, grenades, bomb events, buy phase). The round number is read from
    location_context["round"] or the event's phase field.

    ChainBuilder trims each round's event list to N=10.

    Constraint (C-1): respawn-disabled (eliminated players sit out the round)
    + bomb-objective (Terrorists detonate; Counter-Terrorists defuse or
    eliminate).
    """

    @property
    def cell(self) -> str:
        return "csgo"

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        events = stream.events
        if not events:
            return []

        by_round: dict[int, list[GameEvent]] = defaultdict(list)
        for ev in events:
            rnum = int(ev.location_context.get("round", -1))
            if rnum < 0 and ev.phase and ev.phase.startswith("round_"):
                try:
                    rnum = int(ev.phase.split("_", 1)[1])
                except (IndexError, ValueError):
                    rnum = -1
            if rnum >= 0:
                by_round[rnum].append(ev)

        chains: list[ChainCandidate] = []
        for rnum in sorted(by_round):
            r_events = by_round[rnum]
            if len(r_events) < 2:
                continue
            chains.append(ChainCandidate(
                chain_id=self._chain_id(stream.game_id, f"csgo_r{rnum}"),
                game_id=stream.game_id,
                cell="csgo",
                events=r_events,
                chain_metadata={
                    "chain_type": "full_round",
                    "round": rnum,
                    "n_events": len(r_events),
                },
            ))

        return chains


# ---------------------------------------------------------------------------
# Rocket League T
# Phase B decisions: R-1 boost-economy + goal-causality, R-2 play =
#   possession-to-possession or goal, R-3 N=12
# Note: per-player chain variant flagged as ME-RL-1 for v6.
# ---------------------------------------------------------------------------

class RocketLeagueT(TranslationFunction):
    """
    Extract play-level chains from a Rocket League event stream.

    A "play" is the sequence of events between two consecutive goal events
    (objective_capture). The first play starts at index 0; the last play ends
    at the final event. Each play becomes one ChainCandidate; ChainBuilder
    trims to N=12 (keeping a representative mix of hit + boost events).

    Constraint (R-1): boost-economy (max 100, depletes on use, replenished
    from pads) + goal-causality (goal = ball crosses line, resets position).

    Micro-experiment ME-RL-1 (v6): per-player chains within each play to test
    individual boost-decision detection.
    """

    @property
    def cell(self) -> str:
        return "rocket_league"

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        events = stream.events
        if not events:
            return []

        # Goal events mark play boundaries
        goal_indices = [
            i for i, e in enumerate(events)
            if e.event_type == "objective_capture"
        ]

        # Build play slices between goals
        boundaries = [0] + [gi + 1 for gi in goal_indices] + [len(events)]
        chains: list[ChainCandidate] = []
        for play_num, (start, end) in enumerate(zip(boundaries, boundaries[1:], )):
            play_events = events[start:end]
            if len(play_events) < 2:
                continue
            chains.append(ChainCandidate(
                chain_id=self._chain_id(stream.game_id, f"rl_play_{play_num}"),
                game_id=stream.game_id,
                cell="rocket_league",
                events=play_events,
                chain_metadata={
                    "chain_type": "play_sequence",
                    "play": play_num,
                    "n_events": len(play_events),
                    "ends_in_goal": play_num < len(goal_indices),
                },
            ))

        return chains


# ---------------------------------------------------------------------------
# Rocket League per-player T  — ME-RL-1 (pre-registered micro-experiment)
# Within each play, one ChainCandidate per unique actor.
# ---------------------------------------------------------------------------

class RocketLeaguePlayerT(TranslationFunction):
    """
    ME-RL-1: per-player chains within each Rocket League play.

    Identical play-boundary logic to RocketLeagueT (objective_capture events
    delimit plays), but within each play the events are partitioned by actor
    so each player's actions become a separate ChainCandidate.

    This lets us test whether the boost-economy constraint (R-1) is detectable
    at the individual-player level rather than the whole-team level. A player
    who picks up boost and immediately expends it without rotating back is
    individually violating the soft rotation constraint even if teammates
    cover — per-player chains surface this.

    Chain ID: {game_id}__rl_pp_{play}_{actor_slot} where actor_slot is a
    zero-based index into the sorted set of actors in that play (CF-4=B safe;
    no real names in the ID).

    ChainBuilder trims to N=12 (R-3).
    """

    @property
    def cell(self) -> str:
        return "rocket_league"

    def translate(self, stream: EventStream) -> list[ChainCandidate]:
        events = stream.events
        if not events:
            return []

        goal_indices = [
            i for i, e in enumerate(events)
            if e.event_type == "objective_capture"
        ]
        boundaries = [0] + [gi + 1 for gi in goal_indices] + [len(events)]

        chains: list[ChainCandidate] = []
        for play_num, (start, end) in enumerate(zip(boundaries, boundaries[1:], )):
            play_events = events[start:end]
            if len(play_events) < 2:
                continue

            # Split by actor (deterministic order: sorted actor set for stable IDs)
            by_actor: dict[str, list[GameEvent]] = {}
            for ev in play_events:
                by_actor.setdefault(ev.actor, []).append(ev)

            for actor_slot, actor in enumerate(sorted(by_actor)):
                actor_events = by_actor[actor]
                if len(actor_events) < 2:
                    continue
                chains.append(ChainCandidate(
                    chain_id=self._chain_id(
                        stream.game_id, f"rl_pp_{play_num}_{actor_slot}"
                    ),
                    game_id=stream.game_id,
                    cell="rocket_league",
                    events=actor_events,
                    chain_metadata={
                        "chain_type": "per_player_play",
                        "play": play_num,
                        "actor_slot": actor_slot,
                        "n_events": len(actor_events),
                        "ends_in_goal": play_num < len(goal_indices),
                        "me": "ME-RL-1",
                    },
                ))

        return chains


# ---------------------------------------------------------------------------
# Poker T
# Phase B decisions: P-1 player-hand pair, P-2 decision events only,
#   P-3 N=8, P-4 ≥3 actions filter, P-5 NLHE constraint, P-6 overrides
# ---------------------------------------------------------------------------

from ..cells.poker.poker_t import PokerPerSessionT, PokerT  # noqa: E402


# ---------------------------------------------------------------------------
# Registry — maps cell_id → stub instance (for tests; use real T in eval)
# ---------------------------------------------------------------------------

DOMAIN_T_STUBS: dict[str, TranslationFunction] = {
    "pubg": PUBGT(),                # active battle-royale cell (per A2/D-35)
    "nba": NBAT(),
    "csgo": CSGOT(),
    "rocket_league": RocketLeagueT(),
    "poker": PokerT(),
    # FortniteT kept as legacy for tests but no longer in the active 5 cells
    # per A2 (Epic CDN locked down public chunk access). Do not include in
    # ALL_CELLS or in evaluation runs.
    "fortnite": FortniteT(),
}

# ME-RL-1 variant registry — per-player RL chains (micro-experiment).
# Use in place of DOMAIN_T_STUBS["rocket_league"] to run the ME-RL-1 comparison.
DOMAIN_T_ME_RL1: dict[str, TranslationFunction] = {
    **DOMAIN_T_STUBS,
    "rocket_league": RocketLeaguePlayerT(),
}

# ME-FN-1 variant registry — build-cost Fortnite windows (micro-experiment).
# Use in place of DOMAIN_T_STUBS["fortnite"] to test build-cost constraint signal.
DOMAIN_T_ME_FN1: dict[str, TranslationFunction] = {
    **DOMAIN_T_STUBS,
    "fortnite": FortniteBuildCostT(),
}

# ME-PK-1 variant registry — per-session stack-evolution chains (micro-experiment).
# Raises NotImplementedError when translate() is called; scaffolded for future work.
DOMAIN_T_ME_PK1: dict[str, TranslationFunction] = {
    **DOMAIN_T_STUBS,
    "poker": PokerPerSessionT(),
}
