"""
NBA event extractor.

Per SPEC Q6 sign-off (D-21): groups play-by-play actions into possession-level
events. Each possession contains one or more plays, with one summary GameEvent
emitted per possession boundary.

Possession boundaries are detected via:
  - Made shot (ends offensive possession)
  - Defensive rebound (rebounder team != shooter team — ends offensive possession)
  - Turnover (ends offensive possession)
  - End of period (ends current possession)

PlayByPlayV3 response structure:
  record["game"]["actions"]: list of action dicts. Notable fields:
    actionType: "Made Shot" | "Missed Shot" | "Free Throw" | "Rebound" |
                "Turnover" | "Foul" | "Violation" | "Substitution" |
                "Timeout" | "Jump Ball" | "period" | ""
    subType:    further qualifies the action (e.g., "start"/"end" for period)
    period:     1..4 regulation, 5+ overtime
    clock:      ISO 8601 duration like "PT12M00.00S" — minutes/seconds remaining
    personId:   primary actor
    teamId:     primary actor's team
    scoreHome / scoreAway: running score after the action
    description: human-readable summary
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ...common.schema import EventStream, GameEvent

logger = logging.getLogger(__name__)

# V3 actionType → normalized event type. Names taken from the live API; the
# legacy V2 EVENTMSGTYPE codes are gone in V3.
NBA_ACTIONTYPE_MAP: dict[str, str] = {
    "Made Shot": "engage_decision",
    "Missed Shot": "engage_decision",
    "Free Throw": "ability_use",
    "Rebound": "resource_gain",
    "Turnover": "resource_spend",
    "Foul": "risk_accept",
    "Violation": "disengage_decision",
    "Substitution": "team_coordinate",
    "Timeout": "team_coordinate",
    "Jump Ball": "timing_commit",
    "period": "timing_commit",
}

# Possession-ending action types under Q6-A:
#   Made Shot  → made FG, possession changes
#   Turnover   → possession changes
#   period:end → possession ends artificially
# Rebound is conditional (defensive vs offensive); handled via team-id heuristic.
POSSESSION_ENDING_ACTIONS = {"Made Shot", "Turnover"}

# Pattern for ISO 8601 duration "PT<MIN>M<SEC>.<frac>S"
_CLOCK_RE = re.compile(r"PT(\d+)M(\d+(?:\.\d+)?)S")


def parse_clock(period: int, clock: str) -> float:
    """Convert period + ISO 8601 clock string to seconds from game start."""
    m = _CLOCK_RE.match(clock or "PT12M00.00S")
    if not m:
        return (period - 1) * 720.0
    minutes = int(m.group(1))
    seconds = float(m.group(2))
    remaining = minutes * 60 + seconds
    period_length = 720 if period <= 4 else 300  # OT = 5 min
    elapsed_in_period = period_length - remaining
    base = sum(720 if p <= 4 else 300 for p in range(1, period))
    return float(base + elapsed_in_period)


class NBAExtractor:

    def extract(self, record: dict[str, Any]) -> EventStream:
        """
        Possession-level extraction over PlayByPlayV3's `game.actions` array.
        Each possession yields one summary GameEvent representing the
        offensive trip's primary decision (the possession-ending action).

        A4 (2026-04-29): also surfaces per-actor cumulative foul counts and
        possession-elapsed time so the locked NBA constraint context
        (24-second shot clock, 6-foul ejection) becomes verifiable from the
        rendered chain.
        """
        game_id = self._get_game_id(record)
        stream = EventStream(game_id=game_id, cell="nba")

        actions = record.get("game", {}).get("actions", [])
        if not actions:
            logger.warning(f"No game.actions in NBA record for {game_id}")
            return stream

        play_events: list[dict] = []
        for action in actions:
            parsed = self._parse_action(action)
            if parsed:
                play_events.append(parsed)

        # A4: pre-compute running foul counts per actor across all plays so
        # _make_possession_event can attach the count for the actor whose
        # decision the possession event is attributed to.
        foul_counts_at_play: list[dict[str, int]] = []
        running: dict[str, int] = {}
        for play in play_events:
            if play.get("actiontype") == "Foul":
                actor = play.get("actor")
                if actor:
                    running[actor] = running.get(actor, 0) + 1
            # Snapshot the cumulative counts AFTER this play.
            foul_counts_at_play.append(dict(running))

        possessions = self._group_into_possessions(play_events)

        # Re-index possessions so each one knows its global play-index range,
        # which lets _make_possession_event read the foul snapshot at the
        # terminal play.
        play_index = 0
        for seq, plays in enumerate(possessions):
            terminal_play_idx = play_index + len(plays) - 1
            ev = self._make_possession_event(
                plays, game_id, seq,
                foul_counts_at_terminal=foul_counts_at_play[terminal_play_idx]
                if foul_counts_at_play else {},
            )
            if ev:
                stream.append(ev)
            play_index += len(plays)

        return stream

    @staticmethod
    def _group_into_possessions(plays: list[dict]) -> list[list[dict]]:
        """Group parsed play dicts into possessions."""
        if not plays:
            return []
        possessions: list[list[dict]] = []
        current: list[dict] = []
        for play in plays:
            current.append(play)
            atype = play.get("actiontype", "")

            if atype in POSSESSION_ENDING_ACTIONS:
                possessions.append(current)
                current = []
                continue

            # End of period: actionType=="period", subType=="end"
            if atype == "period" and play.get("subtype") == "end":
                possessions.append(current)
                current = []
                continue

            # Defensive rebound: rebounder team != prior shooter team.
            if atype == "Rebound" and len(current) >= 2:
                prev = current[-2]
                if (
                    prev.get("team_id")
                    and play.get("team_id")
                    and prev["team_id"] != play["team_id"]
                ):
                    possessions.append(current)
                    current = []
                    continue
        if current:
            possessions.append(current)
        return possessions

    def _make_possession_event(
        self, plays: list[dict], game_id: str, seq: int,
        foul_counts_at_terminal: dict[str, int] | None = None,
    ) -> GameEvent | None:
        if not plays:
            return None
        terminal = plays[-1]

        # A4: possession-elapsed time = wall time between possession start
        # and terminal action, in seconds. parse_clock returns absolute
        # seconds-from-game-start, so the difference is well-defined and
        # handles period boundaries correctly.
        possession_elapsed_s = max(0.0, terminal["timestamp"] - plays[0]["timestamp"])

        # A4: foul count for the actor attributed to this possession's
        # terminal action. Used by NBAPromptBuilder to render the 6-foul
        # ejection rule's verifiable variable.
        terminal_actor_foul_count = 0
        if foul_counts_at_terminal:
            terminal_actor_foul_count = foul_counts_at_terminal.get(
                terminal.get("actor", ""), 0
            )

        return GameEvent(
            timestamp=plays[0]["timestamp"],
            event_type=terminal["event_type"],
            actor=terminal["actor"],
            location_context={
                "period": terminal["period"],
                "clock_start": plays[0]["clock"],
                "clock_end": terminal["clock"],
                "n_plays": len(plays),
                "play_descriptions": [p.get("description", "")[:80] for p in plays],
                "score_home": terminal.get("score_home", ""),
                "score_away": terminal.get("score_away", ""),
                "terminal_action": terminal["actiontype"],
                # A4 fields:
                # Renamed from `possession_elapsed_s` 2026-04-29 (D-44 prep): the
                # original name was ambiguous — model misread "elapsed" as
                # "duration so far in this possession" leading to the single
                # NBA FP under strict grounding (CoT cited 24-second rule on
                # an event with elapsed=14.8, well under threshold). The new
                # name explicitly states "seconds into possession when this
                # action was taken" — directly comparable to the 24s shot-clock.
                "time_in_possession_s": round(possession_elapsed_s, 1),
                "actor_foul_count_after": terminal_actor_foul_count,
            },
            raw_data_blob={"plays": [p["raw"] for p in plays]},
            cell="nba",
            game_id=game_id,
            sequence_idx=seq,
            actor_team=terminal.get("team_id") or None,
            phase="regular_season",
        )

    @staticmethod
    def _parse_action(action: dict) -> dict | None:
        """Parse one V3 action dict into a flat intermediate record."""
        try:
            atype = str(action.get("actionType", "") or "")
            event_type = NBA_ACTIONTYPE_MAP.get(atype)
            if event_type is None:
                return None
            period = int(action.get("period", 1) or 1)
            clock = str(action.get("clock", "PT12M00.00S") or "PT12M00.00S")
            timestamp = parse_clock(period, clock)

            person_id = str(action.get("personId", "") or "")
            player_name = str(action.get("playerName", "") or "")
            team_id = str(action.get("teamId", "") or "")

            return {
                "timestamp": timestamp,
                "period": period,
                "clock": clock,
                "actiontype": atype,
                "subtype": str(action.get("subType", "") or ""),
                "event_type": event_type,
                "actor": person_id or player_name or "unknown",
                "team_id": team_id,
                "description": str(action.get("description", "") or ""),
                "score_home": str(action.get("scoreHome", "") or ""),
                "score_away": str(action.get("scoreAway", "") or ""),
                "raw": action,
            }
        except Exception as e:
            logger.debug(f"NBA action parse error: {e}")
            return None

    @staticmethod
    def _get_game_id(record: dict) -> str:
        gid = record.get("game", {}).get("gameId") or ""
        return f"nba_{gid}" if gid else "nba_unknown"
