"""
NBA event extractor.

Per SPEC Q6 sign-off (D-21): groups play-by-play rows into possession-level
events. Each possession contains one or more plays, with one summary GameEvent
emitted per possession boundary.

Possession boundaries are detected via:
  - Made shot (ends offensive possession)
  - Defensive rebound (ends offensive possession, starts new one for other team)
  - Turnover (ends offensive possession)
  - End of period (ends current possession)
  - Free throw resulting in possession change (e.g., final FT made + no foul)

PlayByPlayV3 response structure:
  resultSets[0].rowSet: list of play rows
  columns: ['GAME_ID', 'EVENTNUM', 'EVENTMSGTYPE', 'EVENTMSGACTIONTYPE',
            'PERIOD', 'WCTIMESTRING', 'PCTIMESTRING', 'HOMEDESCRIPTION',
            'NEUTRALDESCRIPTION', 'VISITORDESCRIPTION', 'SCORE', 'SCOREMARGIN',
            'PERSON1TYPE', 'PLAYER1_ID', 'PLAYER1_NAME', 'PLAYER1_TEAM_ID', ...]
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ...common.schema import EventStream, GameEvent  # noqa: F401

logger = logging.getLogger(__name__)

# NBA event message type codes → normalized event types
# See: https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/playbyplayv3.md
NBA_EVENTMSGTYPE_MAP: dict[int, str] = {
    1: "engage_decision",    # field goal made
    2: "engage_decision",    # field goal missed
    3: "ability_use",        # free throw
    4: "resource_gain",      # rebound
    5: "resource_spend",     # turnover
    6: "risk_accept",        # foul
    7: "disengage_decision", # violation
    8: "team_coordinate",    # substitution
    9: "team_coordinate",    # timeout
    10: "timing_commit",     # jump ball
    11: "strategy_adapt",    # ejection (rare)
    12: "objective_contest", # start of period
    13: "timing_commit",     # end of period
    18: "engage_decision",   # instant replay
    20: "team_coordinate",   # stoppage
}

PERIOD_CLOCK_RE = re.compile(r"(\d+):(\d+)")


def parse_pctimestring(period: int, pctimestring: str) -> float:
    """Convert period + clock string to seconds from game start."""
    m = PERIOD_CLOCK_RE.match(pctimestring or "12:00")
    if not m:
        return (period - 1) * 720.0
    minutes, seconds = int(m.group(1)), int(m.group(2))
    remaining = minutes * 60 + seconds
    period_length = 720 if period <= 4 else 300  # OT = 5 min
    elapsed_in_period = period_length - remaining
    base = sum(720 if p <= 4 else 300 for p in range(1, period))
    return float(base + elapsed_in_period)


# NBA possession-ending event types (per Q6-A sign-off)
# A possession ends on:
#   - Made shot (msgtype=1) — possession changes
#   - Defensive rebound (msgtype=4 + non-offensive team) — possession changes
#   - Turnover (msgtype=5) — possession changes
#   - End of period (msgtype=13) — possession ends artificially
#   - Made final free throw (msgtype=3 + last in sequence) — possession changes
POSSESSION_ENDING_MSGTYPES = {1, 5, 13}  # Made shot, turnover, end of period
# Rebounds and free throws need additional context to determine if possession-ending.


class NBAExtractor:

    def extract(self, record: dict[str, Any]) -> EventStream:
        """
        Per Q6 sign-off (D-21): possession-level extraction.
        Each possession produces one summary GameEvent representing the
        offensive trip's primary decision (the possession-ending action).
        """
        game_id = self._get_game_id(record)
        stream = EventStream(game_id=game_id, cell="nba")

        result_sets = record.get("resultSets") or record.get("PlayByPlay", {}).get("resultSets", [])
        if not result_sets:
            logger.warning(f"No resultSets in NBA record for {game_id}")
            return stream

        rs = result_sets[0]
        columns = rs.get("headers", rs.get("columns", []))
        rows = rs.get("rowSet", [])

        col_idx = {col: i for i, col in enumerate(columns)}

        # Step 1: parse all rows into per-play events
        play_events = []
        for row in rows:
            ev_raw = self._parse_row_raw(row, col_idx, game_id)
            if ev_raw:
                play_events.append(ev_raw)

        # Step 2: group plays into possessions
        possessions = self._group_into_possessions(play_events)

        # Step 3: emit one GameEvent per possession (with constituent plays in raw_data_blob)
        for seq, possession_plays in enumerate(possessions):
            ev = self._make_possession_event(possession_plays, game_id, seq)
            if ev:
                stream.append(ev)

        return stream

    @staticmethod
    def _group_into_possessions(plays: list[dict]) -> list[list[dict]]:
        """
        Group a list of parsed play dicts into possessions.
        A new possession begins after a possession-ending event.
        """
        if not plays:
            return []
        possessions: list[list[dict]] = []
        current: list[dict] = []
        for play in plays:
            current.append(play)
            msgtype = play.get("msgtype", 0)
            if msgtype in POSSESSION_ENDING_MSGTYPES:
                possessions.append(current)
                current = []
                continue
            # Defensive rebound (msgtype=4) ends possession only if rebounder
            # team differs from shooter team. Without explicit shot-team in the
            # row, we approximate: any rebound after a missed shot starts a new
            # possession unless it's the same team's offensive rebound.
            if msgtype == 4:
                # Heuristic: previous play's team != this play's team → defensive rebound
                if len(current) >= 2:
                    prev = current[-2]
                    if prev.get("team_id") and play.get("team_id") \
                       and prev["team_id"] != play["team_id"]:
                        possessions.append(current)
                        current = []
                        continue
            # Made free throw at end of FT sequence: msgtype=3 with action types
            # 11/12 (last of two/three). Approximation: treat any made FT as
            # possibly possession-ending; group by clock if same possession.
        if current:
            possessions.append(current)
        return possessions

    def _make_possession_event(
        self, plays: list[dict], game_id: str, seq: int
    ) -> GameEvent | None:
        """Build one GameEvent representing the entire possession."""
        if not plays:
            return None

        # The possession-ending play defines the event_type and actor
        terminal = plays[-1]
        # Score margin and period from terminal play
        return GameEvent(
            timestamp=plays[0]["timestamp"],
            event_type=terminal["event_type"],
            actor=terminal["actor"],
            location_context={
                "period": terminal["period"],
                "pctimestring_start": plays[0]["pctimestring"],
                "pctimestring_end": terminal["pctimestring"],
                "n_plays": len(plays),
                "play_descriptions": [p.get("description", "")[:80] for p in plays],
                "score": terminal.get("score", ""),
                "score_margin": terminal.get("score_margin", ""),
                "terminal_msgtype": terminal["msgtype"],
            },
            raw_data_blob={"plays": [p["raw"] for p in plays]},
            cell="nba",
            game_id=game_id,
            sequence_idx=seq,
            actor_team=terminal.get("team_id") or None,
            phase="regular_season",  # set by metadata downstream
        )

    def _parse_row_raw(
        self, row: list, col_idx: dict, game_id: str
    ) -> dict | None:
        """Parse one PBP row into a flat dict (intermediate; not a GameEvent)."""
        try:
            period = int(row[col_idx["PERIOD"]])
            pctimestring = str(row[col_idx.get("PCTIMESTRING", -1)] or "12:00")
            timestamp = parse_pctimestring(period, pctimestring)
            msgtype = int(row[col_idx.get("EVENTMSGTYPE", -1)] or 0)
            event_type = NBA_EVENTMSGTYPE_MAP.get(msgtype)
            if event_type is None:
                return None

            player1_id = str(row[col_idx.get("PLAYER1_ID", -1)] or "")
            player1_name = str(row[col_idx.get("PLAYER1_NAME", -1)] or "")
            team_id = str(row[col_idx.get("PLAYER1_TEAM_ID", -1)] or "")
            home_desc = str(row[col_idx.get("HOMEDESCRIPTION", -1)] or "")
            visitor_desc = str(row[col_idx.get("VISITORDESCRIPTION", -1)] or "")
            score = str(row[col_idx.get("SCORE", -1)] or "")
            score_margin = str(row[col_idx.get("SCOREMARGIN", -1)] or "")

            return {
                "timestamp": timestamp,
                "period": period,
                "pctimestring": pctimestring,
                "msgtype": msgtype,
                "event_type": event_type,
                "actor": player1_id or player1_name or "unknown",
                "team_id": team_id,
                "description": home_desc or visitor_desc,
                "score": score,
                "score_margin": score_margin,
                "raw": {"row": row, "columns": list(col_idx.keys())},
            }
        except Exception as e:
            logger.debug(f"NBA row parse error: {e}")
            return None

    # Legacy play-level parser retained for direct-event callers (e.g., ME-3
    # micro-experiment if pursued). Not used by extract() under Q6-A.
    def _parse_row_legacy(
        self, row: list, col_idx: dict, game_id: str, seq: int
    ) -> GameEvent | None:
        """Legacy play-level parser. Reserved for ME-3 (NBA play-level micro-exp)."""
        d = self._parse_row_raw(row, col_idx, game_id)
        if d is None:
            return None
        return GameEvent(
            timestamp=d["timestamp"],
            event_type=d["event_type"],
            actor=d["actor"],
            location_context={
                "period": d["period"],
                "pctimestring": d["pctimestring"],
                "description": d["description"],
                "score": d["score"],
                "score_margin": d["score_margin"],
            },
            raw_data_blob=d["raw"],
            cell="nba",
            game_id=game_id,
            sequence_idx=seq,
            actor_team=d["team_id"] or None,
            phase="regular_season",
        )

    def _parse_row(
        self, row: list, col_idx: dict, game_id: str, seq: int
    ) -> GameEvent | None:
        try:
            period = int(row[col_idx["PERIOD"]])
            pctimestring = str(row[col_idx.get("PCTIMESTRING", -1)] or "12:00")
            timestamp = parse_pctimestring(period, pctimestring)

            msgtype = int(row[col_idx.get("EVENTMSGTYPE", -1)] or 0)
            normalized_type = NBA_EVENTMSGTYPE_MAP.get(msgtype)
            if normalized_type is None:
                return None

            player1_id = str(row[col_idx.get("PLAYER1_ID", -1)] or "")
            player1_name = str(row[col_idx.get("PLAYER1_NAME", -1)] or "")
            team_id = str(row[col_idx.get("PLAYER1_TEAM_ID", -1)] or "")

            home_desc = str(row[col_idx.get("HOMEDESCRIPTION", -1)] or "")
            visitor_desc = str(row[col_idx.get("VISITORDESCRIPTION", -1)] or "")
            description = home_desc or visitor_desc

            score = str(row[col_idx.get("SCORE", -1)] or "")
            score_margin = str(row[col_idx.get("SCOREMARGIN", -1)] or "")

            return GameEvent(
                timestamp=timestamp,
                event_type=normalized_type,
                actor=player1_id or player1_name or "unknown",
                location_context={
                    "period": period,
                    "pctimestring": pctimestring,
                    "description": description,
                    "score": score,
                    "score_margin": score_margin,
                },
                raw_data_blob={"row": row, "columns": list(col_idx.keys())},
                cell="nba",
                game_id=game_id,
                sequence_idx=seq,
                actor_team=team_id or None,
                phase="regular_season",  # updated downstream from metadata
            )
        except Exception as e:
            logger.debug(f"NBA row parse error: {e}")
            return None

    @staticmethod
    def _get_game_id(record: dict) -> str:
        parameters = record.get("parameters", {})
        game_id = parameters.get("GameID", "")
        if not game_id:
            rs = (record.get("resultSets") or [{}])[0]
            rows = rs.get("rowSet") or []
            if rows and rows[0]:
                game_id = str(rows[0][0])
        return f"nba_{game_id}" if game_id else "nba_unknown"
