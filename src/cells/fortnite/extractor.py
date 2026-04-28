"""
Fortnite event extractor.

Converts parsed Type-3 event chunks (from Epic's datastorage replay API)
into normalized GameEvent streams.

Each input record has the shape:
    {
        "match_id": "...",
        "events": [
            {
                "id": "...",
                "group": "playerElim" | "AthenaMatchStats" | ...,
                "metadata": {...},   # parsed JSON from FString Metadata field
                "time1_ms": int,
                "time2_ms": int,
            },
            ...
        ]
    }

Known Fortnite event groups:
  playerElim       → elimination events (eliminator, eliminated, weapon, knocked)
  AthenaMatchStats → end-of-match stats per player
  AthenaMatchTeamStats → team-level stats
  PlayerLogin      → player entry (gives actor ID mapping)
  stateEvent       → storm phase transitions (when present)
  PhaseChange      → storm phase advance
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from ...common.schema import EventStream, GameEvent

logger = logging.getLogger(__name__)

# Map from Fortnite event group name → normalized event type
_GROUP_TO_EVENT_TYPE: dict[str, str] = {
    "playerElim": "engage_decision",
    "PhaseChange": "zone_enter",
    "stateEvent": "zone_enter",
    "AthenaMatchTeamStats": "strategy_adapt",
    "PlayerLogin": None,           # metadata only, not a chain event
    "AthenaMatchStats": None,
}

# Map raw weapon / action strings found in metadata
_WEAPON_MAP: dict[str, str] = {
    "fall": "risk_accept",
    "storm": "zone_exit",
    "environment": "risk_accept",
}


class FortniteExtractor:
    """
    Extracts normalized GameEvents from a replay event-chunk record.
    """

    def extract(self, record: dict[str, Any]) -> EventStream:
        match_id = record.get("match_id", "")
        game_id = f"fnc_{match_id}" if match_id else _hash_id(record)
        stream = EventStream(game_id=game_id, cell="fortnite")

        seq = 0
        for chunk in record.get("events", []):
            group = chunk.get("group", "")
            meta = chunk.get("metadata", {})
            t1 = chunk.get("time1_ms", 0)
            t2 = chunk.get("time2_ms", 0)
            # Use midpoint of time range as event timestamp (seconds)
            ts = ((t1 + t2) / 2.0) / 1000.0

            ev = self._parse_chunk(group, meta, ts, game_id, seq)
            if ev is not None:
                stream.append(ev)
                seq += 1

        stream.events.sort(key=lambda e: e.timestamp)
        for i, e in enumerate(stream.events):
            e.sequence_idx = i

        return stream

    def _parse_chunk(
        self,
        group: str,
        meta: dict,
        ts: float,
        game_id: str,
        seq: int,
    ) -> GameEvent | None:
        if group == "playerElim":
            return self._parse_elim(meta, ts, game_id, seq)
        if group in ("PhaseChange", "stateEvent"):
            return self._parse_phase(group, meta, ts, game_id, seq)
        if group in ("AthenaMatchStats", "PlayerLogin", "AthenaMatchTeamStats"):
            return None  # skip non-chain events
        # Unknown group: emit as position_commit if it has a player actor
        actor = _extract_actor(meta)
        if actor and actor != "unknown":
            return GameEvent(
                timestamp=ts,
                event_type="position_commit",
                actor=actor,
                location_context={"group": group},
                raw_data_blob=meta,
                cell="fortnite",
                game_id=game_id,
                sequence_idx=seq,
            )
        return None

    def _parse_elim(
        self, meta: dict, ts: float, game_id: str, seq: int
    ) -> GameEvent | None:
        try:
            eliminator = (
                meta.get("eliminator")
                or meta.get("EliminatorId")
                or meta.get("killerId")
                or "unknown"
            )
            eliminated = (
                meta.get("eliminated")
                or meta.get("EliminatedId")
                or meta.get("victimId")
                or "unknown"
            )
            weapon = str(meta.get("weapon", meta.get("gunType", "unknown"))).lower()
            knocked = bool(meta.get("knocked", meta.get("isKnocked", False)))

            # Storm/fall kills indicate zone constraint violation
            event_type = _WEAPON_MAP.get(weapon, "engage_decision")

            return GameEvent(
                timestamp=ts,
                event_type=event_type,
                actor=str(eliminator),
                location_context={
                    "victim": str(eliminated),
                    "weapon": weapon,
                    "knocked": knocked,
                },
                raw_data_blob=meta,
                cell="fortnite",
                game_id=game_id,
                sequence_idx=seq,
            )
        except Exception as e:
            logger.debug(f"[fortnite] elim parse error: {e}")
            return None

    def _parse_phase(
        self, group: str, meta: dict, ts: float, game_id: str, seq: int
    ) -> GameEvent | None:
        try:
            phase = int(meta.get("phase", meta.get("Phase", meta.get("stormPhase", 0))))
            return GameEvent(
                timestamp=ts,
                event_type="zone_enter",
                actor="storm",
                location_context={
                    "storm_phase": phase,
                    "circle_x": float(meta.get("circleCenterX", meta.get("X", 0))),
                    "circle_y": float(meta.get("circleCenterY", meta.get("Y", 0))),
                    "circle_radius": float(meta.get("circleRadius", meta.get("radius", 0))),
                },
                raw_data_blob=meta,
                cell="fortnite",
                game_id=game_id,
                sequence_idx=seq,
                phase=f"storm_phase_{phase}",
            )
        except Exception as e:
            logger.debug(f"[fortnite] phase parse error: {e}")
            return None


def _extract_actor(meta: dict) -> str:
    for key in ("playerId", "actorId", "accountId", "player", "actor"):
        v = meta.get(key)
        if v and isinstance(v, str):
            return v
    return "unknown"


def _hash_id(record: dict) -> str:
    return "fnc_" + hashlib.md5(str(record)[:256].encode()).hexdigest()[:16]
