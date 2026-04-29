"""
PUBG event extractor.

Converts PUBG telemetry events (gzipped JSON arrays from Krafton's CDN) into
normalized GameEvent streams.

Input record shape (produced by PUBGPipeline.parse):
    {
        "match_id":    "<uuid>",
        "match_attrs": {"gameMode": ..., "mapName": ..., "duration": ...,
                        "createdAt": "...Z", "matchType": ...},
        "telemetry":   [ {"_T": "LogPlayerKillV2", "_D": "...", ...}, ... ]
    }

Mapping rules (minimum viable):
    LogPlayerKillV2  / LogPlayerKill   → engage_decision
    LogPlayerMakeGroggy                → engage_decision (knock)
    LogPlayerTakeDamage (≥30 dmg)      → risk_accept
    LogGameStatePeriodic (zone shrink) → zone_enter
    LogParachuteLanding                → position_commit
    LogVehicleRide / LogVehicleLeave   → position_commit
    LogItemPickup                      → resource_gain
    LogItemUse                         → resource_spend

Heavy events (LogPlayerPosition, LogPlayerAttack, LogHeal) are intentionally
NOT mapped — they would dominate the stream without adding decision-level
information and would break the chain_length=8 budget.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ...common.schema import EventStream, GameEvent

logger = logging.getLogger(__name__)

# Damage threshold below which take-damage events are skipped (chip damage,
# zone tick damage). Above this threshold the event represents a meaningful
# risk-acceptance moment.
_DAMAGE_THRESHOLD = 30.0

# Telemetry events have ISO8601 timestamps in `_D`. Match createdAt is also ISO.
_ISO_FMT_FALLBACK = "%Y-%m-%dT%H:%M:%S.%fZ"


class PUBGExtractor:
    """Map PUBG telemetry → GameEvent streams."""

    def extract(self, record: dict[str, Any]) -> EventStream:
        match_id = record.get("match_id", "")
        match_attrs = record.get("match_attrs", {})
        telemetry = record.get("telemetry", []) or []

        match_created_iso = match_attrs.get("createdAt", "")
        match_created = _parse_iso(match_created_iso)

        game_id = f"pubg_{match_id}"
        stream = EventStream(
            game_id=game_id,
            cell="pubg",
            metadata={
                "game_mode": match_attrs.get("gameMode"),
                "map_name": match_attrs.get("mapName"),
                "duration_s": match_attrs.get("duration"),
                "match_type": match_attrs.get("matchType"),
                "raw_event_count": len(telemetry),
            },
        )

        # Track previous safety zone radius so we only emit zone_enter on shrink
        prev_zone_radius: float | None = None

        seq = 0
        for ev in telemetry:
            etype = ev.get("_T", "")
            game_event, new_zone_radius = self._map(
                etype, ev, match_created, game_id, seq, prev_zone_radius
            )
            if new_zone_radius is not None:
                prev_zone_radius = new_zone_radius
            if game_event is None:
                continue
            stream.append(game_event)
            seq += 1

        # Sort defensively (telemetry should already be ordered, but be safe)
        stream.events.sort(key=lambda e: e.timestamp)
        for i, e in enumerate(stream.events):
            e.sequence_idx = i

        return stream

    # ------------------------------------------------------------------
    # Per-event mapping. Returns (GameEvent | None, new_zone_radius | None).
    # ------------------------------------------------------------------

    def _map(
        self,
        etype: str,
        ev: dict,
        match_created: datetime | None,
        game_id: str,
        seq: int,
        prev_zone_radius: float | None,
    ) -> tuple[GameEvent | None, float | None]:
        ts = self._compute_ts(ev, match_created)

        if etype in ("LogPlayerKillV2", "LogPlayerKill"):
            return self._kill_event(ev, ts, game_id, seq), None

        if etype == "LogPlayerMakeGroggy":
            return self._knock_event(ev, ts, game_id, seq), None

        if etype == "LogPlayerTakeDamage":
            return self._damage_event(ev, ts, game_id, seq), None

        if etype == "LogGameStatePeriodic":
            return self._zone_event(ev, ts, game_id, seq, prev_zone_radius)

        if etype == "LogParachuteLanding":
            return self._landing_event(ev, ts, game_id, seq), None

        if etype in ("LogVehicleRide", "LogVehicleLeave"):
            return self._vehicle_event(ev, etype, ts, game_id, seq), None

        if etype == "LogItemPickup":
            return self._item_event(ev, "resource_gain", ts, game_id, seq), None

        if etype == "LogItemUse":
            return self._item_event(ev, "resource_spend", ts, game_id, seq), None

        return None, None

    # ------------------------------------------------------------------
    # Specific event constructors
    # ------------------------------------------------------------------

    def _kill_event(self, ev: dict, ts: float, game_id: str, seq: int) -> GameEvent | None:
        killer = _get_player(ev, "killer") or _get_player(ev, "finisher")
        victim = _get_player(ev, "victim")
        if not killer or not _is_human(killer):
            return None
        return GameEvent(
            timestamp=ts,
            event_type="engage_decision",
            actor=killer.get("accountId", "unknown"),
            actor_team=str(killer.get("teamId", "")) or None,
            location_context={
                "killer_loc": killer.get("location", {}),
                "victim_loc": (victim or {}).get("location", {}),
                "victim_id": (victim or {}).get("accountId"),
                "weapon": ev.get("damageCauserName", ""),
                "is_groggy_kill": bool(ev.get("dBNOId", -1) != -1),
            },
            raw_data_blob=ev,
            cell="pubg",
            game_id=game_id,
            sequence_idx=seq,
            metadata={"sub_type": "kill"},
        )

    def _knock_event(self, ev: dict, ts: float, game_id: str, seq: int) -> GameEvent | None:
        attacker = _get_player(ev, "attacker")
        victim = _get_player(ev, "victim")
        if not attacker or not _is_human(attacker):
            return None
        return GameEvent(
            timestamp=ts,
            event_type="engage_decision",
            actor=attacker.get("accountId", "unknown"),
            actor_team=str(attacker.get("teamId", "")) or None,
            location_context={
                "attacker_loc": attacker.get("location", {}),
                "victim_loc": (victim or {}).get("location", {}),
                "victim_id": (victim or {}).get("accountId"),
                "weapon": ev.get("damageCauserName", ""),
                "distance_cm": ev.get("distance"),
            },
            raw_data_blob=ev,
            cell="pubg",
            game_id=game_id,
            sequence_idx=seq,
            metadata={"sub_type": "groggy"},
        )

    def _damage_event(self, ev: dict, ts: float, game_id: str, seq: int) -> GameEvent | None:
        damage = float(ev.get("damage", 0) or 0)
        if damage < _DAMAGE_THRESHOLD:
            return None
        victim = _get_player(ev, "victim")
        if not victim or not _is_human(victim):
            return None
        return GameEvent(
            timestamp=ts,
            event_type="risk_accept",
            actor=victim.get("accountId", "unknown"),
            actor_team=str(victim.get("teamId", "")) or None,
            location_context={
                "victim_loc": victim.get("location", {}),
                "damage": damage,
                "damage_cause": ev.get("damageCauserName", ""),
                "damage_type": ev.get("damageTypeCategory", ""),
                "damage_reason": ev.get("damageReason", ""),
                "is_blue_zone": bool((ev.get("damageTypeCategory", "") == "Damage_BlueZone")),
            },
            raw_data_blob=ev,
            cell="pubg",
            game_id=game_id,
            sequence_idx=seq,
        )

    def _zone_event(
        self,
        ev: dict,
        ts: float,
        game_id: str,
        seq: int,
        prev_radius: float | None,
    ) -> tuple[GameEvent | None, float | None]:
        gs = ev.get("gameState", {}) or {}
        radius = gs.get("safetyZoneRadius")
        if radius is None:
            return None, None
        # Only emit on first observation or when zone shrinks (phase advance)
        if prev_radius is not None and radius >= prev_radius - 1.0:
            return None, radius
        evt = GameEvent(
            timestamp=ts,
            event_type="zone_enter",
            actor="zone",
            location_context={
                "safety_zone_position": gs.get("safetyZonePosition", {}),
                "safety_zone_radius": radius,
                "poison_zone_position": gs.get("poisonGasWarningPosition", {}),
                "poison_zone_radius": gs.get("poisonGasWarningRadius"),
                "elapsed_time_s": gs.get("elapsedTime"),
                "num_alive_players": gs.get("numAlivePlayers"),
                "num_alive_teams": gs.get("numAliveTeams"),
            },
            raw_data_blob=ev,
            cell="pubg",
            game_id=game_id,
            sequence_idx=seq,
            phase=f"zone_{int(gs.get('elapsedTime', 0))}s",
        )
        return evt, radius

    def _landing_event(self, ev: dict, ts: float, game_id: str, seq: int) -> GameEvent | None:
        char = _get_player(ev, "character")
        if not char or not _is_human(char):
            return None
        return GameEvent(
            timestamp=ts,
            event_type="position_commit",
            actor=char.get("accountId", "unknown"),
            actor_team=str(char.get("teamId", "")) or None,
            location_context={
                "loc": char.get("location", {}),
                "distance_parachute_cm": ev.get("distance"),
            },
            raw_data_blob=ev,
            cell="pubg",
            game_id=game_id,
            sequence_idx=seq,
            metadata={"sub_type": "parachute_land"},
        )

    def _vehicle_event(
        self, ev: dict, etype: str, ts: float, game_id: str, seq: int
    ) -> GameEvent | None:
        char = _get_player(ev, "character")
        if not char or not _is_human(char):
            return None
        sub = "vehicle_ride" if etype == "LogVehicleRide" else "vehicle_leave"
        return GameEvent(
            timestamp=ts,
            event_type="position_commit",
            actor=char.get("accountId", "unknown"),
            actor_team=str(char.get("teamId", "")) or None,
            location_context={
                "loc": char.get("location", {}),
                "vehicle_type": (ev.get("vehicle") or {}).get("vehicleType", ""),
                "vehicle_id": (ev.get("vehicle") or {}).get("vehicleId", ""),
                "fuel_pct": (ev.get("vehicle") or {}).get("fuelPercent"),
            },
            raw_data_blob=ev,
            cell="pubg",
            game_id=game_id,
            sequence_idx=seq,
            metadata={"sub_type": sub},
        )

    def _item_event(
        self, ev: dict, event_type: str, ts: float, game_id: str, seq: int
    ) -> GameEvent | None:
        char = _get_player(ev, "character")
        if not char or not _is_human(char):
            return None
        item = ev.get("item") or {}
        return GameEvent(
            timestamp=ts,
            event_type=event_type,
            actor=char.get("accountId", "unknown"),
            actor_team=str(char.get("teamId", "")) or None,
            location_context={
                "loc": char.get("location", {}),
                "item_id": item.get("itemId", ""),
                "category": item.get("category", ""),
                "subcategory": item.get("subCategory", ""),
                "stack_count": item.get("stackCount"),
            },
            raw_data_blob=ev,
            cell="pubg",
            game_id=game_id,
            sequence_idx=seq,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_ts(ev: dict, match_created: datetime | None) -> float:
        """Seconds elapsed from match start. Falls back to 0 on parse error."""
        evt_dt = _parse_iso(ev.get("_D", ""))
        if not evt_dt or not match_created:
            return 0.0
        return max(0.0, (evt_dt - match_created).total_seconds())


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _get_player(ev: dict, key: str) -> dict | None:
    val = ev.get(key)
    return val if isinstance(val, dict) else None


def _is_human(player: dict | None) -> bool:
    """
    Return True if this player object represents a real human user.

    PUBG player objects carry a `type` field — `"user"` for humans, `"user_ai"`
    for bots, `"npc"` for special-mode NPCs. /samples-discovered matches mix
    bots and humans; tournament matches do not. Filter at extraction time so
    chains attributed to bot decisions don't pollute the corpus.
    """
    if not isinstance(player, dict):
        return False
    return player.get("type", "user") == "user"


def _parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    s_clean = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s_clean)
    except ValueError:
        try:
            return datetime.strptime(s, _ISO_FMT_FALLBACK)
        except ValueError:
            return None


# ----------------------------------------------------------------------
# Mock event types — used by mock data generation in pipeline.py
# ----------------------------------------------------------------------

PUBG_MOCK_EVENT_TYPES = [
    "engage_decision",
    "zone_enter",
    "zone_exit",
    "position_commit",
    "rotation_commit",
    "resource_gain",
    "resource_spend",
    "risk_accept",
    "risk_reject",
    "timing_commit",
    "objective_contest",
    "team_coordinate",
    "strategy_adapt",
]
