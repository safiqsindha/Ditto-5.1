"""
Bot-filter invariant test for the PUBG cell (D-36).

The PUBG /samples endpoint mixes human players and AI bots. Per D-36, the
extractor must drop events where the *actor* (the player whose decision
the GameEvent attributes) has `type != "user"`. This file pins that
behavior with a synthetic telemetry fixture so a regression in
_is_human() or any per-event constructor will trip CI.
"""
from __future__ import annotations

import pytest

from src.cells.pubg.extractor import PUBGExtractor


def _make_telemetry():
    """
    Synthetic PUBG telemetry stream containing:
      - 3 human-killer kill events (should be kept)
      - 2 bot-killer kill events (should be dropped)
      - 1 zone shrink event (no actor, should be kept)
      - 1 human knock event (kept)
      - 1 bot knock event (dropped)
      - 1 human pickup (kept)
      - 1 bot pickup (dropped)
    Expected: 6 events extracted, zero with `user_ai` actor.
    """
    base_t = "2026-04-01T12:00:00.000Z"
    return [
        # Match start time anchor for zone events
        {"_T": "LogGameStatePeriodic", "_D": "2026-04-01T12:00:00.000Z",
         "gameState": {"safetyZoneRadius": 1000.0}},
        # Human kill (kept)
        {"_T": "LogPlayerKillV2", "_D": "2026-04-01T12:00:30.000Z",
         "killer": {"accountId": "h1", "name": "human1", "type": "user"},
         "victim": {"accountId": "v1", "name": "victim1", "type": "user"},
         "damageReason": "HeadShot", "damageCauserName": "WeapM416_C"},
        # Bot kill (dropped)
        {"_T": "LogPlayerKillV2", "_D": "2026-04-01T12:00:45.000Z",
         "killer": {"accountId": "b1", "name": "bot1", "type": "user_ai"},
         "victim": {"accountId": "v2", "name": "victim2", "type": "user"},
         "damageReason": "TorsoShot", "damageCauserName": "WeapAK_C"},
        # Human knock (kept)
        {"_T": "LogPlayerMakeGroggy", "_D": "2026-04-01T12:01:00.000Z",
         "attacker": {"accountId": "h2", "name": "human2", "type": "user"},
         "victim": {"accountId": "v3", "name": "victim3", "type": "user"},
         "damageReason": "PelvisShot", "damageCauserName": "WeapM16A4_C"},
        # Bot knock (dropped)
        {"_T": "LogPlayerMakeGroggy", "_D": "2026-04-01T12:01:15.000Z",
         "attacker": {"accountId": "b2", "name": "bot2", "type": "user_ai"},
         "victim": {"accountId": "v4", "name": "victim4", "type": "user"},
         "damageReason": "ArmShot", "damageCauserName": "WeapPistol_C"},
        # Zone shrink (kept — actor=zone, not a player)
        {"_T": "LogGameStatePeriodic", "_D": "2026-04-01T12:01:30.000Z",
         "gameState": {"safetyZoneRadius": 800.0}},
        # Human kill (kept)
        {"_T": "LogPlayerKillV2", "_D": "2026-04-01T12:02:00.000Z",
         "killer": {"accountId": "h3", "name": "human3", "type": "user"},
         "victim": {"accountId": "v5", "name": "victim5", "type": "user"},
         "damageReason": "HeadShot", "damageCauserName": "WeapKar98k_C"},
        # Human pickup (kept)
        {"_T": "LogItemPickup", "_D": "2026-04-01T12:02:30.000Z",
         "character": {"accountId": "h1", "name": "human1", "type": "user"},
         "item": {"itemId": "Item_Weapon_M416", "stackCount": 1}},
        # Bot pickup (dropped)
        {"_T": "LogItemPickup", "_D": "2026-04-01T12:02:45.000Z",
         "character": {"accountId": "b1", "name": "bot1", "type": "user_ai"},
         "item": {"itemId": "Item_Weapon_AK", "stackCount": 1}},
        # Human kill (kept)
        {"_T": "LogPlayerKillV2", "_D": "2026-04-01T12:03:00.000Z",
         "killer": {"accountId": "h2", "name": "human2", "type": "user"},
         "victim": {"accountId": "v6", "name": "victim6", "type": "user"},
         "damageReason": "TorsoShot", "damageCauserName": "WeapM16A4_C"},
    ]


def _make_record():
    return {
        "match_id": "mock_pubg_bot_filter_test",
        "match_attrs": {
            "createdAt": "2026-04-01T12:00:00Z",
            "gameMode": "squad-fpp",
            "mapName": "Erangel_Main",
            "duration": 1800,
            "matchType": "official",
        },
        "telemetry": _make_telemetry(),
    }


class TestPUBGBotFilter:
    """D-36 invariant: zero `user_ai`-attributed events leak through extract()."""

    def test_kill_with_bot_killer_is_dropped(self):
        record = _make_record()
        stream = PUBGExtractor().extract(record)
        # Of 4 kill events in fixture, 2 are bot-killers (should be filtered).
        # 3 human kills + 1 human knock = 4 engage_decisions kept;
        # plus 1 zone_enter (shrink) + 1 resource_gain (human pickup) = 6 total.
        kills = [e for e in stream.events if e.event_type == "engage_decision"]
        assert len(kills) == 4, f"Expected 4 engage_decisions (3 kills + 1 knock); got {len(kills)}"

    def test_knock_with_bot_attacker_is_dropped(self):
        record = _make_record()
        stream = PUBGExtractor().extract(record)
        # No actor in any kept event should be a known bot ID.
        bot_ids = {"b1", "b2"}
        for ev in stream.events:
            assert ev.actor not in bot_ids, (
                f"Bot-attributed event leaked: actor={ev.actor} "
                f"event_type={ev.event_type} sequence_idx={ev.sequence_idx}"
            )

    def test_pickup_with_bot_character_is_dropped(self):
        record = _make_record()
        stream = PUBGExtractor().extract(record)
        # Of 2 pickup events, 1 is bot (filtered); 1 is human (kept).
        pickups = [e for e in stream.events if e.event_type == "resource_gain"]
        assert len(pickups) == 1
        assert pickups[0].actor == "h1"

    def test_zone_event_kept_without_player_filter(self):
        record = _make_record()
        stream = PUBGExtractor().extract(record)
        # Zone events have no human actor (actor="zone"); must not be filtered.
        zones = [e for e in stream.events if e.event_type == "zone_enter"]
        assert len(zones) >= 1, "Zone events must survive the bot filter"

    def test_total_event_count_after_bot_filter(self):
        record = _make_record()
        stream = PUBGExtractor().extract(record)
        # 10-event fixture, of which 3 are bot-attributed (must be dropped).
        # Surviving: 4 engage_decisions + 2 zone_enters + 1 resource_gain = 7.
        assert len(stream.events) == 7, (
            f"Expected 7 kept events after bot filter (10 fixture - 3 bot); "
            f"got {len(stream.events)} "
            f"(types={[e.event_type for e in stream.events]})"
        )

    def test_no_user_ai_actor_anywhere(self):
        """Strongest invariant: regardless of event type, no `user_ai` ever survives."""
        record = _make_record()
        stream = PUBGExtractor().extract(record)
        # Compose a forbidden set: every actor ID we attached `type=user_ai` to.
        forbidden = {"b1", "b2"}
        leaked = [e for e in stream.events if e.actor in forbidden]
        assert not leaked, (
            f"D-36 INVARIANT VIOLATED: {len(leaked)} bot-attributed events leaked: "
            f"{[(e.event_type, e.actor) for e in leaked]}"
        )
