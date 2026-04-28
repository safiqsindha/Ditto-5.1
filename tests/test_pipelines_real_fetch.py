"""
Pipeline real-fetch tests with HTTP and subprocess mocks.

Exercises the fetch+parse paths of each pipeline without making real network
calls or invoking external binaries. Covers code paths that pure-mock-data
tests can't reach.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from src.cells.csgo.pipeline import CSGOPipeline
from src.cells.fortnite.pipeline import FortnitePipeline
from src.cells.poker.pipeline import PokerPipeline
from src.cells.nba.pipeline import NBAPipeline
from src.cells.rocket_league.pipeline import RocketLeaguePipeline
from src.common.config import load_cell_configs


@pytest.fixture
def cfgs():
    return load_cell_configs()


# ---------------------------------------------------------------------------
# CS:GO — HLTV demo download via requests
# ---------------------------------------------------------------------------

_FAKE_FACEIT_STATS = {
    "rounds": [
        {
            "match_id": "1-faceit-match-abc",
            "round_stats": {"Map": "de_dust2", "Rounds": "29", "Score": "16:13", "Winner": "team_a"},
            "teams": [
                {
                    "team_id": "team_a",
                    "players": [
                        {
                            "player_id": "faceit_player_001",
                            "nickname": "sniper42",
                            "player_stats": {
                                "Kills": "20", "Assists": "3", "Deaths": "15",
                                "Headshots": "8", "Flash Count": "6",
                                "Entry Count": "5", "MVPs": "2",
                            },
                        }
                    ],
                }
            ],
        }
    ]
}


class TestCSGORealFetch:
    def test_fetch_returns_empty_without_api_key(self, cfgs, tmp_path):
        pipeline = CSGOPipeline(config=cfgs["csgo"], data_root=tmp_path)
        pipeline.api_key = ""
        paths = pipeline.fetch()
        assert paths == []

    def test_fetch_skips_existing_json(self, cfgs, tmp_path, monkeypatch):
        monkeypatch.setenv("FACEIT_API_KEY", "fake_key")
        pipeline = CSGOPipeline(config=cfgs["csgo"], data_root=tmp_path)
        pipeline.config.sample_target = 1
        existing = pipeline.raw_dir / "1-existing-match.json"
        existing.write_text("{}")

        champ_resp = MagicMock(status_code=200)
        champ_resp.json.return_value = {"items": [{"championship_id": "champ1"}]}
        match_resp = MagicMock(status_code=200)
        match_resp.json.return_value = {"items": [{"match_id": "1-existing-match"}]}

        with patch.object(pipeline.session, "get", side_effect=[champ_resp, match_resp]):
            paths = pipeline.fetch()
        assert existing in paths

    def test_fetch_downloads_match_stats(self, cfgs, tmp_path, monkeypatch):
        monkeypatch.setenv("FACEIT_API_KEY", "fake_key")
        pipeline = CSGOPipeline(config=cfgs["csgo"], data_root=tmp_path)
        pipeline.config.sample_target = 1

        champ_resp = MagicMock(status_code=200)
        champ_resp.json.return_value = {"items": [{"championship_id": "champ1"}]}
        match_resp = MagicMock(status_code=200)
        match_resp.json.return_value = {"items": [{"match_id": "1-new-match"}]}
        stats_resp = MagicMock(status_code=200)
        stats_resp.json.return_value = _FAKE_FACEIT_STATS

        with patch.object(pipeline.session, "get", side_effect=[champ_resp, match_resp, stats_resp]):
            paths = pipeline.fetch()
        assert len(paths) == 1
        assert paths[0].name == "1-new-match.json"

    def test_fetch_handles_stats_404(self, cfgs, tmp_path, monkeypatch):
        monkeypatch.setenv("FACEIT_API_KEY", "fake_key")
        pipeline = CSGOPipeline(config=cfgs["csgo"], data_root=tmp_path)
        pipeline.config.sample_target = 1

        champ_resp = MagicMock(status_code=200)
        champ_resp.json.return_value = {"items": [{"championship_id": "champ1"}]}
        match_resp = MagicMock(status_code=200)
        match_resp.json.return_value = {"items": [{"match_id": "1-missing-match"}]}
        not_found_resp = MagicMock(status_code=404)

        with patch.object(pipeline.session, "get", side_effect=[champ_resp, match_resp, not_found_resp]):
            paths = pipeline.fetch()
        assert paths == []

    def test_parse_loads_json(self, cfgs, tmp_path):
        pipeline = CSGOPipeline(config=cfgs["csgo"], data_root=tmp_path)
        f = pipeline.raw_dir / "match1.json"
        f.write_text('{"rounds": []}')
        records = pipeline.parse([f])
        assert len(records) == 1
        assert records[0] == {"rounds": []}

    def test_extract_events_faceit_format(self, cfgs, tmp_path):
        pipeline = CSGOPipeline(config=cfgs["csgo"], data_root=tmp_path)
        streams = pipeline.extract_events([_FAKE_FACEIT_STATS])
        assert len(streams) == 1
        # Kills(20) + Assists(3) + FlashCount(6) + EntryCount(5) + MVPs(2) = 36 events
        assert len(streams[0].events) == 36

    def test_extract_events_awpy_format(self, cfgs, tmp_path):
        pipeline = CSGOPipeline(config=cfgs["csgo"], data_root=tmp_path)
        awpy_record = {
            "matchID": "test_match",
            "mapName": "de_mirage",
            "rounds": [{"roundNum": 1, "startTick": 0, "kills": [], "grenades": [],
                        "bombEvents": [], "ctEqVal": 4250, "tEqVal": 4350}],
        }
        streams = pipeline.extract_events([awpy_record, None, awpy_record])
        assert len(streams) == 2

    def test_get_championship_ids_returns_list(self, cfgs, tmp_path, monkeypatch):
        monkeypatch.setenv("FACEIT_API_KEY", "fake_key")
        pipeline = CSGOPipeline(config=cfgs["csgo"], data_root=tmp_path)
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"items": [{"championship_id": "c1"}, {"championship_id": "c2"}]}
        with patch.object(pipeline.session, "get", return_value=resp):
            ids = pipeline._get_championship_ids()
        assert ids == ["c1", "c2"]


# ---------------------------------------------------------------------------
# NBA — nba_api PlayByPlayV3
# ---------------------------------------------------------------------------

class TestNBARealFetch:
    def test_fetch_handles_missing_nba_api(self, cfgs, tmp_path):
        pipeline = NBAPipeline(config=cfgs["nba"], data_root=tmp_path)
        # Force ImportError by patching the import path
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def raising_import(name, *args, **kwargs):
            if "nba_api" in name:
                raise ImportError("nba_api not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=raising_import):
            paths = pipeline.fetch()
        assert paths == []

    def test_fetch_with_mocked_nba_api(self, cfgs, tmp_path):
        pipeline = NBAPipeline(config=cfgs["nba"], data_root=tmp_path)
        pipeline.config.sample_target = 2

        mock_endpoint = MagicMock()
        mock_endpoint.get_dict.return_value = {
            "parameters": {"GameID": "test"},
            "resultSets": [{"headers": [], "rowSet": []}],
        }
        mock_module = MagicMock()
        mock_module.PlayByPlayV3.return_value = mock_endpoint

        with patch.dict("sys.modules", {
            "nba_api": MagicMock(),
            "nba_api.stats": MagicMock(),
            "nba_api.stats.endpoints": MagicMock(playbyplayv3=mock_module),
        }), patch("src.cells.nba.pipeline.time.sleep"):
            paths = pipeline.fetch()
        # Should have written at least one file
        assert len(paths) >= 1

    def test_fetch_skips_existing(self, cfgs, tmp_path):
        pipeline = NBAPipeline(config=cfgs["nba"], data_root=tmp_path)
        pipeline.config.sample_target = 1
        # Pre-create the first game file
        game_ids = pipeline._get_target_game_ids()
        existing = pipeline.raw_dir / f"{game_ids[0]}.json"
        existing.write_text('{"foo": "bar"}')
        # Mock nba_api so the import doesn't fail
        mock_module = MagicMock()
        with patch.dict("sys.modules", {
            "nba_api": MagicMock(),
            "nba_api.stats": MagicMock(),
            "nba_api.stats.endpoints": MagicMock(playbyplayv3=mock_module),
        }), patch("src.cells.nba.pipeline.time.sleep"):
            paths = pipeline.fetch()
        assert any(p == existing for p in paths)

    def test_parse_handles_bad_files(self, cfgs, tmp_path):
        pipeline = NBAPipeline(config=cfgs["nba"], data_root=tmp_path)
        good = pipeline.raw_dir / "good.json"
        good.write_text('{"resultSets": []}')
        bad = pipeline.raw_dir / "bad.json"
        bad.write_text("not valid json")
        records = pipeline.parse([good, bad])
        # Bad file logs warning but doesn't crash
        assert len(records) == 1

    def test_get_target_game_ids(self, cfgs, tmp_path):
        pipeline = NBAPipeline(config=cfgs["nba"], data_root=tmp_path)
        ids = pipeline._get_target_game_ids()
        assert len(ids) > 0


# ---------------------------------------------------------------------------
# Rocket League — BallChasing API + carball/rrrocket
# ---------------------------------------------------------------------------

class TestRLRealFetch:
    # Ballchasing API JSON format (no binary download, no carball/rrrocket).
    # fetch() → GET /replays (list) + GET /replays/{id} (JSON stats per replay).
    # parse() → load saved JSON files from raw_dir.

    _FAKE_REPLAY_JSON = {
        "id": "rep1",
        "duration": 300,
        "blue": {"goals": 2, "players": [
            {"id": {"platform": "steam", "id": "111"}, "name": "P1",
             "stats": {"core": {"shots": 3, "goals": 1, "saves": 1, "assists": 0,
                                "demo": {"inflicted": 0}},
                       "boost": {"bcpm": 200, "count_collected_big": 5,
                                 "count_collected_small": 10}}},
        ]},
        "orange": {"goals": 1, "players": [
            {"id": {"platform": "steam", "id": "222"}, "name": "P2",
             "stats": {"core": {"shots": 2, "goals": 1, "saves": 2, "assists": 1,
                                "demo": {"inflicted": 1}},
                       "boost": {"bcpm": 180, "count_collected_big": 4,
                                 "count_collected_small": 8}}},
        ]},
    }

    def test_fetch_no_token_silently_returns_empty(self, cfgs, tmp_path, monkeypatch):
        monkeypatch.delenv("BALLCHASING_TOKEN", raising=False)
        pipeline = RocketLeaguePipeline(config=cfgs["rocket_league"], data_root=tmp_path)
        # No token → connection fails (realistic: requests.ConnectionError)
        import requests as req_module
        with patch.object(pipeline.session, "get",
                          side_effect=req_module.ConnectionError("no auth")), \
             patch("src.cells.rocket_league.pipeline.time.sleep"):
            paths = pipeline.fetch()
        assert paths == []

    def test_fetch_with_mocked_api(self, cfgs, tmp_path, monkeypatch):
        monkeypatch.setenv("BALLCHASING_TOKEN", "fake_token")
        pipeline = RocketLeaguePipeline(config=cfgs["rocket_league"], data_root=tmp_path)
        pipeline.config.sample_target = 2

        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.raise_for_status = MagicMock()
        list_resp.json.return_value = {"list": [{"id": "rep1"}, {"id": "rep2"}]}

        replay_resp = MagicMock()
        replay_resp.status_code = 200
        replay_resp.raise_for_status = MagicMock()
        replay_resp.json.return_value = self._FAKE_REPLAY_JSON

        def get_side_effect(url, *args, **kwargs):
            if "/replays/rep" in url:
                return replay_resp
            return list_resp

        with patch.object(pipeline.session, "get", side_effect=get_side_effect), \
             patch("src.cells.rocket_league.pipeline.time.sleep"):
            paths = pipeline.fetch()
        assert len(paths) == 2
        # Files are .json, not .replay
        assert all(p.suffix == ".json" for p in paths)

    def test_fetch_handles_rate_limit(self, cfgs, tmp_path, monkeypatch):
        monkeypatch.setenv("BALLCHASING_TOKEN", "fake")
        pipeline = RocketLeaguePipeline(config=cfgs["rocket_league"], data_root=tmp_path)
        pipeline.config.sample_target = 1

        list_resp = MagicMock()
        list_resp.status_code = 200
        list_resp.raise_for_status = MagicMock()
        list_resp.json.return_value = {"list": [{"id": "rep1"}]}

        rate_resp = MagicMock()
        rate_resp.status_code = 429
        rate_resp.headers = {"Retry-After": "1"}

        def get_side_effect(url, *args, **kwargs):
            # Rate-limit on individual replay fetch
            if "/replays/rep" in url:
                return rate_resp
            return list_resp

        with patch.object(pipeline.session, "get", side_effect=get_side_effect), \
             patch("src.cells.rocket_league.pipeline.time.sleep"):
            paths = pipeline.fetch()
        # All retries exhausted on 429 → no paths saved
        assert len(paths) == 0

    def test_parse_loads_json_from_raw_dir(self, cfgs, tmp_path):
        pipeline = RocketLeaguePipeline(config=cfgs["rocket_league"], data_root=tmp_path)
        # Write a JSON file directly into raw_dir (as fetch() would produce)
        json_path = pipeline.raw_dir / "rep1.json"
        json_path.write_text(json.dumps(self._FAKE_REPLAY_JSON))
        records = pipeline.parse([json_path])
        assert len(records) == 1
        assert records[0]["id"] == "rep1"

    def test_parse_skips_corrupt_json(self, cfgs, tmp_path):
        pipeline = RocketLeaguePipeline(config=cfgs["rocket_league"], data_root=tmp_path)
        bad_path = pipeline.raw_dir / "bad.json"
        bad_path.write_text("not valid json")
        records = pipeline.parse([bad_path])
        assert records == []

    def test_extract_events_from_ballchasing_json(self, cfgs, tmp_path):
        pipeline = RocketLeaguePipeline(config=cfgs["rocket_league"], data_root=tmp_path)
        streams = pipeline.extract_events([self._FAKE_REPLAY_JSON])
        assert len(streams) == 1
        stream = streams[0]
        # Goals come from per-player stats: P1=1 goal, P2=1 goal → 2 total
        goals = [e for e in stream.events if e.event_type == "objective_capture"]
        assert len(goals) == 2
        # Events should be sorted by timestamp
        ts = [e.timestamp for e in stream.events]
        assert ts == sorted(ts)


# ---------------------------------------------------------------------------
# Hearthstone — HSReplay API + hslog
# ---------------------------------------------------------------------------

class TestPokerRealFetch:
    """Poker: PHH public dataset, no auth required. Real fetch tested at parse layer."""

    def test_fetch_returns_list(self, cfgs, tmp_path, monkeypatch):
        """fetch() returns a list (may be empty if network unavailable in CI)."""
        import src.cells.poker.pipeline as pm
        monkeypatch.setattr(pm, "_PHH_TARBALL_URL", "http://localhost:0/nope")
        pipeline = PokerPipeline(config=cfgs["poker"], data_root=tmp_path)
        paths = pipeline.fetch()
        assert isinstance(paths, list)

    def test_parse_returns_empty_without_pokerkit(self, cfgs, tmp_path, monkeypatch):
        """parse() returns [] gracefully when pokerkit is absent."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pokerkit":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        pipeline = PokerPipeline(config=cfgs["poker"], data_root=tmp_path)
        records = pipeline.parse([tmp_path / "nonexistent.phh"])
        assert records == []

    def test_generate_mock_data_hits_sample_target(self, cfgs, tmp_path):
        pipeline = PokerPipeline(config=cfgs["poker"], data_root=tmp_path)
        pipeline.config.sample_target = 10
        streams = pipeline.generate_mock_data()
        assert len(streams) == 10

    def test_mock_streams_have_poker_streets(self, cfgs, tmp_path):
        pipeline = PokerPipeline(config=cfgs["poker"], data_root=tmp_path)
        pipeline.config.sample_target = 2
        streams = pipeline.generate_mock_data()
        phases = {e.phase for s in streams for e in s.events}
        assert "preflop" in phases


# ---------------------------------------------------------------------------
# Fortnite — replay-downloader subprocess + decompressor
# ---------------------------------------------------------------------------

class TestFortniteRealFetch:
    # Pure Python Epic API implementation: OAuth client_credentials + JSON event chunks.
    # No Node.js / dotnet subprocess needed.

    _FAKE_EVENTS = [
        {"id": "e1", "group": "PhaseChange",
         "metadata": {"phase": 1, "circleCenterX": 0, "circleCenterY": 0, "circleRadius": 5000},
         "time1_ms": 90000, "time2_ms": 90000},
        {"id": "e2", "group": "playerElim",
         "metadata": {"eliminator": "p1", "eliminated": "p2", "weapon": "AR", "knocked": False},
         "time1_ms": 120000, "time2_ms": 120000},
    ]
    _FAKE_RECORD = {"match_id": "abc123", "events": _FAKE_EVENTS}

    def _mock_pipeline(self, cfgs, tmp_path):
        pipeline = FortnitePipeline(config=cfgs["fortnite"], data_root=tmp_path)
        pipeline.config.sample_target = 1
        return pipeline

    def test_fetch_auth_failure_returns_empty(self, cfgs, tmp_path):
        pipeline = self._mock_pipeline(cfgs, tmp_path)
        # Simulate OAuth failure
        with patch.object(pipeline, "_get_token", return_value=""), \
             patch("src.cells.fortnite.pipeline.time.sleep"):
            paths = pipeline.fetch()
        assert paths == []

    def test_fetch_expired_replay_skipped(self, cfgs, tmp_path):
        pipeline = self._mock_pipeline(cfgs, tmp_path)
        # Metadata returns 404 → replay expired
        with patch.object(pipeline, "_get_token", return_value="tok"), \
             patch.object(pipeline, "_get_metadata", return_value=None), \
             patch("src.cells.fortnite.pipeline.time.sleep"):
            paths = pipeline.fetch()
        assert paths == []

    def test_fetch_writes_json_on_success(self, cfgs, tmp_path):
        pipeline = self._mock_pipeline(cfgs, tmp_path)
        fake_meta = {"Events": [{"Id": "chunk1"}]}
        with patch.object(pipeline, "_get_token", return_value="tok"), \
             patch.object(pipeline, "_get_metadata", return_value=fake_meta), \
             patch.object(pipeline, "_download_chunk", return_value=b""), \
             patch("src.cells.fortnite.pipeline._parse_event_chunk",
                   return_value=self._FAKE_EVENTS[0]), \
             patch("src.cells.fortnite.pipeline.time.sleep"):
            paths = pipeline.fetch()
        assert len(paths) == 1
        assert paths[0].suffix == ".json"

    def test_fetch_skips_existing_file(self, cfgs, tmp_path):
        pipeline = self._mock_pipeline(cfgs, tmp_path)
        # Pre-write the output file
        norm_id = "0fbcece6e1774c9abdd1ebd88fbad4ec"
        existing = pipeline.raw_dir / f"{norm_id}.json"
        existing.write_text(json.dumps(self._FAKE_RECORD))
        with patch.object(pipeline, "_get_token", return_value="tok"):
            paths = pipeline.fetch()
        # Should not call _get_token at all for already-downloaded match
        assert existing in paths

    def test_parse_loads_json_file(self, cfgs, tmp_path):
        pipeline = self._mock_pipeline(cfgs, tmp_path)
        p = pipeline.raw_dir / "abc.json"
        p.write_text(json.dumps(self._FAKE_RECORD))
        records = pipeline.parse([p])
        assert len(records) == 1
        assert records[0]["match_id"] == "abc123"

    def test_parse_skips_corrupt_file(self, cfgs, tmp_path):
        pipeline = self._mock_pipeline(cfgs, tmp_path)
        p = pipeline.raw_dir / "bad.json"
        p.write_text("not json")
        records = pipeline.parse([p])
        assert records == []

    def test_extract_events_from_event_chunks(self, cfgs, tmp_path):
        pipeline = self._mock_pipeline(cfgs, tmp_path)
        streams = pipeline.extract_events([self._FAKE_RECORD])
        assert len(streams) == 1
        stream = streams[0]
        zone = [e for e in stream.events if e.event_type == "zone_enter"]
        engage = [e for e in stream.events if e.event_type == "engage_decision"]
        assert len(zone) == 1
        assert len(engage) == 1
