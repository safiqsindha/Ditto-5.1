"""
Rocket League data acquisition pipeline for v5.

Data source: Ballchasing.com API.
API key required: env var BALLCHASING_TOKEN (get at ballchasing.com → Settings).

Targeting: ranked-standard (3v3), Diamond I+ rank — above-average players who
engage seriously with boost-economy and rotation constraints.

Fetch strategy: JSON metadata only (GET /replays/{id}) — no binary .replay
download, no carball/rrrocket dependency. Per-player aggregate stats are
converted to a synthetic event stream by the extractor.

Rate limits (free tier): 3.6 s between requests. Respect Retry-After on 429.
Pagination: cursor-based via `next` field in list response.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import requests

from ...common.config import CellConfig
from ...common.schema import EventStream
from ..base_pipeline import BasePipeline
from .extractor import RocketLeagueExtractor

logger = logging.getLogger(__name__)

BALLCHASING_API = "https://ballchasing.com/api"
# Free-tier rate limit: 3.6 s between requests. Higher tiers can lower this.
BALLCHASING_RATE_LIMIT_DELAY = 3.6
_MAX_RETRIES = 8


class RocketLeaguePipeline(BasePipeline):
    """
    Rocket League pipeline: lists ranked replays, fetches JSON stats, extracts events.

    fetch()          → downloads JSON metadata per replay from Ballchasing API
    parse()          → loads JSON files from disk
    extract_events() → RocketLeagueExtractor.extract() per record
    """

    def __init__(self, config: CellConfig, data_root: Path | None = None):
        super().__init__(config, data_root or Path(__file__).parent.parent.parent.parent / "data")
        self.extractor = RocketLeagueExtractor()
        self.api_token = os.getenv("BALLCHASING_TOKEN", "")
        self.session = requests.Session()
        if self.api_token:
            # Ballchasing auth: bare token, no "Bearer" prefix
            self.session.headers.update({"Authorization": self.api_token})

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def fetch(self) -> list[Path]:
        """Fetch replay JSON metadata from Ballchasing API (no binary download)."""
        replay_ids = self._list_ranked_replays()
        paths: list[Path] = []
        for replay_id in replay_ids:
            out_path = self.raw_dir / f"{replay_id}.json"
            if out_path.exists():
                paths.append(out_path)
                continue
            resp = self._get_with_backoff(f"{BALLCHASING_API}/replays/{replay_id}")
            if resp is None:
                continue
            try:
                with open(out_path, "w") as f:
                    json.dump(resp.json(), f)
                paths.append(out_path)
                logger.info(f"[rocket_league] Fetched replay {replay_id}")
            except Exception as e:
                logger.error(f"[rocket_league] Failed to save {replay_id}: {e}")
            time.sleep(BALLCHASING_RATE_LIMIT_DELAY)
        return paths

    def parse(self, raw_paths: list[Path]) -> list[dict]:
        """Load replay JSON files from disk."""
        records: list[dict] = []
        for path in raw_paths:
            try:
                with open(path) as f:
                    records.append(json.load(f))
            except Exception as e:
                logger.warning(f"[rocket_league] Failed to load {path.name}: {e}")
        return records

    def extract_events(self, game_records: list[dict]) -> list[EventStream]:
        return [self.extractor.extract(record) for record in game_records if record]

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def _list_ranked_replays(self) -> list[str]:
        """
        Paginate Ballchasing for ranked-standard replays at Diamond I+ rank.

        Returns up to sample_target replay IDs. Follows the `next` cursor
        until the target is reached or no more pages.
        """
        ids: list[str] = []
        url = f"{BALLCHASING_API}/replays"
        params: dict | None = {
            "playlist": "ranked-standard",
            "min-rank": "diamond-i",
            "count": 200,
            "sort-by": "replay-date",
            "sort-dir": "desc",
        }
        while url and len(ids) < self.config.sample_target:
            resp = self._get_with_backoff(url, params=params)
            if resp is None:
                break
            try:
                data = resp.json()
            except Exception as e:
                logger.error(f"[rocket_league] JSON decode error: {e}")
                break

            page_ids = [r["id"] for r in data.get("list", []) if "id" in r]
            ids.extend(page_ids)
            logger.info(
                f"[rocket_league] Listed {len(ids)}/{self.config.sample_target} replays"
            )

            next_url = data.get("next")
            if not next_url or not page_ids:
                break
            # `next` is a full URL with cursor; clear params to avoid duplication
            url = next_url
            params = None
            time.sleep(BALLCHASING_RATE_LIMIT_DELAY)

        return ids[: self.config.sample_target]

    def _get_with_backoff(
        self, url: str, params: dict | None = None
    ) -> requests.Response | None:
        """GET with exponential backoff on 429 / 5xx."""
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(
                        f"[rocket_league] Rate limited; sleeping {retry_after}s"
                    )
                    time.sleep(retry_after)
                    continue
                if resp.status_code >= 500:
                    wait = 2 ** attempt
                    logger.warning(
                        f"[rocket_league] Server error {resp.status_code}; "
                        f"retrying in {wait}s"
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt == _MAX_RETRIES - 1:
                    logger.error(
                        f"[rocket_league] Request failed after {_MAX_RETRIES} "
                        f"attempts: {e}"
                    )
                    return None
                time.sleep(2 ** attempt)
        return None

    # ------------------------------------------------------------------
    # Mock data
    # ------------------------------------------------------------------

    def generate_mock_data(self) -> list[EventStream]:
        """
        Generate mock Rocket League event streams.
        250 ranked-standard replays at Diamond/Champion/GC distribution.
        ~200 events per replay (boost pickups, shots, saves, demos, rotation).
        """
        streams = []
        rank_map = {
            "diamond": int(self.config.sample_target * 0.50),
            "champion": int(self.config.sample_target * 0.35),
            "grand_champion": (
                self.config.sample_target
                - int(self.config.sample_target * 0.50)
                - int(self.config.sample_target * 0.35)
            ),
        }
        i = 0
        for rank, count in rank_map.items():
            for j in range(count):
                stream = self._make_mock_stream(
                    game_id=f"mock_rl_{rank}_{j:04d}",
                    cell="rocket_league",
                    n_events=200,
                    event_types=RL_MOCK_EVENT_TYPES,
                    actors=[
                        "blue_0", "blue_1", "blue_2",
                        "orange_0", "orange_1", "orange_2",
                    ],
                    seed=i,
                )
                stream.metadata.update({"rank_tier": rank, "playlist": "ranked-standard"})
                streams.append(stream)
                i += 1
        logger.info(f"[rocket_league] Generated {len(streams)} mock streams")
        return streams


RL_MOCK_EVENT_TYPES = [
    "resource_gain",         # boost pickup
    "resource_spend",        # boost use
    "resource_budget",       # low-boost decision
    "engage_decision",       # shot attempt
    "disengage_decision",    # clearing / saving
    "objective_contest",     # 50/50 challenge
    "objective_capture",     # goal scored
    "position_commit",       # rotating back / positioning
    "rotation_commit",       # team rotation
    "team_coordinate",       # passing / setting up teammate
    "risk_accept",           # demo attempt
    "risk_reject",           # avoiding challenge
    "timing_commit",         # aerial timing
    "ability_use",           # power slide / dodge
    "strategy_adapt",        # defensive shift after goal
    "zone_enter",            # entering boost pad zone
    "zone_exit",             # leaving boost pad zone
]
