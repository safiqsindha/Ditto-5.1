"""
CS:GO / CS2 data acquisition pipeline for v5.

Data source: FACEIT API v4 (https://developers.faceit.com/docs/apis/data-v4).

Fetches CS2 match stats from FACEIT championships at level 5+ skill.
FACEIT_API_KEY environment variable required.

Match stats are per-map aggregate player stats (kills, assists, deaths,
headshots, entry frags, flash count). The extractor synthesizes round-level
events from these aggregates, distributed across the actual round count.

Sample target: 150 CS2 maps from recent FACEIT championships.
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
from .extractor import CSGOExtractor

logger = logging.getLogger(__name__)

FACEIT_API = "https://open.faceit.com/data/v4"
FACEIT_RATE_LIMIT_DELAY = 0.5   # 5 req/s sustained; stay conservative


class CSGOPipeline(BasePipeline):
    """
    CS2 pipeline: fetches FACEIT championship match stats, extracts events.

    fetch()          → downloads per-map stats JSON from FACEIT API
    parse()          → loads saved JSON files
    extract_events() → CSGOExtractor.extract() per record
    """

    def __init__(self, config: CellConfig, data_root: Path | None = None):
        super().__init__(config, data_root or Path(__file__).parent.parent.parent.parent / "data")
        self.extractor = CSGOExtractor()
        self.api_key = os.getenv("FACEIT_API_KEY", "")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    # ------------------------------------------------------------------
    # fetch / parse / extract
    # ------------------------------------------------------------------

    def fetch(self) -> list[Path]:
        if not self.api_key:
            logger.error("[csgo] FACEIT_API_KEY not set — cannot fetch real data")
            return []

        match_ids = self._list_match_ids()
        paths: list[Path] = []

        for match_id in match_ids[:self.config.sample_target]:
            out_path = self.raw_dir / f"{match_id}.json"
            if out_path.exists():
                paths.append(out_path)
                continue
            stats = self._fetch_match_stats(match_id)
            if stats:
                out_path.write_text(json.dumps(stats))
                paths.append(out_path)
                logger.debug(f"[csgo] Saved {match_id}")
            time.sleep(FACEIT_RATE_LIMIT_DELAY)

        logger.info(f"[csgo] Fetched {len(paths)} match stat files")
        return paths

    def parse(self, raw_paths: list[Path]) -> list[dict]:
        records = []
        for path in raw_paths:
            try:
                records.append(json.loads(path.read_text()))
            except Exception as e:
                logger.warning(f"[csgo] Parse error for {path.name}: {e}")
        return records

    def extract_events(self, game_records: list[dict]) -> list[EventStream]:
        return [self.extractor.extract(record) for record in game_records if record]

    # ------------------------------------------------------------------
    # FACEIT API helpers
    # ------------------------------------------------------------------

    def _list_match_ids(self) -> list[str]:
        """Collect CS2 match IDs from recent FACEIT championships."""
        ids: list[str] = []

        championship_ids = self._get_championship_ids()
        logger.info(f"[csgo] Found {len(championship_ids)} championships")

        for champ_id in championship_ids:
            new_ids = self._get_championship_match_ids(champ_id)
            ids.extend(new_ids)
            if len(ids) >= self.config.sample_target:
                break
            time.sleep(FACEIT_RATE_LIMIT_DELAY)

        logger.info(f"[csgo] Collected {len(ids)} match IDs")
        return ids

    def _get_championship_ids(self) -> list[str]:
        url = f"{FACEIT_API}/championships"
        params = {"game": "cs2", "limit": 20}
        resp = self._get_with_backoff(url, params=params)
        if resp is None or resp.status_code != 200:
            logger.warning(f"[csgo] Championships list failed: {getattr(resp, 'status_code', 'no response')}")
            return []
        return [c["championship_id"] for c in resp.json().get("items", [])]

    def _get_championship_match_ids(self, championship_id: str) -> list[str]:
        url = f"{FACEIT_API}/championships/{championship_id}/matches"
        params = {"type": "past", "limit": 20}
        resp = self._get_with_backoff(url, params=params)
        if resp is None or resp.status_code != 200:
            return []
        return [m["match_id"] for m in resp.json().get("items", [])]

    def _fetch_match_stats(self, match_id: str) -> dict | None:
        url = f"{FACEIT_API}/matches/{match_id}/stats"
        resp = self._get_with_backoff(url)
        if resp and resp.status_code == 200:
            return resp.json()
        return None

    def _get_with_backoff(
        self,
        url: str,
        params: dict | None = None,
        max_retries: int = 8,
    ) -> requests.Response | None:
        delays = [1, 2, 4, 8, 16, 30, 60, 120]
        for attempt, delay in enumerate(delays[:max_retries]):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", delay))
                    logger.warning(f"[csgo] Rate limited; sleeping {wait}s")
                    time.sleep(wait)
                    continue
                return resp
            except requests.RequestException as e:
                logger.warning(f"[csgo] Request error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
        return None

    # ------------------------------------------------------------------
    # Mock data
    # ------------------------------------------------------------------

    def generate_mock_data(self) -> list[EventStream]:
        """
        Generate mock CS2 event streams.
        150 tournament maps, ~300 events per map (30 rounds × ~10 events/round).
        Events carry location_context["round"] so CSGOT can group by round.
        """
        streams = []
        tournaments = ["IEM_Katowice_2024", "BLAST_Premier_2024", "ESL_One_2024"]
        per_tournament = self.config.sample_target // len(tournaments)
        i = 0
        for tournament in tournaments:
            for j in range(per_tournament):
                game_id = f"mock_csgo_{tournament.lower()}_{j:04d}"
                stream = self._make_mock_stream(
                    game_id=game_id,
                    cell="csgo",
                    n_events=300,
                    event_types=CSGO_MOCK_EVENT_TYPES,
                    actors=[f"ct_player_{k}" for k in range(5)] +
                           [f"t_player_{k}" for k in range(5)],
                    seed=i,
                )
                stream.metadata.update({
                    "tournament": tournament,
                    "map": ["de_mirage", "de_dust2", "de_inferno",
                            "de_nuke", "de_overpass", "de_ancient"][j % 6],
                    "format": "CS2",
                    "source": "mock",
                })
                _stamp_csgo_rounds(stream, n_rounds=30)
                streams.append(stream)
                i += 1
        logger.info(f"[csgo] Generated {len(streams)} mock streams")
        return streams


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

CSGO_MOCK_EVENT_TYPES = [
    "engage_decision",
    "disengage_decision",
    "resource_gain",
    "resource_spend",
    "resource_budget",
    "position_commit",
    "ability_use",
    "rotation_commit",
    "target_select",
    "zone_enter",
    "zone_exit",
    "objective_contest",
    "objective_capture",
    "team_coordinate",
    "timing_commit",
    "risk_accept",
    "risk_reject",
    "strategy_adapt",
]


def _stamp_csgo_rounds(stream, n_rounds: int = 30) -> None:
    """Distribute events across rounds so CSGOT can group by round."""
    n = len(stream.events)
    if n == 0:
        return
    per_round = max(1, n // n_rounds)
    for idx, ev in enumerate(stream.events):
        rnum = min(n_rounds - 1, idx // per_round)
        ev.location_context["round"] = rnum
