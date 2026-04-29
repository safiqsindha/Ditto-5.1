"""
NBA data acquisition pipeline for v5.

Data source: NBA Stats API via nba_api Python package.
Endpoint: PlayByPlayV3 (publicly accessible, no API key required).

Tracking data (Second Spectrum spatial tracking): NOT publicly accessible.
This cell uses play-by-play events only (decision D-N1).

Sample target: 300 games — 240 regular season + 60 playoffs, 2023-24 season.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ...common.config import CellConfig
from ...common.schema import EventStream
from ..base_pipeline import BasePipeline
from .extractor import NBAExtractor

logger = logging.getLogger(__name__)

# 2023-24 season game ID ranges (NBA game IDs are 10 digits: 002YYXXXXX)
SEASON_2024 = "2023-24"
PLAYOFF_SEASON_2024 = "2023-24"  # Same season for API

# Sample game IDs for seeding (will be overridden by real fetch)
# 2023-24 regular season had 1,230 games (`0022300001`–`0022301230`).
# Range goes up to 400 here to allow playoff-slot backfill (NBA-1 workaround).
SAMPLE_RS_GAME_IDS = [f"002230{i:04d}" for i in range(1, 401)]   # regular season
SAMPLE_PO_GAME_IDS = [f"004230{i:04d}" for i in range(1, 61)]    # playoffs (KNOWN BUG NBA-1)


class NBAPipeline(BasePipeline):
    """
    NBA pipeline: fetches play-by-play via nba_api, extracts game events.

    fetch()         → queries PlayByPlayV3 for each target game
    parse()         → returns raw play-by-play dicts unchanged
    extract_events()→ NBAExtractor.extract() per game
    """

    def __init__(self, config: CellConfig, data_root: Path | None = None):
        super().__init__(config, data_root or Path(__file__).parent.parent.parent.parent / "data")
        self.extractor = NBAExtractor()

    def fetch(self) -> list[Path]:
        """
        Fetch play-by-play data for all target games.
        Saves raw JSON to data/raw/nba/{game_id}.json.
        """
        try:
            from nba_api.stats.endpoints import playbyplayv3
        except ImportError:
            logger.error("nba_api not installed. Run: pip install nba_api")
            return []

        game_ids = self._get_target_game_ids()
        paths = []
        for game_id in game_ids:
            out_path = self.raw_dir / f"{game_id}.json"
            if out_path.exists():
                paths.append(out_path)
                continue
            try:
                pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
                data = pbp.get_dict()
                import json
                with open(out_path, "w") as f:
                    json.dump(data, f)
                paths.append(out_path)
                time.sleep(0.6)  # NBA API rate limit courtesy delay
            except Exception as e:
                logger.warning(f"Failed to fetch game {game_id}: {e}")
        return paths

    def parse(self, raw_paths: list[Path]) -> list[dict]:
        import json
        records = []
        for path in raw_paths:
            try:
                with open(path) as f:
                    records.append(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to parse {path}: {e}")
        return records

    def extract_events(self, game_records: list[dict]) -> list[EventStream]:
        return [self.extractor.extract(record) for record in game_records if record]

    def generate_mock_data(self) -> list[EventStream]:
        """
        Generate mock NBA play-by-play event streams.
        300 games: 240 regular season + 60 playoff.
        ~180 events per game (NBA average ~220 plays/game, filtered to decision events).
        Events carry location_context["period"] so NBAT can group by quarter.
        """
        streams = []
        n_rs = int(self.config.sample_target * 0.80)
        n_po = self.config.sample_target - n_rs

        for i in range(n_rs):
            stream = self._make_mock_stream(
                game_id=f"mock_nba_rs_{i:04d}",
                cell="nba",
                n_events=180,
                event_types=NBA_MOCK_EVENT_TYPES,
                actors=[f"player_{j}" for j in range(10)],
                seed=i,
            )
            stream.metadata.update({"phase": "regular_season", "season": "2023-24"})
            _stamp_nba_periods(stream, n_periods=4)
            streams.append(stream)

        for i in range(n_po):
            stream = self._make_mock_stream(
                game_id=f"mock_nba_po_{i:04d}",
                cell="nba",
                n_events=200,
                event_types=NBA_MOCK_EVENT_TYPES,
                actors=[f"player_{j}" for j in range(10)],
                seed=10000 + i,
            )
            stream.metadata.update({"phase": "playoffs", "season": "2023-24"})
            _stamp_nba_periods(stream, n_periods=4)
            streams.append(stream)

        logger.info(f"[nba] Generated {len(streams)} mock streams")
        return streams

    def _get_target_game_ids(self) -> list[str]:
        """
        Return list of game IDs to fetch based on stratification config.

        KNOWN BUG (NBA-1): SAMPLE_PO_GAME_IDS generates playoff IDs sequentially
        (`004230{i:04d}` for i in 1..60) but NBA playoff IDs are not sequential —
        they follow a series-based format (`0042300101` = round 1 series 1 game 1,
        `0042300401` = Finals game 1, etc.). The hardcoded sequential IDs
        produce 60 invalid IDs that all 404 from NBA Stats and crash nba_api's
        parser with `list index out of range`.

        TODO: replace with `nba_api.stats.endpoints.LeagueGameFinder` query
        for season "2023-24" + season_type_nullable in {"Regular Season", "Playoffs"}
        to get actual valid game IDs. For now, restrict to regular season so the
        cell can produce real data without the playoff bug blocking everything.
        """
        strat = self.config.stratification
        game_ids: list[str] = []
        for s in strat:
            phase = s.get("phase", "")
            fraction = s.get("fraction", 0.5)
            n = int(self.config.sample_target * fraction)
            if phase == "regular_season":
                game_ids.extend(SAMPLE_RS_GAME_IDS[:n])
            elif phase == "playoffs":
                # SKIPPED — see NBA-1 above. Backfill the playoff slots with
                # additional regular-season games so we still hit sample_target.
                logger.warning(
                    "[nba] Playoff game IDs are sequentially-generated and invalid; "
                    "skipping %d playoff slots and using regular-season IDs instead. "
                    "See NBA-1 in pipeline.py.",
                    n,
                )
                game_ids.extend(SAMPLE_RS_GAME_IDS[len(game_ids): len(game_ids) + n])
        return game_ids or SAMPLE_RS_GAME_IDS[: self.config.sample_target]


NBA_MOCK_EVENT_TYPES = [
    "engage_decision",       # shot attempt
    "target_select",         # defensive assignment
    "position_commit",       # pick-and-roll, off-ball position
    "rotation_commit",       # defensive rotation
    "resource_gain",         # offensive rebound
    "resource_spend",        # turnover / possession loss
    "team_coordinate",       # timeout, play call
    "timing_commit",         # shot clock management
    "risk_accept",           # fouling decision
    "risk_reject",           # letting clock run / intentional miss
    "objective_contest",     # contested shot / box out
    "strategy_adapt",        # in-game scheme adjustment
    "ability_use",           # free throw (set piece)
    "disengage_decision",    # passing up shot opportunity
]


def _stamp_nba_periods(stream, n_periods: int = 4) -> None:
    """Distribute events across periods so NBAT can group by quarter."""
    n = len(stream.events)
    if n == 0:
        return
    per_period = max(1, n // n_periods)
    for idx, ev in enumerate(stream.events):
        period = min(n_periods, idx // per_period + 1)
        ev.location_context["period"] = period
