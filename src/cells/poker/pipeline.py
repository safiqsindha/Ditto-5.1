"""
Texas Hold'em poker pipeline for v5.

Data source: PHH dataset (https://github.com/uoftcprg/phh-dataset), v3 release.
Primary subsets: Pluribus (10,000 hands, Brown & Sandholm 2019) and
  WSOP 2023 $50K Players Championship (83 hands).
No API keys required; MIT-licensed public dataset.

Pipeline stages:
  fetch()          — download and extract PHH v3 tarball into raw dir
  parse()          — load each .phh file via pokerkit, anonymise names (CF-4=B)
  extract_events() — PokerExtractor converts hand dicts → EventStreams
  generate_mock_data() — deterministic synthetic hands for dry-run / CI

ME-PK-2 stub: HandHQ skill-stratified cell uses a separate fetch path.
"""

from __future__ import annotations

import logging
import random
import tarfile
import time
import urllib.request
from pathlib import Path

from ...common.config import CellConfig
from ...common.schema import EventStream
from ..base_pipeline import BasePipeline
from .extractor import PokerExtractor

logger = logging.getLogger(__name__)

_PHH_TARBALL_URL = (
    "https://github.com/uoftcprg/phh-dataset/archive/refs/tags/v3.tar.gz"
)
# Directory name inside the tarball
_PHH_EXTRACT_SUBDIR = "phh-dataset-3"
# Subsets to ingest (in order of priority)
_PHH_SUBSETS = ["pluribus", "wsop-2023-50k"]

# Normalized mock event types — ordered so common actions come first
_POKER_EVENT_TYPES = [
    "engage_decision",       # call / bet / raise
    "engage_decision",       # duplicated to raise probability
    "engage_decision",
    "disengage_decision",    # fold / check
    "disengage_decision",
    "risk_accept",           # all-in shove
    "resource_spend",        # large bet
    "resource_budget",       # stack management decision
    "position_commit",       # commit to street action
    "strategy_adapt",        # adjust line based on board
    "timing_commit",         # clock pressure
    "objective_contest",     # showdown
]

_STREETS = ["preflop", "flop", "turn", "river"]
_POSITIONS_6MAX = ["SB", "BB", "UTG", "HJ", "CO", "BTN"]


class PokerPipeline(BasePipeline):
    """
    Pipeline for Texas Hold'em poker chains from the PHH dataset.

    Fetches the PHH v3 tarball from GitHub (no auth required).
    Uses pokerkit (the official PHH parser) for .phh loading.
    Falls back to mock data when pokerkit is absent or credentials are unset.
    """

    def __init__(self, config: CellConfig, data_root: Path | None = None):
        super().__init__(config, data_root or Path(__file__).parent.parent.parent.parent / "data")
        self.extractor = PokerExtractor()

    # ------------------------------------------------------------------
    # BasePipeline abstract methods
    # ------------------------------------------------------------------

    def fetch(self) -> list[Path]:
        """
        Download and extract the PHH v3 dataset tarball.

        Returns list of .phh file paths from Pluribus + WSOP subsets.
        Download is skipped when files already exist.
        """
        raw_dir = self.raw_dir / "phh"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Check whether we already have extracted .phh files
        all_paths: list[Path] = []
        need_download = False

        for subset in _PHH_SUBSETS:
            # Try both pre-extracted path and within-tarball path
            for candidate in [
                raw_dir / subset,
                raw_dir / _PHH_EXTRACT_SUBDIR / "data" / subset,
            ]:
                if candidate.exists():
                    found = sorted(candidate.glob("*.phh"))
                    if found:
                        logger.info(f"[poker] {subset}: {len(found)} cached .phh files")
                        all_paths.extend(found)
                        break
            else:
                need_download = True

        if need_download:
            tarball = raw_dir / "phh_v3.tar.gz"
            if not tarball.exists():
                logger.info(f"[poker] Downloading PHH dataset from {_PHH_TARBALL_URL}")
                try:
                    _download_with_retry(_PHH_TARBALL_URL, tarball)
                except Exception as exc:
                    logger.warning(f"[poker] Download failed: {exc}")
                    return []

            logger.info(f"[poker] Extracting tarball to {raw_dir}")
            try:
                with tarfile.open(tarball) as tf:
                    tf.extractall(raw_dir)
            except Exception as exc:
                logger.warning(f"[poker] Extraction failed: {exc}")
                return []

            # Re-scan after extraction
            all_paths = []
            for subset in _PHH_SUBSETS:
                for candidate in [
                    raw_dir / _PHH_EXTRACT_SUBDIR / "data" / subset,
                    raw_dir / subset,
                ]:
                    found = sorted(candidate.glob("*.phh"))
                    if found:
                        all_paths.extend(found)
                        break

        logger.info(f"[poker] {len(all_paths)} .phh files available")
        return all_paths

    def parse(self, raw_paths: list[Path]) -> list[dict]:
        """
        Load each .phh file via pokerkit and anonymise player names (CF-4=B).

        Returns a list of hand dicts with keys:
          game_id, players (anonymised), starting_stacks, blinds,
          big_blind, actions, subset
        """
        try:
            from pokerkit import HandHistory  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "[poker] pokerkit not installed. "
                "Install with: pip install pokerkit. "
                "Returning empty parse."
            )
            return []

        records: list[dict] = []
        target = self.config.sample_target

        for path in raw_paths:
            if len(records) >= target:
                break
            try:
                record = _load_phh_record(HandHistory, path)
                if record:
                    records.append(record)
            except Exception as exc:
                logger.debug(f"[poker] parse error {path.name}: {exc}")

        logger.info(f"[poker] Parsed {len(records)} hands")
        return records

    def extract_events(self, records: list[dict]) -> list[EventStream]:
        """Convert parsed hand dicts → EventStreams via PokerExtractor."""
        streams: list[EventStream] = []
        for rec in records:
            try:
                stream = self.extractor.extract(rec)
                if stream.events:
                    streams.append(stream)
            except Exception as exc:
                logger.debug(f"[poker] extract error {rec.get('game_id', '?')}: {exc}")
        return streams

    def generate_mock_data(self) -> list[EventStream]:
        """
        Generate deterministic synthetic poker hands for dry-run / CI.

        300 hands × ~24 events each (~4 events × 6 players).
        Actors labelled actor_0 … actor_5 (CF-4=B compliant).
        Uses _make_mock_stream so e.metadata["mock"] = True on every event.
        Streets stamped as phase= for PokerT grouping.
        """
        n_hands = self.config.sample_target
        # 60% Pluribus-style 6-max, 40% WSOP 6-handed final-table style
        n_pluribus = int(n_hands * 0.60)
        n_wsop = n_hands - n_pluribus

        streams: list[EventStream] = []
        actors = [f"actor_{j}" for j in range(6)]

        for i in range(n_pluribus):
            stream = self._make_mock_stream(
                game_id=f"mock_pk_pluribus_{i:05d}",
                cell="poker",
                n_events=24,
                event_types=_POKER_EVENT_TYPES,
                actors=actors,
                seed=i,
            )
            stream.metadata.update({
                "subset": "pluribus",
                "n_players": 6,
                "big_blind": 100,
            })
            _stamp_poker_streets(stream)
            streams.append(stream)

        for i in range(n_wsop):
            stream = self._make_mock_stream(
                game_id=f"mock_pk_wsop_{i:05d}",
                cell="poker",
                n_events=20,
                event_types=_POKER_EVENT_TYPES,
                actors=actors,
                seed=50000 + i,
            )
            stream.metadata.update({
                "subset": "wsop-2023-50k",
                "n_players": 6,
                "big_blind": 400,
            })
            _stamp_poker_streets(stream)
            streams.append(stream)

        logger.info(f"[poker] Generated {len(streams)} mock hands")
        return streams


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _stamp_poker_streets(stream: EventStream) -> None:
    """
    Stamp events with phase = street (preflop/flop/turn/river).

    Distribution: 50% preflop, 25% flop, 15% turn, 10% river —
    mirrors real NLHE action distribution.
    """
    n = len(stream.events)
    if n == 0:
        return
    street_splits = [int(n * 0.50), int(n * 0.75), int(n * 0.90), n]
    for idx, ev in enumerate(stream.events):
        if idx < street_splits[0]:
            ev.phase = "preflop"
        elif idx < street_splits[1]:
            ev.phase = "flop"
        elif idx < street_splits[2]:
            ev.phase = "turn"
        else:
            ev.phase = "river"
        # Mirror phase into location_context for PokerT
        ev.location_context["street"] = ev.phase


def _load_phh_record(HandHistory, path: Path) -> dict | None:
    """
    Load one .phh file via pokerkit and return an anonymised hand dict.

    Tries both context-manager and direct-load API patterns to stay
    compatible with pokerkit >= 0.5.
    """
    hh = None
    # Try the newer from_file_path class method
    load_fn = getattr(HandHistory, "from_file_path", None)
    if load_fn:
        try:
            hh = load_fn(str(path))
        except Exception:
            hh = None

    # Fallback: open binary and pass to load()
    if hh is None:
        load_bin = getattr(HandHistory, "load", None)
        if load_bin:
            with open(path, "rb") as f:
                hh = load_bin(f)

    if hh is None:
        return None

    players = list(getattr(hh, "players", []) or [])
    starting_stacks = [int(x) for x in getattr(hh, "starting_stacks", []) or []]
    blinds = [int(x) for x in getattr(hh, "blinds_or_straddles", []) or []]
    actions = list(getattr(hh, "actions", []) or [])

    # CF-4=B: replace player names with stable actor_N IDs
    actor_map = {name: f"actor_{i}" for i, name in enumerate(players)}
    anon_players = [actor_map.get(p, f"actor_{i}") for i, p in enumerate(players)]

    big_blind = max(blinds) if blinds else 100

    return {
        "game_id": f"pk_{path.stem}",
        "players": anon_players,
        "n_players": len(players),
        "starting_stacks": starting_stacks,
        "blinds": blinds,
        "big_blind": big_blind,
        "actions": actions,
        "subset": path.parent.name,
    }


def _download_with_retry(
    url: str,
    dest: Path,
    max_retries: int = 4,
    backoff_base: float = 2.0,
) -> None:
    """Download url to dest with exponential-backoff retries."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = backoff_base ** attempt
                logger.warning(f"[poker] Download attempt {attempt + 1} failed: {exc}; retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Download failed after {max_retries} attempts") from last_exc
