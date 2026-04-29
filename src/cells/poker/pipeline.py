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
import re
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
# Subsets to ingest. Per D-37 (DECISION_LOG): Pluribus is dropped because
# the bot occupies one of the 6 seats in every hand, contaminating ~17% of
# actions with superhuman-bot decisions that fall outside the human population
# under study. We use HandHQ (anonymised human cash games) plus the 83 WSOP
# 2023 $50K Players Championship hands.
#
# PHH dataset v3 layout (nested):
#   data/handhq/<site-stake>/<bb>/*.phhs   — multi-hand TOML streams (~1k hands/file)
#   data/wsop/2023/43/5/*.phh              — 83 single-hand TOML files
# We rglob each subset root for both .phh and .phhs files at any depth.
_PHH_SUBSETS = ["wsop/2023", "handhq"]
# Only keep handhq hands with this seat count — full-ring 6-max is the standard
# 6-handed cash format and matches Pluribus's structure for downstream comparison.
_HANDHQ_TARGET_SEATS = 6

# Per D-37 clarification: filter HandHQ to mid/high stakes (200NL–1000NL).
# Lower brackets (25NL/50NL/100NL) reflect highly exploitative recreational
# play that is structurally distinct from the GTO-aware decision distribution
# the experiment is meant to characterise. 200NL+ also mirrors the standard
# threshold in modern poker training data. HandHQ encodes stake in the
# parent dir name as `_<NLH-bracket>NLH_`, e.g. `..._600NLH_OBFU`.
_HANDHQ_ALLOWED_STAKES = frozenset({"200", "400", "600", "1000"})
_HANDHQ_STAKE_RE = re.compile(r"_(\d+)NLH_")

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

        Returns list of file paths (both .phh and .phhs) from the active
        subsets per D-37: handhq (multi-hand .phhs streams) and wsop/2023
        (single-hand .phh files). Download is skipped when files exist.
        """
        raw_dir = self.raw_dir / "phh"
        raw_dir.mkdir(parents=True, exist_ok=True)

        all_paths: list[Path] = []
        need_download = False

        for subset in _PHH_SUBSETS:
            for candidate in [
                raw_dir / subset,
                raw_dir / _PHH_EXTRACT_SUBDIR / "data" / subset,
            ]:
                if candidate.exists():
                    found = _scan_subset(candidate)
                    if found:
                        logger.info(f"[poker] {subset}: {len(found)} cached files")
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

            all_paths = []
            for subset in _PHH_SUBSETS:
                for candidate in [
                    raw_dir / _PHH_EXTRACT_SUBDIR / "data" / subset,
                    raw_dir / subset,
                ]:
                    found = _scan_subset(candidate)
                    if found:
                        all_paths.extend(found)
                        break

        logger.info(f"[poker] {len(all_paths)} files available")
        return all_paths

    def parse(self, raw_paths: list[Path]) -> list[dict]:
        """
        Parse PHH files as TOML and anonymise player names (CF-4=B).

        Both .phh (single hand) and .phhs (multi-hand stream) are plain TOML,
        so we sidestep pokerkit entirely. handhq files are .phhs streams of
        ~1k hands; we filter to _HANDHQ_TARGET_SEATS (6-max) per D-37.

        Returns a list of hand dicts with keys:
          game_id, players (anonymised), starting_stacks, blinds,
          big_blind, actions, subset
        """
        try:
            import tomllib as _toml  # 3.11+ stdlib
        except ImportError:
            try:
                import tomli as _toml  # 3.9/3.10 backport
            except ImportError:
                logger.warning(
                    "[poker] Neither tomllib nor tomli is installed. "
                    "Install with: pip install tomli. Returning empty parse."
                )
                return []

        records: list[dict] = []
        target = self.config.sample_target

        # Per-stake budgets so the sample is stratified across the four
        # allowed HandHQ brackets rather than consumed entirely from whichever
        # stake sorts first lexicographically. WSOP hands are uncapped (small
        # premium subset, processed before handhq via _PHH_SUBSETS ordering).
        wsop_quota_estimate = 83
        per_stake_budget = max(
            1,
            (target - wsop_quota_estimate) // len(_HANDHQ_ALLOWED_STAKES),
        )
        budgets: dict[str, int] = {s: per_stake_budget for s in _HANDHQ_ALLOWED_STAKES}

        for path in raw_paths:
            if len(records) >= target:
                break

            stake = None
            if "handhq" in path.parts:
                m = _HANDHQ_STAKE_RE.search(str(path))
                if not m or m.group(1) not in _HANDHQ_ALLOWED_STAKES:
                    continue
                stake = m.group(1)
                if budgets.get(stake, 0) <= 0:
                    continue

            try:
                for rec in _iter_records_from_path(_toml, path):
                    records.append(rec)
                    if stake is not None:
                        budgets[stake] -= 1
                        if budgets[stake] <= 0:
                            break
                    if len(records) >= target:
                        break
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


def _scan_subset(root: Path) -> list[Path]:
    """
    Return sorted list of .phh and .phhs files under `root`.
    Both extensions coexist in v3: handhq uses .phhs (multi-hand streams),
    wsop and pluribus use .phh (single-hand files).
    """
    return sorted(list(root.rglob("*.phh")) + list(root.rglob("*.phhs")))


def _iter_records_from_path(toml_module, path: Path):
    """
    Yield record dicts from a .phh or .phhs file.

    .phh  → one record from the file's top-level table.
    .phhs → one record per [N] sub-table (each a separate hand).

    For handhq paths, hands not matching _HANDHQ_TARGET_SEATS are filtered.
    """
    is_handhq = "handhq" in path.parts

    # Per D-37 clarification: drop entire HandHQ files whose stake bracket
    # is outside the allowed set. This filter operates at the file level
    # because each .phhs file is homogeneous on stake.
    if is_handhq:
        m = _HANDHQ_STAKE_RE.search(str(path))
        if not m or m.group(1) not in _HANDHQ_ALLOWED_STAKES:
            return

    with open(path, "rb") as f:
        data = toml_module.load(f)

    subset = _derive_subset(path)

    if path.suffix == ".phhs":
        # Multi-hand stream: top-level keys are stringified hand indices ("1", "2", ...).
        for idx, hand in data.items():
            if not isinstance(hand, dict):
                continue
            if is_handhq and len(hand.get("players", []) or []) != _HANDHQ_TARGET_SEATS:
                continue
            rec = _record_from_hand_dict(hand, path.stem + f"_{idx}", subset)
            if rec:
                yield rec
    else:
        # .phh single-hand file: data is the hand itself.
        rec = _record_from_hand_dict(data, path.stem, subset)
        if rec:
            yield rec


def _record_from_hand_dict(hand: dict, base_id: str, subset: str) -> dict | None:
    """
    Build the record dict the extractor expects from a parsed PHH hand.

    Anonymises player names (CF-4=B). HandHQ names are already obfuscated
    base64 hashes; WSOP names are real and replaced here.
    """
    players = list(hand.get("players", []) or [])
    actions = list(hand.get("actions", []) or [])
    if not players or not actions:
        return None

    # PHH stacks/stakes can be float for cash games or int for tournaments.
    # Cash games occasionally use `inf` for unknown starting stacks; tomli
    # passes those through as float('inf'), which int() would crash on.
    starting_stacks = [_safe_int(x) for x in hand.get("starting_stacks", []) or []]
    blinds = [_safe_int(x) for x in hand.get("blinds_or_straddles", []) or []]

    actor_map = {name: f"actor_{i}" for i, name in enumerate(players)}
    anon_players = [actor_map.get(p, f"actor_{i}") for i, p in enumerate(players)]
    big_blind = max(blinds) if blinds else 100

    return {
        "game_id": f"pk_{base_id}",
        "players": anon_players,
        "n_players": len(players),
        "starting_stacks": starting_stacks,
        "blinds": blinds,
        "big_blind": big_blind,
        "actions": actions,
        "subset": subset,
    }


def _safe_int(x) -> int:
    """Coerce TOML numeric (incl. float('inf')) to int with a sane default."""
    try:
        v = int(x)
        return v
    except (ValueError, TypeError, OverflowError):
        # `inf` or non-numeric — use a large stand-in so the extractor's
        # stack arithmetic doesn't blow up. Real-world stacks rarely exceed
        # 1M big blinds.
        return 1_000_000


def _derive_subset(path: Path) -> str:
    """First dir below v3's data/ root — e.g. 'handhq', 'wsop', 'pluribus'."""
    parts = path.parts
    # The path contains 'data' twice: once as the project root (data/raw/...)
    # and once as the dataset's data/ dir under phh-dataset-3/. We want the
    # one under phh-dataset-3, which is the LAST occurrence.
    last_data = max((i for i, p in enumerate(parts) if p == "data"), default=-1)
    if last_data >= 0 and last_data + 1 < len(parts):
        return parts[last_data + 1]
    return "unknown"


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
