"""
Fortnite data acquisition pipeline for v5.

⚠ LEGACY — superseded by `src/cells/pubg/pipeline.py` per A2 (D-35).
Epic locked down public chunk-data CDN access in 2025–2026 (returns 403 for
all chunk URLs even when metadata APIs return valid `readLink` values), so
this pipeline can no longer fetch real data. Kept as a code reference and
test fixture for the F-1/F-2/F-3 design decisions; not in active evaluation.

Original docstring follows.
---

Downloads tournament match replays from Epic's public datastorage API using
the same embedded client credentials as the open-source ServerReplay Downloader
(github.com/yuyutti/Fortnite_ServerReplay_Downloader).

No EPIC_ACCOUNT_ID or EPIC_ACCESS_TOKEN required — uses client_credentials
OAuth flow with the Fortnite game client ID/secret (public, embedded in
open-source tooling).

Optimization: only downloads Type-3 (Event) chunks from each replay, not the
full data/checkpoint stream. Event chunks are small (~KB each) and contain all
structured game events: eliminations, storm phases, match stats.

Match IDs are provided explicitly (sourced from fortnitetracker.com). Only
tournament matches have server-side replays; replays expire after ~14 days.
"""

from __future__ import annotations

import json
import logging
import struct
import time
from pathlib import Path

import requests

from ...common.config import CellConfig
from ...common.schema import EventStream
from ..base_pipeline import BasePipeline
from .extractor import FortniteExtractor

logger = logging.getLogger(__name__)

# Public Fortnite game client credentials (embedded in open-source replay tools;
# used only for accessing public tournament replay data via client_credentials).
_FN_CLIENT_ID = "3f69e56c7649492c8cc29f1af08a8a12"
_FN_CLIENT_SECRET = "b51ee9cb12234f50a69efa67ef53812e"

_OAUTH_URL = (
    "https://account-public-service-prod.ol.epicgames.com"
    "/account/api/oauth/token"
)
_META_URL = (
    "https://datastorage-public-service-live.ol.epicgames.com"
    "/api/v1/data/fnreplaysmetadata/public/{match_id}.json"
)
_CHUNK_URL = (
    "https://datastorage-public-service-live.ol.epicgames.com"
    "/api/v1/access/fnreplays/public/{match_id}/{chunk_id}"
)

_MAX_RETRIES = 5
_RETRY_BACKOFF = [1, 2, 4, 8, 16]

# Match IDs sourced from fortnitetracker.com (tournament / Cash Cup replays).
# Replays are valid for ~14 days after the match. Add new IDs here as needed.
KNOWN_MATCH_IDS: list[str] = [
    "0fbcece6-e177-4c9a-bdd1-ebd88fbad4ec",
    "845dc7dd-0502-42a8-8e50-50afdd22c0fe",
    "56765d94-3341-4e44-9b0b-27135fc9563d",
    "0e55d827-a6c2-4363-8802-0e28b1df7d81",
    "8e9c5d1225cd471ab370be9c57a3bfbc",  # from fortnitetracker session URL
]


class FortnitePipeline(BasePipeline):
    """
    Fortnite pipeline: downloads tournament replay event chunks, extracts events.

    fetch()          → downloads Type-3 event chunk files per match
    parse()          → parses binary event chunks into structured dicts
    extract_events() → FortniteExtractor.extract() per parsed record
    """

    def __init__(self, config: CellConfig, data_root: Path | None = None):
        super().__init__(
            config, data_root or Path(__file__).parent.parent.parent.parent / "data"
        )
        self.extractor = FortniteExtractor()
        self._token: str = ""
        self._token_expiry: float = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "fortnite-replay-downloader",
            "Content-Type": "application/x-www-form-urlencoded",
        })

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def fetch(self) -> list[Path]:
        """
        Download Type-3 event chunks for each known match ID.

        Saves one JSON file per match containing all parsed event chunks.
        Skips matches where the file already exists (idempotent).
        """
        match_ids = self._get_match_ids()
        paths: list[Path] = []

        for match_id in match_ids[: self.config.sample_target]:
            norm_id = _normalize_match_id(match_id)
            out_path = self.raw_dir / f"{norm_id}.json"
            if out_path.exists():
                paths.append(out_path)
                logger.info(f"[fortnite] {norm_id}: already downloaded, skipping")
                continue

            logger.info(f"[fortnite] Fetching match {norm_id}")
            events = self._fetch_match_events(norm_id)
            if events is None:
                logger.warning(f"[fortnite] {norm_id}: no replay data (expired or not a tournament match)")
                continue

            with open(out_path, "w") as f:
                json.dump({"match_id": norm_id, "events": events}, f)
            paths.append(out_path)
            logger.info(f"[fortnite] {norm_id}: saved {len(events)} event chunks")

        return paths

    def parse(self, raw_paths: list[Path]) -> list[dict]:
        """Load event JSON files from disk."""
        records: list[dict] = []
        for path in raw_paths:
            try:
                with open(path) as f:
                    records.append(json.load(f))
            except Exception as e:
                logger.warning(f"[fortnite] Failed to load {path.name}: {e}")
        return records

    def extract_events(self, game_records: list[dict]) -> list[EventStream]:
        return [self.extractor.extract(r) for r in game_records if r]

    # ------------------------------------------------------------------
    # Epic API: auth + metadata + chunk download
    # ------------------------------------------------------------------

    def _fetch_match_events(self, match_id: str) -> list[dict] | None:
        """
        Download and parse all Type-3 event chunks for one match.
        Returns list of parsed event dicts, or None if the match is unavailable.
        """
        token = self._get_token()
        if not token:
            return None

        # 1. Fetch metadata to get list of event chunk IDs
        meta = self._get_metadata(match_id, token)
        if meta is None:
            return None

        event_chunk_ids: list[str] = [
            e["Id"] for e in meta.get("Events", []) if "Id" in e
        ]
        if not event_chunk_ids:
            logger.warning(f"[fortnite] {match_id}: no event chunks in metadata")
            return []

        # 2. Download each event chunk (small — KB each)
        parsed_events: list[dict] = []
        for chunk_id in event_chunk_ids:
            chunk_bytes = self._download_chunk(match_id, chunk_id, token)
            if chunk_bytes:
                parsed = _parse_event_chunk(chunk_bytes)
                if parsed:
                    parsed_events.append(parsed)

        return parsed_events

    def _get_token(self) -> str:
        """Return a valid bearer token, refreshing if needed."""
        now = time.time()
        if self._token and now < self._token_expiry - 30:
            return self._token

        try:
            resp = self.session.post(
                _OAUTH_URL,
                data={
                    "grant_type": "client_credentials",
                    "token_type": "eg1",
                },
                auth=(_FN_CLIENT_ID, _FN_CLIENT_SECRET),
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            self._token = data["access_token"]
            self._token_expiry = now + data.get("expires_in", 3600)
            logger.info("[fortnite] OAuth token obtained")
            return self._token
        except Exception as e:
            logger.error(f"[fortnite] OAuth failed: {e}")
            return ""

    def _get_metadata(self, match_id: str, token: str) -> dict | None:
        url = _META_URL.format(match_id=match_id)
        resp = self._get(url, token)
        if resp is None:
            return None
        try:
            return resp.json()
        except Exception as e:
            logger.error(f"[fortnite] Metadata JSON decode error for {match_id}: {e}")
            return None

    def _download_chunk(self, match_id: str, chunk_id: str, token: str) -> bytes | None:
        # Step 1: get presigned download URL
        link_url = _CHUNK_URL.format(match_id=match_id, chunk_id=chunk_id)
        resp = self._get(link_url, token)
        if resp is None:
            return None
        try:
            body = resp.json()
        except Exception:
            return None
        # Epic's actual response shape:
        #   {"files": {"<path>": {"readLink": "<presigned-cloudfront-url>", ...}},
        #    "folderThrottled": ..., "maxFileSizeBytes": ..., "expiresAt": ...}
        # The download URL lives in files[<single-key>]["readLink"], NOT at the
        # top level. The dict has a single entry per chunk-access call.
        files = body.get("files", {})
        if isinstance(files, dict):
            entries = list(files.values())
        elif isinstance(files, list):
            entries = files
        else:
            entries = []
        if not entries or not isinstance(entries[0], dict):
            return None
        download_url = entries[0].get("readLink", "")
        if not download_url:
            return None

        # Step 2: download from presigned S3 URL (no auth header needed)
        for attempt, wait in enumerate(_RETRY_BACKOFF):
            try:
                r = requests.get(download_url, timeout=20)
                if r.status_code == 200:
                    return r.content
                logger.warning(
                    f"[fortnite] Chunk {chunk_id} returned {r.status_code}, "
                    f"retry {attempt + 1}/{_MAX_RETRIES}"
                )
            except requests.RequestException as e:
                logger.debug(f"[fortnite] Chunk download error: {e}")
            if attempt < _MAX_RETRIES - 1:
                time.sleep(wait)
        return None

    def _get(self, url: str, token: str) -> requests.Response | None:
        """Authenticated GET with retry."""
        headers = {"Authorization": f"Bearer {token}"}
        for attempt, wait in enumerate(_RETRY_BACKOFF):
            try:
                resp = self.session.get(url, headers=headers, timeout=15)
                if resp.status_code == 404:
                    return None  # not available; don't retry
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", wait))
                    logger.warning(f"[fortnite] Rate limited; sleeping {retry_after}s")
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt == _MAX_RETRIES - 1:
                    logger.error(f"[fortnite] GET failed after retries: {url} — {e}")
                    return None
                time.sleep(wait)
        return None

    # ------------------------------------------------------------------
    # Match ID management
    # ------------------------------------------------------------------

    def _get_match_ids(self) -> list[str]:
        """
        Return match IDs to fetch. Checks for a match_ids.json file in the
        data dir first (so new IDs can be added without code changes), then
        falls back to KNOWN_MATCH_IDS.
        """
        ids_file = self.raw_dir / "match_ids.json"
        if ids_file.exists():
            try:
                extra = json.loads(ids_file.read_text())
                if isinstance(extra, list):
                    combined = list(dict.fromkeys(KNOWN_MATCH_IDS + extra))
                    logger.info(
                        f"[fortnite] {len(combined)} match IDs "
                        f"({len(extra)} from match_ids.json)"
                    )
                    return combined
            except Exception:
                pass
        return KNOWN_MATCH_IDS

    # ------------------------------------------------------------------
    # Mock data
    # ------------------------------------------------------------------

    def generate_mock_data(self) -> list[EventStream]:
        """
        Generate mock Fortnite event streams.
        200 tournament matches at FNCS/Cash Cup distribution.
        Top-10 player actors per match; ~120 events per stream.
        """
        streams = []
        for i in range(self.config.sample_target):
            tier = "FNCS" if i < int(self.config.sample_target * 0.70) else "Cash_Cup"
            stream = self._make_mock_stream(
                game_id=f"mock_fortnite_{tier.lower()}_{i:04d}",
                cell="fortnite",
                n_events=120,
                event_types=FORTNITE_MOCK_EVENT_TYPES,
                actors=[f"player_{j}" for j in range(100)],
                seed=i,
            )
            stream.metadata.update({
                "tier": tier,
                "season": "Ch5S1",
                "region": ["NA_East", "NA_West", "Europe", "Brazil", "Asia"][i % 5],
            })
            streams.append(stream)
        logger.info(f"[fortnite] Generated {len(streams)} mock streams")
        return streams


# ------------------------------------------------------------------
# Binary event chunk parser (Unreal Engine replay Type-3 format)
# ------------------------------------------------------------------

def _parse_event_chunk(data: bytes) -> dict | None:
    """
    Parse a Type-3 event chunk from an Unreal Engine replay binary.

    Structure:
        Id       (FString)
        Group    (FString)
        Metadata (FString) — often JSON
        Time1    (uint32 LE, ms)
        Time2    (uint32 LE, ms)
        [remaining bytes = event-specific binary payload]
    """
    try:
        offset = 0
        event_id, offset = _read_fstring(data, offset)
        group, offset = _read_fstring(data, offset)
        metadata_raw, offset = _read_fstring(data, offset)
        time1 = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        time2 = struct.unpack_from("<I", data, offset)[0]
        offset += 4

        metadata: dict = {}
        if metadata_raw:
            try:
                metadata = json.loads(metadata_raw)
            except Exception:
                metadata = {"raw": metadata_raw}

        return {
            "id": event_id,
            "group": group,
            "metadata": metadata,
            "time1_ms": time1,
            "time2_ms": time2,
        }
    except Exception as e:
        logger.debug(f"[fortnite] Event chunk parse error: {e}")
        return None


def _read_fstring(data: bytes, offset: int) -> tuple[str, int]:
    """Read an Unreal Engine FString (length-prefixed)."""
    if offset + 4 > len(data):
        return "", offset
    length = struct.unpack_from("<i", data, offset)[0]
    offset += 4
    if length == 0:
        return "", offset
    if length < 0:
        # UTF-16 LE
        byte_count = (-length) * 2
        if offset + byte_count > len(data):
            return "", offset
        s = data[offset: offset + byte_count].decode("utf-16-le", errors="replace")
        s = s.rstrip("\x00")
        return s, offset + byte_count
    else:
        # ASCII / Latin-1
        if offset + length > len(data):
            return "", offset
        s = data[offset: offset + length].decode("latin-1", errors="replace")
        s = s.rstrip("\x00")
        return s, offset + length


def _normalize_match_id(match_id: str) -> str:
    """Strip hyphens → 32-char hex string."""
    return match_id.replace("-", "").lower()


# Fortnite-realistic event types for mock data
FORTNITE_MOCK_EVENT_TYPES = [
    "rotation_commit",
    "zone_enter",
    "zone_exit",
    "engage_decision",
    "disengage_decision",
    "resource_gain",
    "resource_spend",
    "resource_budget",
    "build_decision",
    "position_commit",
    "ability_use",
    "item_use",
    "target_select",
    "risk_accept",
    "risk_reject",
    "timing_commit",
    "objective_contest",
    "team_coordinate",
    "strategy_adapt",
]
