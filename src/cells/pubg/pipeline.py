"""
PUBG data acquisition pipeline.

Replaces the Fortnite cell after Epic locked down chunk-CDN access. PUBG offers
a documented public API at https://api.pubg.com with no rate limit on the
/matches and telemetry endpoints — exactly the endpoints we need.

Flow:
    1. Discover match IDs via /shards/<shard>/samples (or override file)
    2. For each match: GET /shards/<shard>/matches/<id>
    3. Parse `included[]` for the telemetry asset URL
    4. Download gzipped JSON telemetry from the CDN
    5. Save match_attrs + telemetry to data/raw/pubg/<id>.json

Auth: PUBG_API_KEY in environment or .env file.
Idempotent: skips matches whose JSON files already exist.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
from pathlib import Path

import requests

from ...common.config import CellConfig
from ...common.schema import EventStream
from ..base_pipeline import BasePipeline
from .extractor import PUBG_MOCK_EVENT_TYPES, PUBGExtractor

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.pubg.com"
_DEFAULT_SHARD = "steam"

# /matches and telemetry endpoints are unrestricted (per PUBG docs); we still
# add a small polite delay between match fetches.
_PER_MATCH_DELAY_S = 0.3
_MAX_RETRIES = 4
_RETRY_BACKOFF = [1, 2, 4, 8]


class PUBGPipeline(BasePipeline):
    """
    PUBG pipeline: fetch sample matches → download telemetry → extract events.

    fetch()          → saves one .json file per match (match_attrs + telemetry)
    parse()          → loads each saved .json into a structured dict
    extract_events() → PUBGExtractor.extract() per record
    """

    def __init__(self, config: CellConfig, data_root: Path | None = None):
        super().__init__(config, data_root or Path(__file__).parent.parent.parent.parent / "data")
        self.extractor = PUBGExtractor()
        self.shard = config.extra.get("shard", _DEFAULT_SHARD)
        self._api_key = self._load_api_key()
        self.session = requests.Session()
        if self._api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/vnd.api+json",
            })

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def fetch(self) -> list[Path]:
        if not self._api_key:
            logger.warning(
                "[pubg] PUBG_API_KEY not set; skipping real fetch. "
                "Set it in .env or env var to enable real data acquisition."
            )
            return []

        match_ids = self._get_match_ids()
        if not match_ids:
            logger.warning("[pubg] No match IDs to fetch")
            return []

        logger.info(
            f"[pubg] Will fetch up to {self.config.sample_target} of "
            f"{len(match_ids)} candidate match IDs"
        )

        paths: list[Path] = []
        n_attempted = 0
        for match_id in match_ids:
            if n_attempted >= self.config.sample_target:
                break
            n_attempted += 1

            out_path = self.raw_dir / f"{match_id}.json"
            if out_path.exists():
                logger.info(f"[pubg] {match_id}: already downloaded, skipping")
                paths.append(out_path)
                continue

            logger.info(f"[pubg] Fetching match {match_id}")
            record = self._fetch_match(match_id)
            if record is None:
                continue

            # Persist
            with open(out_path, "w") as f:
                json.dump(record, f)
            paths.append(out_path)
            logger.info(
                f"[pubg] {match_id}: saved "
                f"{len(record.get('telemetry', []))} telemetry events "
                f"({record.get('match_attrs', {}).get('gameMode')}, "
                f"{record.get('match_attrs', {}).get('duration')}s)"
            )

            time.sleep(_PER_MATCH_DELAY_S)

        return paths

    def parse(self, raw_paths: list[Path]) -> list[dict]:
        records: list[dict] = []
        for path in raw_paths:
            try:
                with open(path) as f:
                    records.append(json.load(f))
            except Exception as e:
                logger.warning(f"[pubg] Failed to load {path.name}: {e}")
        return records

    def extract_events(self, game_records: list[dict]) -> list[EventStream]:
        return [self.extractor.extract(r) for r in game_records if r]

    # ------------------------------------------------------------------
    # PUBG API helpers
    # ------------------------------------------------------------------

    def _fetch_match(self, match_id: str) -> dict | None:
        """Fetch one match's metadata + telemetry. Returns None on failure."""
        match_url = f"{_BASE_URL}/shards/{self.shard}/matches/{match_id}"
        match_resp = self._get(match_url)
        if match_resp is None:
            return None

        try:
            match_body = match_resp.json()
        except Exception as e:
            logger.warning(f"[pubg] {match_id}: bad JSON from /matches: {e}")
            return None

        attrs = match_body.get("data", {}).get("attributes", {})

        # Find telemetry asset URL
        telemetry_url = self._find_telemetry_url(match_body)
        if not telemetry_url:
            logger.warning(f"[pubg] {match_id}: no telemetry asset in included[]")
            return None

        # Download telemetry (no auth header needed for CDN)
        try:
            tel_resp = requests.get(telemetry_url, timeout=120)
            if tel_resp.status_code != 200:
                logger.warning(
                    f"[pubg] {match_id}: telemetry CDN returned {tel_resp.status_code}"
                )
                return None
            raw = tel_resp.content
        except requests.RequestException as e:
            logger.warning(f"[pubg] {match_id}: telemetry request failed: {e}")
            return None

        # Decompress (Krafton typically serves gzip; sometimes raw)
        try:
            decompressed = gzip.decompress(raw)
        except (gzip.BadGzipFile, OSError):
            decompressed = raw

        try:
            telemetry = json.loads(decompressed)
        except Exception as e:
            logger.warning(f"[pubg] {match_id}: telemetry JSON decode failed: {e}")
            return None

        if not isinstance(telemetry, list):
            logger.warning(
                f"[pubg] {match_id}: unexpected telemetry shape {type(telemetry).__name__}"
            )
            return None

        return {
            "match_id": match_id,
            "match_attrs": attrs,
            "telemetry": telemetry,
        }

    @staticmethod
    def _find_telemetry_url(match_body: dict) -> str:
        """Walk the JSON:API `included[]` looking for the telemetry asset URL."""
        for item in match_body.get("included", []):
            if item.get("type") != "asset":
                continue
            attrs = item.get("attributes", {}) or {}
            name = (attrs.get("name") or "").lower()
            if name == "telemetry":
                return attrs.get("URL", "")
        # Fallback: first asset URL we can find
        for item in match_body.get("included", []):
            if item.get("type") == "asset":
                url = (item.get("attributes") or {}).get("URL")
                if url:
                    return url
        return ""

    def _get(self, url: str) -> requests.Response | None:
        for attempt, wait in enumerate(_RETRY_BACKOFF):
            try:
                resp = self.session.get(url, timeout=20)
                if resp.status_code == 404:
                    logger.warning(f"[pubg] 404 from {url}")
                    return None
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", wait))
                    logger.warning(f"[pubg] 429 rate-limited; sleep {retry_after}s")
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt == _MAX_RETRIES - 1:
                    logger.error(f"[pubg] GET failed after retries: {url} — {e}")
                    return None
                time.sleep(wait)
        return None

    # ------------------------------------------------------------------
    # Match ID discovery
    # ------------------------------------------------------------------

    def _get_match_ids(self) -> list[str]:
        """
        Return list of match IDs to fetch. Priority:
          1. data/raw/pubg/match_ids.json (manual override / curated tournament IDs)
          2. /shards/<shard>/samples (PUBG-provided rolling sample of public matches)
        """
        override_path = self.raw_dir / "match_ids.json"
        if override_path.exists():
            try:
                ids = json.loads(override_path.read_text())
                if isinstance(ids, list) and ids:
                    logger.info(f"[pubg] Using {len(ids)} curated IDs from {override_path.name}")
                    return [str(x) for x in ids]
            except Exception as e:
                logger.warning(f"[pubg] Could not parse {override_path}: {e}")

        # Fallback: PUBG /samples endpoint
        url = f"{_BASE_URL}/shards/{self.shard}/samples"
        resp = self._get(url)
        if resp is None:
            return []
        try:
            data = resp.json()
        except Exception:
            return []
        rels = data.get("data", {}).get("relationships", {}).get("matches", {}).get("data", [])
        ids = [m["id"] for m in rels if m.get("type") == "match"]
        logger.info(f"[pubg] Discovered {len(ids)} sample match IDs from {self.shard}/samples")
        return ids

    # ------------------------------------------------------------------
    # API key loading: env var first, then .env file (no python-dotenv dep)
    # ------------------------------------------------------------------

    def _load_api_key(self) -> str:
        from_env = os.environ.get("PUBG_API_KEY", "").strip()
        if from_env:
            return from_env
        # Walk up from cwd / file location to find .env
        for base in (Path.cwd(), Path(__file__).resolve().parent.parent.parent.parent):
            env_path = base / ".env"
            if env_path.exists():
                key = _read_dotenv(env_path).get("PUBG_API_KEY", "").strip()
                if key:
                    return key
        return ""

    # ------------------------------------------------------------------
    # Mock data
    # ------------------------------------------------------------------

    def generate_mock_data(self) -> list[EventStream]:
        """
        Generate deterministic mock streams for harness testing.
        Sized at ~250 events/match × sample_target matches → enough to
        reach the 1,200-chain target with chain_length=8 and 50% retention.
        """
        streams = []
        for i in range(self.config.sample_target):
            game_mode = "squad-fpp" if i % 2 == 0 else "solo-fpp"
            stream = self._make_mock_stream(
                game_id=f"mock_pubg_{game_mode}_{i:04d}",
                cell="pubg",
                n_events=250,
                event_types=PUBG_MOCK_EVENT_TYPES,
                actors=[f"player_{j}" for j in range(100)],
                seed=i,
            )
            stream.metadata.update({
                "game_mode": game_mode,
                "map": ["Erangel_Main", "Miramar_Main", "Sanhok_Main",
                        "Tiger_Main", "Vikendi_Main"][i % 5],
                "match_type": "official",
            })
            streams.append(stream)
        logger.info(f"[pubg] Generated {len(streams)} mock streams")
        return streams


# ----------------------------------------------------------------------
# Local helpers
# ----------------------------------------------------------------------


def _read_dotenv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out
