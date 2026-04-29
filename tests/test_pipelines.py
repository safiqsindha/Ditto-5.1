"""
Tests for v5 data acquisition pipelines (mock paths only — no network).
"""

import pytest
from src.cells.csgo.pipeline import CSGOPipeline
from src.cells.fortnite.pipeline import FortnitePipeline  # legacy fixture
from src.cells.nba.pipeline import NBAPipeline
from src.cells.poker.pipeline import PokerPipeline
from src.cells.pubg.pipeline import PUBGPipeline
from src.cells.rocket_league.pipeline import RocketLeaguePipeline
from src.common.config import load_cell_configs
from src.common.schema import EventStream, GameEvent

# Per A2 (D-35): pubg replaces fortnite as the active battle-royale cell.
# fortnite kept here so legacy tests still resolve a class, but parametrize
# only over the 5 active cells (which is what cells.yaml exposes).
PIPELINE_CLASSES = {
    "pubg": PUBGPipeline,
    "nba": NBAPipeline,
    "csgo": CSGOPipeline,
    "rocket_league": RocketLeaguePipeline,
    "poker": PokerPipeline,
    "fortnite": FortnitePipeline,
}
ACTIVE_CELLS = ["pubg", "nba", "csgo", "rocket_league", "poker"]


@pytest.fixture(scope="module")
def configs():
    return load_cell_configs()


@pytest.mark.parametrize("cell", ACTIVE_CELLS)
class TestPipelineMockPath:
    def test_pipeline_constructible(self, configs, cell, tmp_path):
        config = configs[cell]
        pipeline = PIPELINE_CLASSES[cell](config=config, data_root=tmp_path)
        assert pipeline.cell == cell

    def test_generate_mock_data_produces_streams(self, configs, cell, tmp_path):
        config = configs[cell]
        pipeline = PIPELINE_CLASSES[cell](config=config, data_root=tmp_path)
        streams = pipeline.generate_mock_data()
        assert len(streams) > 0
        assert all(isinstance(s, EventStream) for s in streams)
        assert all(s.cell == cell for s in streams)

    def test_mock_streams_have_events(self, configs, cell, tmp_path):
        config = configs[cell]
        pipeline = PIPELINE_CLASSES[cell](config=config, data_root=tmp_path)
        streams = pipeline.generate_mock_data()
        # Sample a few streams and check they have events
        for s in streams[:3]:
            assert len(s) > 0
            assert all(isinstance(e, GameEvent) for e in s.events)
            assert all(e.cell == cell for e in s.events)
            assert all(e.metadata.get("mock") is True for e in s.events)

    def test_mock_run_persists_streams(self, configs, cell, tmp_path):
        """Verify pipeline.run(force_mock=True) writes streams to disk."""
        config = configs[cell]
        pipeline = PIPELINE_CLASSES[cell](config=config, data_root=tmp_path)
        streams = pipeline.run(force_mock=True)
        # Persisted files
        events_dir = tmp_path / "events" / cell
        files = list(events_dir.glob("*.jsonl"))
        assert len(files) == len(streams)
        # Spot-check loadable
        loaded = EventStream.from_jsonl(files[0])
        assert loaded.cell == cell
        assert len(loaded) > 0


def test_force_mock_overrides_credentials(configs, tmp_path, monkeypatch):
    """Even if credentials are set, force_mock=True must use mock data."""
    # Per A2 (D-35): use pubg in place of the legacy fortnite fixture.
    monkeypatch.setenv("PUBG_API_KEY", "fake")
    config = configs["pubg"]
    pipeline = PUBGPipeline(config=config, data_root=tmp_path)
    streams = pipeline.run(force_mock=True)
    assert all(s.metadata.get("mock") is True for s in streams)


def test_mock_fallback_when_credentials_missing(configs, tmp_path, monkeypatch):
    """Pipelines with required env vars should fall back to mock when missing."""
    monkeypatch.delenv("PUBG_API_KEY", raising=False)
    config = configs["pubg"]
    pipeline = PUBGPipeline(config=config, data_root=tmp_path)
    assert config.should_use_mock()
    streams = pipeline.run()
    assert all(s.metadata.get("mock") is True for s in streams)
