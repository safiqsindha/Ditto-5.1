"""
Tests for v5 config loading.
"""


from src.common.config import (
    CellConfig,
    HarnessConfig,
    load_cell_configs,
    load_harness_config,
)


class TestCellConfig:
    def test_load_all_five_cells(self):
        # Per A2 (D-35): pubg replaces fortnite as the active battle-royale cell.
        configs = load_cell_configs()
        expected = {"pubg", "nba", "csgo", "rocket_league", "poker"}
        assert set(configs.keys()) == expected

    def test_each_cell_has_sample_target(self):
        configs = load_cell_configs()
        for cell_id, config in configs.items():
            assert config.sample_target > 0, f"{cell_id} has zero sample target"

    def test_each_cell_has_time_range(self):
        configs = load_cell_configs()
        for cell_id, config in configs.items():
            assert config.time_range_start, f"{cell_id} missing start"
            assert config.time_range_end, f"{cell_id} missing end"

    def test_env_satisfied_with_no_required_vars(self):
        config = CellConfig(
            cell_id="test", display_name="test", data_source="test",
            sample_target=10, time_range_start="2024-01-01",
            time_range_end="2024-12-31", env_vars=[], mock_fallback=True,
            stratification=[],
        )
        # all([]) is True — vacuous truth
        assert config.env_satisfied()
        # But mock_fallback path should still work; should_use_mock returns False
        assert not config.should_use_mock()

    def test_env_not_satisfied_with_missing_var(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR_FOO", raising=False)
        config = CellConfig(
            cell_id="test", display_name="test", data_source="test",
            sample_target=10, time_range_start="2024-01-01",
            time_range_end="2024-12-31", env_vars=["MISSING_VAR_FOO"],
            mock_fallback=True, stratification=[],
        )
        assert not config.env_satisfied()
        assert config.should_use_mock()

    def test_env_satisfied_with_set_var(self, monkeypatch):
        monkeypatch.setenv("PRESENT_VAR_FOO", "value")
        config = CellConfig(
            cell_id="test", display_name="test", data_source="test",
            sample_target=10, time_range_start="2024-01-01",
            time_range_end="2024-12-31", env_vars=["PRESENT_VAR_FOO"],
            mock_fallback=True, stratification=[],
        )
        assert config.env_satisfied()
        assert not config.should_use_mock()


class TestHarnessConfig:
    def test_load_default_harness(self):
        config = load_harness_config()
        assert isinstance(config, HarnessConfig)
        assert config.alpha == 0.05
        assert config.bonferroni_divisor == 5
        assert config.gate2_retention_floor == 0.50

    def test_default_construction(self):
        config = HarnessConfig()
        assert config.alpha == 0.05
        assert config.bonferroni_divisor == 5
        assert config.bootstrap_iterations == 10000
        assert config.bootstrap_seed == 42
