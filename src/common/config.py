"""
Configuration loading for v5. Reads cells.yaml and harness.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


@dataclass
class CellConfig:
    cell_id: str
    display_name: str
    data_source: str
    sample_target: int          # matches / games / replays / maps
    time_range_start: str
    time_range_end: str
    env_vars: list[str]
    mock_fallback: bool
    stratification: list[dict]
    extra: dict = field(default_factory=dict)

    def env_satisfied(self) -> bool:
        return all(os.getenv(v) for v in self.env_vars)

    def should_use_mock(self) -> bool:
        return self.mock_fallback and not self.env_satisfied()


@dataclass
class HarnessConfig:
    alpha: float = 0.05
    bonferroni_divisor: int = 5
    continuity_correction: bool = True
    two_sided: bool = True
    min_discordant_pairs: int = 10
    bootstrap_iterations: int = 10000
    bootstrap_seed: int = 42
    confidence_level: float = 0.95
    gate2_retention_floor: float = 0.50
    cell_reporting: bool = True
    aggregate_reporting: bool = True


def load_cell_configs() -> dict[str, CellConfig]:
    if not _YAML_AVAILABLE:
        return _default_cell_configs()
    path = CONFIG_DIR / "cells.yaml"
    if not path.exists():
        return _default_cell_configs()
    with open(path) as f:
        raw = yaml.safe_load(f)
    configs = {}
    for cell_id, data in raw.get("cells", {}).items():
        tr = data.get("time_range", {})
        # Determine sample target key (varies by domain)
        sample_target = (
            data.get("sample_target_matches")
            or data.get("sample_target_games")
            or data.get("sample_target_replays")
            or data.get("sample_target_maps")
            or 200
        )
        configs[cell_id] = CellConfig(
            cell_id=cell_id,
            display_name=data.get("display_name", cell_id),
            data_source=data.get("data_source", "unknown"),
            sample_target=sample_target,
            time_range_start=tr.get("start", "2024-01-01"),
            time_range_end=tr.get("end", "2024-12-31"),
            env_vars=data.get("env_vars", []),
            mock_fallback=data.get("mock_fallback", True),
            stratification=data.get("stratification", []),
            extra=data,
        )
    return configs


def load_harness_config() -> HarnessConfig:
    if not _YAML_AVAILABLE:
        return HarnessConfig()
    path = CONFIG_DIR / "harness.yaml"
    if not path.exists():
        return HarnessConfig()
    with open(path) as f:
        raw = yaml.safe_load(f)
    h = raw.get("harness", {})
    mcn = h.get("mcnemar", {})
    var = h.get("variance", {})
    g2 = h.get("gate2", {})
    return HarnessConfig(
        alpha=mcn.get("alpha", 0.05),
        bonferroni_divisor=mcn.get("bonferroni_divisor", 5),
        continuity_correction=mcn.get("continuity_correction", True),
        two_sided=mcn.get("two_sided", True),
        min_discordant_pairs=mcn.get("min_discordant_pairs", 10),
        bootstrap_iterations=var.get("bootstrap_iterations", 10000),
        bootstrap_seed=var.get("bootstrap_seed", 42),
        confidence_level=var.get("confidence_level", 0.95),
        gate2_retention_floor=g2.get("retention_floor", 0.50),
        cell_reporting=h.get("reporting", {}).get("cell_level", True),
        aggregate_reporting=h.get("reporting", {}).get("aggregate", True),
    )


def _default_cell_configs() -> dict[str, CellConfig]:
    # Per A2 (D-35): pubg is the active battle-royale cell; fortnite kept
    # as a legacy fixture. Per A3 (D-37): poker sample target is 3,500.
    defaults = {
        "pubg": (25, ["PUBG_API_KEY"]),
        "nba": (300, []),
        "csgo": (150, ["FACEIT_API_KEY"]),
        "rocket_league": (250, ["BALLCHASING_TOKEN"]),
        "poker": (3500, []),
        "fortnite": (200, ["EPIC_ACCOUNT_ID", "EPIC_ACCESS_TOKEN"]),  # legacy
    }
    return {
        cell_id: CellConfig(
            cell_id=cell_id, display_name=cell_id,
            data_source="unknown", sample_target=target,
            time_range_start="2024-01-01", time_range_end="2024-12-31",
            env_vars=env_vars, mock_fallback=True, stratification=[],
        )
        for cell_id, (target, env_vars) in defaults.items()
    }
