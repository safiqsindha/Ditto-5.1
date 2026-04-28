"""
Tests for CLI entry points (main() functions).

Each CLI is exercised via argparse using monkeypatched sys.argv. Captures
stdout via pytest's capsys fixture. No subprocess invocation needed.
"""

from __future__ import annotations

import json
import sys

import pytest

# Subjects under test
import check_config as check_config_module
import run_pilot as run_pilot_module
from scripts import benchmark_pilot as benchmark_module
from src.harness import cost_estimator as cost_estimator_module
from src.pilot import render_report as render_report_module


class TestCheckConfigCLI:
    def test_default_run(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["check_config", "--no-color"])
        with pytest.raises(SystemExit) as exc:
            check_config_module.main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "Configuration Sanity Check" in out
        assert "Bonferroni divisor" in out

    def test_strict_with_missing_credentials_exits_nonzero(self, monkeypatch, capsys):
        # Strict mode + no credentials → yellow → exit 1
        monkeypatch.delenv("EPIC_ACCOUNT_ID", raising=False)
        monkeypatch.delenv("EPIC_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("BALLCHASING_TOKEN", raising=False)
        monkeypatch.delenv("HSREPLAY_API_KEY", raising=False)
        monkeypatch.setattr(sys, "argv", ["check_config", "--no-color", "--strict"])
        with pytest.raises(SystemExit) as exc:
            check_config_module.main()
        assert exc.value.code == 1

    def test_single_cell(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["check_config", "--no-color", "--cell", "nba"])
        with pytest.raises(SystemExit) as exc:
            check_config_module.main()
        assert exc.value.code == 0
        out = capsys.readouterr().out
        # Only nba header should appear, not other cells
        assert "[nba]" in out
        assert "[fortnite]" not in out

    def test_unknown_cell_exits_2(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["check_config", "--no-color", "--cell", "bogus"])
        with pytest.raises(SystemExit) as exc:
            check_config_module.main()
        assert exc.value.code == 2

    def test_status_for_cell_red(self, monkeypatch):
        from src.common.config import CellConfig
        cfg = CellConfig(
            cell_id="x", display_name="x", data_source="x", sample_target=1,
            time_range_start="2024-01-01", time_range_end="2024-12-31",
            env_vars=["MISSING_VAR"], mock_fallback=False, stratification=[],
        )
        monkeypatch.delenv("MISSING_VAR", raising=False)
        color, light, status = check_config_module.status_for_cell(cfg)
        assert "🔴" in light or "blocked" in status


class TestRunPilotCLI:
    def test_run_pilot_main(self, monkeypatch, tmp_path):
        out = tmp_path / "report.json"
        monkeypatch.setattr(sys, "argv", [
            "run_pilot", "--cells", "nba", "--output", str(out)
        ])
        with pytest.raises(SystemExit) as exc:
            run_pilot_module.main()
        # Pilot passes → exit 0
        assert exc.value.code == 0
        assert out.exists()

    def test_run_pilot_function_returns_bool(self):
        success = run_pilot_module.run_pilot(cells=["nba"], output_path=None)
        assert isinstance(success, bool)


class TestCostEstimatorCLI:
    def test_default(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["cost_estimator"])
        cost_estimator_module.main()
        out = capsys.readouterr().out
        assert "Total cost:" in out
        assert "12,000" in out

    def test_custom_args(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", [
            "cost_estimator",
            "--chains", "500",
            "--cells", "3",
            "--calls", "1",
        ])
        cost_estimator_module.main()
        out = capsys.readouterr().out
        # 500 × 3 × 1 = 1,500 calls
        assert "1,500" in out


class TestRenderReportCLI:
    def test_main_creates_md(self, monkeypatch, tmp_path, capsys):
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "all_passed": True,
            "cells": [{
                "cell": "nba", "n_streams": 10, "n_events_total": 100,
                "n_chains_raw": 30, "n_chains_post_gate2": 25,
                "retention_rate": 0.83, "gate2_floor": 0.5, "gate2_pass": True,
                "chain_length_mean": 5, "chain_length_min": 5, "chain_length_max": 5,
                "chain_length_median": 5, "actionable_frac_mean": 1.0,
                "actionable_frac_min": 1.0, "actionable_frac_max": 1.0,
                "event_type_distribution": {"engage_decision": 50},
                "n_sample_chains": 10, "warnings": [], "errors": [],
                "passed": True,
            }],
        }))
        monkeypatch.setattr(sys, "argv", [
            "render_report", str(report)
        ])
        render_report_module.main()
        md = report.with_suffix(".md")
        assert md.exists()
        assert "v5 Pilot" in md.read_text()

    def test_main_missing_input_exits(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(sys, "argv", [
            "render_report", str(tmp_path / "nonexistent.json")
        ])
        with pytest.raises(SystemExit) as exc:
            render_report_module.main()
        assert exc.value.code == 1


class TestBenchmarkCLI:
    def test_main_one_cell(self, monkeypatch, tmp_path, capsys):
        out_md = tmp_path / "bench.md"
        out_json = tmp_path / "bench.json"
        monkeypatch.setattr(sys, "argv", [
            "benchmark_pilot",
            "--cells", "nba",
            "--output", str(out_md),
            "--json", str(out_json),
        ])
        benchmark_module.main()
        assert out_md.exists()
        assert out_json.exists()
        text = out_md.read_text()
        assert "v5 Pilot Performance Benchmark" in text
        # JSON should have structured results
        data = json.loads(out_json.read_text())
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["cell"] == "nba"
