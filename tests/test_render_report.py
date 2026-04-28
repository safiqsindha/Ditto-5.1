"""
Tests for the pilot report markdown renderer.
"""

import json

from src.pilot.render_report import render

SAMPLE_REPORT = {
    "all_passed": True,
    "cells": [
        {
            "cell": "fortnite",
            "n_streams": 200,
            "n_events_total": 24000,
            "n_chains_raw": 8000,
            "n_chains_post_gate2": 8000,
            "retention_rate": 1.0,
            "gate2_floor": 0.5,
            "gate2_pass": True,
            "chain_length_mean": 5.0,
            "chain_length_min": 3,
            "chain_length_max": 5,
            "chain_length_median": 5.0,
            "actionable_frac_mean": 1.0,
            "actionable_frac_min": 1.0,
            "actionable_frac_max": 1.0,
            "event_type_distribution": {"engage_decision": 2256, "rotation_commit": 2253},
            "n_sample_chains": 50,
            "warnings": [],
            "errors": [],
            "passed": True,
        }
    ],
}


def test_render_basic():
    md = render(SAMPLE_REPORT)
    assert "# v5 Pilot Validation Report" in md
    assert "ALL PASS" in md
    assert "fortnite" in md
    assert "100.0%" in md
    assert "engage_decision" in md
    assert "2,256" in md  # comma-formatted count


def test_render_with_failure():
    failed = json.loads(json.dumps(SAMPLE_REPORT))  # deep copy
    failed["all_passed"] = False
    failed["cells"][0]["passed"] = False
    failed["cells"][0]["gate2_pass"] = False
    failed["cells"][0]["errors"] = ["T not implemented"]
    md = render(failed)
    assert "SOME FAILURES" in md
    assert "FAIL" in md
    assert "T not implemented" in md


def test_render_with_warnings():
    warned = json.loads(json.dumps(SAMPLE_REPORT))
    warned["cells"][0]["warnings"] = ["Only 100 chains; target is 1200"]
    md = render(warned)
    assert "Warnings" in md
    assert "Only 100 chains" in md


def test_render_empty_cells():
    md = render({"all_passed": True, "cells": []})
    assert "# v5 Pilot Validation Report" in md
    assert "ALL PASS" in md
