"""
v5 pilot validation entry point.

Runs mock data through all five pipelines and validates:
  - Event stream generation
  - Gate 2 retention rate (>= 50% floor)
  - Distributional sanity checks

Usage:
  cd /path/to/Ditto-V5
  python run_pilot.py [--cells fortnite nba csgo rocket_league poker]
  python run_pilot.py --output RESULTS/pilot_report.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_pilot")

ALL_CELLS = ["fortnite", "nba", "csgo", "rocket_league", "poker"]


def run_pilot(cells: list[str], output_path: Path | None = None) -> bool:
    from src.common.config import load_cell_configs, load_harness_config
    from src.interfaces.translation import DOMAIN_T_STUBS
    from src.pilot.validator import PilotValidator

    cell_configs = load_cell_configs()
    harness_config = load_harness_config()

    # Import pipelines
    from src.cells.csgo.pipeline import CSGOPipeline
    from src.cells.fortnite.pipeline import FortnitePipeline
    from src.cells.nba.pipeline import NBAPipeline
    from src.cells.poker.pipeline import PokerPipeline
    from src.cells.rocket_league.pipeline import RocketLeaguePipeline

    pipeline_map = {
        "fortnite": FortnitePipeline,
        "nba": NBAPipeline,
        "csgo": CSGOPipeline,
        "rocket_league": RocketLeaguePipeline,
        "poker": PokerPipeline,
    }

    validator = PilotValidator(gate2_floor=harness_config.gate2_retention_floor)
    streams_by_cell: dict = {}

    for cell in cells:
        if cell not in pipeline_map:
            logger.warning(f"Unknown cell '{cell}', skipping")
            continue

        logger.info(f"Running pipeline for cell: {cell}")
        config = cell_configs.get(cell)
        if config is None:
            logger.error(f"No config found for cell '{cell}'")
            continue

        PipelineClass = pipeline_map[cell]
        pipeline = PipelineClass(config=config)

        # Always use mock data in pilot (no real API calls)
        streams = pipeline.run(force_mock=True)
        streams_by_cell[cell] = streams
        logger.info(f"  {cell}: {len(streams)} streams, "
                    f"{sum(len(s) for s in streams)} events")

        # Phase B complete — register real T for each cell
        t_fn = DOMAIN_T_STUBS.get(cell)
        if t_fn is None:
            logger.error(f"No T registered for cell '{cell}'")
            continue
        validator.register_cell(cell, t_fn)

    logger.info("Running pilot validation with real T (Phase B)...")
    report = validator.run(streams_by_cell, sample_size=75)
    report.print_summary()

    if output_path:
        report.save(output_path)
        logger.info(f"Report saved to {output_path}")

    return report.all_passed


def main():
    parser = argparse.ArgumentParser(description="v5 Pilot Validation")
    parser.add_argument(
        "--cells", nargs="+", default=ALL_CELLS,
        choices=ALL_CELLS, help="Cells to validate"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to save JSON report (optional)"
    )
    args = parser.parse_args()

    success = run_pilot(cells=args.cells, output_path=args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
