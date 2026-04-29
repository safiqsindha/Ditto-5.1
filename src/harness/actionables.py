"""
ACTIONABLE_TYPES whitelist for v5 — v1.1 amendments applied.

Amendments vs v1.0:
  1. ResourceBudget included in actionable types.
  2. phase_ prefix is stripped from event_type before comparison
     (handled in GameEvent.__post_init__, duplicated here for clarity).

Gate 2: a chain passes Gate 2 if the fraction of its events that are
actionable meets or exceeds the retention floor (default 0.50).
"""

from __future__ import annotations

from ..common.schema import ChainCandidate, GameEvent

# ---------------------------------------------------------------------------
# Core ACTIONABLE_TYPES whitelist (v1.1)
# ---------------------------------------------------------------------------
# These are the normalized event_type strings (phase_ prefix already stripped)
# that qualify an event as "actionable" — i.e., the model is expected to
# make a meaningful constraint-respecting decision in response to the event.
#
# The whitelist is domain-agnostic at the harness level. Per-cell T
# implementations may narrow or extend this list via the cell_overrides
# mechanism; any such change must be pre-registered.
# ---------------------------------------------------------------------------

ACTIONABLE_TYPES: frozenset[str] = frozenset([
    # Resource and economy events (v1.1: ResourceBudget added)
    "resource_gain",
    "resource_spend",
    "resource_budget",          # v1.1 amendment
    "resource_trade",
    "resource_depleted",

    # Position and movement decisions
    "position_commit",
    "zone_enter",
    "zone_exit",
    "route_decision",
    "rotation_commit",

    # Engagement / combat decisions
    "engage_decision",
    "disengage_decision",
    "target_select",
    "ability_use",
    "item_use",

    # Strategic / macro decisions
    "objective_contest",
    "objective_capture",
    "objective_abandon",
    "team_coordinate",
    "draft_pick",               # Hearthstone deck/card play
    "draft_ban",

    # State transitions that represent agent decisions
    "strategy_adapt",
    "risk_accept",
    "risk_reject",
    "concede",

    # Temporal / phase decisions
    "timing_commit",
    "delay_action",
    "force_action",
])

# Per-cell type overrides: additional domain-specific types that qualify
# as actionable. These MUST be pre-registered before use in real evaluation.
# Populated here as empty dicts so the interface is defined.

# Phase B locked 2026-04-27 (both authors). See DECISION_LOG D-24 through D-28.
CELL_ACTIONABLE_OVERRIDES: dict[str, frozenset[str]] = {
    "fortnite": frozenset([         # F-6 locked — legacy fixture, superseded by pubg per A2
        "storm_rotation",           # storm-zone rotation decision
        "build_decision",           # build structure (consumes material)
    ]),
    "pubg": frozenset([],),         # active battle-royale cell per A2 (D-35)
                                    # PUBG extractor maps to base whitelist types
                                    # (engage_decision, zone_enter, position_commit,
                                    # resource_gain/spend, risk_accept) — no
                                    # PUBG-specific event types need to be added.
    "nba": frozenset([              # N-6 locked
        "shot_selection",           # shot attempt decision
        "clutch_decision",          # high-leverage end-game decision
    ]),
    "csgo": frozenset([             # C-6 locked
        "buy_decision",             # pre-round equipment purchase
        "utility_deploy",           # grenade / utility deployment decision
        "bombsite_commit",          # bomb plant or defuse commitment
    ]),
    "rocket_league": frozenset([    # R-6 locked
        "aerial_commit",            # aerial approach commitment
        "boost_steal",              # opponent boost pad denial
        "rotation_back",            # defensive rotation decision
    ]),
    "poker": frozenset([             # P-6 locked
        "pot_odds_decision",        # EV-driven fold / call / raise
        "stack_pressure",           # short-stack push / fold constraint
        "hand_strength_commitment", # equity-based commitment to pot
    ]),
}


def is_actionable(event: GameEvent, cell: str | None = None) -> bool:
    """Return True if this event qualifies as actionable under v1.1 rules."""
    etype = event.event_type  # phase_ already stripped by GameEvent.__post_init__
    if etype in ACTIONABLE_TYPES:
        return True
    if cell and cell in CELL_ACTIONABLE_OVERRIDES:
        if etype in CELL_ACTIONABLE_OVERRIDES[cell]:
            return True
    return False


def gate2_check(chain: ChainCandidate, floor: float = 0.50) -> bool:
    """
    Gate 2: True if actionable-event fraction >= floor.
    Applied to ChainCandidate.events list.
    """
    if not chain.events:
        return False
    actionable_count = sum(1 for e in chain.events if is_actionable(e, chain.cell))
    retention_rate = actionable_count / len(chain.events)
    chain.chain_metadata["gate2_actionable_fraction"] = retention_rate
    chain.is_actionable = retention_rate >= floor
    return chain.is_actionable


def compute_retention_rate(chains: list[ChainCandidate], floor: float = 0.50) -> dict:
    """
    Batch Gate 2 check. Returns summary statistics.
    """
    if not chains:
        return {"n_total": 0, "n_passed": 0, "retention_rate": 0.0, "floor": floor,
                "gate2_pass": False}
    n_passed = sum(1 for c in chains if gate2_check(c, floor))
    retention_rate = n_passed / len(chains)
    return {
        "n_total": len(chains),
        "n_passed": n_passed,
        "retention_rate": retention_rate,
        "floor": floor,
        "gate2_pass": retention_rate >= floor,
    }
