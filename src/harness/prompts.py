"""
Evaluation prompt template + response parser for v5.

Per Q1 sign-off (D-16): two separate API calls per chain — one baseline (no
constraint context), one intervention (with constraint context). This module
defines the prompt template that turns a ChainCandidate into the user-message
string sent to the subject model.

PER-CELL PROMPT CONTENT IS DELIBERATELY GENERIC HERE.
The cell-specific phrasing (constraint description, prediction question) requires
T-design context and lands in Phase B (joint authoring). Per-cell PromptBuilder
subclasses can override `format_constraint_context()` and `format_question()`
once T is designed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from ..common.schema import ChainCandidate, GameEvent

logger = logging.getLogger(__name__)


@dataclass
class PromptPair:
    """The two prompts (baseline + intervention) for one chain."""
    chain_id: str
    cell: str
    baseline_prompt: str
    intervention_prompt: str
    metadata: dict


class PromptBuilder:
    """
    Builds baseline + intervention prompts for one cell.

    Subclasses override per-cell hooks:
      - format_constraint_context(chain) → str
      - format_question(chain) → str
      - format_event(event, idx) → str (optional)
    """

    def __init__(self, cell: str):
        self.cell = cell

    # --- Public entry point -------------------------------------------------
    def build(self, chain: ChainCandidate) -> PromptPair:
        if chain.cell != self.cell:
            raise ValueError(
                f"PromptBuilder for cell '{self.cell}' got chain for cell '{chain.cell}'"
            )

        chain_block = self.format_chain(chain)
        question = self.format_question(chain)
        constraint_block = self.format_constraint_context(chain)

        baseline_prompt = self._compose(
            chain_block=chain_block,
            constraint_block=None,
            question=question,
        )
        intervention_prompt = self._compose(
            chain_block=chain_block,
            constraint_block=constraint_block,
            question=question,
        )

        return PromptPair(
            chain_id=chain.chain_id,
            cell=self.cell,
            baseline_prompt=baseline_prompt,
            intervention_prompt=intervention_prompt,
            metadata={
                "n_events": len(chain.events),
                "chain_length": len(chain.events),
                "actor_count": len({e.actor for e in chain.events}),
            },
        )

    # --- Composition --------------------------------------------------------
    @staticmethod
    def _compose(chain_block: str, constraint_block: str | None, question: str) -> str:
        parts: list[str] = []
        if constraint_block:
            parts.append("## Constraint Context\n" + constraint_block.strip())
        parts.append("## Event Chain\n" + chain_block.strip())
        parts.append("## Question\n" + question.strip())
        return "\n\n".join(parts)

    # --- Per-cell hooks (override in subclasses) ----------------------------
    def format_chain(self, chain: ChainCandidate) -> str:
        """Default: render each event on its own line. Subclasses can prettify."""
        actor_map = _build_actor_map(chain.events)
        lines = []
        for i, ev in enumerate(chain.events):
            lines.append(self.format_event(ev, i, actor_map=actor_map))
        return "\n".join(lines)

    def format_event(self, event: GameEvent, idx: int, actor_map: dict | None = None) -> str:
        """
        Default: 'idx | t=TIME | TYPE by Player_N (context_summary)'.

        CF-4=B: actor is anonymised to 'Player_N' using the chain-local actor_map
        built in format_chain(). This prevents player/team identity leaking into
        prompts and ensures the model reasons from constraint rules alone.
        """
        anon = actor_map.get(event.actor, "Player_?") if actor_map else "Player_?"
        ctx_summary = self._summarize_context(event.location_context)
        return (
            f"{idx:>3}. t={event.timestamp:>7.1f}s | "
            f"{event.event_type:<22} | actor={anon:<12} "
            f"| {ctx_summary}"
        )

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        """
        TBD per cell — pre-registered at T-design time (Phase B).

        Default placeholder explains why this is unset.
        """
        return (
            f"[CONSTRAINT CONTEXT FOR {self.cell.upper()} — TO BE DEFINED AT T-DESIGN]\n"
            "Per SPEC §6 (T design), the constraint context language is locked "
            "during the T-design joint authoring session. This default is a "
            "placeholder so prompt construction can be tested infrastructure-side."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        """
        TBD per cell — pre-registered at T-design time (Phase B).
        """
        return (
            "[QUESTION TBD AT T-DESIGN]\n"
            "Predict the actor or event-type that follows. "
            "Reply with a single token only."
        )

    @staticmethod
    def _summarize_context(ctx: dict) -> str:
        if not ctx:
            return ""
        # Cap at 6 keys, 80 chars total to keep prompt compact
        items = list(ctx.items())[:6]
        s = ", ".join(f"{k}={_short(v)}" for k, v in items)
        return s[:120]


def _short(v) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    s = str(v)
    return s if len(s) <= 32 else s[:29] + "..."


def _build_actor_map(events: list) -> dict[str, str]:
    """
    CF-4=B anonymisation: map each unique actor ID to a stable 'Player_N' slot.

    Order is first-appearance within the chain, ensuring the same actor always
    gets the same slot across all events in the chain. The mapping is chain-local
    so different chains do not share slot assignments.
    """
    seen: dict[str, str] = {}
    for ev in events:
        if ev.actor not in seen:
            seen[ev.actor] = f"Player_{len(seen)}"
    return seen


# ---------------------------------------------------------------------------
# Per-cell PromptBuilder stubs.
# These hold the per-cell hooks; cell-specific prompt content is filled in
# at T-design time (Phase B).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-cell PromptBuilders — constraint context locked in Phase B 2026-04-27.
# CF-1=A (natural language), CF-2=D (binary classify), CF-4=B (domain only).
# ---------------------------------------------------------------------------

_CLASSIFY_QUESTION = (
    "Is the sequence of events above consistent with the rules of {domain}?\n"
    "Reply with exactly one word: YES or NO."
)


class FortnitePromptBuilder(PromptBuilder):
    """F-5 constraint context locked Phase B."""

    def __init__(self):
        super().__init__(cell="fortnite")

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        return (
            "In Fortnite, players must remain within the safe storm zone or take "
            "damage over time. An eliminated player cannot generate further actions. "
            "Building structures consumes exactly one material per piece."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        return _CLASSIFY_QUESTION.format(domain="Fortnite")


class NBAPromptBuilder(PromptBuilder):
    """N-5 constraint context locked Phase B."""

    def __init__(self):
        super().__init__(cell="nba")

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        return (
            "In NBA basketball, the offensive team must attempt a shot within 24 "
            "seconds of gaining possession. Personal fouls accumulate; a player with "
            "six fouls is ejected and cannot return. Possession changes on made shots, "
            "turnovers, and defensive rebounds."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        return _CLASSIFY_QUESTION.format(domain="NBA basketball")


class CSGOPromptBuilder(PromptBuilder):
    """C-5 constraint context locked Phase B."""

    def __init__(self):
        super().__init__(cell="csgo")

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        return (
            "In Counter-Strike, each round one team plants and detonates the bomb "
            "(Terrorist win) or the other defuses it or eliminates all attackers "
            "(Counter-Terrorist win). Eliminated players do not respawn until the next "
            "round. The bomb may only be planted at designated sites A or B."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        return _CLASSIFY_QUESTION.format(domain="Counter-Strike")


class RocketLeaguePromptBuilder(PromptBuilder):
    """R-5 constraint context locked Phase B."""

    def __init__(self):
        super().__init__(cell="rocket_league")

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        return (
            "In Rocket League, teams of three score by hitting a ball into the opposing "
            "goal. Each car has a boost meter capped at 100 that depletes with use; "
            "boost is collected from pads on the field. A goal resets ball position to "
            "center and ends the current play."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        return _CLASSIFY_QUESTION.format(domain="Rocket League")


class PokerPromptBuilder(PromptBuilder):
    """P-5 constraint context locked Phase B."""

    def __init__(self):
        super().__init__(cell="poker")

    def format_constraint_context(self, chain: ChainCandidate) -> str:
        return (
            "In No-Limit Texas Hold'em poker, a player may fold, check (only if no "
            "bet has been made on this street), call the current bet, or bet/raise up "
            "to the full amount of their remaining chips (their stack). A player may "
            "not wager more chips than they hold. Blinds are posted before cards are "
            "dealt and count toward the first betting round. Action proceeds clockwise; "
            "on post-flop streets the first active player to the left of the dealer "
            "button acts first. A player who folds may not act again in the hand. At "
            "showdown the player with the best five-card hand wins the pot."
        )

    def format_question(self, chain: ChainCandidate) -> str:
        return _CLASSIFY_QUESTION.format(domain="poker")


PER_CELL_PROMPT_BUILDERS = {
    "fortnite": FortnitePromptBuilder,
    "nba": NBAPromptBuilder,
    "csgo": CSGOPromptBuilder,
    "rocket_league": RocketLeaguePromptBuilder,
    "poker": PokerPromptBuilder,
}


# ---------------------------------------------------------------------------
# Response parser — converts model text to a normalized prediction string.
# ---------------------------------------------------------------------------

# Common abstain phrases the model might emit
_ABSTAIN_PATTERNS = re.compile(
    r"^\s*(i\s*don'?t\s*know|abstain|n/?a|unknown|cannot\s*determine|insufficient|"
    r"no\s*answer|skip)\s*\.?\s*$",
    re.IGNORECASE,
)


def parse_model_response(raw: str, allowed_predictions: list[str] | None = None) -> str:
    """
    Parse a model response string into a normalized prediction.

    Returns:
      - "" if the response is empty / abstain (scoring layer treats this as -1)
      - First matched token if allowed_predictions is given and one matches
      - The first stripped non-empty line of the response otherwise

    Per Q1 sign-off (separate calls), parsing is shared between baseline and
    intervention responses.
    """
    if not raw or not raw.strip():
        return ""

    s = raw.strip()
    if _ABSTAIN_PATTERNS.match(s):
        return ""

    # If the model wrapped the answer in JSON, try to extract it
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for k in ("answer", "prediction", "response", "value"):
                    if k in obj:
                        return str(obj[k]).strip().lower()
        except json.JSONDecodeError:
            pass

    # First line, lowercased, stripped of trailing punctuation
    first_line = s.splitlines()[0].strip().rstrip(".!?,").lower()

    if allowed_predictions:
        # Try exact match first
        for pred in allowed_predictions:
            if first_line == pred.lower():
                return pred.lower()
        # Fall through to substring match
        for pred in allowed_predictions:
            if pred.lower() in first_line:
                return pred.lower()
        # No match → return empty (will score as abstain)
        logger.debug(f"Response '{first_line[:60]}' did not match any allowed prediction")
        return ""

    return first_line
